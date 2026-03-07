"""
stage1_nlp.py
=============
Stage 1: NLP ingestion, graph construction, and ASHA field writeback.

Node types (7):  Patient, ASHA Worker, ANM Worker, PHC, WelfareScheme, Village, Contact
Edge types (8):  household_contact, workplace_contact, shared_contact,
                 assigned_to, attends, supervises, enrolled_in, social(stub)

Key changes from stage1_ingest.py:
  - UPSERT throughout — graph is never destroyed between runs
  - ANM nodes + supervises edges
  - PHC node replaces DOTS centre; attends edge replaces treats edge
  - WelfareScheme node + enrolled_in edges
  - Load score denominator: 15 (was 50)
  - Phase-adaptive silence: 5d intensive / 6d late continuation / 7d continuation
  - ASHA writeback functions close the feedback loop to Cosmos immediately:
      writeback_dose_confirmed, writeback_dose_missed,
      writeback_contact_screened, writeback_risk_scores,
      writeback_pagerank_scores, promote_contact_to_patient
  - inject_silence_events NOT called inside ingest_all (was causing double injection)
"""

import os, sys, json, asyncio
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ── Azure AI Language ─────────────────────────────────────────────────────────

def get_language_client():
    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential
        ep  = os.getenv("LANGUAGE_ENDPOINT")
        key = os.getenv("LANGUAGE_KEY")
        if not ep or not key:
            print("  [NER] LANGUAGE_ENDPOINT/KEY not set — NER disabled")
            return None
        return TextAnalyticsClient(endpoint=ep, credential=AzureKeyCredential(key))
    except ImportError:
        return None

SAMPLE_NOTES = {
    "NIK-100001": "Patient lives with wife Meena Devi (42) and son Karthik (16). Works at Tondiarpet tannery.",
    "NIK-100002": "Household: mother Savitri (68, diabetic) and brother Rajan (35).",
    "NIK-100003": "Wife Anitha (30) and daughter Priya (8) at home. Father Murugan (72) also present.",
    "NIK-100004": "Gave dose today. Wife has been coughing for 3 days. Patient reluctant to continue.",
}

def extract_contacts_from_note(lc, note: str) -> list:
    if lc is None:
        return []
    result = lc.recognize_entities([note])[0]
    if result.is_error:
        return []
    contacts, current = [], {}
    for ent in result.entities:
        if ent.category == "Person":
            if current: contacts.append(current)
            current = {"name": ent.text, "confidence": round(ent.confidence_score, 3)}
        elif ent.category == "Age" and current:
            current["age_text"] = ent.text
        elif ent.category == "PersonType" and current:
            current["relationship"] = ent.text
    if current:
        contacts.append(current)
    return contacts

def extract_update_intent(lc, note: str) -> dict:
    n = note.lower()
    return {
        "dose_confirmed":         any(w in n for w in ["took dose","gave dose","confirmed"]),
        "dose_missed":            any(w in n for w in ["could not visit","not home","missed"]),
        "patient_reluctance":     any(w in n for w in ["reluctant","feels better","doesn't want"]),
        "new_symptom_in_contact": any(w in n for w in ["coughing","fever","symptoms"]),
    }

def demo_ner(lc):
    if lc is None:
        print("  [NER] Skipped — not configured")
        return
    for pid, note in list(SAMPLE_NOTES.items())[:2]:
        contacts = extract_contacts_from_note(lc, note)
        intent   = extract_update_intent(lc, note)
        print(f"    {pid}: {len(contacts)} contacts, intent={intent}")

def transcribe_voice_note(audio_path: str, language: str = "Tamil") -> str:
    bhashini = {"Tamil","Telugu","Kannada","Bengali"}
    if language in bhashini:
        return f"[Bhashini ASR stub — {language} transcription pending]"
    return f"[Azure Speech stub — {language} transcription pending]"


# ── Azure Event Hubs ──────────────────────────────────────────────────────────

def get_eventhub_producer():
    try:
        from azure.eventhub import EventHubProducerClient
        conn = os.getenv("EVENTHUB_CONNECTION_STRING")
        hub  = os.getenv("EVENTHUB_NAME", "graph-events")
        if not conn: return None
        return EventHubProducerClient.from_connection_string(conn, eventhub_name=hub)
    except ImportError:
        return None

def publish_event(producer, event_type: str, source_id: str, target_id: str, features: dict):
    if producer is None: return
    try:
        from azure.eventhub import EventData
        payload = json.dumps({
            "event_type": event_type, "source_node": source_id, "target_node": target_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "features": features,
        })
        batch = producer.create_batch()
        batch.add(EventData(payload))
        producer.send_batch(batch)
    except Exception as e:
        print(f"  [EventHubs] {e}")


# ── Gremlin helpers ───────────────────────────────────────────────────────────

def safe(val) -> str:
    return str(val).replace("'", "\\'")

def run_query(gc, query: str):
    try:
        return gc.submit(query).all().result()
    except Exception as e:
        print(f"  [Gremlin] {e}")
        return []


# ── ASHA / ANM summaries ──────────────────────────────────────────────────────

def build_asha_summaries(records: list) -> dict:
    """
    Load score denominator is 15 (system-defined max caseload).
    At 15 patients → caseload component = 0.40 (ceiling).
    At  5 patients → caseload component = 0.13 (low).
    """
    groups = defaultdict(list)
    for r in records:
        groups[r["operational"]["asha_id"]].append(r)

    summaries = {}
    for asha_id, pts in groups.items():
        n          = len(pts)
        avg_dist   = sum(p["adherence"]["distance_to_center_km"] for p in pts) / n
        visited_7d = sum(1 for p in pts if p["operational"]["last_asha_visit_days_ago"] <= 7)
        high_risk  = sum(1 for p in pts if p.get("risk_score", 0) > 0.65)
        avg_missed = sum(p["adherence"]["days_since_last_dose"] for p in pts) / n
        load = min(1.0, (
            (n / 15)           * 0.4 +   # caseload pressure — denominator 15
            (avg_missed / 14)  * 0.3 +   # patient engagement decline
            (high_risk / max(n,1)) * 0.3  # proportion high-risk
        ))
        summaries[asha_id] = {
            "asha_id":        asha_id,
            "anm_id":         pts[0]["operational"].get("anm_id", ""),
            "caseload":       n,
            "avg_distance_km":round(avg_dist, 2),
            "visit_freq_7d":  round(visited_7d / n, 3),
            "high_risk_count":high_risk,
            "load_score":     round(load, 4),
            "district":       pts[0]["location"]["district"],
            "block":          pts[0]["location"]["block"],
        }
    return summaries

def build_anm_summaries(records: list, asha_summaries: dict) -> dict:
    anm_groups = defaultdict(list)
    for summary in asha_summaries.values():
        anm_id = summary.get("anm_id", "")
        if anm_id:
            anm_groups[anm_id].append(summary)
    result = {}
    for anm_id, ashas in anm_groups.items():
        result[anm_id] = {
            "anm_id":        anm_id,
            "asha_count":    len(ashas),
            "total_patients":sum(a["caseload"] for a in ashas),
            "avg_load_score":round(sum(a["load_score"] for a in ashas) / len(ashas), 4),
            "high_risk_total":sum(a["high_risk_count"] for a in ashas),
            "district":      ashas[0]["district"],
            "block":         ashas[0]["block"],
        }
    return result

def build_village_summaries(records: list) -> dict:
    groups = defaultdict(list)
    for r in records:
        key = f"{r['location']['district']}_{r['location']['block']}"
        groups[key].append(r)
    result = {}
    for vid, pts in groups.items():
        avg_dist = sum(p["adherence"]["distance_to_center_km"] for p in pts) / len(pts)
        low_edu  = sum(1 for p in pts if p["social"]["low_education"]) / len(pts)
        result[vid] = {
            "village_id": vid,
            "district":   pts[0]["location"]["district"],
            "block":      pts[0]["location"]["block"],
            "patient_count":   len(pts),
            "avg_distance_km": round(avg_dist, 2),
            "low_edu_rate":    round(low_edu, 3),
            "connectivity_score": round(max(0, 1 - avg_dist / 20), 3),
        }
    return result


# ── Phase-adaptive silence detection ─────────────────────────────────────────

def _silence_threshold(record: dict) -> int:
    """
    Intensive phase: flag at 5 days (daily doses expected).
    Late continuation (week 17+): flag at 6 days.
    Continuation: flag at 7 days.
    """
    phase = record["clinical"]["phase"]
    week  = min(record["clinical"]["total_treatment_days"] // 7, 26)
    if phase == "Intensive":
        return 5
    elif week >= 17:
        return 6
    return 7

def inject_silence_events(records: list, producer) -> list:
    count = 0
    for r in records:
        effective = max(r["adherence"]["days_since_last_dose"],
                        r["operational"]["last_asha_visit_days_ago"])
        threshold = _silence_threshold(r)
        if effective >= threshold:
            r["silence_event"] = {
                "duration_days": effective,
                "type": "complete" if r["adherence"]["days_since_last_dose"] >= 14 else "partial",
                "threshold_used": threshold,
            }
            publish_event(producer, "silence", r["patient_id"], r["patient_id"],
                          {"duration_days": effective, "threshold": threshold})
            count += 1
    print(f"  Silence events: {count} patients (thresholds: 5d intensive / 6d late / 7d continuation)")
    return records


# ── Node upserts (coalesce pattern — safe to run multiple times) ──────────────

def _upsert(gc, label: str, vid: str, props: dict, district: str):
    """Generic upsert: update if exists, create if not. Never destroys data."""
    set_str    = "".join(f".property('{k}', {v if isinstance(v,(int,float,bool)) else repr(str(v))})"
                         for k, v in props.items())
    create_str = set_str + f".property('pk', '{safe(district)}')"
    _q = (
        f"g.V('{safe(vid)}').fold().coalesce("
        f"unfold(){set_str},"
        f"addV('{label}').property('id','{safe(vid)}'){create_str}"
        f")"
    )
    run_query(gc, _q)

def ingest_patient_node(gc, producer, record: dict):
    pid      = record["patient_id"]
    district = record["location"]["district"]
    silence  = str(record.get("silence_event") is not None).lower()
    s_days   = record.get("silence_event", {}).get("duration_days", 0)
    mem_init = json.dumps([0.0] * 64)

    # Build the full upsert query manually to handle the memory_vector init carefully
    run_query(gc,
        f"g.V('{safe(pid)}').fold().coalesce("
        f"unfold()"
        f".property('risk_score',       {record['risk_score']})"
        f".property('phase',            '{safe(record['clinical']['phase'])}')"
        f".property('regimen',          '{safe(record['clinical'].get('regimen','Cat_I'))}')"
        f".property('treatment_week',   {min(record['clinical']['total_treatment_days']//7,26)})"
        f".property('days_missed',      {record['adherence']['days_since_last_dose']})"
        f".property('adherence_rate',   {record['adherence']['adherence_rate_30d']})"
        f".property('phase_adherence',  {record['adherence'].get('phase_adherence_rate', record['adherence']['adherence_rate_30d'])})"
        f".property('distance_km',      {record['adherence']['distance_to_center_km']})"
        f".property('silence',          {silence})"
        f".property('silence_days',     {s_days})"
        f".property('welfare_enrolled', {str(record['operational'].get('welfare_enrolled',False)).lower()})"
        f".property('risk_velocity',    {record.get('risk_velocity',0.0)})"
        f".property('asha_id',          '{safe(record['operational']['asha_id'])}')"
        f".property('anm_id',           '{safe(record['operational'].get('anm_id',''))}')"
        f".property('block',            '{safe(record['location']['block'])}')"
        f".property('district',         '{safe(district)}')"
        f","
        f"addV('patient')"
        f".property('id',               '{safe(pid)}')"
        f".property('risk_score',       {record['risk_score']})"
        f".property('phase',            '{safe(record['clinical']['phase'])}')"
        f".property('regimen',          '{safe(record['clinical'].get('regimen','Cat_I'))}')"
        f".property('treatment_week',   {min(record['clinical']['total_treatment_days']//7,26)})"
        f".property('days_missed',      {record['adherence']['days_since_last_dose']})"
        f".property('adherence_rate',   {record['adherence']['adherence_rate_30d']})"
        f".property('phase_adherence',  {record['adherence'].get('phase_adherence_rate', record['adherence']['adherence_rate_30d'])})"
        f".property('distance_km',      {record['adherence']['distance_to_center_km']})"
        f".property('silence',          {silence})"
        f".property('silence_days',     {s_days})"
        f".property('welfare_enrolled', {str(record['operational'].get('welfare_enrolled',False)).lower()})"
        f".property('risk_velocity',    {record.get('risk_velocity',0.0)})"
        f".property('asha_id',          '{safe(record['operational']['asha_id'])}')"
        f".property('anm_id',           '{safe(record['operational'].get('anm_id',''))}')"
        f".property('block',            '{safe(record['location']['block'])}')"
        f".property('district',         '{safe(district)}')"
        f".property('memory_vector',    '{safe(mem_init)}')"
        f".property('pk',               '{safe(district)}')"
        f")"
    )
    publish_event(producer, "patient_upserted", pid, pid,
                  {"risk_score": record["risk_score"]})

def ingest_asha_node(gc, producer, summary: dict):
    aid = summary["asha_id"]
    run_query(gc,
        f"g.V('{safe(aid)}').fold().coalesce("
        f"unfold()"
        f".property('caseload',        {summary['caseload']})"
        f".property('load_score',      {summary['load_score']})"
        f".property('visit_freq_7d',   {summary['visit_freq_7d']})"
        f".property('high_risk_count', {summary['high_risk_count']}),"
        f"addV('asha_worker')"
        f".property('id',              '{safe(aid)}')"
        f".property('caseload',        {summary['caseload']})"
        f".property('load_score',      {summary['load_score']})"
        f".property('avg_distance_km', {summary['avg_distance_km']})"
        f".property('visit_freq_7d',   {summary['visit_freq_7d']})"
        f".property('high_risk_count', {summary['high_risk_count']})"
        f".property('anm_id',          '{safe(summary.get('anm_id',''))}')"
        f".property('district',        '{safe(summary['district'])}')"
        f".property('block',           '{safe(summary['block'])}')"
        f".property('pk',              '{safe(summary['district'])}')"
        f")"
    )

def ingest_anm_node(gc, producer, summary: dict):
    """ANM supervisor node — new node type from ASHA guidelines."""
    anm_id = summary["anm_id"]
    run_query(gc,
        f"g.V('{safe(anm_id)}').fold().coalesce("
        f"unfold()"
        f".property('asha_count',      {summary['asha_count']})"
        f".property('avg_load_score',  {summary['avg_load_score']})"
        f".property('high_risk_total', {summary['high_risk_total']}),"
        f"addV('anm_worker')"
        f".property('id',              '{safe(anm_id)}')"
        f".property('asha_count',      {summary['asha_count']})"
        f".property('total_patients',  {summary['total_patients']})"
        f".property('avg_load_score',  {summary['avg_load_score']})"
        f".property('high_risk_total', {summary['high_risk_total']})"
        f".property('district',        '{safe(summary['district'])}')"
        f".property('block',           '{safe(summary['block'])}')"
        f".property('pk',              '{safe(summary['district'])}')"
        f")"
    )
    publish_event(producer, "anm_upserted", anm_id, anm_id,
                  {"asha_count": summary["asha_count"]})

def ingest_phc_node(gc, producer, district: str, block: str) -> str:
    """
    PHC node replaces DOTS centre.
    drug_available and staffed are operational state properties.
    A stockout sets drug_available=false → risk propagates to all connected patients.
    """
    phc_id = f"PHC_{block}"
    run_query(gc,
        f"g.V('{safe(phc_id)}').fold().coalesce("
        f"unfold(),"
        f"addV('phc')"
        f".property('id',             '{safe(phc_id)}')"
        f".property('block',          '{safe(block)}')"
        f".property('district',       '{safe(district)}')"
        f".property('drug_available', true)"
        f".property('staffed',        true)"
        f".property('pk',             '{safe(district)}')"
        f")"
    )
    return phc_id

def ingest_welfare_scheme_node(gc, producer, district: str) -> str:
    """
    Nikshay Poshan Yojana node.
    payment_status: 'active' or 'delayed'.
    When delayed → enrolled_in edge weight drops → risk rises for all enrolled patients.
    """
    scheme_id = "SCHEME_NikshayPoshanYojana"
    run_query(gc,
        f"g.V('{safe(scheme_id)}').fold().coalesce("
        f"unfold(),"
        f"addV('welfare_scheme')"
        f".property('id',             '{safe(scheme_id)}')"
        f".property('name',           'Nikshay Poshan Yojana')"
        f".property('payment_status', 'active')"
        f".property('amount_inr',     500)"
        f".property('district',       '{safe(district)}')"
        f".property('pk',             '{safe(district)}')"
        f")"
    )
    return scheme_id

def ingest_village_node(gc, producer, summary: dict):
    vid = summary["village_id"]
    run_query(gc,
        f"g.V('{safe(vid)}').fold().coalesce("
        f"unfold(),"
        f"addV('village')"
        f".property('id',                 '{safe(vid)}')"
        f".property('district',           '{safe(summary['district'])}')"
        f".property('block',              '{safe(summary['block'])}')"
        f".property('patient_count',      {summary['patient_count']})"
        f".property('avg_distance_km',    {summary['avg_distance_km']})"
        f".property('connectivity_score', {summary['connectivity_score']})"
        f".property('low_edu_rate',       {summary['low_edu_rate']})"
        f".property('pk',                 '{safe(summary['district'])}')"
        f")"
    )

def ingest_contact_node(gc, producer, record: dict, contact: dict,
                        contact_registry: dict) -> tuple:
    pid      = record["patient_id"]
    district = record["location"]["district"]
    name     = contact["name"]
    cid      = f"CONTACT_{safe(name).replace(' ','_').replace('.','')}"

    if name not in contact_registry:
        run_query(gc,
            f"g.V('{safe(cid)}').fold().coalesce("
            f"unfold(),"
            f"addV('contact')"
            f".property('id',            '{safe(cid)}')"
            f".property('name',          '{safe(name)}')"
            f".property('age',           {contact['age']})"
            f".property('vulnerability', {contact['vulnerability_score']})"
            f".property('rel',           '{safe(contact['rel'])}')"
            f".property('screened',      {str(contact['screened']).lower()})"
            f".property('patient_count', 1)"
            f".property('district',      '{safe(district)}')"
            f".property('pk',            '{safe(district)}')"
            f")"
        )
        contact_registry[name] = (cid, pid)
        return cid, True
    else:
        existing_cid, existing_pid = contact_registry[name]
        bridge_w = round(contact["vulnerability_score"] * record.get("risk_score", 0.1), 4)
        run_query(gc,
            f"g.V('{safe(pid)}').addE('shared_contact')"
            f".to(g.V('{safe(existing_pid)}'))"
            f".property('weight',       {bridge_w})"
            f".property('contact_name', '{safe(name)}')"
            f".property('decay_rate',   0.04)"
        )
        publish_event(producer, "shared_contact_bridge", pid, existing_pid,
                      {"contact_name": name, "weight": bridge_w})
        return existing_cid, False


# ── Edge ingestion ────────────────────────────────────────────────────────────

def ingest_contact_edge(gc, producer, record: dict, contact: dict, cid: str):
    pid   = record["patient_id"]
    base  = 0.9 if contact["rel"] == "Household" else 0.6
    w     = round(base * contact["vulnerability_score"], 4)
    label = "household_contact" if contact["rel"] == "Household" else "workplace_contact"
    decay = 0.02 if contact["rel"] == "Household" else 0.05
    run_query(gc,
        f"g.V('{safe(pid)}').outE('{label}').where(inV().hasId('{safe(cid)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',                    {w})"
        f".property('last_interaction_days_ago', 0),"
        f"g.V('{safe(pid)}').addE('{label}')"
        f".to(g.V('{safe(cid)}'))"
        f".property('weight',                    {w})"
        f".property('base_weight',               {base})"
        f".property('last_interaction_days_ago', 0)"
        f".property('decay_rate',                {decay})"
        f")"
    )
    publish_event(producer, label, pid, cid, {"weight": w})

def ingest_asha_patient_edge(gc, producer, record: dict, asha_summary: dict):
    pid        = record["patient_id"]
    aid        = record["operational"]["asha_id"]
    load       = asha_summary["load_score"]
    days_since = record["operational"]["last_asha_visit_days_ago"]
    recency    = max(0, 1 - days_since / 30)
    w          = round((1 - load) * recency, 4)
    run_query(gc,
        f"g.V('{safe(aid)}').outE('assigned_to').where(inV().hasId('{safe(pid)}')).fold().coalesce("
        f"unfold()"
        f".property('weight',          {w})"
        f".property('load_score',      {load})"
        f".property('days_since_visit',{days_since}),"
        f"g.V('{safe(aid)}').addE('assigned_to')"
        f".to(g.V('{safe(pid)}'))"
        f".property('weight',          {w})"
        f".property('load_score',      {load})"
        f".property('days_since_visit',{days_since})"
        f".property('decay_rate',      0.03)"
        f")"
    )
    publish_event(producer, "asha_patient_assigned", aid, pid,
                  {"weight": w, "load_score": load})

def ingest_anm_asha_edge(gc, producer, anm_id: str, asha_id: str):
    """Static supervision edge — weight 1.0, decay 0.0."""
    run_query(gc,
        f"g.V('{safe(anm_id)}').addE('supervises')"
        f".to(g.V('{safe(asha_id)}'))"
        f".property('weight',    1.0)"
        f".property('decay_rate', 0.0)"
    )
    publish_event(producer, "anm_supervises_asha", anm_id, asha_id, {})

def ingest_phc_patient_edge(gc, producer, record: dict, phc_id: str):
    """attends edge replaces treats/DOTS edge."""
    pid     = record["patient_id"]
    dist    = record["adherence"]["distance_to_center_km"]
    tf      = 1.0 if record["operational"]["phone_type"] == "Smartphone" else 0.8
    w       = round((1 / (1 + dist * 0.1)) * tf, 4)
    run_query(gc,
        f"g.V('{safe(pid)}').addE('attends')"
        f".to(g.V('{safe(phc_id)}'))"
        f".property('weight',      {w})"
        f".property('distance_km', {dist})"
        f".property('decay_rate',  0.01)"
    )
    publish_event(producer, "patient_attends_phc", pid, phc_id,
                  {"weight": w, "distance_km": dist})

def ingest_welfare_edge(gc, producer, record: dict, scheme_id: str):
    """enrolled_in edge — only created if patient is enrolled."""
    if not record["operational"].get("welfare_enrolled", False):
        return
    pid = record["patient_id"]
    run_query(gc,
        f"g.V('{safe(pid)}').addE('enrolled_in')"
        f".to(g.V('{safe(scheme_id)}'))"
        f".property('weight',         1.0)"
        f".property('payment_status', 'active')"
    )
    publish_event(producer, "patient_enrolled_welfare", pid, scheme_id, {})


# ── ASHA writeback functions (close the feedback loop) ────────────────────────

def writeback_dose_confirmed(gc, producer, patient_id: str, asha_id: str):
    """
    ASHA tapped Done.
    Resets days_missed to 0, clears silence, refreshes ASHA-Patient edge recency.
    """
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('days_missed', 0)"
        f".property('silence',     false)"
        f".property('silence_days', 0)"
    )
    # Refresh edge recency — weight recalculated at max (full recency, use existing load)
    run_query(gc,
        f"g.V('{safe(asha_id)}').outE('assigned_to')"
        f".where(inV().has('id', '{safe(patient_id)}'))"
        f".property('days_since_visit', 0)"
    )
    publish_event(producer, "dose_confirmed", asha_id, patient_id,
                  {"days_missed_reset": True, "silence_cleared": True})
    print(f"  [Writeback] dose_confirmed: {patient_id}")

def writeback_dose_missed(gc, producer, patient_id: str, asha_id: str):
    """ASHA tapped Could Not Visit. Increments silence_days."""
    result   = run_query(gc, f"g.V('{safe(patient_id)}').values('silence_days')")
    new_days = (int(result[0]) if result else 0) + 1
    run_query(gc,
        f"g.V('{safe(patient_id)}')"
        f".property('silence_days', {new_days})"
        f".property('silence',      true)"
    )
    publish_event(producer, "dose_missed", asha_id, patient_id,
                  {"silence_days": new_days})
    print(f"  [Writeback] dose_missed: {patient_id} silence_days={new_days}")

def writeback_contact_screened(gc, producer, patient_id: str,
                                contact_name: str, asha_id: str):
    """ASHA reports contact screened. Sets contact.screened = true in graph."""
    cid = f"CONTACT_{safe(contact_name).replace(' ','_').replace('.','')}"
    run_query(gc, f"g.V('{safe(cid)}').property('screened', true)")
    publish_event(producer, "contact_screened", asha_id, cid,
                  {"patient_id": patient_id, "contact_name": contact_name})
    print(f"  [Writeback] contact_screened: {contact_name}")

def writeback_risk_scores(gc, patients: list):
    """
    Write Stage 3 scores back to patient nodes.
    Stores previous_risk_score to enable velocity computation next cycle.
    Called by main.py after score_all_patients().
    """
    if gc is None:
        return
    updated = 0
    for p in patients:
        pid      = p["patient_id"]
        score    = p.get("risk_score", 0)
        level    = p.get("risk_level", "LOW")
        velocity = p.get("risk_velocity", 0.0)
        run_query(gc,
            f"g.V('{safe(pid)}')"
            f".property('risk_score',          {score})"
            f".property('risk_level',          '{safe(level)}')"
            f".property('risk_velocity',       {velocity})"
            f".property('previous_risk_score', {score})"
        )
        updated += 1
    print(f"  [Writeback] Risk scores written to {updated} patient nodes")

def writeback_pagerank_scores(gc, pagerank_scores: dict):
    """Write PageRank scores to nodes after Stage 3b."""
    if gc is None:
        return
    updated = 0
    for node_id, score in pagerank_scores.items():
        run_query(gc,
            f"g.V('{safe(node_id)}')"
            f".property('pagerank_score', {round(score, 8)})"
        )
        updated += 1
    print(f"  [Writeback] PageRank scores written to {updated} nodes")

def promote_contact_to_patient(gc, producer, contact_name: str,
                                source_patient_id: str, asha_id: str,
                                anm_id: str, district: str, block: str) -> str:
    """
    Contact tests positive for TB → promote contact node to patient.
    Draws a transmission_likely edge from source patient.
    Returns new patient_id.
    """
    import random
    new_pid = f"NIK-NEW-{contact_name.replace(' ','')[:8].upper()}-{random.randint(1000,9999)}"
    cid     = f"CONTACT_{safe(contact_name).replace(' ','_').replace('.','')}"

    # Cosmos DB does not allow mutating a vertex's id after creation.
    # Correct approach: copy contact properties to a new patient vertex,
    # draw transmission edge from source, then drop the old contact node.
    contact_props = run_query(gc, f"g.V('{safe(cid)}').valueMap()")
    props = contact_props[0] if contact_props else {}
    age   = props.get('age', [30])
    age   = age[0] if isinstance(age, list) else age
    vuln  = props.get('vulnerability', [1.0])
    vuln  = vuln[0] if isinstance(vuln, list) else vuln
    mem_init = json.dumps([0.0] * 64)
    run_query(gc,
        f"g.V('{safe(new_pid)}').fold().coalesce("
        f"unfold(),"
        f"addV('patient')"
        f".property('id',           '{safe(new_pid)}')"
        f".property('node_type',    'patient')"
        f".property('asha_id',      '{safe(asha_id)}')"
        f".property('anm_id',       '{safe(anm_id)}')"
        f".property('district',     '{safe(district)}')"
        f".property('block',        '{safe(block)}')"
        f".property('age',          {age})"
        f".property('risk_score',   0.0)"
        f".property('days_missed',  0)"
        f".property('silence',      false)"
        f".property('memory_vector','{safe(mem_init)}')"
        f".property('pk',           '{safe(district)}')"
        f")"
    )
    # Drop old contact node — it is now represented as a patient
    run_query(gc, f"g.V('{safe(cid)}').drop()")
    run_query(gc,
        f"g.V('{safe(source_patient_id)}').addE('transmission_likely')"
        f".to(g.V('{safe(new_pid)}'))"
        f".property('weight',    0.85)"
        f".property('confirmed', false)"
    )
    publish_event(producer, "new_tb_case_from_contact", source_patient_id, new_pid,
                  {"contact_name": contact_name})
    print(f"  [Promote] {contact_name} → {new_pid} (from {source_patient_id})")
    return new_pid


def process_asha_reply(gc, producer, message_text: str,
                       patient_id: str, asha_id: str) -> dict:
    """Route ASHA reply to correct writeback function and publish to Event Hubs."""
    msg   = message_text.strip().lower()
    event = {"asha_id": asha_id, "patient_id": patient_id, "raw_message": message_text}

    if msg in ["done", "✓", "1"]:
        event["event_type"] = "dose_confirmed"
        if gc: writeback_dose_confirmed(gc, producer, patient_id, asha_id)
    elif msg in ["could not visit", "not home", "2"]:
        event["event_type"] = "dose_missed"
        if gc: writeback_dose_missed(gc, producer, patient_id, asha_id)
    elif msg in ["issue to report", "issue", "3"]:
        event["event_type"]   = "issue_flagged"
        event["flag_officer"] = True
        publish_event(producer, "issue_flagged", asha_id, patient_id, {})
    else:
        event["event_type"]   = "free_text_update"
        event["requires_ner"] = True

    return event


# ── Main ingestion orchestrator ───────────────────────────────────────────────

def ingest_all(gc, producer, records: list, limit: int = 1000):
    """
    Upsert-based — never calls clear_graph(). Safe to run multiple times.
    silence injection is done in main.py, not here, to avoid double injection.
    """
    records = records[:limit]
    print(f"  Building summaries for {len(records)} records...")

    asha_summaries    = build_asha_summaries(records)
    anm_summaries     = build_anm_summaries(records, asha_summaries)
    village_summaries = build_village_summaries(records)
    district          = records[0]["location"]["district"]
    block             = records[0]["location"]["block"]

    print(f"  Upserting PHC node ({block})...")
    phc_id    = ingest_phc_node(gc, producer, district, block)

    print(f"  Upserting welfare scheme node...")
    scheme_id = ingest_welfare_scheme_node(gc, producer, district)

    print(f"  Upserting {len(anm_summaries)} ANM nodes...")
    for s in anm_summaries.values():
        ingest_anm_node(gc, producer, s)

    print(f"  Upserting {len(asha_summaries)} ASHA nodes...")
    for s in asha_summaries.values():
        ingest_asha_node(gc, producer, s)

    print(f"  Creating ANM→ASHA supervision edges...")
    for asha_id, s in asha_summaries.items():
        anm_id = s.get("anm_id", "")
        if anm_id:
            ingest_anm_asha_edge(gc, producer, anm_id, asha_id)

    print(f"  Upserting {len(village_summaries)} village nodes...")
    for s in village_summaries.values():
        ingest_village_node(gc, producer, s)

    print(f"  Upserting {len(records)} patient nodes + edges...")
    contact_registry = {}
    new_c = bridges = 0

    for i, record in enumerate(records):
        try:
            aid      = record["operational"]["asha_id"]
            asha_sum = asha_summaries.get(aid, {"load_score": 0.3, "caseload": 10, "anm_id": ""})

            ingest_patient_node(gc, producer, record)

            for contact in record["contact_network"]:
                cid, is_new = ingest_contact_node(gc, producer, record,
                                                   contact, contact_registry)
                ingest_contact_edge(gc, producer, record, contact, cid)
                if is_new: new_c += 1
                else:      bridges += 1

            ingest_asha_patient_edge(gc, producer, record, asha_sum)
            ingest_phc_patient_edge(gc, producer, record, phc_id)
            ingest_welfare_edge(gc, producer, record, scheme_id)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(records)} patients upserted...")
        except Exception as e:
            print(f"  Error on {record.get('patient_id','?')}: {e}")
            continue

    print(f"\n  Ingestion complete (upsert — existing data preserved)")
    print(f"  {len(records)} patients · {len(asha_summaries)} ASHAs · {len(anm_summaries)} ANMs")
    print(f"  {new_c} contact nodes · {bridges} shared_contact bridges")
    print(f"  Node types (7): Patient, ASHA, ANM, PHC, WelfareScheme, Village, Contact")
    print(f"  Edge types (8): household_contact, workplace_contact, shared_contact,")
    print(f"                  assigned_to, attends, supervises, enrolled_in, social(stub)")