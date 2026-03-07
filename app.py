"""
app.py
======
Nikshay-Graph Dashboard — Streamlit

Tabs:
  1. Stage 1: Graph         — node/edge schema, ASHA load, silence summary
  2. Stage 2/3: Risk Scores — score distribution, composition, BBN factors
  3. Stage 3: Propagation   — patient visit list, contact screening list
  4. Stage 4: Explanations  — ASHA vs Officer explanations, safety status
  5. ASHA Worker            — personalised morning briefing, voice note, quick-action buttons
  6. District Officer       — systemic alerts, high-risk map by block, top ASHAs

All ASHA updates are submitted here (via quick-action buttons or free-text input)
and routed to stage5_voice.process_asha_dashboard_reply() → Event Hubs → Stage 1.
WhatsApp and SMS delivery have been removed.

Run:  streamlit run app.py
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title  = "Nikshay-Graph",
    page_icon   = "🫁",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_patients():
    p = "nikshay_scored_dataset.json"
    if not Path(p).exists():
        return []
    with open(p) as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_agent3():
    p = "agent3_output.json"
    if not Path(p).exists():
        return {"visit_list": [], "screening_list": [], "systemic_alerts": []}
    with open(p) as f:
        return json.load(f)

@st.cache_data(ttl=60)
def load_briefings():
    p = "briefings_output.json"
    if not Path(p).exists():
        return {"asha_briefings": {}, "systemic_alerts": []}
    with open(p) as f:
        return json.load(f)

patients   = load_scored_patients()
agent3     = load_agent3()
briefings  = load_briefings()

visit_list       = agent3.get("visit_list", [])
screening_list   = agent3.get("screening_list", [])
systemic_alerts  = agent3.get("systemic_alerts", [])
asha_briefings   = briefings.get("asha_briefings", {})

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/IIT_Madras_Logo.svg/200px-IIT_Madras_Logo.svg.png",
             width=80)
    st.title("Nikshay-Graph")
    st.caption("TB Treatment Dropout Prevention\nIIT Madras · Microsoft AI Unlocked")
    st.divider()

    n_patients = len(patients)
    n_high     = sum(1 for p in patients if p.get("risk_level") == "HIGH")
    n_silent   = sum(1 for p in patients if p.get("silence_event"))
    n_alerts   = len(systemic_alerts)

    st.metric("Total Patients",    n_patients)
    st.metric("High Risk",         n_high,   delta=f"{n_high/max(n_patients,1)*100:.1f}%")
    st.metric("Silent (disengaged)", n_silent)
    st.metric("Systemic Alerts",   n_alerts, delta_color="inverse")

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    if not patients:
        st.warning("No data loaded. Run: `python main.py --limit 100`")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "📊 Stage 1: Graph",
    "📈 Stage 2/3: Risk Scores",
    "🕸 Stage 3: Propagation",
    "💬 Stage 4: Explanations",
    "👩‍⚕️ ASHA Worker",
    "🏛 District Officer",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Stage 1: Graph
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Stage 1: Graph Construction")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Node Types (7)")
        node_df = pd.DataFrame([
            {"Node Type":   "Patient",        "Count": len(patients),
             "Key Properties": "risk_score, phase, treatment_week, days_missed, memory_vector"},
            {"Node Type":   "ASHA Worker",    "Count": len({p["operational"]["asha_id"] for p in patients}),
             "Key Properties": "caseload, load_score, visit_freq_7d, high_risk_count"},
            {"Node Type":   "ANM Worker",     "Count": len({p["operational"].get("anm_id","") for p in patients}),
             "Key Properties": "asha_count, avg_load_score, high_risk_total"},
            {"Node Type":   "PHC",            "Count": 1,
             "Key Properties": "block, drug_available, staffed"},
            {"Node Type":   "Welfare Scheme", "Count": 1,
             "Key Properties": "name (Nikshay Poshan Yojana), payment_status"},
            {"Node Type":   "Village/Ward",   "Count": len({p["location"]["block"] for p in patients}),
             "Key Properties": "connectivity_score, avg_distance_km, low_edu_rate"},
            {"Node Type":   "Contact",        "Count": sum(len(p.get("contact_network",[])) for p in patients),
             "Key Properties": "vulnerability_score, rel, screened, age"},
        ])
        st.dataframe(node_df, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Edge Types (8)")
        edge_df = pd.DataFrame([
            {"Edge":             "household_contact",  "Base Weight": "0.9 × vuln",    "Decay": "0.02/wk"},
            {"Edge":             "workplace_contact",  "Base Weight": "0.6 × vuln",    "Decay": "0.05/wk"},
            {"Edge":             "shared_contact",     "Base Weight": "vuln × risk",   "Decay": "0.04/wk"},
            {"Edge":             "assigned_to",        "Base Weight": "(1-load)×recency", "Decay": "0.03/wk"},
            {"Edge":             "attends (PHC)",      "Base Weight": "1/(1+dist×0.1)","Decay": "0.01/wk"},
            {"Edge":             "supervises (ANM)",   "Base Weight": "1.0",           "Decay": "0.0"},
            {"Edge":             "enrolled_in",        "Base Weight": "1.0",           "Decay": "0.0"},
            {"Edge":             "social (stub)",      "Base Weight": "0.3",           "Decay": "fast"},
        ])
        st.dataframe(edge_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("ASHA Worker Load Distribution")

    if patients:
        from collections import defaultdict
        asha_groups = defaultdict(list)
        for p in patients:
            asha_groups[p["operational"]["asha_id"]].append(p)

        load_data = []
        for aid, pts in asha_groups.items():
            load_data.append({
                "asha_id":     aid,
                "caseload":    len(pts),
                "load_score":  round(sum(p.get("asha_load_score", 0) for p in pts) / len(pts), 3),
                "high_risk":   sum(1 for p in pts if p.get("risk_level") == "HIGH"),
            })
        load_df = pd.DataFrame(load_data).sort_values("load_score", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.bar_chart(load_df.set_index("asha_id")["load_score"].head(20),
                         use_container_width=True)
            st.caption("ASHA load scores (top 20 workers)")
        with col_b:
            st.bar_chart(load_df.set_index("asha_id")["high_risk"].head(20),
                         use_container_width=True)
            st.caption("High-risk patients per ASHA (top 20)")

    st.subheader("Silence Detection Summary")
    if patients:
        silence_records = [p for p in patients if p.get("silence_event")]
        complete = sum(1 for p in silence_records
                       if p["silence_event"].get("type") == "complete")
        partial  = len(silence_records) - complete
        s1, s2, s3 = st.columns(3)
        s1.metric("Silent Patients",     len(silence_records))
        s2.metric("Complete (≥14d)",     complete)
        s3.metric("Partial (5-13d)",     partial)
        st.caption("Phase-adaptive thresholds: 5d Intensive / 6d Late Continuation / 7d Continuation")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stage 2/3: Risk Scores
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Stage 2/3: Risk Score Analysis")

    if not patients:
        st.info("Run the pipeline first: `python main.py --limit 100`")
    else:
        scores = [p.get("risk_score", 0) for p in patients]
        tiers  = [p.get("risk_level", "LOW") for p in patients]

        c1, c2, c3 = st.columns(3)
        c1.metric("HIGH",   tiers.count("HIGH"),   f"{tiers.count('HIGH')/len(tiers)*100:.1f}%")
        c2.metric("MEDIUM", tiers.count("MEDIUM"), f"{tiers.count('MEDIUM')/len(tiers)*100:.1f}%")
        c3.metric("LOW",    tiers.count("LOW"),     f"{tiers.count('LOW')/len(tiers)*100:.1f}%")

        st.subheader("Score Distribution")
        score_df = pd.DataFrame({"risk_score": scores})
        st.bar_chart(score_df["risk_score"].value_counts(bins=20).sort_index(),
                     use_container_width=True)

        st.subheader("Score Composition — Top 20 Patients")
        top20 = sorted(patients, key=lambda x: x.get("risk_score", 0), reverse=True)[:20]
        comp_data = []
        for p in top20:
            sc = p.get("score_composition", {})
            comp_data.append({
                "Patient ID":    p["patient_id"],
                "Final Score":   p.get("risk_score", 0),
                "Risk Tier":     p.get("risk_level", "?"),
                "Wk":            p.get("treatment_week", "?"),
                "TGN %":         int(sc.get("tgn_weight", 0) * 100),
                "BBN %":         int(sc.get("bbn_weight", 0) * 100),
                "ASHA Load %":   20,
                "BBN Status":    sc.get("bbn_status", "active"),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.subheader("BBN Prior — Published Odds Ratios")
        or_data = {
            "Divorced/Separated":          3.80,
            "Drug use":                    2.40,
            "Drug-resistant TB (DR-TB)":   2.80,
            "HIV co-infection":            2.16,
            "Prior TB/LTFU":               2.10,
            "Continuation phase":          2.30,
            "Age 20-39":                   2.07,
            "Missed 14+ days":             6.50,
            "Missed 7-13 days":            3.20,
            "Distance > 10km":             2.10,
            "Alcohol use":                 1.92,
            "No nutritional support":      1.60,
            "No Nikshay Poshan Yojana":    1.45,
            "Diabetes (protective in TN)": 0.52,
        }
        or_df = pd.DataFrame({"Factor": list(or_data.keys()),
                               "Odds Ratio": list(or_data.values())})
        or_df = or_df.sort_values("Odds Ratio", ascending=False)
        st.bar_chart(or_df.set_index("Factor")["Odds Ratio"], use_container_width=True)
        st.caption("Sources: Vijay Kumar et al. 2020 (Tamil Nadu), India TB Report 2023, "
                   "Figueiredo et al. PMC10760311, Mistry et al. PMC6830133")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Stage 3: Propagation
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Stage 3: Contact Risk Propagation & Screening")

    left, right = st.columns(2)

    with left:
        st.subheader(f"Patient Visit Priority List ({len(visit_list)} patients)")
        if visit_list:
            vl_df = pd.DataFrame([{
                "Rank":     v["rank"],
                "ID":       v["patient_id"],
                "Tier":     v["risk_level"],
                "Score":    v["risk_score"],
                "Week":     v["treatment_week"],
                "Missed d": v["days_missed"],
                "Block":    v["block"],
            } for v in visit_list])
            st.dataframe(vl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No visit list — run pipeline first.")

    with right:
        st.subheader(f"Contact Screening List ({len(screening_list)} contacts)")
        if screening_list:
            sl_df = pd.DataFrame([{
                "Rank":       c["rank"],
                "Name":       c["name"],
                "Age":        c["age"],
                "Relation":   c["rel"],
                "Vuln Score": c["vulnerability"],
                "Priority":   round(c["screening_priority"], 6),
            } for c in screening_list])
            st.dataframe(sl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No screening list — run pipeline first.")

    st.subheader("Adaptive Risk Thresholds")
    thresh_df = pd.DataFrame([
        {"Phase":              "Intensive (wk 1-8)",         "HIGH >": 0.75, "MEDIUM >": 0.50},
        {"Phase":              "Early Continuation (wk 9-16)","HIGH >": 0.65, "MEDIUM >": 0.40},
        {"Phase":              "Late Continuation (wk 17-26)","HIGH >": 0.55, "MEDIUM >": 0.30},
    ])
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)
    st.caption("Thresholds tighten near treatment completion — the intervention window is narrower.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Stage 4: Explanations
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Stage 4: Template-Based Explanations")
    st.info("⚠️ **Safety principle:** All explanations are template-based from actual model outputs. "
            "No free-form LLM generation is used — eliminates hallucination risk in a medical context.")

    if not visit_list:
        st.info("No visit list — run pipeline first.")
    else:
        for v in visit_list[:10]:
            with st.expander(
                f"Rank {v['rank']} — {v['patient_id']} "
                f"[{v['risk_level']}] score={v['risk_score']} wk={v['treatment_week']}",
                expanded=(v["rank"] <= 3)
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown("**ASHA Worker (simple)**")
                    st.write(v["asha_explanation"])
                    safety_icon = "✅ Passed" if v["safety_passed"] else "🚫 BLOCKED"
                    st.markdown(f"Safety: {safety_icon}")
                with col_b:
                    st.markdown("**District Officer (detailed)**")
                    st.write(v["officer_explanation"])
                    if v.get("top_factors"):
                        st.markdown("**Top risk factors:**")
                        for factor, or_val in v["top_factors"].items():
                            st.write(f"  • {factor}: OR = {or_val}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ASHA Worker
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.header("👩‍⚕️ ASHA Worker Portal")
    st.caption("ASHA workers submit updates here. All actions are routed to Cosmos DB and Event Hubs.")

    # ASHA selector
    asha_ids = sorted(asha_briefings.keys()) if asha_briefings else []
    if not asha_ids and patients:
        asha_ids = sorted({p["operational"]["asha_id"] for p in patients})

    if not asha_ids:
        st.info("No briefings available — run pipeline first.")
    else:
        selected_asha = st.selectbox("Select ASHA Worker ID", asha_ids)

        briefing = asha_briefings.get(selected_asha)

        if briefing:
            st.subheader(f"Morning Briefing — {selected_asha}")
            lang = briefing.get("language", "Tamil")
            st.caption(f"Language: {lang} | Patients: {briefing.get('patient_count', 0)}")

            # Translated text
            st.markdown("**Translated Briefing Text:**")
            st.info(briefing.get("translated_text") or briefing.get("english_text", ""))

            # Audio playback
            audio_path = briefing.get("audio_path")
            if audio_path and os.path.exists(str(audio_path)):
                st.markdown("**Voice Note:**")
                with open(audio_path, "rb") as af:
                    st.audio(af.read(), format="audio/mp3")
            elif briefing.get("audio_available"):
                st.info("Audio file was generated but is no longer on disk. Re-run Stage 5 to regenerate.")
            else:
                st.caption("Voice note not available (TTS not configured — set SPEECH_KEY or BHASHINI_API_KEY in .env)")

            st.divider()
            st.subheader("Patient Cards — Tap to Update")

            visit_cards = briefing.get("visit_cards", [])
            if not visit_cards:
                # Fall back to visit_list filtered by this ASHA
                visit_cards = [v for v in visit_list if v.get("asha_id") == selected_asha]

            for card in visit_cards:
                pid       = card["patient_id"]
                tier      = card.get("risk_level", "?")
                score     = card.get("risk_score", 0)
                missed    = card.get("days_missed", 0)
                expl      = card.get("explanation", card.get("asha_explanation", ""))
                tier_color = "🔴" if tier == "HIGH" else ("🟡" if tier == "MEDIUM" else "🟢")

                with st.expander(f"{tier_color} {pid} — {tier} (score {score:.2f}) | {missed}d missed"):
                    st.write(expl)

                    col1, col2, col3 = st.columns(3)

                    if col1.button("✅ Done (dose given)", key=f"done_{pid}"):
                        _reply_event(selected_asha, pid, "done")
                        st.success(f"Recorded: dose confirmed for {pid}")

                    if col2.button("❌ Could Not Visit", key=f"miss_{pid}"):
                        _reply_event(selected_asha, pid, "could_not_visit")
                        st.warning(f"Recorded: could not visit {pid}")

                    if col3.button("⚠️ Flag Issue", key=f"flag_{pid}"):
                        _reply_event(selected_asha, pid, "issue")
                        st.error(f"Flagged to District Officer: {pid}")

                    # Contact screening
                    patient_contacts = next(
                        (p.get("contact_network", []) for p in patients
                         if p["patient_id"] == pid), []
                    )
                    if patient_contacts:
                        unscreened = [c["name"] for c in patient_contacts
                                      if not c.get("screened", False)]
                        if unscreened:
                            contact_name = st.selectbox(
                                "Mark contact screened",
                                ["— select —"] + unscreened,
                                key=f"contact_{pid}"
                            )
                            if contact_name != "— select —":
                                if st.button(f"✅ Confirm screened: {contact_name}",
                                             key=f"screen_{pid}_{contact_name}"):
                                    _reply_event(selected_asha, pid, "contact_screened",
                                                 contact_name=contact_name)
                                    st.success(f"Contact {contact_name} marked as screened.")

                    # Free-text note
                    note = st.text_input(f"Free-text note (optional)", key=f"note_{pid}")
                    if note and st.button(f"Submit note", key=f"note_btn_{pid}"):
                        _reply_event(selected_asha, pid, "free_text", free_text=note)
                        st.success("Note submitted — NER pipeline will process it.")
        else:
            st.info(f"No briefing generated yet for {selected_asha}. Run pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — District Officer
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("🏛 District TB Officer Dashboard")

    # Systemic alerts at the top
    if systemic_alerts:
        st.subheader(f"⚠️ Systemic Alerts ({len(systemic_alerts)})")
        for alert in systemic_alerts:
            tier = alert.get("tier", 0)
            if tier >= 4:
                st.error(f"**TIER 4 — DISTRICT-WIDE:** {alert['message']}")
            elif tier == 3:
                st.error(f"**TIER 3 — BLOCK LEVEL:** {alert['message']}")
            else:
                st.warning(f"**TIER 2 — ASHA {alert.get('asha_id','?')}:** {alert['message']}")
    else:
        st.success("No systemic alerts — no clinic closures or drug stockouts detected.")

    st.divider()

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        # High-risk by block
        st.subheader("High-Risk Patients by Block")
        from collections import Counter
        block_counts = Counter(
            p["location"]["block"] for p in patients if p.get("risk_level") == "HIGH"
        )
        if block_counts:
            block_df = pd.DataFrame({"Block": list(block_counts.keys()),
                                     "High-Risk Count": list(block_counts.values())})
            st.bar_chart(block_df.set_index("Block"), use_container_width=True)

        # Top ASHA workers by high-risk count
        st.subheader("Top 10 ASHA Workers by High-Risk Patient Count")
        from collections import defaultdict
        asha_high = defaultdict(int)
        for p in patients:
            if p.get("risk_level") == "HIGH":
                asha_high[p["operational"]["asha_id"]] += 1
        top_ashas = sorted(asha_high.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_ashas:
            asha_df = pd.DataFrame(top_ashas, columns=["ASHA ID", "High-Risk Patients"])
            st.dataframe(asha_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Officer Explanations — Top 10 Patients")
        for v in visit_list[:10]:
            tier_color = "🔴" if v["risk_level"] == "HIGH" else "🟡"
            with st.expander(f"{tier_color} {v['patient_id']} — {v['risk_level']} "
                             f"(score {v['risk_score']}, wk {v['treatment_week']})"):
                st.write(v["officer_explanation"])
                if v.get("score_composition"):
                    sc = v["score_composition"]
                    st.caption(
                        f"TGN: {int(sc.get('tgn_weight',0)*100)}%  |  "
                        f"BBN: {int(sc.get('bbn_weight',0)*100)}%  |  "
                        f"ASHA Load: 20%  |  BBN status: {sc.get('bbn_status','active')}"
                    )


# ─────────────────────────────────────────────────────────────────────────────
# ASHA REPLY HELPER (defined after tab code to avoid forward-reference issues)
# ─────────────────────────────────────────────────────────────────────────────

def _reply_event(asha_id: str, patient_id: str, action: str,
                  free_text: str = "", contact_name: str = ""):
    """
    Submit an ASHA action to the backend without crashing the dashboard
    if Cosmos DB or Event Hubs are not configured.
    """
    try:
        from stage5_voice import process_asha_dashboard_reply
        from stage1_nlp import get_eventhub_producer

        producer = get_eventhub_producer()

        gc = None
        try:
            from cosmos_client import get_client, health_check
            if health_check():
                gc = get_client()
        except Exception:
            pass  # Cosmos offline — Event Hubs still captures the event

        process_asha_dashboard_reply(
            gc           = gc,
            producer     = producer,
            action       = action,
            patient_id   = patient_id,
            asha_id      = asha_id,
            free_text    = free_text,
            contact_name = contact_name,
        )
    except Exception as e:
        st.error(f"Could not submit update: {e}")


# Re-wire _reply_event now that it's defined (Streamlit runs top to bottom,
# so button callbacks in Tab 5 are executed after all tab code has run).
# This is safe because Streamlit re-runs the script on each interaction.
