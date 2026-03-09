"""
Nikshay-Graph — Azure Functions (Python v2 programming model)
=============================================================
Two functions:

  1. nikshay_overnight_processor  — Timer trigger, runs daily at 22:00 IST (16:30 UTC)
     Full pipeline: silence detection → TGN inference → scoring → PageRank →
     explanations → ASHA voice briefings.
     Saves nikshay_scored_dataset.json + agent3_output.json + briefings_output.json
     for the Streamlit dashboard to pick up.

  2. nikshay_note_ingestor  — Event Hub trigger on graph-events hub.
     Fires when ASHA submits a free-text note from the dashboard.
     Runs NER, queues contact extraction, updates graph edge weights.
     Does NOT touch BBN ORs — those are schedule-gated only.

DEPLOYMENT NOTES:
  - Runtime: Python 3.12, Functions v4
  - This file must be at the repo ROOT (same level as app.py, main.py)
  - requirements.txt at repo root must include azure-functions
"""

import azure.functions as func
import logging
import json
import os
import sys

# Make all pipeline modules importable (they live at repo root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Overnight pipeline
# Timer: every day at 22:00 IST = 16:30 UTC
# ─────────────────────────────────────────────────────────────────────────────

@app.timer_trigger(
    schedule="0 30 16 * * *",
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False,
)
def nikshay_overnight_processor(myTimer: func.TimerRequest) -> None:
    """
    Nightly full pipeline run.

    What this touches in the graph:
      - Patient node: days_missed, silence, silence_days, risk_score, memory_vector
      - ASHA→Patient edge: weight, days_since_visit, load_score  (via writeback_*)
      - BBN OR update: ONLY if check_and_run_scheduled_update() decides it's due
        (schedule-gated — not triggered by ASHA worker actions, only confirmed
         dropouts from the District Officer tab feed into this)

    What this does NOT touch:
      - confirmed_dropouts.json — Officer-only action from the dashboard
      - BBN ORs directly — only via the schedule gate
    """
    if myTimer.past_due:
        logging.info("Timer is past due — running immediately.")
    logging.info("Nikshay overnight processor started.")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        # ── Imports ────────────────────────────────────────────────────────
        from cosmos_client import get_client, health_check
        from stage1_nlp   import (get_eventhub_producer, build_asha_summaries,
                                   inject_silence_events, ingest_all)
        from stage2_tgn   import run_tgn_inference
        from stage3_score import (score_all_patients, detect_systemic_failures,
                                   check_and_run_scheduled_update)
        from stage4_explain import get_patient_visit_list, get_contact_screening_list
        from stage5_voice  import run_morning_briefings
        import networkx as nx

        # ── Resolve data directory ─────────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # ── Load patient dataset ───────────────────────────────────────────
        dataset_path = os.path.join(data_dir, "nikshay_grounded_dataset.json")
        if not os.path.exists(dataset_path):
            # Fallback: try root
            dataset_path = os.path.join(base_dir, "nikshay_grounded_dataset.json")
        if not os.path.exists(dataset_path):
            logging.error("Dataset not found. Commit data/nikshay_grounded_dataset.json.")
            return

        with open(dataset_path, encoding="utf-8") as f:
            patients = json.load(f)
        logging.info(f"Loaded {len(patients)} patient records.")

        # ── Cosmos DB ──────────────────────────────────────────────────────
        gc = None
        try:
            if health_check():
                gc = get_client()
                logging.info("Cosmos DB: connected.")
        except Exception as e:
            logging.warning(f"Cosmos DB unavailable — offline run. ({e})")

        # ── Event Hubs ─────────────────────────────────────────────────────
        producer = None
        try:
            producer = get_eventhub_producer()
        except Exception as e:
            logging.warning(f"Event Hubs unavailable — events skipped. ({e})")

        # ── BBN schedule gate ──────────────────────────────────────────────
        # The ONLY place BBN ORs are updated.
        # ASHA worker updates (dose/visit actions) never touch ORs.
        try:
            result = check_and_run_scheduled_update()
            logging.info(f"BBN schedule: {result.get('status', 'checked')}")
        except Exception as e:
            logging.warning(f"BBN schedule check failed (non-fatal): {e}")

        # ── Stage 1 ────────────────────────────────────────────────────────
        asha_summaries = build_asha_summaries(patients)
        inject_silence_events(patients, producer)
        if gc:
            ingest_all(gc, producer, patients, asha_summaries)
        logging.info("Stage 1 complete.")

        # ── Stage 2 ────────────────────────────────────────────────────────
        tgn_scores, attention_weights = run_tgn_inference(patients, gc=gc)
        logging.info(f"Stage 2 complete. {len(tgn_scores)} patients scored.")

        # ── Stage 3 ────────────────────────────────────────────────────────
        patients = score_all_patients(patients, tgn_scores, asha_summaries)
        systemic_alerts = detect_systemic_failures(patients)
        logging.info(f"Stage 3 complete. Systemic alerts: {len(systemic_alerts)}")

        # Persist scored dataset
        scored_path = os.path.join(data_dir, "nikshay_scored_dataset.json")
        with open(scored_path, "w", encoding="utf-8") as f:
            json.dump(patients, f, indent=2, default=str)

        # ── Stage 3b — PageRank ────────────────────────────────────────────
        G = nx.Graph()
        for p in patients:
            pid = p["patient_id"]
            G.add_node(pid, node_type="patient",
                       risk_score=p.get("risk_score", 0))
            for c in p.get("contact_network", []):
                cid = f"CONTACT_{c['name'].replace(' ', '_')}"
                G.add_node(cid, node_type="contact",
                           vulnerability=c.get("vulnerability_score", 1.0),
                           screened=c.get("screened", False))
                G.add_edge(pid, cid,
                           weight=0.9 if c.get("rel") == "Household" else 0.6)

        high_risk = {p["patient_id"]: p["risk_score"]
                     for p in patients if p.get("risk_level") == "HIGH"}
        personalization = {n: high_risk.get(n, 0.001) for n in G.nodes()}
        try:
            pagerank_scores = nx.pagerank(
                G, alpha=0.85, personalization=personalization,
                weight="weight", max_iter=200)
        except Exception:
            pagerank_scores = {}
        logging.info(f"Stage 3b PageRank: {len(pagerank_scores)} nodes scored.")

        # ── Stage 4 ────────────────────────────────────────────────────────
        visit_list      = get_patient_visit_list(patients, top_n=10)
        screening_list  = get_contact_screening_list(G, pagerank_scores, top_n=10)
        agent3_output   = {
            "visit_list":      visit_list,
            "screening_list":  screening_list,
            "systemic_alerts": systemic_alerts,
        }
        agent3_path = os.path.join(data_dir, "agent3_output.json")
        with open(agent3_path, "w", encoding="utf-8") as f:
            json.dump(agent3_output, f, indent=2, default=str)
        logging.info("Stage 4 complete.")

        # ── Stage 5 ────────────────────────────────────────────────────────
        try:
            briefing_result = run_morning_briefings(
                visit_list=visit_list,
                screening_list=screening_list,
                systemic_alerts=systemic_alerts,
                patients=patients,
            )
            briefings_path = os.path.join(data_dir, "briefings_output.json")
            with open(briefings_path, "w", encoding="utf-8") as f:
                json.dump(briefing_result, f, indent=2, default=str)
            logging.info(f"Stage 5 complete. "
                         f"{len(briefing_result.get('asha_briefings', {}))} briefings.")
        except Exception as e:
            logging.warning(f"Stage 5 failed (non-fatal — dashboard still works): {e}")

        logging.info("Overnight pipeline complete.")

    except Exception as e:
        logging.exception(f"Overnight pipeline FAILED: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — ASHA note ingestor (Event Hub trigger)
# Fires when ASHA submits a free-text note from the dashboard.
# Runs NER → extracts contacts → updates graph edge weights.
# Does NOT update BBN ORs.
# ─────────────────────────────────────────────────────────────────────────────

@app.event_hub_message_trigger(
    arg_name="azeventhub",
    event_hub_name="graph-events",
    connection="EVENTHUB_CONNECTION_STRING",
)
def nikshay_note_ingestor(azeventhub: func.EventHubEvent) -> None:
    """
    Triggered by free_text_update events on the graph-events Event Hub.
    Runs Azure AI Language NER on the note text.
    Writes any new contact nodes and updates edge weights in Cosmos DB.

    This is the async closure of the ASHA portal feedback loop:
      ASHA types note → dashboard publishes event → this function fires →
      NER extracts contacts → graph updated → next morning briefing reflects it.
    """
    logging.info("nikshay_note_ingestor triggered.")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        raw = azeventhub.get_body().decode("utf-8")
        event = json.loads(raw)
        logging.info(f"Event received: {event.get('event_type')} "
                     f"| patient={event.get('target_node')} "
                     f"| asha={event.get('source_node')}")

        # Only process free-text notes — other event types are handled synchronously
        if event.get("event_type") != "free_text_update":
            logging.info(f"Skipping event type: {event.get('event_type')}")
            return

        note       = event.get("features", {}).get("text", "").strip()
        patient_id = event.get("target_node", "")
        asha_id    = event.get("source_node", "")

        if not note or not patient_id:
            logging.warning("Empty note or missing patient_id — skipping.")
            return

        from stage1_nlp   import (get_language_client, extract_contacts_from_note,
                                   get_eventhub_producer, ingest_contact_edge,
                                   publish_event)
        from cosmos_client import get_client, health_check

        lc = get_language_client()
        contacts = extract_contacts_from_note(lc, note)
        logging.info(f"NER extracted {len(contacts)} contacts from note.")

        gc = None
        try:
            if health_check():
                gc = get_client()
        except Exception as e:
            logging.warning(f"Cosmos DB unavailable — contacts not written. ({e})")

        producer = None
        try:
            producer = get_eventhub_producer()
        except Exception:
            pass

        if gc and contacts:
            # Build a minimal patient record stub so ingest_contact_edge works
            stub_record = {"patient_id": patient_id, "district": "Tondiarpet"}
            for c in contacts:
                cid = f"CONTACT_{c['name'].replace(' ', '_').replace('.', '')}"
                ingest_contact_edge(gc, producer, stub_record, c, cid)
                logging.info(f"  Contact edge written: {cid}")

        # Publish a processed event back so the dashboard activity feed updates
        if producer:
            publish_event(producer, "note_processed", asha_id, patient_id,
                          {"contacts_found": len(contacts), "source": "function"})

        logging.info("nikshay_note_ingestor complete.")

    except Exception as e:
        logging.exception(f"nikshay_note_ingestor FAILED: {e}")
        raise
