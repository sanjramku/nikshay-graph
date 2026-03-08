"""
function_app.py — Nikshay-Graph Azure Functions
================================================

Two functions:

1. nikshay_note_ingestor  [Event Hubs trigger]
   Fires whenever an ASHA submits a note or action via the dashboard.
   Queues the note for overnight NER — does NOT run NER immediately.
   This keeps the dashboard responsive (no 2-second NER delay per tap).

2. nikshay_overnight_processor  [Timer trigger — 22:00 IST daily]
   Runs NER on all queued notes from the day.
   Updates graph nodes and edges in Cosmos DB.
   Rescores affected patients.
   Saves results to data/overnight_results.json for the dashboard to read.
   The next morning's briefings are generated AFTER this runs.

Deployment:
    az functionapp create \
        --resource-group nikshaydbgraph \
        --name nikshay-pipeline-func \
        --storage-account nikshaydbgraph8a14 \
        --runtime python \
        --runtime-version 3.12 \
        --functions-version 4 \
        --os-type linux

    az functionapp config appsettings set \
        --name nikshay-pipeline-func \
        --resource-group nikshaydbgraph \
        --settings @env_settings.json

Environment variables required (same as .env):
    EVENTHUB_CONNECTION_STRING
    EVENTHUB_NAME
    COSMOS_ENDPOINT
    COSMOS_DATABASE
    COSMOS_GRAPH
    COSMOS_KEY
    LANGUAGE_ENDPOINT
    LANGUAGE_KEY
    SPEECH_KEY
    SPEECH_REGION
    TRANSLATOR_KEY
    TRANSLATOR_REGION
"""

import azure.functions as func
import json
import logging
import os
from datetime import datetime, timezone

app = func.FunctionApp()


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Event Hubs trigger: queue every ASHA action for overnight NER
# Fires on every event published by the dashboard (_reply_event in app.py)
# ─────────────────────────────────────────────────────────────────────────────

@app.event_hub_message_trigger(
    arg_name="event",
    event_hub_name="%EVENTHUB_NAME%",
    connection="EVENTHUB_CONNECTION_STRING",
    cardinality="many",          # batch mode — processes multiple events per call
    consumer_group="$Default",
)
def nikshay_note_ingestor(event: func.EventHubEvent) -> None:
    """
    Receives ASHA actions from Event Hubs.
    Routes them based on event_type:
      - dose_confirmed / dose_missed → immediate writeback to Cosmos DB
        (these are time-critical — done instantly, not queued overnight)
      - free_text_update → queued for overnight NER
      - contact_screened → immediate writeback
      - issue_flagged    → logged for District Officer

    Why immediate for dose actions?
    Dose confirmation changes the patient's silence status. If we wait until
    22:00 to process it, the dashboard shows wrong silence counts all day.
    NER on free text is expensive and not time-critical — overnight is fine.
    """
    try:
        body = event.get_body().decode("utf-8")
        payload = json.loads(body)
    except Exception as e:
        logging.error(f"[Ingestor] Failed to parse event: {e}")
        return

    event_type = payload.get("event_type", "")
    patient_id = payload.get("target_node", payload.get("features", {}).get("patient_id", ""))
    source_id  = payload.get("source_node", "")
    features   = payload.get("features", {})

    logging.info(f"[Ingestor] Received: {event_type} | patient={patient_id} | source={source_id}")

    # Get graph client
    gc, producer = _get_clients()

    try:
        if event_type == "dose_confirmed":
            # Immediate — resets silence, updates edge weight
            from stage1_nlp import writeback_dose_confirmed
            delta = writeback_dose_confirmed(gc, producer, patient_id, source_id)
            logging.info(f"[Ingestor] dose_confirmed writeback: {delta}")

        elif event_type == "dose_missed":
            # Immediate — increments silence_days, decays edge weight
            from stage1_nlp import writeback_dose_missed
            delta = writeback_dose_missed(gc, producer, patient_id, source_id)
            logging.info(f"[Ingestor] dose_missed writeback: {delta}")

        elif event_type == "contact_screened":
            # Immediate — sets contact.screened = true in graph
            from stage1_nlp import writeback_contact_screened
            contact_name = features.get("contact_name", "")
            writeback_contact_screened(gc, producer, patient_id, contact_name, source_id)
            logging.info(f"[Ingestor] contact_screened: {contact_name}")

        elif event_type == "free_text_update":
            # Queue for overnight NER — not immediate
            note = features.get("text", "")
            if note:
                from stage1_nlp import queue_note_for_overnight
                queue_note_for_overnight(
                    patient_id=patient_id,
                    asha_id=source_id,
                    note=note,
                    action="free_text",
                )
                logging.info(f"[Ingestor] Note queued for overnight: {patient_id} — '{note[:50]}'")
            else:
                logging.warning(f"[Ingestor] free_text_update with empty text — skipped")

        elif event_type == "issue_flagged":
            # Log for District Officer — no graph change needed
            logging.info(
                f"[Ingestor] Issue flagged by {source_id} for patient {patient_id}. "
                f"District Officer will see this in their dashboard."
            )

        else:
            logging.info(f"[Ingestor] Unhandled event_type: {event_type} — logged only")

    except Exception as e:
        logging.error(f"[Ingestor] Error processing {event_type} for {patient_id}: {e}", exc_info=True)

    finally:
        if producer:
            try:
                producer.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2 — Timer trigger: overnight batch NER + graph update + rescore
# Runs at 22:00 IST (16:30 UTC) every day
# After this runs, main.py --briefings generates the next morning's briefings
# ─────────────────────────────────────────────────────────────────────────────

@app.timer_trigger(
    arg_name="timer",
    schedule="0 30 16 * * *",   # 22:00 IST = 16:30 UTC, every day
    run_on_startup=False,        # don't run on deploy — only on schedule
    use_monitor=True,            # prevents duplicate runs if function restarts
)
def nikshay_overnight_processor(timer: func.TimerRequest) -> None:
    """
    End-of-day NER batch processor.

    Sequence:
    1. Load all pending notes from data/pending_notes.json
    2. For each note:
       a. Run NER → extract contacts + intents
       b. writeback_note_to_patient() — store note on patient node
       c. writeback_new_contact() — add any new contacts + edges
       d. writeback_symptom_flag() — boost vulnerability on symptomatic contacts
       e. Rescore patient → writeback_risk_scores()
    3. Save summary to data/overnight_results.json
    4. Clear processed notes from queue
    5. Trigger morning briefing generation for updated patients

    ASHA workers get their updated briefings the next morning.
    The briefing will mention new contacts found from their notes.
    """
    if timer.past_due:
        logging.warning("[Overnight] Timer is past due — running now but may be delayed")

    run_start = datetime.now(timezone.utc)
    logging.info(f"[Overnight] Starting at {run_start.isoformat()}")

    gc, producer = _get_clients()

    try:
        from stage1_nlp import process_overnight_notes
        results = process_overnight_notes(gc, producer)

        logging.info(
            f"[Overnight] Complete — "
            f"processed={results['processed']} "
            f"contacts_added={results['contacts_added']} "
            f"symptoms_flagged={results['symptoms_flagged']} "
            f"tier_changes={results['tier_changes']} "
            f"errors={len(results['errors'])}"
        )

        # Save results for dashboard to display
        _save_overnight_results(results, run_start)

        # If any patients were rescored, regenerate morning briefings
        if results["processed"] > 0:
            _regenerate_briefings(results["graph_deltas"])

    except Exception as e:
        logging.error(f"[Overnight] Fatal error: {e}", exc_info=True)
        _save_overnight_results({"error": str(e), "processed": 0}, run_start)

    finally:
        if producer:
            try:
                producer.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_clients():
    """Return (gremlin_client, eventhub_producer). Either may be None if not configured."""
    gc = None
    producer = None
    try:
        from cosmos_client import get_client, health_check
        if health_check():
            gc = get_client()
    except Exception as e:
        logging.warning(f"[Clients] Cosmos DB unavailable: {e}")

    try:
        from stage1_nlp import get_eventhub_producer
        producer = get_eventhub_producer()
    except Exception as e:
        logging.warning(f"[Clients] Event Hubs unavailable: {e}")

    return gc, producer


def _save_overnight_results(results: dict, run_start: datetime):
    """
    Persist overnight results to disk.
    The Officer dashboard reads this to show what the overnight run changed.
    """
    import os
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    results["run_start"] = run_start.isoformat()
    results["run_end"]   = datetime.now(timezone.utc).isoformat()

    with open("data/overnight_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info("[Overnight] Results saved to data/overnight_results.json")


def _regenerate_briefings(graph_deltas: list):
    """
    Regenerate morning briefings for ASHA workers whose patients were updated.
    Called at the end of the overnight processor so workers get fresh briefings
    with contacts found from their notes included.
    """
    try:
        import json as _json
        from pathlib import Path

        # Load the scored dataset (updated by overnight processor)
        scored_path = Path("nikshay_scored_dataset.json")
        if not scored_path.exists():
            logging.warning("[Briefings] nikshay_scored_dataset.json not found — skipping")
            return

        with open(scored_path) as f:
            patients = _json.load(f)

        # Only regenerate briefings for ASHAs whose patients changed
        affected_pids  = {d["patient_id"] for d in graph_deltas}
        affected_ashas = {
            p["operational"]["asha_id"]
            for p in patients
            if p["patient_id"] in affected_pids
        }
        logging.info(f"[Briefings] Regenerating for {len(affected_ashas)} ASHA workers: {affected_ashas}")

        # Load existing agent3 output for visit/screening lists
        agent3_path = Path("agent3_output.json")
        if not agent3_path.exists():
            logging.warning("[Briefings] agent3_output.json not found — skipping briefing regen")
            return

        with open(agent3_path) as f:
            agent3 = _json.load(f)

        from stage5_voice import run_morning_briefings
        briefings = run_morning_briefings(
            visit_list      = agent3.get("visit_list", []),
            screening_list  = agent3.get("screening_list", []),
            systemic_alerts = agent3.get("systemic_alerts", []),
            patients        = patients,
        )

        with open("briefings_output.json", "w") as f:
            _json.dump(briefings, f, indent=2)

        logging.info("[Briefings] briefings_output.json regenerated with overnight updates")

    except Exception as e:
        logging.error(f"[Briefings] Regeneration failed: {e}", exc_info=True)
