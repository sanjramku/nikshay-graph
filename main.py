"""
main.py
=======
Nikshay-Graph Pipeline Orchestrator

Runs all five stages in sequence:
  Stage 1 — NLP ingestion, graph construction, silence detection
  Stage 2 — TGN temporal risk inference
  Stage 3 — Dropout risk classification (3-component score)
  Stage 3b — Personalised PageRank over NetworkX graph
  Stage 4 — Explanation generation + safety validation
  Stage 5 — Morning briefing generation (dashboard delivery)

FIXES APPLIED vs original codebase:
  - Imports stage1_nlp (not stage1_ingest — that file does not exist)
  - inject_silence_events called ONCE here, not inside ingest_all (was doubling)
  - save_memory_to_cosmos uses datetime.now(timezone.utc).isoformat() — no os.popen()
  - stage5_voice.run_morning_briefings returns a dict; no variable overwrite bug
  - WhatsApp/SMS delivery removed; briefings saved to briefings_output.json for dashboard

Usage:
    python main.py                         # full run, 100 records, Cosmos enabled
    python main.py --skip-cosmos           # skip all Cosmos DB writes
    python main.py --skip-comms            # (no-op — comms now go to dashboard)
    python main.py --limit 20              # use first 20 records only
    python main.py --generate              # regenerate synthetic dataset first
"""

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Nikshay-Graph pipeline runner")
    p.add_argument("--skip-cosmos", action="store_true",
                   help="Skip all Cosmos DB operations")
    p.add_argument("--skip-comms",  action="store_true",
                   help="No-op (communications now go to dashboard, not WhatsApp/SMS)")
    p.add_argument("--limit",       type=int, default=100,
                   help="Number of patient records to process (default: 100)")
    p.add_argument("--generate",    action="store_true",
                   help="Regenerate synthetic dataset before running pipeline")
    p.add_argument("--confirmed-cases", type=int, default=0,
                   help="Number of confirmed real dropout cases (affects BBN weight)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_or_generate_dataset(generate: bool, limit: int) -> list:
    dataset_path = "data/nikshay_grounded_dataset.json"

    if generate or not Path(dataset_path).exists():
        print("\n=== Generating synthetic dataset ===")
        from dataset_gen import generate_and_save
        generate_and_save()

    print(f"\n=== Loading dataset (limit: {limit}) ===")
    with open(dataset_path) as f:
        patients = json.load(f)

    patients = patients[:limit]
    print(f"  Loaded {len(patients)} records from {dataset_path}")
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — NLP, Graph Construction, Silence Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_stage1(patients: list, skip_cosmos: bool):
    print("\n=== Stage 1: NLP + Graph Construction ===")

    # NOTE: imports from stage1_nlp (not stage1_ingest)
    from stage1_nlp import (
        get_language_client,
        get_eventhub_producer,
        demo_ner,
        build_asha_summaries,
        inject_silence_events,
        ingest_all,
    )

    lc       = get_language_client()
    producer = get_eventhub_producer()

    # NER demo on sample notes
    print("\n  Running NER demo...")
    demo_ner(lc)

    # Build ASHA/ANM summaries
    print("\n  Building ASHA summaries...")
    asha_summaries = build_asha_summaries(patients)
    print(f"  {len(asha_summaries)} ASHA workers indexed")

    # Inject silence events ONCE (ingest_all does NOT call this internally)
    print("\n  Detecting silence events...")
    patients = inject_silence_events(patients, producer)

    # Graph upsert in Cosmos DB
    gc = None
    if not skip_cosmos:
        try:
            from cosmos_client import get_client, health_check
            print("\n  Connecting to Cosmos DB...")
            if health_check():
                gc = get_client()
                print("\n  Upserting graph nodes + edges...")
                ingest_all(gc, producer, patients, limit=len(patients))
            else:
                print("  ⚠ Cosmos DB health check failed — skipping graph writes")
        except Exception as e:
            print(f"  ⚠ Cosmos DB error: {e} — continuing without graph writes")
    else:
        print("  [--skip-cosmos] Skipping Cosmos DB operations")

    return patients, asha_summaries, gc, producer


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — TGN Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_stage2(patients: list, gc):
    print("\n=== Stage 2: TGN Temporal Risk Inference ===")
    from stage2_tgn import run_tgn_inference

    tgn_scores, attention_weights = run_tgn_inference(patients, gc=gc)
    print(f"  TGN inference complete. {len(tgn_scores)} patients scored.")

    # Sample output
    sample = list(tgn_scores.items())[:3]
    for pid, score in sample:
        print(f"    {pid}: tgn_score={score}")

    return tgn_scores, attention_weights


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Risk Classification + Systemic Failure Detection
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3(patients: list, tgn_scores: dict, asha_summaries: dict,
               confirmed_cases: int):
    print("\n=== Stage 3: Risk Classification ===")
    from stage3_score import score_all_patients, detect_systemic_failures

    patients = score_all_patients(
        patients,
        tgn_scores      = tgn_scores,
        asha_summaries  = asha_summaries,
        confirmed_cases = confirmed_cases,
    )
    systemic_alerts = detect_systemic_failures(patients)

    # Persist scores to Cosmos (if available)
    # (writeback_risk_scores is called from stage1_nlp)
    try:
        from stage1_nlp import writeback_risk_scores
        from cosmos_client import get_client
        _gc = get_client()
        writeback_risk_scores(_gc, patients)
    except Exception:
        pass  # Cosmos not configured or unreachable — non-fatal

    # Save scored dataset
    with open("nikshay_scored_dataset.json", "w") as f:
        json.dump(patients, f, indent=2)
    print(f"  Saved nikshay_scored_dataset.json ({len(patients)} records)")

    return patients, systemic_alerts


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3b — Personalised PageRank (NetworkX)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage3b(patients: list, gc):
    """
    Build an in-memory NetworkX graph and run Personalised PageRank.
    High-risk patients seed the personalisation vector.
    Propagates risk to contacts and ASHA workers.

    NOTE: This logic lives in main.py because it requires both the scored
    patient list (Stage 3 output) and a graph structure. A future refactor
    should move it to stage3_score.py.
    """
    print("\n=== Stage 3b: Personalised PageRank ===")
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not installed — skipping PageRank. pip install networkx")
        return {}, None

    G = nx.DiGraph()

    # Add patient nodes
    for p in patients:
        pid = p["patient_id"]
        G.add_node(pid, node_type="patient",
                   risk_score=p.get("risk_score", 0),
                   asha_id=p["operational"]["asha_id"],
                   block=p["location"]["block"])

        # Add contact nodes + edges
        for c in p.get("contact_network", []):
            cid = f"CONTACT_{c['name'].replace(' ', '_')}"
            if not G.has_node(cid):
                G.add_node(cid, node_type="contact",
                           name=c["name"], age=c.get("age", 30),
                           rel=c.get("rel", "Household"),
                           vulnerability=c.get("vulnerability_score", 1.0),
                           screened=c.get("screened", False),
                           source_patient=pid)
            base  = 0.9 if c.get("rel") == "Household" else 0.6
            weight= base * c.get("vulnerability_score", 1.0)
            G.add_edge(pid, cid, weight=weight)
            G.add_edge(cid, pid, weight=weight * 0.5)

        # ASHA → Patient edge
        aid = p["operational"]["asha_id"]
        if not G.has_node(aid):
            G.add_node(aid, node_type="asha_worker")
        load = p.get("asha_load_score", 0.3)
        recency = max(0, 1 - p["operational"]["last_asha_visit_days_ago"] / 30)
        G.add_edge(aid, pid, weight=(1 - load) * recency)

    # Build personalisation vector: high-risk patients get weight 1.0
    high_risk = {p["patient_id"]: p["risk_score"]
                 for p in patients if p.get("risk_score", 0) > 0.65}
    if not high_risk:
        print("  No high-risk patients — using uniform personalisation")
        personalisation = None
    else:
        total = sum(high_risk.values())
        personalisation = {n: (high_risk[n] / total if n in high_risk else 0.0)
                           for n in G.nodes()}

    try:
        pagerank_scores = nx.pagerank(G, alpha=0.85,
                                       personalization=personalisation,
                                       weight="weight", max_iter=200)
    except Exception as e:
        print(f"  PageRank failed: {e}")
        pagerank_scores = {}

    # Write PageRank scores back to Cosmos
    try:
        from stage1_nlp import writeback_pagerank_scores
        writeback_pagerank_scores(gc, pagerank_scores)
    except Exception:
        pass

    high_risk_contacts = sum(
        1 for n, d in G.nodes(data=True)
        if d.get("node_type") == "contact"
        and not d.get("screened", False)
        and pagerank_scores.get(n, 0) > 0.001
    )
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  PageRank complete. High-risk unscreened contacts: {high_risk_contacts}")

    return pagerank_scores, G


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Explanation Generation + Safety Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_stage4(patients: list, pagerank_scores: dict, G, top_n: int = 10):
    print("\n=== Stage 4: Explanation & Safety Validation ===")
    from stage4_explain import get_patient_visit_list, get_contact_screening_list

    visit_list = get_patient_visit_list(patients, top_n=top_n)
    print(f"\n  Top {len(visit_list)} patients for ASHA visit:")
    for v in visit_list[:5]:
        print(f"    Rank {v['rank']}: {v['patient_id']} [{v['risk_level']}]")
        print(f"      ASHA:    {v['asha_explanation']}")
        safety_str = "✓ passed" if v["safety_passed"] else "✗ BLOCKED"
        print(f"      Safety:  {safety_str}")

    screening_list = []
    if G is not None and pagerank_scores:
        screening_list = get_contact_screening_list(G, pagerank_scores, top_n=top_n)
        print(f"\n  Top {len(screening_list)} contacts for TB screening:")
        for c in screening_list[:3]:
            print(f"    Rank {c['rank']}: {c['name']} (age {c['age']}, {c['rel']}) "
                  f"— priority={c['screening_priority']:.6f}")

    output = {
        "visit_list":       visit_list,
        "screening_list":   screening_list,
        "systemic_alerts":  [],  # filled in by caller
    }
    with open("agent3_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved agent3_output.json")

    return visit_list, screening_list


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Morning Briefings (dashboard delivery)
# ─────────────────────────────────────────────────────────────────────────────

def run_stage5(visit_list: list, screening_list: list,
               systemic_alerts: list, patients: list):
    print("\n=== Stage 5: Morning Briefings (dashboard) ===")
    from stage5_voice import run_morning_briefings

    briefing_output = run_morning_briefings(
        visit_list      = visit_list,
        screening_list  = screening_list,
        systemic_alerts = systemic_alerts,
        patients        = patients,
    )

    # Serialise audio paths (may be temp file paths or None)
    serialisable = {}
    for asha_id, b in briefing_output["asha_briefings"].items():
        serialisable[asha_id] = {k: v for k, v in b.items() if k != "audio_path"}
        serialisable[asha_id]["audio_available"] = b.get("audio_path") is not None

    with open("briefings_output.json", "w") as f:
        json.dump({
            "asha_briefings":  serialisable,
            "systemic_alerts": briefing_output["systemic_alerts"],
        }, f, indent=2)
    print(f"  Saved briefings_output.json")
    print(f"  Launch dashboard: streamlit run app.py")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print("Nikshay-Graph Pipeline")
    print(f"  Records:          {args.limit}")
    print(f"  Cosmos DB:        {'SKIP' if args.skip_cosmos else 'ENABLED'}")
    print(f"  Confirmed cases:  {args.confirmed_cases}")
    print("=" * 60)

    # Stage 0 — Data
    patients = load_or_generate_dataset(args.generate, args.limit)

    # Stage 1 — NLP + Graph
    patients, asha_summaries, gc, producer = run_stage1(patients, args.skip_cosmos)

    # Stage 2 — TGN
    tgn_scores, attention_weights = run_stage2(patients, gc)

    # Stage 3 — Scoring
    patients, systemic_alerts = run_stage3(
        patients, tgn_scores, asha_summaries, args.confirmed_cases
    )

    # Stage 3b — PageRank
    pagerank_scores, G = run_stage3b(patients, gc)

    # Stage 4 — Explanations
    visit_list, screening_list = run_stage4(patients, pagerank_scores, G)

    # Patch systemic_alerts into agent3_output.json
    with open("agent3_output.json") as f:
        ao = json.load(f)
    ao["systemic_alerts"] = systemic_alerts
    with open("agent3_output.json", "w") as f:
        json.dump(ao, f, indent=2)

    # Stage 5 — Briefings
    run_stage5(visit_list, screening_list, systemic_alerts, patients)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  nikshay_scored_dataset.json  — scored patients")
    print(f"  agent3_output.json           — visit + screening lists")
    print(f"  briefings_output.json        — ASHA briefings for dashboard")
    print(f"\nLaunch dashboard:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
