"""
app.py — Nikshay-Graph Dashboard (fixed)

Changes from original:
  1. _reply_event() moved BEFORE tab code — was crashing all button clicks
  2. Tab names changed to user-facing labels (not pipeline stage numbers)
  3. Score Distribution x-axis fixed (was showing raw Python dict strings)
  4. Score Composition table: explains uniform weights when BBN not yet faded
  5. High Risk sidebar delta now red (inverse) — 49% high risk is not a good trend
  6. ASHA load: single combined chart instead of two disconnected bar charts
  7. Contact screening "Priority" column replaced with readable rank + description
  8. BBN Odds Ratios chart: coloured bars + plain-English annotation
  9. ASHA portal subtitle changed from internal architecture to user-facing copy
 10. District Officer tab: full workload + hotspot + gap analysis (from officer_dashboard.py)
 11. Audio file: shows "Generate Audio" note instead of stale error message
 12. Sidebar logo uses emoji fallback instead of broken image path
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Nikshay-Graph",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# _reply_event — MUST be defined BEFORE any tab code
# Original bug: defined at line 440, called in Tab 5 → NameError on every click
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# RESCORE PATIENT — updates JSON on disk after ASHA action, triggers rerun
# This is the core of the real-time graph update system.
# Called after every patient card submission — no pipeline re-run needed.
# ─────────────────────────────────────────────────────────────────────────────

def rescore_patient_locally(patient_id: str, action: str, note: str = "") -> dict:
    """
    Apply an ASHA update to a patient and immediately re-score them.
    Writes the result back to nikshay_scored_dataset.json so the next
    cache refresh picks up the new score and tier.

    Returns {old_score, new_score, old_tier, new_tier, changed: bool}
    """
    import json
    from pathlib import Path
    from stage3_score import (compute_bbn_prior, compose_final_score,
                               apply_urgency_multiplier, assign_risk_tier,
                               get_adaptive_thresholds)

    json_path = Path("nikshay_scored_dataset.json")
    if not json_path.exists():
        json_path = Path("data/nikshay_scored_dataset.json")
    if not json_path.exists():
        return {}

    with open(json_path, encoding="utf-8") as f:
        all_patients = json.load(f)

    patient = next((p for p in all_patients if p["patient_id"] == patient_id), None)
    if not patient:
        return {}

    old_score = patient.get("risk_score", 0)
    old_tier  = patient.get("risk_level", "?")

    # Apply the action to the patient record
    if action == "done":
        patient["adherence"]["days_since_last_dose"] = 0
        patient["operational"]["last_asha_visit_days_ago"] = 0
        if "silence_event" in patient:
            del patient["silence_event"]
    elif action == "could_not_visit":
        patient["operational"]["last_asha_visit_days_ago"] = (
            patient["operational"].get("last_asha_visit_days_ago", 0) + 1
        )
    elif action == "free_text" and note:
        # Append note to free_text_note for next NER run
        existing = patient.get("free_text_note", "")
        patient["free_text_note"] = (existing + " " + note).strip()

    # Re-score with updated data
    bbn_result   = compute_bbn_prior(patient)
    bbn_score    = bbn_result["score"]
    tgn_score    = patient.get("tgn_score", bbn_score)
    asha_load    = patient.get("asha_load_score", 0.3)
    composition  = compose_final_score(tgn_score, bbn_score, asha_load)
    week         = min(patient["clinical"]["total_treatment_days"] // 7, 26)
    new_score    = apply_urgency_multiplier(composition["composite_score"], week)
    new_tier     = assign_risk_tier(new_score, week)
    thresholds   = get_adaptive_thresholds(week)

    patient["previous_risk_score"] = old_score
    patient["risk_score"]          = new_score
    patient["risk_level"]          = new_tier
    patient["risk_velocity"]       = round(new_score - old_score, 4)
    patient["score_composition"]   = composition
    patient["thresholds"]          = thresholds
    patient["all_factors"]         = bbn_result["all_factors"]
    patient["top_factors"]         = dict(
        sorted(bbn_result["all_factors"].items(), key=lambda x: x[1], reverse=True)[:3]
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_patients, f, indent=2)

    return {
        "old_score": round(old_score, 3),
        "new_score": round(new_score, 3),
        "old_tier":  old_tier,
        "new_tier":  new_tier,
        "changed":   old_tier != new_tier,
        "velocity":  round(new_score - old_score, 4),
    }

def _reply_event(asha_id: str, patient_id: str, action: str,
                 free_text: str = "", contact_name: str = ""):
    """
    Submit an ASHA action to the backend.
    Degrades gracefully if Cosmos DB / Event Hubs are not configured.
    """
    try:
        from stage5_voice import process_asha_dashboard_reply
        from stage1_nlp import get_eventhub_producer
        producer  = get_eventhub_producer()
        gc        = None
        cosmos_ok = False
        try:
            from cosmos_client import get_client, health_check
            if health_check():
                gc        = get_client()
                cosmos_ok = True
        except Exception:
            pass

        result = process_asha_dashboard_reply(
            gc=gc, producer=producer, action=action,
            patient_id=patient_id, asha_id=asha_id,
            free_text=free_text, contact_name=contact_name,
        )

        # Build a precise log message from the graph delta
        delta = result.get("graph_delta") if result else None

        if action == "done":
            if delta and delta.get("edge_weight_old") is not None:
                w_old = delta["edge_weight_old"]
                w_new = delta["edge_weight_new"]
                arrow = "↑" if w_new > w_old else ("↓" if w_new < w_old else "=")
                detail = (
                    f"✅ Dose confirmed — silence cleared · "
                    f"ASHA→patient edge weight {w_old:.3f} {arrow} {w_new:.3f} "
                    f"(recency restored, load={delta['load_score']:.2f})"
                )
            else:
                detail = "✅ Dose confirmed — days_missed reset to 0, silence cleared"

        elif action == "could_not_visit":
            if delta and delta.get("edge_weight_old") is not None:
                w_old  = delta["edge_weight_old"]
                w_new  = delta["edge_weight_new"]
                d_old  = delta["days_since_visit_old"]
                d_new  = delta["days_since_visit_new"]
                arrow  = "↓" if w_new < w_old else "="
                sdays  = delta["node_changes"].get("silence_days", "")
                detail = (
                    f"❌ Could not visit — silence_days {sdays} · "
                    f"ASHA→patient edge weight {w_old:.3f} {arrow} {w_new:.3f} "
                    f"(days_since_visit {d_old}→{d_new}, recency decayed)"
                )
            else:
                detail = "❌ Visit missed — silence_days incremented, edge weight decayed"

        elif action == "contact_screened":
            detail = f"👁 Contact '{contact_name}' screened — contact node updated in graph"
        elif action == "issue":
            detail = "⚠️ Issue flagged for District Officer"
        elif action == "free_text":
            # Queue the note for overnight NER processing
            if free_text.strip():
                try:
                    from stage1_nlp import queue_note_for_overnight
                    queue_note_for_overnight(patient_id, asha_id, free_text)
                    detail = "📝 Note queued for overnight NER — contacts & symptoms will be extracted tonight"
                except Exception as qe:
                    detail = f"📝 Note saved locally (queue unavailable: {qe})"
            else:
                detail = "📝 Empty note — not queued"
        else:
            detail = action

        # Rescore locally — scores refresh fully overnight after NER runs
        rescore = rescore_patient_locally(patient_id, action, free_text)
        if rescore and rescore.get("changed"):
            detail += f" 🔄 TIER CHANGE: {rescore['old_tier']} → {rescore['new_tier']}"

        # Always queue any note for overnight NER regardless of action type
        if free_text.strip():
            try:
                from stage1_nlp import queue_note_for_overnight
                queue_note_for_overnight(patient_id, asha_id, free_text, action)
                detail += " · 📝 Note queued for overnight NER"
            except Exception as qe:
                detail += f" · 📝 Note saved locally (queue error: {qe})"

        if cosmos_ok:
            detail += " · Cosmos DB ✓"
            st.toast(f"Graph updated: {patient_id}", icon="✅")
            if rescore and rescore.get("changed"):
                st.toast(f"⚡ {patient_id}: {rescore['old_tier']} → {rescore['new_tier']}", icon="🔄")
        else:
            detail += " · Saved locally (scores refresh overnight)"
            st.toast(f"Saved: {patient_id} — scores refresh overnight", icon="📋")
        log_graph_activity(action, patient_id, detail)
        # Clear cache so sidebar and all tabs see the new score immediately
        st.cache_data.clear()

    except Exception as e:
        st.toast(f"Backend not connected — action logged locally. ({e})", icon="⚠️")
        log_graph_activity(action, patient_id, f"OFFLINE: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_scored_patients():
    for p in ["nikshay_scored_dataset.json", "data/nikshay_scored_dataset.json"]:
        if Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return []

@st.cache_data(ttl=60)
def load_agent3():
    for p in ["agent3_output.json", "data/agent3_output.json"]:
        if Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return {"visit_list": [], "screening_list": [], "systemic_alerts": []}

@st.cache_data(ttl=60)
def load_briefings():
    for p in ["briefings_output.json", "data/briefings_output.json"]:
        if Path(p).exists():
            with open(p) as f:
                return json.load(f)
    return {"asha_briefings": {}, "systemic_alerts": []}

patients        = load_scored_patients()
agent3          = load_agent3()
briefings       = load_briefings()
visit_list      = agent3.get("visit_list", [])
screening_list  = agent3.get("screening_list", [])
systemic_alerts = agent3.get("systemic_alerts", [])
asha_briefings  = briefings.get("asha_briefings", {})


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# Fix: removed broken image URL (was showing "0"), use emoji + text instead
# Fix: High Risk delta now red (inverse) — high risk is not a positive trend
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🫁 Nikshay-Graph")
    st.caption("TB Treatment Dropout Prevention\nIIT Madras · Microsoft AI Unlocked")
    st.divider()

    n_patients = len(patients)
    n_high     = sum(1 for p in patients if p.get("risk_level") == "HIGH")
    n_silent   = sum(1 for p in patients
                     if max(p["adherence"]["days_since_last_dose"],
                            p["operational"]["last_asha_visit_days_ago"]) >= 7
                     ) if patients else 0
    n_alerts   = len(systemic_alerts)

    st.metric("Total Patients",     n_patients)
    # delta_color="inverse" → red arrow for high numbers (bad, not good)
    st.metric("High Risk",          n_high,
              delta=f"{n_high/max(n_patients,1)*100:.1f}% of district",
              delta_color="inverse")
    st.metric("Silent (7d+)",       n_silent,
              delta=f"{n_silent/max(n_patients,1)*100:.1f}%",
              delta_color="inverse")
    st.metric("Systemic Alerts",    n_alerts, delta_color="inverse")

    st.divider()
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    if not patients:
        st.warning("No data loaded.\nRun: `python main.py --limit 100`")


    # Live update feed in sidebar — visible from any tab
    recent = st.session_state.get("graph_activity", [])[:5]
    if recent:
        st.divider()
        st.markdown("**📡 Recent graph updates**")
        for ev in recent:
            tier_icon = "🔄" if "TIER CHANGE" in ev.get("detail","") else "✓"
            st.caption(f"{tier_icon} {ev['time']} · {ev['patient_id']}")


# ─────────────────────────────────────────────────────────────────────────────
# TABS — renamed from "Stage N: ..." to user-facing labels
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "🎯 Patient Risk Board",
    "📊 Risk Analytics",
    "🕸️ Network & Screening",
    "💬 Explanations",
    "👷 ASHA Portal",
    "🏛️ Officer Command View",
])



# ─────────────────────────────────────────────────────────────────────────────
# GRAPH UPDATE ACTIVITY LOG  (session state — shows what changed in Cosmos DB)
# ─────────────────────────────────────────────────────────────────────────────

if "graph_activity" not in st.session_state:
    st.session_state.graph_activity = []   # list of {time, action, patient_id, detail}

def log_graph_activity(action: str, patient_id: str, detail: str):
    from datetime import datetime
    st.session_state.graph_activity.insert(0, {
        "time":       datetime.now().strftime("%H:%M:%S"),
        "action":     action,
        "patient_id": patient_id,
        "detail":     detail,
    })
    # Keep only last 20 events
    st.session_state.graph_activity = st.session_state.graph_activity[:20]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Patient Risk Board
# Was "Stage 1: Graph" — engineers understand "graph construction", judges don't
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.header("Patient Risk Board")
    st.caption("All patients ranked by dropout risk · Updated after each ASHA report")

    if not patients:
        st.info("Run the pipeline first: `python main.py --limit 100`")
    else:
        c1, c2, c3, c4 = st.columns(4)
        n_high_t   = sum(1 for p in patients if p.get("risk_level") == "HIGH")
        n_medium_t = sum(1 for p in patients if p.get("risk_level") == "MEDIUM")
        n_low_t    = sum(1 for p in patients if p.get("risk_level") == "LOW")
        c1.metric("🔴 HIGH",    n_high_t,   f"{n_high_t/n_patients*100:.0f}%")
        c2.metric("🟡 MEDIUM",  n_medium_t, f"{n_medium_t/n_patients*100:.0f}%")
        c3.metric("🟢 LOW",     n_low_t,    f"{n_low_t/n_patients*100:.0f}%")
        c4.metric("💊 DR-TB",   sum(1 for p in patients if p["clinical"]["regimen"] == "DR_TB"))

        st.divider()
        st.subheader("All Patients — sorted by urgency")
        st.caption(
            "rank_score = risk score × treatment-week urgency multiplier. "
            "A 0.70 risk at week 24 ranks above 0.70 at week 4 — "
            "dropout near end of treatment means 24 weeks of progress wasted."
        )

        rows = []
        for p in sorted(patients,
                        key=lambda x: x.get("rank_score", x.get("risk_score", 0)),
                        reverse=True):
            missed  = p["adherence"]["days_since_last_dose"]
            silence = max(missed, p["operational"]["last_asha_visit_days_ago"])
            factors = list(p.get("top_factors", {}).keys())
            rows.append({
                "Risk":          p.get("risk_level", "?"),
                "Patient ID":    p["patient_id"],
                "Risk Score":    round(p.get("risk_score", 0), 3),
                "Rank Score":    round(p.get("rank_score", p.get("risk_score", 0)), 3),
                "Week":          p.get("treatment_week", "?"),
                "Phase":         p["clinical"]["phase"],
                "Days Missed":   missed,
                "Silent (days)": silence,
                "Top Factor":    factors[0] if factors else "—",
                "ASHA":          p["operational"]["asha_id"],
                "Block":         p["location"]["block"],
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True, height=500)

        # Graph schema collapsed — useful for demo Q&A but not the landing view
        st.divider()
        with st.expander("📐 Graph schema — node & edge types (for technical review)", expanded=False):
            nc, ec = st.columns(2)
            with nc:
                st.markdown("**Node Types (7)**")
                node_df = pd.DataFrame([
                    {"Node Type": "Patient",       "Count": len(patients),
                     "Key Properties": "risk_score, phase, treatment_week, days_missed, memory_vector"},
                    {"Node Type": "ASHA Worker",   "Count": len({p["operational"]["asha_id"] for p in patients}),
                     "Key Properties": "caseload, load_score, visit_freq_7d, high_risk_count"},
                    {"Node Type": "ANM Worker",    "Count": len({p["operational"].get("anm_id","") for p in patients}),
                     "Key Properties": "asha_count, avg_load_score, high_risk_total"},
                    {"Node Type": "PHC",           "Count": 1,
                     "Key Properties": "block, drug_available, staffed"},
                    {"Node Type": "Welfare Scheme","Count": 1,
                     "Key Properties": "Nikshay Poshan Yojana, payment_status"},
                    {"Node Type": "Village/Ward",  "Count": len({p["location"]["block"] for p in patients}),
                     "Key Properties": "connectivity_score, avg_distance_km, low_edu_rate"},
                    {"Node Type": "Contact",       "Count": sum(len(p.get("contact_network",[])) for p in patients),
                     "Key Properties": "vulnerability_score, rel, screened, age"},
                ])
                st.dataframe(node_df, use_container_width=True, hide_index=True)
            with ec:
                st.markdown("**Edge Types (8)**")
                edge_df = pd.DataFrame([
                    {"Edge": "household_contact", "Weight Formula": "0.9 × vulnerability", "Decay": "0.02/wk"},
                    {"Edge": "workplace_contact", "Weight Formula": "0.6 × vulnerability", "Decay": "0.05/wk"},
                    {"Edge": "shared_contact",    "Weight Formula": "vulnerability × risk","Decay": "0.04/wk"},
                    {"Edge": "assigned_to",       "Weight Formula": "(1−load) × recency",  "Decay": "0.03/wk"},
                    {"Edge": "attends (PHC)",     "Weight Formula": "1/(1+dist×0.1)",      "Decay": "0.01/wk"},
                    {"Edge": "supervises (ANM)",  "Weight Formula": "1.0",                 "Decay": "0.0"},
                    {"Edge": "enrolled_in",       "Weight Formula": "1.0",                 "Decay": "0.0"},
                    {"Edge": "social (stub)",     "Weight Formula": "0.3",                 "Decay": "fast"},
                ])
                st.dataframe(edge_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk Analytics
# Fix: Score Distribution x-axis was showing raw Python dict bin objects
# Fix: Score Composition explains uniform weights (not a bug — Phase 1)
# Fix: BBN chart uses coloured horizontal bars with plain-English annotation
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.header("Risk Analytics")

    if not patients:
        st.info("Run the pipeline first: `python main.py --limit 100`")
    else:
        import plotly.graph_objects as go

        scores = [p.get("risk_score", 0) for p in patients]
        tiers  = [p.get("risk_level", "LOW") for p in patients]

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 HIGH",   tiers.count("HIGH"),
                  f"{tiers.count('HIGH')/len(tiers)*100:.0f}%")
        c2.metric("🟡 MEDIUM", tiers.count("MEDIUM"),
                  f"{tiers.count('MEDIUM')/len(tiers)*100:.0f}%")
        c3.metric("🟢 LOW",    tiers.count("LOW"),
                  f"{tiers.count('LOW')/len(tiers)*100:.0f}%")

        # ── Score distribution — fixed x-axis ────────────────────────────
        st.subheader("Risk Score Distribution")
        st.caption(
            "Score = calibrated dropout probability (0–1). "
            "0.72 means this patient has a 72% modelled probability of dropping out. "
            "The urgency multiplier is applied separately for visit ordering only."
        )

        bins   = np.arange(0, 1.05, 0.05)
        counts, edges = np.histogram(scores, bins=bins)
        # Fixed: use plain numeric bin labels, not Python dict objects
        bin_labels    = [f"{edges[i]:.2f}" for i in range(len(counts))]
        bar_colours   = [
            "#f87171" if (edges[i]+edges[i+1])/2 >= 0.65 else
            "#fbbf24" if (edges[i]+edges[i+1])/2 >= 0.40 else
            "#4ade80"
            for i in range(len(counts))
        ]

        fig_hist = go.Figure(go.Bar(
            x=bin_labels, y=counts,
            marker_color=bar_colours,
            hovertemplate="Score ≥ %{x}<br>Patients: %{y}<extra></extra>",
        ))
        fig_hist.add_vline(
            x=0.65, line_dash="dash", line_color="#f87171",
            annotation_text="HIGH threshold (late continuation phase)",
            annotation_font_size=11,
        )
        fig_hist.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Number of Patients",
            xaxis_tickangle=-45,
            height=360,
            margin=dict(l=20, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()

        # ── Score composition ─────────────────────────────────────────────
        st.subheader("Score Composition — Top 20 Patients")

        # Detect Phase 1 (all weights identical) and explain it — not hide it
        sample_sc = patients[0].get("score_composition", {})
        all_same  = all(
            p.get("score_composition", {}).get("bbn_weight") == sample_sc.get("bbn_weight")
            for p in patients[:20]
        )
        if all_same:
            bbn_w = sample_sc.get("bbn_weight", 0.4)
            tgn_w = sample_sc.get("tgn_weight", 0.6)
            st.info(
                f"ℹ️ **Why are all weights identical?** "
                f"With 0 confirmed dropouts so far, the system uses its Phase 1 weights: "
                f"TGN {tgn_w*100:.0f}% · BBN {bbn_w*100:.0f}% · ASHA Load 20%. "
                f"The BBN component fades automatically as real dropout cases are confirmed — "
                f"fully retired at 200 confirmed cases."
            )

        top20 = sorted(patients,
                       key=lambda x: x.get("rank_score", x.get("risk_score", 0)),
                       reverse=True)[:20]
        comp_data = []
        for p in top20:
            sc = p.get("score_composition", {})
            comp_data.append({
                "Patient":     p["patient_id"],
                "Risk Score":  round(p.get("risk_score", 0), 3),
                "Rank Score":  round(p.get("rank_score", p.get("risk_score", 0)), 3),
                "Tier":        p.get("risk_level", "?"),
                "Week":        p.get("treatment_week", "?"),
                "Phase":       p["clinical"]["phase"],
                "Days Missed": p["adherence"]["days_since_last_dose"],
                "TGN weight":  f"{sc.get('tgn_weight',0)*100:.0f}%",
                "BBN weight":  f"{sc.get('bbn_weight',0)*100:.0f}%",
                "ASHA weight": "20%",
                "BBN status":  sc.get("bbn_status", "active"),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.divider()

        # ── ASHA Load — combined chart ────────────────────────────────────
        # Fix: was two disconnected unlabelled bar charts side by side
        # Now: one horizontal chart with both load score and HIGH-risk count visible
        st.subheader("ASHA Worker Load Distribution")
        st.caption(
            "Load score = caseload pressure (40%) + missed-dose rate (30%) + "
            "high-risk proportion (30%). Above 0.60 = overloaded."
        )

        asha_groups = defaultdict(list)
        for p in patients:
            asha_groups[p["operational"]["asha_id"]].append(p)

        load_rows = []
        for aid, pts in asha_groups.items():
            n        = len(pts)
            high     = sum(1 for x in pts if x.get("risk_level") == "HIGH")
            avg_miss = sum(x["adherence"]["days_since_last_dose"] for x in pts) / n
            load     = min(1.0, (n/15)*0.4 + (avg_miss/14)*0.3 + (high/max(n,1))*0.3)
            load_rows.append({
                "ASHA ID":    aid,
                "Load Score": round(load, 3),
                "HIGH Risk":  high,
                "Caseload":   n,
                "Status":     "🔴 Overloaded" if load > 0.70 else
                              "🟡 At risk"    if load > 0.50 else
                              "🟢 OK",
            })
        load_df = pd.DataFrame(load_rows).sort_values("Load Score", ascending=True)

        bar_colours_load = [
            "#f87171" if r["Load Score"] > 0.70 else
            "#fb923c" if r["Load Score"] > 0.50 else
            "#4ade80"
            for _, r in load_df.iterrows()
        ]

        fig_load = go.Figure(go.Bar(
            x=load_df["Load Score"],
            y=load_df["ASHA ID"],
            orientation="h",
            marker_color=bar_colours_load,
            text=[f"load={s:.2f}  |  {h} HIGH-risk  |  {c} patients"
                  for s, h, c in zip(load_df["Load Score"],
                                     load_df["HIGH Risk"],
                                     load_df["Caseload"])],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Load score: %{x:.3f}<br>"
                "HIGH risk patients: %{customdata[0]}<br>"
                "Caseload: %{customdata[1]}<extra></extra>"
            ),
            customdata=load_df[["HIGH Risk","Caseload"]].values,
        ))
        fig_load.add_vline(x=0.60, line_dash="dash", line_color="orange",
                           annotation_text="Overload threshold (0.60)",
                           annotation_font_size=11)
        fig_load.update_layout(
            xaxis=dict(range=[0, 1.3], title="Load Score"),
            height=max(300, len(load_df) * 32),
            margin=dict(l=20, r=240, t=10, b=30),
        )
        st.plotly_chart(fig_load, use_container_width=True)

        st.divider()

        # ── Silence detection ─────────────────────────────────────────────
        st.subheader("Silence Detection")
        st.caption(
            "Silence = no dose AND no ASHA visit for ≥7 days. "
            "Detected before a formal dose gap is recorded — the earliest observable dropout signal."
        )
        silence_records = [
            p for p in patients
            if max(p["adherence"]["days_since_last_dose"],
                   p["operational"]["last_asha_visit_days_ago"]) >= 7
        ]
        complete = sum(1 for p in silence_records
                       if p["adherence"]["days_since_last_dose"] >= 14)
        partial  = len(silence_records) - complete
        s1, s2, s3 = st.columns(3)
        s1.metric("Silent Patients (7d+)", len(silence_records),
                  f"{len(silence_records)/max(n_patients,1)*100:.0f}% of district")
        s2.metric("Complete silence (≥14d)", complete,
                  "Escalate immediately" if complete else "None")
        s3.metric("Partial silence (7–13d)", partial)
        st.caption("Phase-adaptive thresholds: 5d Intensive · 6d Late Continuation · 7d Continuation")

        st.divider()

        # ── BBN Odds Ratios — coloured + annotated ────────────────────────
        st.subheader("What drives dropout risk? — Published Odds Ratios")
        st.caption(
            "Each bar shows how much more likely a patient with this factor is to drop out. "
            "OR = 6.5 means 6.5× more likely to miss treatment. "
            "Every number is from peer-reviewed Tamil Nadu / India TB literature — not estimated."
        )
        or_data = {
            "Missed 14+ days":             6.50,
            "Missed 7–13 days":            3.20,
            "Divorced / Separated":        3.80,
            "DR-TB regimen":               2.80,
            "Drug use":                    2.40,
            "Continuation phase":          2.30,
            "HIV co-infection":            2.16,
            "Prior LTFU / TB history":     2.10,
            "Distance > 10 km":            2.10,
            "Age 20–39":                   2.07,
            "Alcohol use":                 1.92,
            "No nutritional support":      1.60,
            "Distance 5–10 km":            1.60,
            "No welfare (NPY) enrolment":  1.45,
            "Male sex":                    1.29,
            "Diabetes (protective in TN)": 0.52,
        }
        or_df = pd.DataFrame({
            "Factor":      list(or_data.keys()),
            "Odds Ratio":  list(or_data.values()),
        }).sort_values("Odds Ratio", ascending=True)

        fig_or = go.Figure(go.Bar(
            x=or_df["Odds Ratio"],
            y=or_df["Factor"],
            orientation="h",
            marker_color=[
                "#4ade80" if v < 1 else
                "#fbbf24" if v < 2.5 else
                "#f87171"
                for v in or_df["Odds Ratio"]
            ],
            hovertemplate="%{y}<br>OR = %{x:.2f}× more likely to drop out<extra></extra>",
        ))
        fig_or.add_vline(x=1.0, line_dash="dash", line_color="gray",
                         annotation_text="No effect (OR = 1.0)",
                         annotation_font_size=10)
        fig_or.update_layout(
            xaxis_title="Odds Ratio (× more likely to drop out)",
            height=480,
            margin=dict(l=20, r=60, t=10, b=40),
        )
        st.plotly_chart(fig_or, use_container_width=True)
        st.caption(
            "Sources: Vijay Kumar et al. 2020 (Tamil Nadu) · India TB Report 2023 · "
            "Figueiredo et al. PMC10760311 · Mistry et al. PMC6830133"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Network & Screening
# Fix: Contact screening "Priority" column — was raw PageRank decimals (0.0322)
#      Now shows rank 1–10 + human-readable screening reason
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.header("Network & Screening")
    st.caption(
        "PageRank propagates risk outward from HIGH-risk patients through the contact graph. "
        "Contacts of the most dangerous patients rise to the top of the screening list."
    )

    left, right = st.columns(2)

    with left:
        st.subheader(f"Priority Visit List ({len(visit_list)} patients)")
        if visit_list:
            vl_df = pd.DataFrame([{
                "Rank":        v["rank"],
                "Patient ID":  v["patient_id"],
                "Risk":        v["risk_level"],
                "Score":       round(v["risk_score"], 3),
                "Week":        v["treatment_week"],
                "Days Missed": v["days_missed"],
                "Block":       v["block"],
            } for v in visit_list])
            st.dataframe(vl_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run pipeline first.")

    with right:
        st.subheader(f"Contact Screening Priority ({len(screening_list)} contacts)")
        st.caption(
            "Unscreened contacts of HIGH/MEDIUM risk patients. "
            "Ranked by: patient risk × contact vulnerability × age risk × relationship. "
            "Call these people in for TB screening this week."
        )
        if screening_list:
            # Fix: replace raw PageRank score with readable label
            sl_df = pd.DataFrame([{
                "Priority":       c["rank"],
                "Name":           c["name"],
                "Age":            c["age"],
                "Relationship":   c["rel"],
                "Vulnerability":  c["vulnerability"],
                "Source Patient": c.get("source_patient", "—"),
                "Why screen?":    (
                    f"{'Household' if c['rel']=='Household' else 'Workplace'} contact of "
                    f"{'HIGH' if c.get('source_patient') else ''}-risk patient"
                    + (" · elderly" if c["age"] > 60 else
                       " · child" if c["age"] < 10 else "")
                ),
            } for c in screening_list])
            st.dataframe(sl_df, use_container_width=True, hide_index=True)
        else:
            st.info("Run pipeline first.")

    st.divider()

    st.subheader("Why thresholds tighten near end of treatment")
    st.caption(
        "A score of 0.60 at week 4 is MEDIUM risk. The same 0.60 at week 22 is HIGH risk. "
        "A dropout at week 22 wastes 22 weeks of treatment and leaves drug-resistant bacteria — "
        "the most direct path to MDR-TB."
    )
    thresh_df = pd.DataFrame([
        {"Treatment Phase":         "Intensive (wk 1–8)",           "HIGH >": "> 0.75", "MEDIUM >": "> 0.50"},
        {"Treatment Phase":         "Early Continuation (wk 9–16)", "HIGH >": "> 0.65", "MEDIUM >": "> 0.40"},
        {"Treatment Phase":         "Late Continuation (wk 17–26)", "HIGH >": "> 0.55", "MEDIUM >": "> 0.30"},
    ])
    st.dataframe(thresh_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Explanations
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.header("Patient Explanations")
    st.info(
        "🔒 **No LLM generation used here.** Every explanation is a fixed template "
        "populated from actual model outputs. Checked by Azure AI Foundry content "
        "safety before delivery. No hallucination risk in a clinical context."
    )

    if not visit_list:
        st.info("Run pipeline first.")
    else:
        for v in visit_list[:10]:
            tier_icon = "🔴" if v["risk_level"] == "HIGH" else "🟡"
            with st.expander(
                f"{tier_icon} Rank {v['rank']} — {v['patient_id']} "
                f"[{v['risk_level']}] · Score {v['risk_score']:.3f} · Week {v['treatment_week']}",
                expanded=(v["rank"] <= 2)
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown("**ASHA Worker message** *(simple, <20 words)*")
                    st.markdown(f"> {v['asha_explanation']}")
                    st.caption(f"Safety: {'✅ Passed' if v['safety_passed'] else '🚫 BLOCKED'}")
                with col_b:
                    st.markdown("**District Officer briefing** *(structured)*")
                    st.markdown(v["officer_explanation"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ASHA Portal
# Fix: subtitle changed from internal architecture to user-facing copy
# Fix: audio shows actionable note instead of stale error message
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    # ── Initialise session state for update tracking ──────────────────────────
    if "asha_updates" not in st.session_state:
        st.session_state.asha_updates = {}   # pid → {"action": str, "note": str}
    if "screened_contacts" not in st.session_state:
        st.session_state.screened_contacts = set()  # set of contact names marked done

    st.header("📋 ASHA Field Update Portal")
    st.caption(
        "Your daily patient visit list. Mark each visit as Done, "
        "record if you could not visit, or flag any concern for your supervisor."
    )

    asha_ids = sorted(asha_briefings.keys()) if asha_briefings else []
    if not asha_ids and patients:
        asha_ids = sorted({p["operational"]["asha_id"] for p in patients})

    if not asha_ids:
        st.info("No data yet — ask your supervisor to run the morning pipeline first.")
    else:
        selected_asha = st.selectbox(
            "Which ASHA worker are you?",
            asha_ids,
            help="Select your ID from the list. Each ASHA sees only their assigned patients."
        )
        briefing = asha_briefings.get(selected_asha)

        # ── Morning briefing box ──────────────────────────────────────────────
        if briefing:
            lang          = briefing.get("language", "Tamil")
            patient_count = briefing.get("patient_count", 0)

            st.info(
                f"**Good morning, {selected_asha}** · "
                f"You have **{patient_count} patients** to visit today · "
                f"Briefing language: {lang}"
            )

            with st.expander("📢 Read your full morning briefing", expanded=False):
                st.write(briefing.get("translated_text") or briefing.get("english_text", ""))

            audio_path      = briefing.get("audio_path")
            audio_available = briefing.get("audio_available", False)
            if audio_path and os.path.exists(str(audio_path)):
                st.markdown("**🔊 Voice briefing:**")
                with open(audio_path, "rb") as af:
                    st.audio(af.read(), format="audio/mp3")
                st.caption(f"Audio file: `{audio_path}`")
            elif audio_available:
                # File generated but not at expected path — offer to regenerate
                st.warning(
                    "🔇 Voice note was generated but the file is no longer on disk. "
                    "Re-run the pipeline to regenerate it."
                )
                if st.button("🔄 Regenerate audio", key=f"regen_{selected_asha}"):
                    try:
                        from stage5_voice import format_morning_briefing, translate_text
                        lang = briefing.get("language","Tamil")
                        eng  = briefing.get("english_text","")
                        if eng:
                            from stage5_voice import generate_voice_note
                            import shutil, pathlib
                            new_path = generate_voice_note(
                                translate_text(eng, lang) if lang != "English" else eng,
                                lang
                            )
                            if new_path and pathlib.Path(new_path).exists():
                                dest = pathlib.Path("data/audio") / f"{selected_asha}.mp3"
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(new_path, dest)
                                st.success(f"Audio regenerated → {dest}")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Regeneration failed: {e}")
            else:
                st.caption("🔇 Voice briefing not enabled — configure SPEECH_KEY in .env to enable Azure Neural TTS.")

            st.divider()

        # ── Visit cards ───────────────────────────────────────────────────────
        visit_cards = briefing.get("visit_cards", []) if briefing else []
        if not visit_cards:
            visit_cards = [v for v in visit_list if v.get("asha_id") == selected_asha]

        # Progress summary
        n_cards   = len(visit_cards)
        n_done    = sum(1 for c in visit_cards
                        if st.session_state.asha_updates.get(c["patient_id"], {}).get("action"))
        if n_cards:
            st.markdown(
                f"**Today's visits — {n_done} of {n_cards} updated** &nbsp;"
                f"{'✅ All done!' if n_done == n_cards and n_cards > 0 else ''}"
            )
            st.progress(n_done / n_cards if n_cards else 0)
            st.write("")

        if not visit_cards:
            st.info(f"No visits assigned to {selected_asha} today.")
        else:
            for card in visit_cards:
                pid    = card["patient_id"]
                tier   = card.get("risk_level", "?")
                score  = card.get("risk_score", 0)
                missed = card.get("days_missed", 0)
                expl   = card.get("explanation", card.get("asha_explanation", ""))
                icon   = "🔴" if tier == "HIGH" else ("🟡" if tier == "MEDIUM" else "🟢")

                # Show already-submitted status in the expander title
                prior = st.session_state.asha_updates.get(pid, {})
                prior_action = prior.get("action", "")
                status_label = {
                    "done":           " ✅ Dose confirmed",
                    "could_not_visit":" ❌ Could not visit",
                    "issue":          " ⚠️ Issue flagged",
                }.get(prior_action, "")

                # Plain language urgency label for ASHA workers
                urgency_label = (
                    "⚠ URGENT" if tier == "HIGH" else
                    "VISIT TODAY" if missed >= 7 else
                    "CHECK IN"
                )
                missed_label = (
                    f"{missed}d missed" if missed > 0 else "dose current"
                )
                with st.expander(
                    f"{icon} [{urgency_label}] {pid}  |  {missed_label}{status_label}",
                    expanded=(tier == "HIGH" and not prior_action)
                ):
                    # Structured visit card for field workers
                    info_col, score_col = st.columns([3, 1])
                    with info_col:
                        st.markdown(f"**📋 Visit reason:** {expl}")
                        # Show top risk factor in plain language
                        top_factors = card.get("top_factors", {})
                        if top_factors:
                            top_name = list(top_factors.keys())[0]
                            top_or   = list(top_factors.values())[0]
                            st.caption(f"Main risk: {top_name} ({top_or:.1f}× more likely to drop out)")
                    with score_col:
                        risk_pct = int(score * 100)
                        st.metric("Dropout risk", f"{risk_pct}%",
                                  help="Modelled probability of dropping out of treatment")
                        st.caption(f"Week {card.get('treatment_week','?')} · {card.get('phase', tier)}")

                    # Already updated — show confirmation, allow undo
                    if prior_action:
                        action_labels = {
                            "done":           "✅ You marked: Dose given",
                            "could_not_visit":"❌ You marked: Could not visit",
                            "issue":          "⚠️ You flagged: Issue for supervisor",
                        }
                        st.success(action_labels.get(prior_action, prior_action))
                        if prior.get("note"):
                            st.caption("Note recorded: " + str(prior.get('note','')))
                        if st.button("↩ Undo this update", key=f"undo_{pid}"):
                            st.session_state.asha_updates.pop(pid, None)
                            st.rerun()
                    else:
                        # ── Staged form: select action first, review, then submit ──
                        st.markdown("**Step 1 — What happened on this visit?**")
                        draft_key  = f"draft_{pid}"
                        note_key   = f"note_{pid}"

                        if draft_key not in st.session_state:
                            st.session_state[draft_key] = None

                        col1, col2, col3 = st.columns(3)
                        if col1.button("✅ Dose given", key=f"sel_done_{pid}",
                                       type="primary" if st.session_state[draft_key]=="done" else "secondary",
                                       help="Patient took their medication today"):
                            st.session_state[draft_key] = "done"
                            st.rerun()
                        if col2.button("❌ Could not visit", key=f"sel_miss_{pid}",
                                       type="primary" if st.session_state[draft_key]=="could_not_visit" else "secondary",
                                       help="Patient was not home or visit not possible"):
                            st.session_state[draft_key] = "could_not_visit"
                            st.rerun()
                        if col3.button("⚠️ Flag for supervisor", key=f"sel_flag_{pid}",
                                       type="primary" if st.session_state[draft_key]=="issue" else "secondary",
                                       help="Report a concern to your ANM or officer"):
                            st.session_state[draft_key] = "issue"
                            st.rerun()

                        # Step 2 — optional note
                        st.markdown("**Step 2 — Add a note (optional):**")
                        note = st.text_area(
                            "e.g. Patient complained of side effects, wife has cough…",
                            key=note_key,
                            height=70,
                            label_visibility="collapsed",
                        )

                        # Step 3 — review & submit
                        draft_action = st.session_state.get(draft_key)
                        if draft_action:
                            action_preview = {
                                "done":           "✅ Dose given",
                                "could_not_visit":"❌ Could not visit",
                                "issue":          "⚠️ Flagged for supervisor",
                            }.get(draft_action, draft_action)
                            st.info(
                                f"**Ready to submit:** {action_preview}"                                + (f" · Note: {note.strip()[:60]}…" if note.strip() else "")
                            )
                            sub1, sub2 = st.columns([1, 2])
                            if sub1.button("📤 Submit update", key=f"submit_{pid}", type="primary"):
                                _reply_event(selected_asha, pid, draft_action, free_text=note.strip())
                                st.session_state.asha_updates[pid] = {
                                    "action": draft_action, "note": note.strip()
                                }
                                st.session_state.pop(draft_key, None)
                                st.rerun()
                            if sub2.button("✏️ Change selection", key=f"clear_{pid}"):
                                st.session_state[draft_key] = None
                                st.rerun()
                        else:
                            st.caption("Select an outcome above to enable submission.")

                    # ── Contact screening ─────────────────────────────────
                    patient_contacts = next(
                        (p.get("contact_network", []) for p in patients
                         if p["patient_id"] == pid), []
                    )
                    unscreened = [
                        c["name"] for c in patient_contacts
                        if not c.get("screened", False)
                        and c["name"] not in st.session_state.screened_contacts
                    ]
                    if unscreened:
                        st.markdown("**Household contacts to screen for TB symptoms:**")
                        contact_name = st.selectbox(
                            "Select contact",
                            ["— select —"] + unscreened,
                            key=f"contact_{pid}"
                        )
                        if contact_name != "— select —":
                            if st.button(
                                f"✅ Mark {contact_name} as screened",
                                key=f"screen_{pid}_{contact_name.replace(' ','_')}"
                            ):
                                _reply_event(selected_asha, pid, "contact_screened",
                                             contact_name=contact_name)
                                st.session_state.screened_contacts.add(contact_name)
                                st.success(f"✅ {contact_name} marked as screened.")
                                st.rerun()

        if not briefing:
            st.info(f"No briefing generated for {selected_asha} yet — run pipeline first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Officer Command View
# Full workload + hotspot + gap analysis
# Was completely blank due to _reply_event crash
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.header("🏛️ District TB Officer — Command View")
    st.caption("Real-time risk across all ASHA workers, ANM zones, and patient contacts.")

    if not patients:
        st.info("No data — run pipeline first.")
    else:
        import plotly.graph_objects as go

        # ── Systemic alerts ───────────────────────────────────────────────
        if systemic_alerts:
            for alert in systemic_alerts:
                tier = alert.get("tier", 2)
                if tier >= 4:
                    st.error(f"🔴 **TIER 4 — DISTRICT-WIDE:** {alert['message']}")
                elif tier == 3:
                    st.warning(f"🟠 **TIER 3 — BLOCK MO:** {alert['message']}")
                else:
                    with st.expander(
                        f"🟡 Tier 2 — ASHA {alert.get('asha_id','?')} → "
                        f"ANM {alert.get('anm_id','?')}"
                    ):
                        st.write(alert["message"])
        else:
            st.success(
                "✅ No systemic failures detected across all ANM zones. "
                "No drug stockouts or clinic closures flagged."
            )

        st.divider()

        # ── KPI row ───────────────────────────────────────────────────────
        asha_groups = defaultdict(list)
        for p in patients:
            asha_groups[p["operational"]["asha_id"]].append(p)

        asha_loads = {}
        for aid, pts in asha_groups.items():
            n        = len(pts)
            high     = sum(1 for x in pts if x.get("risk_level") == "HIGH")
            avg_miss = sum(x["adherence"]["days_since_last_dose"] for x in pts) / n
            load     = min(1.0, (n/15)*0.4 + (avg_miss/14)*0.3 + (high/max(n,1))*0.3)
            asha_loads[aid] = round(load, 3)

        overloaded       = sum(1 for v in asha_loads.values() if v > 0.60)
        unscreened_total = sum(
            1 for p in patients
            for c in p.get("contact_network", [])
            if not c["screened"]
        )
        welfare_gap = sum(1 for p in patients if not p["operational"]["welfare_enrolled"])

        # Show graph update count from this session so officer sees real-time impact
        session_updates = len(st.session_state.get("graph_activity", []))
        session_tier_changes = sum(
            1 for e in st.session_state.get("graph_activity", [])
            if "TIER CHANGE" in e.get("detail", "")
        )
        # Recount HIGH from live patient data (picks up rescored patients)
        live_patients = load_scored_patients()
        live_n_high = sum(1 for p in live_patients if p.get("risk_level") == "HIGH")

        if session_updates > 0:
            baseline_high = st.session_state.get("baseline_high", live_n_high)
            if "baseline_high" not in st.session_state:
                st.session_state.baseline_high = live_n_high
            delta_high = live_n_high - st.session_state.baseline_high
            st.success(
                f"📡 **Graph live this session:** {session_updates} updates · "
                f"{session_tier_changes} tier changes · "
                f"HIGH risk: {st.session_state.baseline_high} → {live_n_high} "
                f"({'↓' if delta_high < 0 else '↑' if delta_high > 0 else '='}{abs(delta_high)})"
            )
        else:
            st.session_state.baseline_high = live_n_high

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔴 HIGH Risk",            live_n_high)
        k2.metric("⚡ Overloaded ASHAs",      overloaded,
                  f"workers above 0.60 load",
                  delta_color="inverse")
        k3.metric("👥 Unscreened Contacts",   unscreened_total)
        k4.metric("💊 Welfare Gap",           welfare_gap,
                  "not enrolled in NPY",
                  delta_color="inverse")

        st.divider()

        # ── ASHA workload ─────────────────────────────────────────────────
        st.subheader("① ASHA Worker Workload")
        st.caption(
            "Load = caseload pressure (40%) + missed-dose rate (30%) + "
            "high-risk proportion (30%). Red = overloaded (>0.70), "
            "orange = at risk (>0.50), green = manageable."
        )

        load_rows = []
        for aid, score in asha_loads.items():
            pts = asha_groups[aid]
            load_rows.append({
                "ASHA ID":   aid,
                "Load":      score,
                "HIGH Risk": sum(1 for p in pts if p.get("risk_level") == "HIGH"),
                "Caseload":  len(pts),
                "ANM Zone":  pts[0]["operational"].get("anm_id", "—"),
            })
        load_df = pd.DataFrame(load_rows).sort_values("Load", ascending=False)

        bar_colours = [
            "#f87171" if s > 0.70 else "#fb923c" if s > 0.50 else "#4ade80"
            for s in load_df["Load"]
        ]

        fig_load = go.Figure(go.Bar(
            x=load_df["Load"],
            y=load_df["ASHA ID"],
            orientation="h",
            marker_color=bar_colours,
            text=[f"{s:.2f}" for s in load_df["Load"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Load: %{x:.3f}<br>"
                "HIGH risk: %{customdata[0]}<br>"
                "Caseload: %{customdata[1]}<br>"
                "ANM Zone: %{customdata[2]}<extra></extra>"
            ),
            customdata=load_df[["HIGH Risk","Caseload","ANM Zone"]].values,
        ))
        fig_load.add_vline(x=0.60, line_dash="dash", line_color="orange",
                           annotation_text="Overload (0.60)",
                           annotation_font_size=11)
        fig_load.update_layout(
            xaxis=dict(range=[0, 1.15], title="Load Score"),
            height=max(280, len(load_df) * 32),
            margin=dict(l=20, r=60, t=10, b=30),
        )
        st.plotly_chart(fig_load, use_container_width=True)

        # Drill-down table
        with st.expander("Drill into an ASHA worker's patient list"):
            drill_asha = st.selectbox(
                "Select ASHA",
                load_df.sort_values("Load", ascending=False)["ASHA ID"].tolist(),
                key="officer_drill"
            )
            drill_pts = asha_groups[drill_asha]
            drill_rows = []
            for p in sorted(drill_pts,
                            key=lambda x: x.get("rank_score", x.get("risk_score",0)),
                            reverse=True):
                silence = max(p["adherence"]["days_since_last_dose"],
                              p["operational"]["last_asha_visit_days_ago"])
                drill_rows.append({
                    "Patient":    p["patient_id"],
                    "Risk":       p.get("risk_level","?"),
                    "Score":      round(p.get("risk_score",0), 3),
                    "Week":       p.get("treatment_week","?"),
                    "Phase":      p["clinical"]["phase"],
                    "Days Missed":p["adherence"]["days_since_last_dose"],
                    "Silent":     silence,
                    "Welfare":    "✓" if p["operational"]["welfare_enrolled"] else "✗",
                    "Nutrition":  "✓" if p["operational"]["nutritional_support"] else "✗",
                })
            st.dataframe(pd.DataFrame(drill_rows), use_container_width=True, hide_index=True)

            # Overload recommendation
            drill_load = asha_loads.get(drill_asha, 0)
            if drill_load > 0.70:
                st.error(
                    f"⚠️ {drill_asha} is critically overloaded (load={drill_load:.2f}). "
                    f"Recommend reassigning patients to a lower-load worker in the same ANM zone."
                )
            elif drill_load > 0.50:
                st.warning(
                    f"⚡ {drill_asha} is approaching overload (load={drill_load:.2f}). "
                    f"Avoid assigning new patients."
                )
            else:
                st.success(f"✓ {drill_asha} workload is manageable (load={drill_load:.2f}).")

        st.divider()

        # ── Block hotspot ─────────────────────────────────────────────────
        st.subheader("② Block-Level TB Hotspots")
        st.caption("Blocks with the highest HIGH-risk proportion need priority resource deployment.")

        block_data = defaultdict(lambda: {"HIGH":0,"MEDIUM":0,"LOW":0,"total":0,
                                          "dr_tb":0,"unscreened":0,"welfare_gap":0})
        for p in patients:
            b = p["location"]["block"]
            block_data[b][p.get("risk_level","LOW")] += 1
            block_data[b]["total"] += 1
            if p["clinical"]["regimen"] == "DR_TB":
                block_data[b]["dr_tb"] += 1
            block_data[b]["unscreened"] += sum(
                1 for c in p.get("contact_network",[]) if not c["screened"])
            if not p["operational"]["welfare_enrolled"]:
                block_data[b]["welfare_gap"] += 1

        block_rows = sorted([{
            "Block":               b,
            "Total Patients":      d["total"],
            "HIGH Risk":           d["HIGH"],
            "HIGH %":              f"{d['HIGH']/max(d['total'],1)*100:.0f}%",
            "DR-TB":               d["dr_tb"],
            "Unscreened Contacts": d["unscreened"],
            "Welfare Gap":         d["welfare_gap"],
        } for b, d in block_data.items()], key=lambda x: x["HIGH Risk"], reverse=True)

        bc1, bc2 = st.columns([2, 3])
        with bc1:
            st.dataframe(pd.DataFrame(block_rows), use_container_width=True, hide_index=True)
        with bc2:
            # If all patients are in one block, a block-level chart has a single bar and is
            # useless. Fall back to an ASHA-level breakdown within that block instead.
            if len(block_rows) <= 1:
                st.caption(
                    f"All patients are in **{block_rows[0]['Block']}** block. "
                    f"Showing per-ASHA risk breakdown within this block."
                )
                asha_block_rows = sorted([{
                    "ASHA": aid,
                    "HIGH":   sum(1 for p in pts if p.get("risk_level") == "HIGH"),
                    "MEDIUM": sum(1 for p in pts if p.get("risk_level") == "MEDIUM"),
                    "LOW":    sum(1 for p in pts if p.get("risk_level") == "LOW"),
                } for aid, pts in asha_groups.items()],
                    key=lambda x: x["HIGH"], reverse=True)
                fig_block = go.Figure()
                for level, colour in [("HIGH","#f87171"),("MEDIUM","#fbbf24"),("LOW","#4ade80")]:
                    fig_block.add_trace(go.Bar(
                        name=level,
                        x=[r["ASHA"] for r in asha_block_rows],
                        y=[r[level]  for r in asha_block_rows],
                        marker_color=colour,
                    ))
                fig_block.update_layout(
                    barmode="stack",
                    xaxis_title="ASHA Worker",
                    yaxis_title="Patients",
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", y=1.1),
                    height=340,
                    margin=dict(l=10, r=10, t=30, b=80),
                )
            else:
                fig_block = go.Figure()
                for level, colour in [("HIGH","#f87171"),("MEDIUM","#fbbf24"),("LOW","#4ade80")]:
                    fig_block.add_trace(go.Bar(
                        name=level,
                        x=[r["Block"] for r in block_rows],
                        y=[block_data[r["Block"]][level] for r in block_rows],
                        marker_color=colour,
                    ))
                fig_block.update_layout(
                    barmode="stack",
                    xaxis_title="Block",
                    yaxis_title="Patients",
                    legend=dict(orientation="h", y=1.1),
                    height=300,
                    margin=dict(l=10, r=10, t=30, b=40),
                )
            st.plotly_chart(fig_block, use_container_width=True)

        if block_rows:
            top = block_rows[0]
            st.info(
                f"🎯 **Priority block: {top['Block']}** — {top['HIGH %']} of patients are HIGH risk "
                f"({top['HIGH Risk']} of {top['Total Patients']}), "
                f"{top['Unscreened Contacts']} unscreened contacts, {top['DR-TB']} DR-TB cases. "
                f"Deploy a targeted screening camp and additional ASHA support here this week."
            )

        st.divider()

        # ── Intervention gaps ─────────────────────────────────────────────
        st.subheader("③ Intervention Gaps")

        g1, g2 = st.columns(2)
        with g1:
            enrolled     = sum(1 for p in patients if p["operational"]["welfare_enrolled"])
            not_enrolled = n_patients - enrolled
            fig_w = go.Figure(go.Pie(
                labels=["Enrolled in NPY", "Gap — not enrolled"],
                values=[enrolled, not_enrolled],
                hole=0.55,
                marker_colors=["#4ade80","#f87171"],
                textinfo="label+percent",
            ))
            fig_w.update_layout(
                title="Nikshay Poshan Yojana Enrolment",
                showlegend=False, height=260,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption(
                "Patients not enrolled lose ₹500/month nutritional support. "
                "Non-enrolment raises dropout risk by 1.45×."
            )
        with g2:
            phase_risk = defaultdict(lambda: {"HIGH":0,"MEDIUM":0,"LOW":0})
            for p in patients:
                phase_risk[p["clinical"]["phase"]][p.get("risk_level","LOW")] += 1
            phases = list(phase_risk.keys())
            fig_p = go.Figure()
            for level, colour in [("HIGH","#f87171"),("MEDIUM","#fbbf24"),("LOW","#4ade80")]:
                fig_p.add_trace(go.Bar(
                    name=level, x=phases,
                    y=[phase_risk[ph][level] for ph in phases],
                    marker_color=colour,
                ))
            fig_p.update_layout(
                barmode="group",
                title="Risk Level by Treatment Phase",
                yaxis_title="Patients",
                legend=dict(orientation="h", y=1.1),
                height=260,
                margin=dict(l=10, r=10, t=40, b=30),
            )
            st.plotly_chart(fig_p, use_container_width=True)
            st.caption(
                "Continuation phase dropout OR = 2.30. "
                "Patients who feel better think they're cured — "
                "peak dropout risk despite near-complete treatment."
            )

        st.divider()

        # ── Confirmed dropout tracking ────────────────────────────────────
        st.subheader("④ Confirm Dropouts — Update BBN Model")
        st.caption(
            "When a patient has officially dropped out of treatment, mark them here. "
            "After 10+ confirmed cases the system recalculates the dropout odds ratios "
            "from your real field data instead of relying solely on published literature."
        )

        # Load confirmed dropouts + schedule status
        try:
            from stage3_score import (load_confirmed_dropouts, load_bbn_schedule,
                                       is_update_due, BBN_UPDATE_FREQUENCY)
            confirmed    = load_confirmed_dropouts()
            n_confirmed  = len(confirmed)
            n_pending    = sum(1 for v in confirmed.values() if not v.get("included_in_update"))
            schedule     = load_bbn_schedule()
            due, due_msg = is_update_due(schedule)
        except Exception:
            confirmed   = {}
            n_confirmed = 0
            n_pending   = 0
            schedule    = {}
            due, due_msg = False, "Could not load schedule"

        # Schedule status card
        sched_col1, sched_col2, sched_col3, sched_col4 = st.columns(4)
        sched_col1.metric("Confirmed dropouts",  n_confirmed, "total recorded")
        sched_col2.metric("Pending next cycle",  n_pending,   "not yet used")
        sched_col3.metric("Cycles completed",    schedule.get("cycles_completed", 0))
        sched_col4.metric(
            "Update frequency",
            schedule.get("frequency", BBN_UPDATE_FREQUENCY).title(),
            "configurable in main.py",
        )

        next_due = schedule.get("next_due_date", "")
        last_ran = schedule.get("last_update_date", "")
        if due:
            st.warning(
                f"⏰ **BBN weight update is due.** {due_msg} "
                f"The update will run automatically the next time `python main.py` is executed. "
                f"All {n_pending} pending cases will be processed."
            )
        else:
            st.info(
                f"✅ **BBN weights are current.** "
                f"Next scheduled update: **{next_due[:10] if next_due else 'not set'}** "
                f"({'last ran ' + last_ran[:10] if last_ran else 'never run yet'}). "
                f"Weights will not change mid-run — every patient in a pipeline run "
                f"is always scored with identical weights."
            )

        bbn_progress = min(n_confirmed / 200, 1.0)
        st.caption(f"BBN retirement progress: **{n_confirmed} / 200** confirmed cases "
                   f"(BBN prior retires at 200 — TGN takes full weight)")
        st.progress(bbn_progress)

        for v in visit_list[:10]:
            icon        = "🔴" if v["risk_level"] == "HIGH" else "🟡"
            pid         = v["patient_id"]
            already_confirmed = pid in confirmed
            with st.expander(
                f"{icon} {pid} — {v['risk_level']} "
                f"· Score {v['risk_score']:.3f} · Week {v['treatment_week']}"
                + (" ✓ CONFIRMED DROPOUT" if already_confirmed else ""),
            ):
                # Structured officer explanation
                sc = v.get("score_composition", {})
                factors = list(v.get("top_factors", {}).items())
                col_info, col_action = st.columns([3, 1])
                with col_info:
                    st.markdown(f"**Patient:** {pid} &nbsp;|&nbsp; **Risk tier:** {v['risk_level']} &nbsp;|&nbsp; **Week:** {v['treatment_week']} of treatment")
                    st.markdown(f"**Phase:** {v['phase']} &nbsp;|&nbsp; **Days missed:** {v['days_missed']} &nbsp;|&nbsp; **Score:** {v['risk_score']:.3f}")
                    st.markdown("**Top risk drivers:**")
                    for fname, for_val in factors[:3]:
                        st.markdown(f"  &nbsp;&nbsp;• {fname} &nbsp;(OR {for_val:.2f}×)")
                    if sc:
                        st.caption(
                            f"Score breakdown — TGN: {sc.get('tgn_weight',0)*100:.0f}%"
                            f" · BBN prior: {sc.get('bbn_weight',0)*100:.0f}%"
                            f" · ASHA load: 20%"
                            f" · BBN status: {sc.get('bbn_status','active')}"
                        )
                with col_action:
                    if already_confirmed:
                        st.success("✓ Dropout confirmed")
                        st.caption(f"Recorded: {confirmed[pid].get('confirmed_at','')[:10]}")
                    else:
                        if st.button(f"Mark as confirmed dropout", key=f"confirm_{pid}",
                                     type="primary", use_container_width=True):
                            try:
                                from stage3_score import save_confirmed_dropout
                                save_confirmed_dropout(pid, v.get("top_factors", {}))
                                log_graph_activity("confirmed_dropout", pid,
                                    "Marked as confirmed dropout · BBN weights will update after 10 cases")
                                st.success("Recorded ✓ — BBN will update")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        st.divider()

        # ── Graph Activity Feed ───────────────────────────────────────────────
        st.subheader("⑤ Live Graph Update Feed")
        st.caption(
            "Every ASHA tap, contact screening, or confirmed dropout that "
            "updates the Cosmos DB graph appears here in real time."
        )
        if st.session_state.graph_activity:
            feed_df = pd.DataFrame(st.session_state.graph_activity)
            st.dataframe(feed_df, use_container_width=True, hide_index=True, height=250)
        else:
            st.info(
                "No graph updates yet this session. "
                "When an ASHA worker marks a dose or you confirm a dropout above, "
                "the update will appear here."
            )

        st.divider()

        # ── Overnight processing results ──────────────────────────────────────
        st.subheader("⑥ Overnight NER — Graph Updates from ASHA Notes")
        st.caption(
            "Every evening at 22:00 IST, the system runs NER on all notes submitted "
            "by ASHA workers during the day. New contacts are added to the graph, "
            "symptomatic contacts are flagged, and patients are rescored. "
            "ASHA workers receive updated briefings the next morning."
        )

        overnight_path = Path("data/overnight_results.json")
        if overnight_path.exists():
            with open(overnight_path) as _of:
                overnight = json.load(_of)

            run_at = overnight.get("run_start", "")[:19].replace("T", " ") + " UTC"
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Notes processed",    overnight.get("processed", 0))
            o2.metric("New contacts added", overnight.get("contacts_added", 0),
                      help="Contacts extracted from ASHA notes and added as graph nodes")
            o3.metric("Symptoms flagged",   overnight.get("symptoms_flagged", 0),
                      help="Contacts whose vulnerability was boosted due to reported symptoms")
            o4.metric("Tier changes",       overnight.get("tier_changes", 0),
                      delta_color="inverse" if overnight.get("tier_changes", 0) > 0 else "off",
                      help="Patients whose HIGH/MEDIUM/LOW classification changed after rescoring")

            st.caption(f"Last overnight run: **{run_at}**")

            if overnight.get("error"):
                st.error(f"⚠️ Overnight run error: {overnight['error']}")

            # Show per-patient graph delta audit trail
            deltas = overnight.get("graph_deltas", [])
            if deltas:
                st.markdown("**What changed in the graph last night:**")
                for d in deltas:
                    pid          = d.get("patient_id", "?")
                    note_preview = d.get("note_preview", "")
                    score_ch     = d.get("score_change") or {}
                    contacts_add = d.get("contacts_added", [])
                    syms_flagged = d.get("symptoms_flagged", [])
                    dose_update  = d.get("dose_update")

                    tier_badge = ""
                    if score_ch.get("changed"):
                        tier_badge = (
                            f" 🔄 **{score_ch['old_tier']} → {score_ch['new_tier']}** "
                            f"(score {score_ch['old_score']:.3f} → {score_ch['new_score']:.3f})"
                        )

                    with st.expander(
                        f'📋 {pid}{tier_badge} — "{note_preview[:60]}...',
                        expanded=score_ch.get("changed", False)
                    ):
                        st.markdown(f"**ASHA note:** _{note_preview}_")

                        if dose_update:
                            dose_icon = "✅" if dose_update == "dose_confirmed" else "❌"
                            st.markdown(
                                f"{dose_icon} **Dose update from note:** "
                                f"{'Dose confirmed — days_missed reset to 0' if dose_update == 'dose_confirmed' else 'Visit missed — silence_days incremented'}"
                            )

                        if contacts_add:
                            st.markdown(f"**New contacts added to graph ({len(contacts_add)}):**")
                            for c in contacts_add:
                                sym_note = f" ⚠️ symptom: *{c.get('symptom')}*" if c.get("has_symptom") else ""
                                st.markdown(
                                    f"  &nbsp;&nbsp;• **{c.get('contact_name', '?')}** "
                                    f"({c.get('rel', '?')}) — "
                                    f"edge `{c.get('edge_label', 'household_contact')}` "
                                    f"weight {c.get('edge_weight', 0):.3f} · "
                                    f"vulnerability {c.get('vulnerability_score', 1.0):.2f}"
                                    + sym_note
                                )

                        if syms_flagged:
                            st.markdown(f"**Symptom flags applied ({len(syms_flagged)}):**")
                            for s in syms_flagged:
                                st.markdown(
                                    f"  &nbsp;&nbsp;• **{s.get('contact_name', '?')}** — "
                                    f"symptom: *{s.get('symptom', '?')}* · "
                                    f"vulnerability {s.get('vuln_old', 0):.2f} → {s.get('vuln_new', 0):.2f} "
                                    f"(edge weight boosted, moves up screening list)"
                                )

                        if score_ch:
                            col_sc1, col_sc2 = st.columns(2)
                            col_sc1.metric(
                                "Risk score",
                                f"{score_ch.get('new_score', 0):.3f}",
                                f"{score_ch.get('new_score', 0) - score_ch.get('old_score', 0):+.3f}",
                                delta_color="inverse"
                            )
                            col_sc2.metric(
                                "Risk tier",
                                score_ch.get("new_tier", "?"),
                                f"was {score_ch.get('old_tier', '?')}",
                                delta_color="off"
                            )

            # Pending notes queue status
            pending_path = Path("data/pending_notes.json")
            if pending_path.exists():
                with open(pending_path) as _pf:
                    pending_all = json.load(_pf)
                pending_count = sum(1 for n in pending_all if not n.get("processed"))
                if pending_count > 0:
                    st.info(
                        f"📥 **{pending_count} notes queued** for tonight's processing "
                        f"({len(pending_all)} total, "
                        f"{len(pending_all) - pending_count} already processed). "
                        f"NER will run at 22:00 IST."
                    )
                else:
                    st.success("✅ All notes from today have been processed.")

        else:
            st.info(
                "No overnight results yet. "
                "Notes submitted by ASHA workers are queued in `data/pending_notes.json` "
                "and processed nightly at 22:00 IST by the Azure Function. "
                "Results appear here the next morning."
            )
            pending_path = Path("data/pending_notes.json")
            if pending_path.exists():
                with open(pending_path) as _pf:
                    pending_all = json.load(_pf)
                pending_count = sum(1 for n in pending_all if not n.get("processed"))
                if pending_count > 0:
                    st.warning(
                        f"📥 **{pending_count} notes waiting** to be processed tonight. "
                        f"To process now (for testing), run: `python main.py --overnight`"
                    )

        # Show learned ORs if they exist
        from pathlib import Path
        if Path("data/learned_ors.json").exists():
            st.divider()
            st.subheader("⑥ BBN Learned Odds Ratios")
            st.caption("These are the updated odds ratios your real confirmed cases have produced. "
                       "The system started from Tamil Nadu literature values and is shifting toward your data.")
            try:
                import json as _json
                with open("data/learned_ors.json") as _f:
                    lor_data = _json.load(_f)
                lor_ors = lor_data.get("ors", {})
                lit_ors = {k: float(__import__("numpy").exp(v))
                           for k, v in __import__("stage3_score").LOG_OR.items()}
                or_rows = []
                for k, learned_v in lor_ors.items():
                    lit_v = lit_ors.get(k, learned_v)
                    delta = learned_v - lit_v
                    or_rows.append({
                        "Factor":          k,
                        "Literature OR":   round(lit_v, 3),
                        "Learned OR":      round(learned_v, 3),
                        "Δ from lit":      f"{'↑' if delta>0.05 else '↓' if delta<-0.05 else '≈'} {delta:+.3f}",
                    })
                st.dataframe(pd.DataFrame(or_rows), use_container_width=True, hide_index=True)
                st.caption(f"Last updated: {lor_data.get('updated_at','')[:19]} · "
                           f"Cases used: {lor_data.get('cases_used', 0)} · "
                           f"Total confirmed: {lor_data.get('total_confirmed', 0)}")
            except Exception as e:
                st.warning(f"Could not load learned ORs: {e}")