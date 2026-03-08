"""
stage3_score.py
===============
Stage 3: Dropout Risk Classification

Implements the exact three-component score from the pipeline document:
  - TGN output        (60% weight)  — temporally-aware, graph-informed
  - BBN prior         (fading weight) — literature-calibrated ORs, retires at 200 cases
  - ASHA load score   (20% weight)  — system-side failure signal

Plus:
  - Urgency multiplier: risk × (1 + treatment_week/26)
  - Adaptive thresholds: HIGH/MEDIUM tighten as treatment progresses
    Intensive (wk 1-8):       HIGH > 0.75, MEDIUM > 0.50
    Early Continuation (9-16): HIGH > 0.65, MEDIUM > 0.40
    Late Continuation (17-26): HIGH > 0.55, MEDIUM > 0.30

Usage:
    from stage3_score import score_all_patients, detect_systemic_failures
"""

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# BBN PRIOR — literature-calibrated odds ratios (fades as real data accumulates)
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_LTFU     = 0.062
BASELINE_LOG_ODDS = np.log(BASELINE_LTFU / (1 - BASELINE_LTFU))

# Published adjusted Odds Ratios — log scale = logistic regression coefficients
LOG_OR = {
    "alcohol_use":            np.log(1.92),
    "divorced_separated":     np.log(3.80),
    "diabetes":               np.log(0.52),   # protective in Tamil Nadu
    "prior_tb":               np.log(2.10),
    "hiv":                    np.log(2.16),
    "low_education":          np.log(1.55),
    "drug_use":               np.log(2.40),
    "male_sex":               np.log(1.29),
    "distance_5_to_10km":    np.log(1.60),
    "distance_over_10km":    np.log(2.10),
    "continuation_phase":     np.log(2.30),
    "no_nutritional_support": np.log(1.60),
    "no_welfare":             np.log(1.45),   # Nikshay Poshan Yojana non-enrollment
    "dr_tb":                  np.log(2.80),   # WHO DR-TB report 2022
    "missed_7_to_13_days":   np.log(3.20),
    "missed_14_plus_days":   np.log(6.50),
    "age_20_to_39":           np.log(2.07),
    "age_over_60":            np.log(1.40),
}

# BBN retires after this many confirmed real dropout cases are observed
BBN_RETIREMENT_THRESHOLD = 200

def compute_bbn_prior(record: dict) -> dict:
    """
    Knowledge-based logistic model using published odds ratios.
    Returns score (0-1) + all contributing factors with their OR values.
    Active during Phase 1 (prototype) and early Phase 2 (pilot).
    Automatically fades as TGN accumulates real evidence.
    """
    # Use learned ORs if available, otherwise fall back to literature values
    effective_log_or = get_effective_log_ors()
    lo = BASELINE_LOG_ODDS
    factors = {}

    d, c, s, a, o = (record["demographics"], record["clinical"],
                     record["social"], record["adherence"], record["operational"])

    if s.get("alcohol_use"):
        lo += effective_log_or["alcohol_use"]
        factors["Alcohol use"] = round(np.exp(effective_log_or["alcohol_use"]), 2)
    if d.get("marital") == "Divorced":
        lo += effective_log_or["divorced_separated"]
        factors["Divorced/separated"] = round(np.exp(effective_log_or["divorced_separated"]), 2)
    if c["comorbidities"].get("diabetes"):
        lo += effective_log_or["diabetes"]
        factors["Diabetes (monitored — protective)"] = round(np.exp(effective_log_or["diabetes"]), 2)
    if c["comorbidities"].get("hiv"):
        lo += effective_log_or["hiv"]
        factors["HIV co-infection"] = round(np.exp(effective_log_or["hiv"]), 2)
    if a.get("prior_lfu_history"):
        lo += effective_log_or["prior_tb"]
        factors["Prior LTFU/TB history"] = round(np.exp(effective_log_or["prior_tb"]), 2)
    if d.get("gender") == "Male":
        lo += effective_log_or["male_sex"]
        factors["Male sex"] = round(np.exp(effective_log_or["male_sex"]), 2)
    if s.get("low_education"):
        lo += effective_log_or["low_education"]
        factors["Low education"] = round(np.exp(effective_log_or["low_education"]), 2)
    if s.get("drug_use"):
        lo += effective_log_or["drug_use"]
        factors["Drug use"] = round(np.exp(effective_log_or["drug_use"]), 2)
    if c.get("phase") == "Continuation":
        lo += effective_log_or["continuation_phase"]
        factors["Continuation phase"] = round(np.exp(effective_log_or["continuation_phase"]), 2)
    if not o.get("nutritional_support"):
        lo += effective_log_or["no_nutritional_support"]
        factors["No nutritional support"] = round(np.exp(effective_log_or["no_nutritional_support"]), 2)
    if not o.get("welfare_enrolled", True):  # NPY non-enrollment
        lo += effective_log_or["no_welfare"]
        factors["Not enrolled in Nikshay Poshan Yojana"] = round(np.exp(effective_log_or["no_welfare"]), 2)
    c_data = record["clinical"]
    if c_data.get("regimen") == "DR_TB":
        lo += effective_log_or["dr_tb"]
        factors["Drug-resistant TB (DR-TB)"] = round(np.exp(effective_log_or["dr_tb"]), 2)

    dist = a.get("distance_to_center_km", 0)
    if 5 <= dist < 10:
        lo += effective_log_or["distance_5_to_10km"]
        factors[f"Distance {dist:.1f}km (5-10km)"] = round(np.exp(effective_log_or["distance_5_to_10km"]), 2)
    elif dist >= 10:
        lo += effective_log_or["distance_over_10km"]
        factors[f"Distance {dist:.1f}km (>10km)"] = round(np.exp(effective_log_or["distance_over_10km"]), 2)

    missed = a.get("days_since_last_dose", 0)
    if 7 <= missed < 14:
        lo += effective_log_or["missed_7_to_13_days"]
        factors[f"{missed} days since last dose"] = round(np.exp(effective_log_or["missed_7_to_13_days"]), 2)
    elif missed >= 14:
        lo += effective_log_or["missed_14_plus_days"]
        factors[f"{missed} days since last dose (CRITICAL)"] = round(np.exp(effective_log_or["missed_14_plus_days"]), 2)

    age = d.get("age", 30)
    if 20 <= age <= 39:
        lo += effective_log_or["age_20_to_39"]
        factors[f"Age {age} (high-risk group 20-39)"] = round(np.exp(effective_log_or["age_20_to_39"]), 2)
    elif age > 60:
        lo += effective_log_or["age_over_60"]
        factors[f"Age {age} (elderly)"] = round(np.exp(effective_log_or["age_over_60"]), 2)

    prob = float(np.clip(1 / (1 + np.exp(-lo)), 0.0, 1.0))
    return {"score": round(prob, 4), "all_factors": factors}


def get_bbn_weight(confirmed_dropout_cases: int = 0) -> float:
    """
    BBN prior weight fades linearly from 0.40 to 0.00 as real cases accumulate.
    At 0 real cases: BBN weight = 0.40, TGN weight = 0.60
    At 200 real cases: BBN weight = 0.00, TGN weight = 0.80 (ASHA load stays at 0.20)
    """
    if confirmed_dropout_cases >= BBN_RETIREMENT_THRESHOLD:
        return 0.0
    return 0.40 * (1 - confirmed_dropout_cases / BBN_RETIREMENT_THRESHOLD)


# ─────────────────────────────────────────────────────────────────────────────
# ASHA LOAD SCORE — system-side risk component
# ─────────────────────────────────────────────────────────────────────────────

def compute_asha_load_score(record: dict, asha_summaries: dict) -> float:
    """
    Computed from the ASHA worker node's properties.
    Reflects caseload pressure, visit frequency decline, geographic spread.
    This component is permanent — system failure is always a real risk factor.
    """
    asha_id = record["operational"]["asha_id"]
    summary = asha_summaries.get(asha_id)
    if not summary:
        return 0.3  # default moderate if no ASHA data

    return summary["load_score"]


# ─────────────────────────────────────────────────────────────────────────────
# THREE-COMPONENT SCORE COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

def compose_final_score(tgn_score: float, bbn_score: float, asha_load: float,
                        confirmed_cases: int = 0) -> dict:
    """
    Final risk score = weighted combination of three components.

    Component         | Weight at 0 cases | Weight at 200 cases
    TGN output        | 0.40              | 0.80
    BBN prior         | 0.40 → fading     | 0.00 → retired
    ASHA load score   | 0.20 (permanent)  | 0.20 (permanent)

    Note: weights always sum to 1.0.
    TGN starts at 40% (not 60%) because BBN holds 40% at prototype stage.
    As real cases accumulate, BBN fades and TGN grows to 80%.
    """
    bbn_weight  = get_bbn_weight(confirmed_cases)
    tgn_weight  = 1.0 - bbn_weight - 0.20
    asha_weight = 0.20

    raw = (tgn_weight * tgn_score) + (bbn_weight * bbn_score) + (asha_weight * asha_load)

    return {
        "composite_score": round(float(np.clip(raw, 0, 1)), 4),
        "tgn_component":   round(tgn_score * tgn_weight, 4),
        "bbn_component":   round(bbn_score * bbn_weight, 4),
        "asha_component":  round(asha_load * asha_weight, 4),
        "tgn_weight":      round(tgn_weight, 3),
        "bbn_weight":      round(bbn_weight, 3),
        "asha_weight":     asha_weight,
        "bbn_status":      "active" if bbn_weight > 0 else "retired",
    }


# ─────────────────────────────────────────────────────────────────────────────
# URGENCY MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def apply_urgency_multiplier(composite_score: float, treatment_week: int) -> float:
    """
    Risk × urgency factor.
    Urgency increases linearly through the 26-week course.
    A score of 0.65 at week 22 is more urgent than 0.65 at week 4 —
    the intervention window is narrower near the end of treatment.
    Max multiplier is 1.5 (at week 26).
    """
    urgency = 1.0 + (treatment_week / 26) * 0.5
    return round(float(np.clip(composite_score * urgency, 0, 1)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLDS — tighten as treatment progresses
# ─────────────────────────────────────────────────────────────────────────────

def get_adaptive_thresholds(treatment_week: int) -> dict:
    """
    Thresholds tighten as patient gets closer to treatment completion.
    Same score triggers different tiers depending on treatment stage.

    Phase             | Treatment Week | HIGH threshold | MEDIUM threshold
    Intensive         | 1-8            | 0.75           | 0.50
    Early Continuation| 9-16           | 0.65           | 0.40
    Late Continuation | 17-26          | 0.55           | 0.30
    """
    if treatment_week <= 8:
        return {"high": 0.75, "medium": 0.50, "phase_label": "Intensive (wk 1-8)"}
    elif treatment_week <= 16:
        return {"high": 0.65, "medium": 0.40, "phase_label": "Early Continuation (wk 9-16)"}
    else:
        return {"high": 0.55, "medium": 0.30, "phase_label": "Late Continuation (wk 17-26)"}


def assign_risk_tier(final_score: float, treatment_week: int) -> str:
    thresholds = get_adaptive_thresholds(treatment_week)
    if final_score >= thresholds["high"]:
        return "HIGH"
    elif final_score >= thresholds["medium"]:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE RISK SCORE (BBN only — used when TGN not available)
# ─────────────────────────────────────────────────────────────────────────────

def compute_risk_score(record: dict) -> dict:
    """Convenience wrapper returning BBN score + factors. Used by stage2 simulation."""
    bbn = compute_bbn_prior(record)
    return {
        "risk_score":  bbn["score"],
        "all_factors": bbn["all_factors"],
        "top_factors": dict(sorted(bbn["all_factors"].items(), key=lambda x: x[1], reverse=True)[:3]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORE ALL PATIENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_all_patients(patients: list, tgn_scores: dict = None,
                       asha_summaries: dict = None,
                       confirmed_cases: int = 0) -> list:
    """
    Full scoring pipeline:
    1. BBN prior for every patient
    2. Compose with TGN score (if available) and ASHA load
    3. Apply urgency multiplier
    4. Assign risk tier using adaptive thresholds
    """
    if asha_summaries is None:
        asha_summaries = {}

    print(f"Scoring {len(patients)} patients...")
    print(f"  BBN weight: {get_bbn_weight(confirmed_cases):.2f} "
          f"({'active' if confirmed_cases < BBN_RETIREMENT_THRESHOLD else 'retired'})")
    print(f"  Confirmed real dropout cases seen: {confirmed_cases}/{BBN_RETIREMENT_THRESHOLD}")

    high = medium = low = 0

    for p in patients:
        bbn_result  = compute_bbn_prior(p)
        bbn_score   = bbn_result["score"]
        tgn_score   = (tgn_scores or {}).get(p["patient_id"], bbn_score)  # fall back to BBN
        asha_load   = compute_asha_load_score(p, asha_summaries)

        composition = compose_final_score(tgn_score, bbn_score, asha_load, confirmed_cases)
        treatment_week = min(p["clinical"]["total_treatment_days"] // 7, 26)
        final_score    = apply_urgency_multiplier(composition["composite_score"], treatment_week)
        risk_tier      = assign_risk_tier(final_score, treatment_week)
        thresholds     = get_adaptive_thresholds(treatment_week)

        # Risk velocity: rate of change since last pipeline run
        prev_score    = p.get("previous_risk_score") or final_score
        risk_velocity = round(final_score - float(prev_score), 4)

        p["risk_score"]          = final_score
        p["previous_risk_score"] = final_score
        p["risk_velocity"]       = risk_velocity
        p["composite_score"]     = composition["composite_score"]
        p["risk_level"]          = risk_tier
        p["tgn_score"]           = tgn_score   # saved so overnight rescore can reuse it
        p["treatment_week"]      = treatment_week
        p["score_composition"]   = composition
        p["thresholds"]          = thresholds
        p["all_factors"]         = bbn_result["all_factors"]
        p["top_factors"]         = dict(sorted(bbn_result["all_factors"].items(),
                                               key=lambda x: x[1], reverse=True)[:3])
        p["asha_load_score"]     = asha_load

        # Velocity override: fast-rising patients escalated to HIGH regardless of absolute score
        if risk_velocity >= 0.12 and risk_tier != "HIGH":
            p["risk_level"]          = "HIGH"
            p["velocity_escalated"]  = True

        # Use p["risk_level"] not risk_tier — captures velocity-escalated patients
        if p["risk_level"] == "HIGH":     high   += 1
        elif p["risk_level"] == "MEDIUM": medium += 1
        else:                             low    += 1

    print(f"  HIGH: {high}  |  MEDIUM: {medium}  |  LOW: {low}")
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEMIC FAILURE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_systemic_failures(patients: list) -> list:
    """
    Four-tier escalation matching the real ASHA supervision chain:

      Tier 1 — ASHA level:   individual patient silent 7+ days → alert to ASHA
      Tier 2 — ANM level:    >50% of ONE ASHA's patients missing → alert to her ANM
      Tier 3 — Block MO:     multiple ASHAs in ONE ANM zone all failing → alert to Block MO
      Tier 4 — DTO level:    >3 ANM zones affected district-wide → alert to District TB Officer

    Tier 2+ requires min 5 patients per ASHA to avoid false positives.
    Tier 1 alerts are per-patient and returned separately so Stage 5 can route them.
    """
    from collections import defaultdict

    # Group by ASHA, then by ANM
    asha_groups = defaultdict(list)
    anm_groups  = defaultdict(list)   # anm_id → list of asha_ids with problems

    for p in patients:
        asha_groups[p["operational"]["asha_id"]].append(p)

    alerts = []

    # ── Tier 2: per-ASHA systemic failure ────────────────────────────────────
    problem_ashas_by_anm = defaultdict(list)

    for asha_id, group in asha_groups.items():
        if len(group) < 5:
            continue
        miss_rate = sum(1 for p in group
                        if p["adherence"]["days_since_last_dose"] > 0) / len(group)
        if miss_rate > 0.50:
            anm_id   = group[0]["operational"].get("anm_id", "UNKNOWN")
            avg_load = sum(p.get("asha_load_score", 0) for p in group) / len(group)
            alerts.append({
                "tier":             2,
                "asha_id":          asha_id,
                "anm_id":           anm_id,
                "patients_affected":len(group),
                "miss_rate_pct":    round(miss_rate * 100, 1),
                "avg_load_score":   round(avg_load, 3),
                "alert_type":       "SYSTEMIC_ASHA",
                "escalate_to":      f"ANM {anm_id}",
                "message": (
                    f"TIER 2 ALERT — {asha_id} ({anm_id}): "
                    f"{miss_rate*100:.0f}% of {len(group)} patients missed doses. "
                    f"Possible worker absence or local disruption. "
                    f"Escalate to ANM {anm_id} — do NOT send to ASHA directly."
                ),
            })
            problem_ashas_by_anm[anm_id].append(asha_id)

    # ── Tier 3: multiple ASHAs failing within one ANM zone ───────────────────
    problem_anms = []
    for anm_id, bad_ashas in problem_ashas_by_anm.items():
        total_ashas = sum(1 for p in patients
                          if p["operational"].get("anm_id") == anm_id
                          and p["operational"]["asha_id"] in asha_groups)
        # use unique ASHA count under this ANM
        all_anm_ashas = {p["operational"]["asha_id"] for p in patients
                         if p["operational"].get("anm_id") == anm_id}
        fail_rate = len(bad_ashas) / max(len(all_anm_ashas), 1)

        if fail_rate >= 0.40 and len(bad_ashas) >= 2:
            problem_anms.append(anm_id)
            alerts.append({
                "tier":             3,
                "anm_id":           anm_id,
                "ashas_affected":   bad_ashas,
                "fail_rate_pct":    round(fail_rate * 100, 1),
                "alert_type":       "SYSTEMIC_BLOCK",
                "escalate_to":      "Block Medical Officer",
                "message": (
                    f"TIER 3 ALERT — ANM zone {anm_id}: "
                    f"{len(bad_ashas)} of {len(all_anm_ashas)} ASHAs showing systemic failures. "
                    f"Likely block-level disruption (drug stockout, clinic closure). "
                    f"Escalate to Block Medical Officer immediately."
                ),
            })

    # ── Tier 4: district-wide — multiple ANM zones affected ──────────────────
    if len(problem_anms) > 3:
        alerts.append({
            "tier":           4,
            "anm_zones":      problem_anms,
            "zones_affected": len(problem_anms),
            "alert_type":     "SYSTEMIC_DISTRICT",
            "escalate_to":    "District TB Officer",
            "message": (
                f"TIER 4 ALERT — DISTRICT-WIDE: "
                f"{len(problem_anms)} ANM zones showing concurrent systemic failures "
                f"({', '.join(problem_anms[:5])}{'...' if len(problem_anms)>5 else ''}). "
                f"Escalate to District TB Officer. Possible district-level drug stockout "
                f"or programme disruption."
            ),
        })

    if alerts:
        tier_counts = {2: 0, 3: 0, 4: 0}
        for a in alerts:
            tier_counts[a["tier"]] = tier_counts.get(a["tier"], 0) + 1
        print(f"\n  Systemic alerts: "
              f"Tier 2 (ASHA→ANM): {tier_counts[2]}  "
              f"Tier 3 (ANM→Block MO): {tier_counts[3]}  "
              f"Tier 4 (District): {tier_counts[4]}")
        for a in alerts:
            print(f"  [{a['escalate_to']}] {a['message'][:90]}...")

    return alerts



# ─────────────────────────────────────────────────────────────────────────────
# CONFIRMED DROPOUT TRACKING & BAYESIAN OR UPDATE
# ─────────────────────────────────────────────────────────────────────────────

CONFIRMED_DROPOUTS_FILE = "data/confirmed_dropouts.json"
LEARNED_ORS_FILE        = "data/learned_ors.json"
BBN_SCHEDULE_FILE       = "data/bbn_update_schedule.json"

PRIOR_WEIGHT            = 20      # literature equivalent to ~20 real observations
MIN_CASES_TO_UPDATE     = 15      # never update an OR with fewer than this many cases
MAX_MOVE_PER_CYCLE      = 0.50    # OR cannot shift more than 50% of current value in one cycle

# How often the BBN weights are allowed to change.
# Options: "monthly", "quarterly", "biannual", "annual"
# The update runs at pipeline STARTUP, before any patient is scored.
# Mid-run updates are never allowed — every patient in a run uses identical weights.
BBN_UPDATE_FREQUENCY    = "biannual"   # default: every 6 months

_FREQUENCY_DAYS = {
    "monthly":   30,
    "quarterly": 91,
    "biannual":  182,
    "annual":    365,
}


def load_confirmed_dropouts() -> dict:
    """
    Load confirmed dropout records from disk.
    Returns dict: {patient_id: {factors: {...}, confirmed_at: iso_timestamp}}
    Creates the file if missing.
    """
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    if Path(CONFIRMED_DROPOUTS_FILE).exists():
        with open(CONFIRMED_DROPOUTS_FILE) as f:
            return json.load(f)
    return {}


def save_confirmed_dropout(patient_id: str, factors: dict):
    """
    Persist a single confirmed dropout record to disk.

    IMPORTANT — this function NEVER triggers a weight update.
    Weight updates only happen through check_and_run_scheduled_update(),
    which is called once at pipeline startup before any patient is scored.
    This guarantees every patient in a pipeline run is evaluated with
    exactly the same OR weights — no mid-run inconsistency.
    """
    from datetime import datetime, timezone
    dropouts = load_confirmed_dropouts()
    if patient_id in dropouts:
        print(f"  [BBN] {patient_id} already recorded — skipping duplicate")
        return
    dropouts[patient_id] = {
        "factors":            factors,
        "confirmed_at":       datetime.now(timezone.utc).isoformat(),
        "included_in_update": False,
    }
    with open(CONFIRMED_DROPOUTS_FILE, "w") as f:
        json.dump(dropouts, f, indent=2)
    new_pending = sum(1 for v in dropouts.values() if not v.get("included_in_update"))
    print(f"  [BBN] Confirmed dropout recorded: {patient_id} "
          f"({len(dropouts)} total, {new_pending} pending next cycle)")




def load_bbn_schedule() -> dict:
    """
    Load the BBN update schedule metadata.
    Returns dict with last_update_date and next_due_date.
    Creates a default schedule (due immediately) if no file exists.
    """
    from pathlib import Path
    from datetime import datetime, timezone, timedelta
    if Path(BBN_SCHEDULE_FILE).exists():
        with open(BBN_SCHEDULE_FILE) as f:
            return json.load(f)
    # First run — schedule is due immediately so the system initialises correctly
    now = datetime.now(timezone.utc)
    return {
        "last_update_date":  None,
        "next_due_date":     now.isoformat(),
        "frequency":         BBN_UPDATE_FREQUENCY,
        "cycles_completed":  0,
    }


def save_bbn_schedule(schedule: dict):
    """Write the updated schedule to disk after a successful cycle."""
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open(BBN_SCHEDULE_FILE, "w") as f:
        json.dump(schedule, f, indent=2)


def is_update_due(schedule: dict = None) -> tuple:
    """
    Check whether the BBN update cycle is due.

    Returns (is_due: bool, reason: str).

    The cycle is due when:
      - It has never run before (no last_update_date), OR
      - The current date is on or after next_due_date

    The cycle is NOT due when:
      - The pipeline was already run today (same calendar day), OR
      - The next due date is in the future

    This means if you run the pipeline 10 times in one day, the weights
    only update on the first run that day. All subsequent runs that day
    use the weights that were locked in at the start of that first run.
    """
    from datetime import datetime, timezone
    if schedule is None:
        schedule = load_bbn_schedule()

    now = datetime.now(timezone.utc)

    if schedule.get("last_update_date") is None:
        return True, "First run — initialising BBN schedule"

    next_due_str = schedule.get("next_due_date")
    if not next_due_str:
        return True, "No next_due_date recorded — running update"

    next_due = datetime.fromisoformat(next_due_str)

    if now >= next_due:
        days_overdue = (now - next_due).days
        return True, (
            f"Update due ({schedule.get('frequency', BBN_UPDATE_FREQUENCY)} cycle). "
            f"Last ran: {schedule.get('last_update_date', 'never')[:10]}. "
            f"Overdue by {days_overdue} days."
        )

    days_remaining = (next_due - now).days
    return False, (
        f"BBN weights current. Next update due: {next_due_str[:10]} "
        f"({days_remaining} days). Frequency: {schedule.get('frequency', BBN_UPDATE_FREQUENCY)}."
    )


def check_and_run_scheduled_update(frequency: str = None) -> dict:
    """
    Call this ONCE at pipeline startup, before score_all_patients() runs.

    Checks whether the calendar-based update cycle is due.
    If due AND enough cases exist: runs the update, locks new weights to disk.
    If not due OR too few cases: does nothing — current weights stay.

    Either way, score_all_patients() then reads whatever is in learned_ors.json
    and every patient in the run uses exactly the same weights.

    Args:
        frequency: override BBN_UPDATE_FREQUENCY (for testing). One of:
                   "monthly", "quarterly", "biannual", "annual"

    Returns dict with keys:
        update_ran      : bool
        reason          : str — why update ran or was skipped
        weights_source  : "learned" | "literature" | "updated"
        new_cases_used  : int
        next_due_date   : str (ISO)
    """
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    freq     = frequency or BBN_UPDATE_FREQUENCY
    freq_key = freq if freq in _FREQUENCY_DAYS else "biannual"
    interval = _FREQUENCY_DAYS[freq_key]

    schedule     = load_bbn_schedule()
    due, reason  = is_update_due(schedule)

    print(f"  [BBN Schedule] {reason}")

    if not due:
        # Weights unchanged — determine source for reporting
        from pathlib import Path
        source = "learned" if Path(LEARNED_ORS_FILE).exists() else "literature"
        return {
            "update_ran":    False,
            "reason":        reason,
            "weights_source": source,
            "new_cases_used": 0,
            "next_due_date": schedule.get("next_due_date", ""),
        }

    # Update is due — check whether enough cases exist
    dropouts  = load_confirmed_dropouts()
    new_cases = [v for v in dropouts.values() if not v.get("included_in_update")]

    if len(new_cases) < MIN_CASES_TO_UPDATE:
        reason_skip = (
            f"Update due but only {len(new_cases)} new confirmed cases "
            f"(minimum {MIN_CASES_TO_UPDATE}). Weights unchanged until next cycle."
        )
        print(f"  [BBN Schedule] {reason_skip}")
        # Still advance the schedule so we check again next period
        _advance_schedule(schedule, freq_key, interval)
        return {
            "update_ran":     False,
            "reason":         reason_skip,
            "weights_source": "learned" if Path(LEARNED_ORS_FILE).exists() else "literature",
            "new_cases_used": 0,
            "next_due_date":  schedule.get("next_due_date", ""),
        }

    # Run the update
    print(f"  [BBN Schedule] Running update with {len(new_cases)} new cases...")
    run_bbn_update_cycle(dropouts)

    # Advance schedule
    _advance_schedule(schedule, freq_key, interval)

    return {
        "update_ran":     True,
        "reason":         f"Scheduled {freq_key} update ran — {len(new_cases)} cases processed",
        "weights_source": "updated",
        "new_cases_used": len(new_cases),
        "next_due_date":  schedule.get("next_due_date", ""),
    }


def _advance_schedule(schedule: dict, freq_key: str, interval_days: int):
    """Update the schedule after a cycle (whether update ran or was skipped)."""
    from datetime import datetime, timezone, timedelta
    now      = datetime.now(timezone.utc)
    next_due = now + timedelta(days=interval_days)
    schedule["last_update_date"] = now.isoformat()
    schedule["next_due_date"]    = next_due.isoformat()
    schedule["frequency"]        = freq_key
    schedule["cycles_completed"] = schedule.get("cycles_completed", 0) + 1
    save_bbn_schedule(schedule)
    print(f"  [BBN Schedule] Next update scheduled: {next_due.strftime('%Y-%m-%d')} "
          f"({freq_key}, {interval_days} days)")


def load_learned_ors() -> dict:
    """
    Load the latest learned OR values.
    Falls back to the hardcoded literature values if no learned file exists.
    """
    from pathlib import Path
    if Path(LEARNED_ORS_FILE).exists():
        with open(LEARNED_ORS_FILE) as f:
            data = json.load(f)
            return data.get("ors", {})
    # Return literature defaults as starting point
    return {k: float(np.exp(v)) for k, v in LOG_OR.items()}


def run_bbn_update_cycle(dropouts: dict = None):
    """
    Bayesian OR update — runs on a batch of confirmed dropouts.

    For each risk factor, computes:
        observed_OR = (dropouts_with_factor / total_with_factor)
                    / (dropouts_without_factor / total_without_factor)

        updated_OR = (prior_OR * PRIOR_WEIGHT + observed_OR * n_cases)
                   / (PRIOR_WEIGHT + n_cases)

    OR cannot move more than MAX_MOVE_PER_CYCLE (50%) per cycle.
    Factors with fewer than MIN_CASES_TO_UPDATE cases are frozen.

    Saves results to data/learned_ors.json and marks cases as processed.
    """
    from datetime import datetime, timezone
    if dropouts is None:
        dropouts = load_confirmed_dropouts()

    new_cases = [v for v in dropouts.values() if not v.get("included_in_update")]
    if len(new_cases) < 10:
        print(f"  [BBN Update] Only {len(new_cases)} new cases — minimum 10 required. Skipping.")
        return

    # Load current ORs (literature or previously learned)
    current_ors = load_learned_ors()
    update_log  = {}

    # Factor name → column in the factors dict (matches keys from compute_bbn_prior)
    factor_map = {
        "alcohol_use":            "Alcohol use",
        "divorced_separated":     "Divorced/separated",
        "hiv":                    "HIV co-infection",
        "prior_tb":               "Prior LTFU/TB history",
        "drug_use":               "Drug use",
        "continuation_phase":     "Continuation phase",
        "no_nutritional_support": "No nutritional support",
        "no_welfare":             "Not enrolled in Nikshay Poshan Yojana",
        "dr_tb":                  "Drug-resistant TB (DR-TB)",
        "low_education":          "Low education",
        "male_sex":               "Male sex",
    }

    n_total = len(new_cases)

    for factor_key, factor_label in factor_map.items():
        lit_or     = float(np.exp(LOG_OR.get(factor_key, 0)))
        current_or = current_ors.get(factor_key, lit_or)

        cases_with    = sum(1 for c in new_cases if factor_label in c.get("factors", {}))
        cases_without = n_total - cases_with

        if cases_with < MIN_CASES_TO_UPDATE:
            update_log[factor_key] = f"FROZEN ({cases_with} cases with factor < {MIN_CASES_TO_UPDATE} minimum)"
            continue

        # Observed OR: how often this factor appeared in real dropouts vs not
        # Guard against division by zero
        p_with    = cases_with    / max(n_total, 1)
        p_without = cases_without / max(n_total, 1)
        if p_without == 0:
            observed_or = current_or  # can't compute meaningfully
        else:
            observed_or = p_with / p_without

        # Weighted Bayesian average
        updated_or = (
            (current_or * PRIOR_WEIGHT) + (observed_or * n_total)
        ) / (PRIOR_WEIGHT + n_total)

        # Sanity clamp: OR can't move more than 50% in one cycle
        max_move   = current_or * MAX_MOVE_PER_CYCLE
        updated_or = max(current_or - max_move, min(current_or + max_move, updated_or))
        updated_or = round(max(0.1, updated_or), 4)  # OR can't go below 0.1

        update_log[factor_key] = (
            f"lit={lit_or:.3f} → prev={current_or:.3f} → updated={updated_or:.3f} "
            f"({cases_with}/{n_total} cases with factor)"
        )
        current_ors[factor_key] = updated_or

    # Mark all processed cases
    for pid in dropouts:
        if not dropouts[pid].get("included_in_update"):
            dropouts[pid]["included_in_update"] = True
    with open(CONFIRMED_DROPOUTS_FILE, "w") as f:
        json.dump(dropouts, f, indent=2)

    # Save learned ORs
    from pathlib import Path
    Path("data").mkdir(exist_ok=True)
    with open(LEARNED_ORS_FILE, "w") as f:
        json.dump({
            "ors":          current_ors,
            "updated_at":   datetime.now(timezone.utc).isoformat(),
            "cases_used":   n_total,
            "total_confirmed": len(dropouts),
            "update_log":   update_log,
        }, f, indent=2)

    print(f"\n  [BBN Update] OR update complete — {n_total} new cases processed")
    for k, v in update_log.items():
        print(f"    {k}: {v}")
    print(f"  Saved → {LEARNED_ORS_FILE}")


def get_effective_log_ors() -> dict:
    """
    Return the current effective log-OR dict.
    Uses learned values if available, falls back to literature values.
    Called by compute_bbn_prior() so the BBN automatically uses updated weights.
    """
    learned = load_learned_ors()
    result  = dict(LOG_OR)  # start from literature
    for k in result:
        if k in learned:
            result[k] = np.log(max(0.01, learned[k]))
    return result


if __name__ == "__main__":
    with open("nikshay_grounded_dataset.json") as f:
        patients = json.load(f)

    scored  = score_all_patients(patients[:100])
    alerts  = detect_systemic_failures(scored)

    top5 = sorted(scored, key=lambda x: x["risk_score"], reverse=True)[:5]
    print("\nTop 5 patients:")
    for p in top5:
        print(f"  {p['patient_id']}  final={p['risk_score']}  tier={p['risk_level']}  "
              f"week={p['treatment_week']}  threshold_H={p['thresholds']['high']}")

    with open("nikshay_scored_dataset.json", "w") as f:
        json.dump(scored, f, indent=2)
    print("\nSaved nikshay_scored_dataset.json")