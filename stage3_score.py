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
    lo = BASELINE_LOG_ODDS
    factors = {}

    d, c, s, a, o = (record["demographics"], record["clinical"],
                     record["social"], record["adherence"], record["operational"])

    if s.get("alcohol_use"):
        lo += LOG_OR["alcohol_use"]
        factors["Alcohol use"] = round(np.exp(LOG_OR["alcohol_use"]), 2)
    if d.get("marital") == "Divorced":
        lo += LOG_OR["divorced_separated"]
        factors["Divorced/separated"] = round(np.exp(LOG_OR["divorced_separated"]), 2)
    if c["comorbidities"].get("diabetes"):
        lo += LOG_OR["diabetes"]
        factors["Diabetes (monitored — protective)"] = round(np.exp(LOG_OR["diabetes"]), 2)
    if c["comorbidities"].get("hiv"):
        lo += LOG_OR["hiv"]
        factors["HIV co-infection"] = round(np.exp(LOG_OR["hiv"]), 2)
    if a.get("prior_lfu_history"):
        lo += LOG_OR["prior_tb"]
        factors["Prior LTFU/TB history"] = round(np.exp(LOG_OR["prior_tb"]), 2)
    if d.get("gender") == "Male":
        lo += LOG_OR["male_sex"]
        factors["Male sex"] = round(np.exp(LOG_OR["male_sex"]), 2)
    if s.get("low_education"):
        lo += LOG_OR["low_education"]
        factors["Low education"] = round(np.exp(LOG_OR["low_education"]), 2)
    if s.get("drug_use"):
        lo += LOG_OR["drug_use"]
        factors["Drug use"] = round(np.exp(LOG_OR["drug_use"]), 2)
    if c.get("phase") == "Continuation":
        lo += LOG_OR["continuation_phase"]
        factors["Continuation phase"] = round(np.exp(LOG_OR["continuation_phase"]), 2)
    if not o.get("nutritional_support"):
        lo += LOG_OR["no_nutritional_support"]
        factors["No nutritional support"] = round(np.exp(LOG_OR["no_nutritional_support"]), 2)
    if not o.get("welfare_enrolled", True):  # NPY non-enrollment
        lo += LOG_OR["no_welfare"]
        factors["Not enrolled in Nikshay Poshan Yojana"] = round(np.exp(LOG_OR["no_welfare"]), 2)
    c_data = record["clinical"]
    if c_data.get("regimen") == "DR_TB":
        lo += LOG_OR["dr_tb"]
        factors["Drug-resistant TB (DR-TB)"] = round(np.exp(LOG_OR["dr_tb"]), 2)

    dist = a.get("distance_to_center_km", 0)
    if 5 <= dist < 10:
        lo += LOG_OR["distance_5_to_10km"]
        factors[f"Distance {dist:.1f}km (5-10km)"] = round(np.exp(LOG_OR["distance_5_to_10km"]), 2)
    elif dist >= 10:
        lo += LOG_OR["distance_over_10km"]
        factors[f"Distance {dist:.1f}km (>10km)"] = round(np.exp(LOG_OR["distance_over_10km"]), 2)

    missed = a.get("days_since_last_dose", 0)
    if 7 <= missed < 14:
        lo += LOG_OR["missed_7_to_13_days"]
        factors[f"{missed} days since last dose"] = round(np.exp(LOG_OR["missed_7_to_13_days"]), 2)
    elif missed >= 14:
        lo += LOG_OR["missed_14_plus_days"]
        factors[f"{missed} days since last dose (CRITICAL)"] = round(np.exp(LOG_OR["missed_14_plus_days"]), 2)

    age = d.get("age", 30)
    if 20 <= age <= 39:
        lo += LOG_OR["age_20_to_39"]
        factors[f"Age {age} (high-risk group 20-39)"] = round(np.exp(LOG_OR["age_20_to_39"]), 2)
    elif age > 60:
        lo += LOG_OR["age_over_60"]
        factors[f"Age {age} (elderly)"] = round(np.exp(LOG_OR["age_over_60"]), 2)

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
    TGN output        | 0.60              | 0.80
    BBN prior         | 0.40 → fading     | 0.00 → retired
    ASHA load score   | 0.20 (permanent)  | 0.20 (permanent)

    Note: weights are normalised to sum to 1.0.
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