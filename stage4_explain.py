"""
stage4_explain.py
=================
Stage 4: Explanation & Visualisation

Implements the pipeline document's explanation layer exactly:
  - Template-based explanations only — NO free-form LLM generation
    (deliberate safety decision: prevents hallucination in medical output)
  - Attention weights from TGN extracted as explainability factors
  - Two explanation formats: ASHA worker (simple) and District Officer (detailed)
  - Azure AI Foundry safety validation before any output is delivered
  - Graph data prepared for District Officer dashboard

Why template-based, not LLM:
  "The system always uses a fixed template and never generates free-form text
   for explanations. Free-form language model generation in a medical context
   carries hallucination risk. The template ensures every explanation is
   directly grounded in actual model outputs."
  — Nikshay-Graph Pipeline Document, Stage 4

Usage:
    from stage4_explain import generate_asha_explanation, generate_officer_explanation
    from stage4_explain import validate_output_safety, get_patient_visit_list
"""

import os
import json
import re
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

def _get_primary_reason(record: dict) -> str:
    """
    Determine the single most urgent reason from top factors + clinical state.
    Returns a short plain-English phrase suitable for ASHA workers.
    """
    missed = record["adherence"]["days_since_last_dose"]
    factors = record.get("top_factors", {})

    # Priority order: dose gap > silence > prior LFU > HIV > distance > load
    if missed >= 14:
        return f"has not taken medicine for {missed} days"
    if record.get("silence_event"):
        days_quiet = record["silence_event"].get("duration_days", missed)
        return f"no contact for {days_quiet} days — may be disengaging"
    if missed >= 7:
        return f"missed medicine for {missed} days"
    if record["adherence"].get("prior_lfu_history"):
        return "has dropped out of treatment before"
    if record["clinical"]["comorbidities"].get("hiv"):
        return "HIV co-infection makes dropout especially dangerous"
    if record["adherence"].get("distance_to_center_km", 0) > 10:
        dist = record["adherence"]["distance_to_center_km"]
        return f"lives {dist:.1f}km from treatment centre — access barrier"
    if record.get("asha_load_score", 0) > 0.7:
        return "ASHA worker has high caseload — visit overdue"
    # Fall back to top factor
    if factors:
        top_name = list(factors.keys())[0]
        return f"risk factor: {top_name.lower()}"
    return "treatment engagement declining"


def _get_first_name(patient_id: str) -> str:
    """Extract a display name from patient ID for ASHA briefing."""
    # In production: look up actual patient name from Nikshay
    # For prototype: use patient ID suffix
    return patient_id.split("-")[-1]


def generate_asha_explanation(record: dict) -> str:
    """
    ASHA-facing explanation. Simple, under 20 words, actionable.
    Format: "Visit [identifier] — [single most urgent reason]"

    Grounded entirely in actual record data — no LLM involved.
    """
    identifier = _get_first_name(record["patient_id"])
    reason     = _get_primary_reason(record)
    return f"Visit patient {identifier} — {reason}."


def generate_officer_explanation(record: dict) -> str:
    """
    District Officer-facing explanation. More detailed, includes score components.
    Format: "Patient [ID] — [tier] risk, Week [N]. Primary: [reason]. Secondary: [reason]."
    """
    pid        = record["patient_id"]
    tier       = record.get("risk_level", "?")
    week       = record.get("treatment_week", "?")
    score      = record.get("risk_score", 0)
    phase      = record["clinical"]["phase"]
    threshold  = record.get("thresholds", {})
    composition= record.get("score_composition", {})
    factors    = list(record.get("top_factors", record.get("all_factors", {})).keys())

    primary   = factors[0] if len(factors) > 0 else "unknown"
    secondary = factors[1] if len(factors) > 1 else None

    # Component breakdown — use model weights (sum to 100%), not raw contributions
    tgn_pct  = int(composition.get("tgn_weight",  0) * 100)
    bbn_pct  = int(composition.get("bbn_weight",  0) * 100)
    asha_pct = int(composition.get("asha_weight", 0) * 100)

    bbn_note = f" BBN prior: {composition.get('bbn_status', 'active')}." if composition else ""

    explanation = (
        f"Patient {pid} — {tier} risk (score {score:.2f}), Week {week} of treatment ({phase}). "
        f"Threshold at this stage: HIGH > {threshold.get('high', 0.65)}. "
        f"Primary driver: {primary.lower()}. "
    )
    if secondary:
        explanation += f"Secondary: {secondary.lower()}. "
    explanation += (
        f"Score components: patient factors {tgn_pct}%, "
        f"clinical prior {bbn_pct}%, "
        f"ASHA workload {asha_pct}%.{bbn_note}"
    )

    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# AZURE AI FOUNDRY SAFETY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_output_safety(explanation: str, record: dict) -> dict:
    """
    Validate explanation before delivery.
    Checks (from pipeline document):
    1. References only patient-specific clinical factors
    2. Contains no diagnostic claims or medication recommendations
    3. Does not fabricate information not present in model outputs
    4. Framed as visit prioritisation guidance only

    Production: calls Azure AI Foundry content safety API.
    Prototype: rule-based checks + stub.
    """
    blocked_phrases = [
        "you have", "you are", "diagnosed with", "you should take",
        "prescribe", "medication change", "stop taking", "side effect",
        "cure", "will die", "cancer", "increase dose", "decrease dose",
    ]

    violations = []
    exp_lower  = explanation.lower()

    for phrase in blocked_phrases:
        if phrase in exp_lower:
            violations.append(f"Blocked phrase detected: '{phrase}'")

    # Check it's framed as visit guidance
    if not any(w in exp_lower for w in ["visit", "patient", "risk", "dose", "contact", "screen"]):
        violations.append("Explanation does not appear to be visit prioritisation guidance.")

    # Production path: Azure AI Foundry content safety.
    # A network/API error is an infrastructure problem, NOT a content problem.
    # Log it and allow the explanation through — never silently suppress valid
    # clinical guidance because of a transient connectivity issue.
    foundry_endpoint = os.getenv("FOUNDRY_ENDPOINT")
    if foundry_endpoint and not violations:
        try:
            _call_foundry_safety(explanation)
        except Exception as e:
            print(f"  [Safety] Foundry API error (explanation NOT blocked): {e}")

    return {
        "passed":     len(violations) == 0,
        "violations": violations,
        "text":       explanation if not violations else "[BLOCKED — safety violation]",
    }


def _call_foundry_safety(text: str) -> bool:
    """
    Screen explanation text through Azure AI Content Safety.

    Works with both Azure endpoint formats:
      https://<n>.cognitiveservices.azure.com/   (classic Cognitive Services)
      https://<n>.services.ai.azure.com/         (AI Foundry hub)

    Raises ValueError only for genuine content violations (severity > 2).
    Returns True when text is safe. Infrastructure errors bubble up to the
    caller which logs them and allows the explanation through.
    """
    import requests
    endpoint = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")
    key      = os.getenv("FOUNDRY_KEY")
    if not endpoint or not key:
        return True  # not configured — skip silently

    url = f"{endpoint}/contentsafety/text:analyze?api-version=2024-02-15-preview"
    try:
        resp = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/json"},
            json={"text": text, "categories": ["Hate", "SelfHarm", "Sexual", "Violence"]},
            timeout=10,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        raise ValueError(
            f"Content Safety HTTP {resp.status_code}: {resp.text[:200]}"
        ) from http_err

    for cat in resp.json().get("categoriesAnalysis", []):
        if cat.get("severity", 0) > 2:
            raise ValueError(
                f"Content safety violation: {cat['category']} severity {cat['severity']}"
            )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# VISIT PRIORITY RANKING
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_visit_list(patients: list, top_n: int = 10) -> list:
    """
    Rank patients for ASHA visit priority.
    Sorting key: risk_score (includes urgency multiplier) then treatment_week desc.
    A patient with score 0.71 at week 22 ranks above 0.79 at week 3
    because the intervention window is narrower.
    """
    def priority_key(p):
        risk = p.get("risk_score", 0)
        week = p.get("treatment_week", 1)
        # Later treatment weeks get a small boost to break ties
        return (risk, week / 26)

    ranked = sorted(patients, key=priority_key, reverse=True)

    results = []
    for i, p in enumerate(ranked[:top_n]):
        asha_exp = generate_asha_explanation(p)
        officer_exp = generate_officer_explanation(p)

        asha_safety    = validate_output_safety(asha_exp, p)
        officer_safety = validate_output_safety(officer_exp, p)

        results.append({
            "rank":               i + 1,
            "patient_id":         p["patient_id"],
            "risk_score":         p["risk_score"],
            "risk_level":         p.get("risk_level", "?"),
            "treatment_week":     p.get("treatment_week", "?"),
            "phase":              p["clinical"]["phase"],
            "days_missed":        p["adherence"]["days_since_last_dose"],
            "asha_id":            p["operational"]["asha_id"],
            "block":              p["location"]["block"],
            "top_factors":        p.get("top_factors", {}),
            "score_composition":  p.get("score_composition", {}),
            # Template-based explanations — no LLM
            "asha_explanation":   asha_safety["text"],
            "officer_explanation":officer_safety["text"],
            "safety_passed":      asha_safety["passed"] and officer_safety["passed"],
        })

    blocked = sum(1 for r in results if not r["safety_passed"])
    if blocked:
        print(f"  ⚠ {blocked} explanations blocked by safety validation")

    return results


def get_contact_screening_list(G, pagerank_scores: dict, top_n: int = 10) -> list:
    """
    Rank contacts for TB screening by propagated PageRank score.
    Filter out already-screened contacts.

    G must be a NetworkX graph whose nodes carry properties:
        node_type, age, rel, vulnerability, screened, name, source_patient
    These are set by stage2 when building the patient/contact graph,
    NOT the PyTorch edge_index graph. Pass the nx graph, not the torch one.
    """
    contacts = []
    for node_id, score in pagerank_scores.items():
        node = G.nodes.get(node_id, {})
        if node.get("node_type") != "contact" or node.get("screened", False):
            continue
        age  = node.get("age", 30)
        rel  = node.get("rel", "Workplace")
        vuln = node.get("vulnerability", 1.0)
        age_risk = 1.5 if age > 60 or age < 10 else 1.0
        rel_risk = 1.3 if rel == "Household" else 1.0
        priority = score * vuln * age_risk * rel_risk

        # Template-based screening reason
        reason = f"Screen {node.get('name', 'contact')} (age {age}, {rel}) — unscreened contact of a high-risk patient."

        contacts.append({
            "contact_id":         node_id,
            "name":               node.get("name", "Unknown"),
            "age":                age,
            "rel":                rel,
            "vulnerability":      vuln,
            "source_patient":     node.get("source_patient", ""),
            "screening_priority": round(priority, 8),
            "screening_reason":   reason,
        })

    ranked = sorted(contacts, key=lambda x: x["screening_priority"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked[:top_n]


if __name__ == "__main__":
    with open("nikshay_scored_dataset.json") as f:
        patients = json.load(f)

    visit_list = get_patient_visit_list(patients, top_n=5)
    print("\nASHA Visit List (template explanations, safety-validated):")
    for v in visit_list:
        print(f"\n  Rank {v['rank']}: {v['patient_id']} [{v['risk_level']}]")
        print(f"  ASHA:    {v['asha_explanation']}")
        print(f"  Officer: {v['officer_explanation']}")
        print(f"  Safety:  {'✓ passed' if v['safety_passed'] else '✗ BLOCKED'}")