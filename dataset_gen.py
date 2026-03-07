"""
dataset_gen.py — single district, 1000 patients, max 15/ASHA, ANMs, PHC, welfare, regimen
"""
import json, os, numpy as np
from pathlib import Path

try:
    from faker import Faker
    fake = Faker("en_IN")
    def random_name(): return fake.name()
except ImportError:
    _NAMES = ["Murugan Selvam","Anitha Rajan","Karthik Kumar","Priya Devi","Rajan Muthu",
              "Savitri Amma","Suresh Babu","Meena Devi","Venkatesh Pillai","Lakshmi N",
              "Senthil Kumar","Kavitha Raj","Balaji Sundaram","Deepa Krishnan","Manoj Prabhu"]
    import random as _r
    def random_name(): return _r.choice(_NAMES)

rng = np.random.default_rng(42)

DISTRICT, BLOCK, STATE, LANGUAGE, PHC_ID = "Chennai","Tondiarpet","Tamil Nadu","Tamil","PHC_Tondiarpet"
N_PATIENTS, MAX_PER_ASHA = 1000, 20
N_ASHAS = N_PATIENTS // MAX_PER_ASHA        # 66
N_ANMS  = max(1, N_ASHAS // 5)             # 13
ASHA_IDS = [f"ASHA-TN-{i+1:03d}" for i in range(N_ASHAS)]
ANM_IDS  = [f"ANM-TN-{i+1:02d}"  for i in range(N_ANMS)]
ASHA_TO_ANM = {a: ANM_IDS[min(i//5, N_ANMS-1)] for i,a in enumerate(ASHA_IDS)}

def _asha(uid): return ASHA_IDS[min(uid//MAX_PER_ASHA, N_ASHAS-1)]
def _anm(aid):  return ASHA_TO_ANM.get(aid, ANM_IDS[0])

BASELINE_LTFU     = 0.062
BASELINE_LOG_ODDS = np.log(BASELINE_LTFU/(1-BASELINE_LTFU))
LOG_OR = {
    "alcohol_use":np.log(1.92),"divorced_separated":np.log(3.80),"diabetes":np.log(0.52),
    "hiv":np.log(2.16),"prior_tb":np.log(2.10),"male_sex":np.log(1.29),
    "low_education":np.log(1.73),"drug_use":np.log(2.40),"continuation_phase":np.log(2.30),
    "no_nutritional_support":np.log(1.60),"no_welfare":np.log(1.45),"dr_tb":np.log(2.80),
    "distance_5_to_10km":np.log(1.60),"distance_over_10km":np.log(2.10),
    "missed_7_to_13_days":np.log(3.20),"missed_14_plus_days":np.log(6.50),
    "age_20_to_39":np.log(2.07),"age_over_60":np.log(1.40),
}
PREV = {"diabetes":0.12,"hiv":0.03,"prior_tb":0.18,"nutritional_support":0.54,
        "welfare_enrolled":0.60,"alcohol_use":0.22,"drug_use":0.08,"low_education":0.45}
TEMPLATES = [
    "Patient lives with {rel1} {name1} ({age1}) and {rel2} {name2} ({age2}). Works at {wp}.",
    "Household: {rel1} {name1} ({age1}). Coworker {name2} at {wp}.",
    "{rel1} {name1} ({age1}) and {rel2} {name2} ({age2}) live with patient.",
    "Patient reported {rel1} {name1} coughing. Also lives with {rel2} {name2} ({age2}).",
]
WORKPLACES = ["Tondiarpet tannery","Royapuram fish market","Basin Bridge rail yard",
              "Tondiarpet tile factory","Washermanpet textile unit","harbour cargo dock"]
RELS = ["wife","husband","mother","father","brother","sister","son","daughter","uncle","aunt"]

def _note(contacts):
    t  = str(rng.choice(TEMPLATES))
    wp = str(rng.choice(WORKPLACES))
    c1 = contacts[0] if contacts else {"name":random_name(),"age":35}
    c2 = contacts[1] if len(contacts)>1 else {"name":random_name(),"age":28}
    return t.format(rel1=str(rng.choice(RELS)),name1=c1["name"],age1=c1["age"],
                    rel2=str(rng.choice(RELS)),name2=c2["name"],age2=c2["age"],wp=wp)

def _ltfu(p):
    lo = BASELINE_LOG_ODDS
    if p["alcohol_use"]:             lo += LOG_OR["alcohol_use"]
    if p["marital"]=="Divorced":     lo += LOG_OR["divorced_separated"]
    if p["diabetes"]:                lo += LOG_OR["diabetes"]
    if p["hiv"]:                     lo += LOG_OR["hiv"]
    if p["prior_tb"]:                lo += LOG_OR["prior_tb"]
    if p["gender"]=="Male":          lo += LOG_OR["male_sex"]
    if p["low_education"]:           lo += LOG_OR["low_education"]
    if p["drug_use"]:                lo += LOG_OR["drug_use"]
    if p["phase"]=="Continuation":   lo += LOG_OR["continuation_phase"]
    if not p["nutritional_support"]: lo += LOG_OR["no_nutritional_support"]
    if not p["welfare_enrolled"]:    lo += LOG_OR["no_welfare"]
    if p["regimen"]=="DR_TB":        lo += LOG_OR["dr_tb"]
    d = p["distance_km"]
    if 5<=d<10:  lo += LOG_OR["distance_5_to_10km"]
    elif d>=10:  lo += LOG_OR["distance_over_10km"]
    m = p["days_missed"]
    if 7<=m<14:  lo += LOG_OR["missed_7_to_13_days"]
    elif m>=14:  lo += LOG_OR["missed_14_plus_days"]
    a = p["age"]
    if 20<=a<=39: lo += LOG_OR["age_20_to_39"]
    elif a>60:    lo += LOG_OR["age_over_60"]
    return float(np.clip(1/(1+np.exp(-lo)) + rng.normal(0,0.025), 0.0, 1.0))

def _history(days_missed, phase):
    if phase=="Intensive":
        h = [1]*30
        for i in range(min(days_missed,30)): h[-(i+1)]=0
        for i in range(30-days_missed):
            if rng.random()<0.04: h[i]=0
    else:
        h = []
        for day in range(30):
            is_dose_day = (day%7) in [0,2,4]
            if not is_dose_day: h.append(0)
            elif day>=(30-days_missed): h.append(0)
            else: h.append(0 if rng.random()<0.04 else 1)
    return h

def _phase_rate(history, phase):
    expected = 30 if phase=="Intensive" else sum(1 for d in range(30) if (d%7) in [0,2,4])
    return round(min(sum(history)/max(expected,1), 1.0), 3)

def _patient(uid):
    aid     = _asha(uid)
    anm_id  = _anm(aid)
    phase   = str(rng.choice(["Intensive","Continuation"], p=[0.35,0.65]))
    regimen = str(rng.choice(["Cat_I","Cat_II","DR_TB"], p=[0.78,0.17,0.05]))
    p = {
        "age":int(rng.integers(18,75)), "gender":str(rng.choice(["Male","Female"],p=[0.65,0.35])),
        "diabetes":bool(rng.random()<PREV["diabetes"]), "hiv":bool(rng.random()<PREV["hiv"]),
        "prior_tb":bool(rng.random()<PREV["prior_tb"]), "phase":phase, "regimen":regimen,
        "nutritional_support":bool(rng.random()<PREV["nutritional_support"]),
        "welfare_enrolled":bool(rng.random()<PREV["welfare_enrolled"]),
        "alcohol_use":bool(rng.random()<PREV["alcohol_use"]), "drug_use":bool(rng.random()<PREV["drug_use"]),
        "low_education":bool(rng.random()<PREV["low_education"]),
        "marital":str(rng.choice(["Married","Single","Divorced","Widowed"],p=[0.60,0.25,0.05,0.10])),
        "distance_km":float(round(float(rng.gamma(2,2.5)),2)),
        "days_missed":int(rng.choice([0,1,2,7,14],p=[0.80,0.12,0.04,0.03,0.01])),
        "total_treatment_days":int(rng.integers(7,183)),
        "phone_type":str(rng.choice(["Smartphone","Basic","None"],p=[0.55,0.38,0.07])),
    }
    hist      = _history(p["days_missed"], phase)
    last_visit= int(rng.choice([1,3,7,14,30],p=[0.20,0.30,0.25,0.15,0.10]))
    contacts  = []
    for _ in range(int(rng.integers(1,6))):
        ca = int(rng.integers(5,75)); af=1.5 if (ca>60 or ca<10) else 1.0
        cm = bool(rng.random()<0.15)
        contacts.append({"name":random_name(),"age":ca,
                         "rel":str(rng.choice(["Household","Workplace"],p=[0.80,0.20])),
                         "has_comorbidity":cm,"vulnerability_score":float(round(af*(1.5 if cm else 1.0),2)),
                         "screened":bool(rng.random()<0.30)})
    return {
        "patient_id": f"NIK-{100001+uid}",
        "location":{"district":DISTRICT,"state":STATE,"block":BLOCK,"phc_id":PHC_ID},
        "demographics":{"age":p["age"],"gender":p["gender"],"marital":p["marital"]},
        "clinical":{"comorbidities":{"diabetes":p["diabetes"],"hiv":p["hiv"]},
                    "phase":phase,"regimen":regimen,
                    "total_treatment_days":p["total_treatment_days"]},
        "social":{"alcohol_use":p["alcohol_use"],"drug_use":p["drug_use"],
                  "low_education":p["low_education"]},
        "adherence":{"days_since_last_dose":p["days_missed"],"dose_history_30d":hist,
                     "adherence_rate_30d":round(sum(hist)/30,3),
                     "phase_adherence_rate":_phase_rate(hist,phase),
                     "prior_lfu_history":p["prior_tb"],
                     "distance_to_center_km":p["distance_km"]},
        "operational":{"nutritional_support":p["nutritional_support"],
                       "welfare_enrolled":p["welfare_enrolled"],
                       "phone_type":p["phone_type"],"language":LANGUAGE,
                       "asha_id":aid,"anm_id":anm_id,
                       "last_asha_visit_days_ago":last_visit},
        "contact_network":contacts,"free_text_note":_note(contacts),
        "risk_score":round(_ltfu(p),4),"risk_velocity":0.0,"previous_risk_score":None,
    }

def generate_and_save(n=N_PATIENTS, path="data/nikshay_grounded_dataset.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} patients — {BLOCK}, {DISTRICT}")
    print(f"  ASHAs:{N_ASHAS} (max {MAX_PER_ASHA} each)  ANMs:{N_ANMS}  PHC:{PHC_ID}")
    data = [_patient(i) for i in range(n)]
    from collections import Counter
    ac = Counter(p["operational"]["asha_id"] for p in data)
    assert max(ac.values())<=MAX_PER_ASHA, "ASHA over-capacity"
    with open(path,"w") as f: json.dump(data,f,indent=2)
    high   = sum(1 for r in data if r["risk_score"]>0.65)
    silent = sum(1 for r in data if max(r["adherence"]["days_since_last_dose"],
                 r["operational"]["last_asha_visit_days_ago"])>=7)
    welfare= sum(1 for r in data if r["operational"]["welfare_enrolled"])
    dr     = sum(1 for r in data if r["clinical"]["regimen"]=="DR_TB")
    print(f"  Saved {n} records → {path}")
    print(f"  HIGH risk:{high}({high/n*100:.1f}%)  Silent:{silent}  Welfare:{welfare}  DR-TB:{dr}")
    print(f"  ASHA load range: {min(ac.values())}–{max(ac.values())} patients")
    return data

if __name__=="__main__":
    generate_and_save()