# -*- coding: utf-8 -*-
"""Research-only KTAS to hospital-capability screening; not a clinical triage rule."""
from math import isfinite

REQUIRED_LEVEL = {1: 3, 2: 3, 3: 2, 4: 1, 5: 1}
HOSPITAL_LEVEL = {
    "권역응급의료센터": 3,
    "지역응급의료센터": 2,
    "지역응급의료기관": 1,
    "응급실운영신고기관": 1,
}

def explain_hospital_eligibility(hospital, ktas, occupancy_limit=0.85):
    """Explain independent hard-mask rejections for one patient/hospital pair.

    Counts are per patient-hospital pair, not unique failed patients.
    """
    k = int(ktas)
    if k not in REQUIRED_LEVEL or float(ktas) != k:
        raise ValueError(f"Invalid KTAS: {ktas!r}")
    reasons = []
    if not hospital.get("reachable", False):
        reasons.append("no_route")
    hospital_type = str(hospital.get("type", "")).strip()
    level = HOSPITAL_LEVEL.get(hospital_type)
    if level is None:
        reasons.append("unknown_hospital_type")
    elif level < REQUIRED_LEVEL[k]:
        reasons.append("insufficient_capability")
    try:
        occupancy = float(hospital.get("occupancy", 1.0))
    except (TypeError, ValueError):
        occupancy = float("nan")
    if (not isfinite(occupancy) or occupancy < 0 or occupancy >= occupancy_limit
            or not hospital.get("can_admit", True)):
        reasons.append("hospital_full_or_invalid_occupancy")
    if hospital.get("accepting", True) is not True:
        reasons.append("not_accepting")
    return reasons


def valid_actions(hospitals, ktas, occupancy_limit=0.85):
    return [i for i, hospital in enumerate(hospitals)
            if not explain_hospital_eligibility(hospital, ktas, occupancy_limit)]
