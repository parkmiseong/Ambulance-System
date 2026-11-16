"""Source-backed hourly demand shared by the dashboard and new CLI training."""
import copy
import json
from pathlib import Path

REFERENCE = json.loads((Path(__file__).resolve().parents[1] / "DATASET" /
                       "gangnam_ems_reference.json").read_text(encoding="utf-8"))
REFERENCE_PER_HOUR = REFERENCE["reference_value_per_day"] / 24.
PATIENTS_PER_HOUR = tuple(REFERENCE_PER_HOUR * multiplier
                          for multiplier in REFERENCE["scenario_multipliers"])
DEFAULT_DURATION = REFERENCE["default_duration_seconds"]


def demand_metadata():
    return {**copy.deepcopy(REFERENCE), "reference_per_hour": REFERENCE_PER_HOUR,
            "patients_per_hour": list(PATIENTS_PER_HOUR), "simulation_unit": "명/시뮬레이션 시간(1시간)",
            "input_modes": ["preset", "rate", "count"], "max_patients_per_hour": 3600}


def dashboard_scenario(options, legacy_scenarios):
    """Convert patients/hour into the shared engine's per-SUMO-second Bernoulli probability."""
    scenario = dict(legacy_scenarios[options["demand"] * 3 + options["traffic"]])
    mode = options.get("demand_mode", "preset")
    rate = PATIENTS_PER_HOUR[options["demand"]]
    profile = demand_metadata()
    if mode == "rate":
        rate = options["custom_patients_per_hour"]
        profile = dict(profile_id="user_hourly_v1", mode=mode, basis="사용자가 직접 입력한 시간당 기대 환자 수")
    elif mode == "count":
        count = options["patient_count"]
        rate = count * 3600. / options["duration"]
        scenario["fixed_patient_count"] = count
        profile = dict(profile_id="user_fixed_count_v1", mode=mode, basis="사용자가 직접 입력한 회차당 총 환자 수")
    elif mode != "preset":
        raise ValueError("환자 발생 설정 방식이 올바르지 않습니다.")
    scenario.update(prob=rate / 3600., patients_per_hour=rate,
                    demand_mode=mode,
                    name=f"{profile['profile_id']}_patients_per_hour{rate:.6f}_traffic{scenario['scale']}",
                    demand_profile=profile)
    return scenario
