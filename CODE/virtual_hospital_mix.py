"""Web hospital composition: two tertiary centers plus a deterministic ratio split."""
from collections import Counter
import random

from reward import HOSPITAL_TYPES

TYPE_KEYS = ("tertiary", "local_center", "local_institution", "reported_er")
TYPE_LABELS = dict(zip(TYPE_KEYS, HOSPITAL_TYPES))
RATIO_KEYS = TYPE_KEYS[1:]
DEFAULT_WEIGHTS = dict(zip(RATIO_KEYS, (5, 4, 1)))
PROFILE_ID = "tertiary2_ratio_v1"


def mix_controls():
    return dict(profile_id=PROFILE_ID, base_tertiary_count=2, default_weights=dict(DEFAULT_WEIGHTS),
                categories=[dict(key=key, label=TYPE_LABELS[key]) for key in RATIO_KEYS])


def hospital_plan(count, weights=None):
    if type(count) is not int or not 0 <= count <= 10 or count == 1:
        raise ValueError("가상 병원은 0개 또는 2~10개입니다. 생성 시 권역응급의료센터 2개를 기본 포함합니다.")
    weights = dict(DEFAULT_WEIGHTS) if weights is None else weights
    if not isinstance(weights, dict) or set(weights) != set(RATIO_KEYS):
        raise ValueError("나머지 가상 병원의 세 유형 비율을 모두 지정하세요.")
    if any(type(value) is not int or not 0 <= value <= 100 for value in weights.values()):
        raise ValueError("가상 병원 비율은 0~100 사이의 정수로 입력하세요.")
    remaining, total = max(0, count-2), sum(weights.values())
    if remaining and not total:
        raise ValueError("추가 가상 병원이 있으면 유형 비율 중 하나 이상은 0보다 커야 합니다.")
    counts = {key: 0 for key in TYPE_KEYS}
    counts["tertiary"] = min(2, count)
    if total:
        remainders = {}
        for key in RATIO_KEYS:
            counts[key], remainders[key] = divmod(remaining * weights[key], total)
        extra = remaining - sum(counts[key] for key in RATIO_KEYS)
        ranked = sorted(RATIO_KEYS, key=lambda key: (-remainders[key], RATIO_KEYS.index(key)))
        for key in ranked[:extra]:
            counts[key] += 1
    return dict(profile_id=PROFILE_ID, enabled=bool(count), base_tertiary_count=2,
                count=count, weights=dict(weights), counts=counts, assignment_seed=42,
                allocation="largest_remainder_stable_type_order")


def apply_hospital_mix(hospitals, weights=None):
    """Change only virtual type/name fields; preserve coordinates, IDs and list order."""
    virtual = sorted((hospital for hospital in hospitals if hospital.get("virtual")), key=lambda h: h["id"])
    plan = hospital_plan(len(virtual), weights)
    additional = [key for key in RATIO_KEYS for _ in range(plan["counts"][key])]
    random.Random(plan["assignment_seed"]).shuffle(additional)
    categories = ["tertiary"] * min(2, len(virtual)) + additional
    indices = Counter()
    for hospital, category in zip(virtual, categories):
        indices[category] += 1
        hospital["type"] = TYPE_LABELS[category]
        hospital["name"] = f"가상 {TYPE_LABELS[category]} {indices[category]}"
        hospital["virtual_mix_category"] = category
        hospital.pop("capability", None)  # Derive capability from the new type.
    return plan
