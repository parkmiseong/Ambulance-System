
"""Pre-KTAS 1(최중증)~5(경증) 연구용 보상. 임상적 보정 전 가정값."""

from dataclasses import dataclass
import math
import os

GOLDEN_TIME_LIMITS = {
    1: 120.0,
    2: 180.0,
    3: 240.0,
    4: 300.0,
    5: 360.0,
}

HOSPITAL_TYPES = (
    "권역응급의료센터",
    "지역응급의료센터",
    "지역응급의료기관",
    "응급실운영신고기관",
)

SUITABILITY = {
    1: (1.0, 0.70, 0.05, 0.0),
    2: (1.0, 0.90, 0.20, 0.0),
    3: (0.65, 1.0, 0.85, 0.30),
    4: (0.20, 0.80, 1.0, 0.90),
    5: (0.05, 0.35, 0.90, 1.0),
}

DEFAULT_UNCERTAINTY = 0.20


@dataclass(frozen=True)
class RewardWeights:
    golden: float = 10.0
    travel: float = 4.0
    overtime: float = 8.0
    mismatch: float = 24.0
    occupancy: float = 8.0
    rejection: float = 12.0
    route_failure: float = 20.0
    overcrowding: float = 5.0


REWARD_PROFILES = {
    "baseline": RewardWeights(),
    "strong_mismatch": RewardWeights(mismatch=48.0),
    "balanced": RewardWeights(
        mismatch=36.0,
        travel=6.0,
        occupancy=12.0,
    ),
}

REWARD_PROFILE = os.getenv(
    "AMBULANCE_REWARD_PROFILE", "baseline"
).strip().lower()

if REWARD_PROFILE not in REWARD_PROFILES:
    raise ValueError(
        f"알 수 없는 보상 프로필: {REWARD_PROFILE}"
    )

WEIGHTS = REWARD_PROFILES[REWARD_PROFILE]


def normalize_ktas(value):
    """Pre-KTAS 1~5만 허용."""
    if isinstance(value, bool):
        raise ValueError(f"유효하지 않은 KTAS: {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"유효하지 않은 KTAS: {value!r}"
        ) from None

    if not math.isfinite(number) or number not in (1, 2, 3, 4, 5):
        raise ValueError(
            f"KTAS는 1~5의 정수여야 합니다: {value!r}"
        )

    return int(number)


def normalize_hospital_type(value):
    """등록된 병원 유형만 허용."""
    s = str(value or "").strip()

    for hospital_type in HOSPITAL_TYPES:
        if s == hospital_type:
            return hospital_type

    raise ValueError(
        f"알 수 없는 병원 유형: {value!r}"
    )


def pre_ktas_distribution(
    pre_ktas,
    uncertainty=DEFAULT_UNCERTAINTY,
):
    k = normalize_ktas(pre_ktas)

    u = float(uncertainty)

    if not math.isfinite(u) or not 0 <= u <= 1:
        raise ValueError(
            f"유효하지 않은 불확실성: {uncertainty}"
        )

    neighbors = [
        x for x in (k - 1, k + 1)
        if 1 <= x <= 5
    ]

    distribution = {k: 1.0 - u}

    for x in neighbors:
        distribution[x] = (
            distribution.get(x, 0.0)
            + u / len(neighbors)
        )

    return distribution


def get_hospital_suitability(
    ktas,
    hospital,
    uncertainty=DEFAULT_UNCERTAINTY,
):
    if not isinstance(hospital, dict):
        raise ValueError("hospital은 딕셔너리여야 합니다.")

    hospital_type = normalize_hospital_type(
        hospital.get("type")
    )

    idx = HOSPITAL_TYPES.index(hospital_type)

    distribution = pre_ktas_distribution(
        ktas, uncertainty
    )

    return sum(
        probability * SUITABILITY[k][idx]
        for k, probability in distribution.items()
    )


def reward_components(
    travel_time,
    ktas,
    hospital,
    golden_limit=None,
    rejection_count=0,
    occupancy=0.0,
    occupancy_at_selection=None,
    route_failed=False,
    pre_ktas=None,
    uncertainty=DEFAULT_UNCERTAINTY,
    weights=WEIGHTS,
):
    # 1. KTAS 검증
    k = normalize_ktas(
        pre_ktas if pre_ktas is not None else ktas
    )

    # 2. 골든타임 검증
    limit = float(
        GOLDEN_TIME_LIMITS[k]
        if golden_limit is None
        else golden_limit
    )

    if not math.isfinite(limit) or limit <= 0:
        raise ValueError(
            f"유효하지 않은 골든타임: {limit}"
        )

    # 3. 이동시간 검증
    t = float(travel_time)

    if not math.isfinite(t) or t < 0:
        raise ValueError(
            f"유효하지 않은 이동시간: {travel_time}"
        )

    ratio = t / limit

    # 4. 병원 점유율 검증
    occ = float(
        occupancy
        if occupancy_at_selection is None
        else occupancy_at_selection
    )

    if not math.isfinite(occ) or not 0 <= occ <= 1:
        raise ValueError(
            f"유효하지 않은 점유율: {occ}"
        )

    # 5. 병원 적합도
    fit = get_hospital_suitability(
        k, hospital, uncertainty
    )

    # 6. 보상 계산
    return {
        "golden": (
            weights.golden
            if not route_failed and ratio <= 1
            else 0.0
        ),
        "travel": (
            -weights.travel * min(ratio, 3.0)
        ),
        "overtime": (
            -weights.overtime
            * min(max(0.0, ratio - 1.0), 2.0)
        ),
        "mismatch": (
            -weights.mismatch * (1.0 - fit) ** 2
        ),
        "occupancy": (
            -weights.occupancy * occ
        ),
        "overcrowding": (
            -weights.overcrowding
            * max(0.0, occ - 0.7) / 0.3
        ),
        "rejection": (
            -weights.rejection
            * max(0, int(rejection_count))
        ),
        "route_failure": (
            -weights.route_failure
            if route_failed else 0.0
        ),
    }


def calculate_reward(
    travel_time,
    ktas=None,
    hospital=None,
    golden_limit=None,
    rejection_count=0,
    occupancy=0.0,
    occupancy_at_selection=None,
    route_failed=False,
    pre_ktas=None,
    uncertainty=DEFAULT_UNCERTAINTY,
    weights=WEIGHTS,
    **kwargs,
):
    if hospital is None:
        raise ValueError("hospital required")

    components = reward_components(
        travel_time=travel_time,
        ktas=ktas,
        hospital=hospital,
        golden_limit=golden_limit,
        rejection_count=rejection_count,
        occupancy=occupancy,
        occupancy_at_selection=occupancy_at_selection,
        route_failed=route_failed,
        pre_ktas=pre_ktas,
        uncertainty=uncertainty,
        weights=weights,
    )

    return float(sum(components.values()))
