"""Pre-KTAS 1(최중증)~5(경증) 연구용 보상. 임상적 보정 전 가정값."""
from dataclasses import dataclass

GOLDEN_TIME_LIMITS = {1:120.,2:180.,3:240.,4:300.,5:360.}  # 연구용, 공식 기준 아님
HOSPITAL_TYPES = ('권역응급의료센터','지역응급의료센터','지역응급의료기관','응급실운영신고기관')
SUITABILITY = {
 1:(1.,.7,.05,0.), 2:(1.,.9,.2,0.), 3:(.9,1.,.7,.2),
 4:(.75,.9,1.,.85), 5:(.65,.8,.95,1.)}
# 불확실성은 인접 중증도로 대칭 분산시키는 실험용 모델. 실제 혼동행렬로 교체할 것.
DEFAULT_UNCERTAINTY = .20

@dataclass(frozen=True)
class RewardWeights:
    golden:float=10.
    travel:float=4.
    overtime:float=8.
    mismatch:float=24.
    occupancy:float=8.
    rejection:float=12.
    route_failure:float=20.
    overcrowding:float=5.

WEIGHTS=RewardWeights()

def normalize_ktas(value):
    try:return max(1,min(5,int(value)))
    except (ValueError,TypeError):return 3

def normalize_hospital_type(value):
    s=str(value or '')
    for t in HOSPITAL_TYPES:
        if t in s:return t
    if '권역' in s:return HOSPITAL_TYPES[0]
    if '지역' in s:return HOSPITAL_TYPES[2]
    return HOSPITAL_TYPES[3]

def pre_ktas_distribution(pre_ktas,uncertainty=DEFAULT_UNCERTAINTY):
    k=normalize_ktas(pre_ktas)
    u=max(0.,min(1.,float(uncertainty)))
    neighbors=[x for x in (k-1,k+1) if 1<=x<=5]
    d={k:1.-u}
    for x in neighbors:d[x]=d.get(x,0.)+u/len(neighbors)
    return d

def get_hospital_suitability(ktas,hospital,uncertainty=DEFAULT_UNCERTAINTY):
    idx=HOSPITAL_TYPES.index(normalize_hospital_type(hospital.get('type')))
    return sum(p*SUITABILITY[k][idx] for k,p in pre_ktas_distribution(ktas,uncertainty).items())

def reward_components(travel_time,ktas,hospital,golden_limit=None,rejection_count=0,
                      occupancy=0.,occupancy_at_selection=None,route_failed=False,
                      pre_ktas=None,uncertainty=DEFAULT_UNCERTAINTY,weights=WEIGHTS):
    k=normalize_ktas(pre_ktas if pre_ktas is not None else ktas)
    limit=float(golden_limit or GOLDEN_TIME_LIMITS[k]);limit=max(1.,limit)
    t=max(0.,float(travel_time));ratio=t/limit
    occ=float(occupancy if occupancy_at_selection is None else occupancy_at_selection)
    occ=max(0.,min(1.,occ))
    fit=get_hospital_suitability(k,hospital,uncertainty)
    # 적합도 미달의 제곱 패널티 + 병상 과밀 패널티. 대형병원 편향 완화.
    return {
      'golden':weights.golden if ratio<=1 else 0.,
      'travel':-weights.travel*min(ratio,3.),
      'overtime':-weights.overtime*min(max(0.,ratio-1.),2.),
      'mismatch':-weights.mismatch*(1.-fit)**2,
      'occupancy':-weights.occupancy*occ,
      'overcrowding':-weights.overcrowding*max(0.,occ-.7)/.3,
      'rejection':-weights.rejection*max(0,int(rejection_count)),
      'route_failure':-weights.route_failure if route_failed else 0.,
    }

def calculate_reward(travel_time,ktas=None,hospital=None,golden_limit=None,rejection_count=0,
                     occupancy=0.,occupancy_at_selection=None,route_failed=False,pre_ktas=None,
                     uncertainty=DEFAULT_UNCERTAINTY,weights=WEIGHTS,**kwargs):
    if hospital is None:raise ValueError('hospital required')
    return float(sum(reward_components(travel_time,ktas,hospital,golden_limit,rejection_count,
        occupancy,occupancy_at_selection,route_failed,pre_ktas,uncertainty,weights).values()))
