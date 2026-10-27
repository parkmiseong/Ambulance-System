"""1단계: reward.py의 수치·경계·병원 선택 민감도 검사 (SUMO 불필요)."""
import csv
import math
from pathlib import Path
from reward import (GOLDEN_TIME_LIMITS, HOSPITAL_TYPES, REWARD_PROFILES,
                    pre_ktas_distribution, reward_components, calculate_reward)

out = Path('results/reward_validation')
out.mkdir(parents=True, exist_ok=True)
rows=[]
for profile, weights in REWARD_PROFILES.items():
    for k in range(1,6):
        for hospital_type in HOSPITAL_TYPES:
            for ratio in (0.5,1.0,1.5,3.0,5.0):
                for occ in (0.0,0.4,0.8,1.0):
                    t=GOLDEN_TIME_LIMITS[k]*ratio
                    h={'type':hospital_type}
                    parts=reward_components(t,k,h,occupancy=occ,weights=weights)
                    total=sum(parts.values())
                    assert math.isfinite(total)
                    assert abs(total-calculate_reward(t,k,h,occupancy=occ,weights=weights))<1e-9
                    rows.append({'profile':profile,'pre_ktas':k,'hospital_type':hospital_type,
                                 'time_seconds':t,'time_ratio':ratio,'occupancy':occ,
                                 **parts,'total_reward':total})
for k in range(1,6):
    d=pre_ktas_distribution(k)
    assert abs(sum(d.values())-1)<1e-12
    assert all(1<=x<=5 and 0<=p<=1 for x,p in d.items())
with (out/'reward_grid.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# 오류가 발생하기 쉬운 입력은 의도적 진단: 고치지 않고 리포트한다.
checks=[]
for label, fn in [
    ('unknown hospital type silently defaults', lambda: reward_components(180,3,{'type':'알 수 없는 병원'})),
    ('negative time silently clamped',lambda: reward_components(-20,3,{'type':HOSPITAL_TYPES[1]})),
    ('zero golden_limit uses fallback',lambda: reward_components(180,3,{'type':HOSPITAL_TYPES[1]},golden_limit=0)),
    ('NaN time',lambda: reward_components(float('nan'),3,{'type':HOSPITAL_TYPES[1]})),
    ('invalid ktas defaults to 3',lambda: reward_components(180,'unknown',{'type':HOSPITAL_TYPES[1]})),
]:
    try:
        v=fn()
        checks.append((label,'accepted',str(sum(v.values()))))
    except Exception as e:
        checks.append((label,type(e).__name__,str(e)))

print('[정상] 보상 조합 검증:',len(rows),'건 /',out/'reward_grid.csv')
print('[점검] 연구용 시간 기준:',GOLDEN_TIME_LIMITS)
for label,status,detail in checks: print('[입력 진단]',label,':',status,detail)
print('[주의] 각 병원의 이동시간/점유율이 다른 실제 환경에서는 선택 순위가 달라질 수 있음')
