# -*- coding: utf-8 -*-
"""7단계 환자별 병원 선택 진단. 표준 라이브러리만 사용."""
import argparse
import csv
import os
from collections import defaultdict, Counter


def number(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write(path, rows, fields):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('[저장]',path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--patients',required=True)
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args()
    with open(args.patients,encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError('빈 환자 CSV')
    out=args.output_dir or os.path.join(os.path.dirname(args.patients),'step7_analysis')
    os.makedirs(out,exist_ok=True)
    grouped=defaultdict(list)
    cross=defaultdict(lambda:Counter())
    for r in rows:
        strategy=r['strategy']
        grouped[strategy].append(r)
        ktas=r.get('pre_ktas','')
        hosp=r.get('hospital_type') or '미배정'
        cross[(strategy,ktas)][hosp]+=1
    summary=[]
    for strategy, group in grouped.items():
        dispatched=[r for r in group if r.get('was_dispatched')=='1']
        arrived=sum(r.get('arrived')=='1' for r in group)
        extra=[number(r.get('extra_estimated_time_sec')) for r in dispatched]
        extra=[v for v in extra if v is not None]
        waits=[number(r.get('wait_sec')) for r in group]
        waits=[v for v in waits if v is not None]
        estimated=[number(r.get('estimated_travel_time_sec')) for r in dispatched]
        estimated=[v for v in estimated if v is not None]
        summary.append({'strategy':strategy,'patients':len(group),'arrived':arrived,
                        'arrival_rate_pct':round(100*arrived/len(group),2),
                        'dispatched_with_payload':len(dispatched),
                        'mean_wait_sec':round(sum(waits)/len(waits),2) if waits else '',
                        'mean_selected_estimated_sec':round(sum(estimated)/len(estimated),2) if estimated else '',
                        'mean_extra_vs_fastest_sec':round(sum(extra)/len(extra),2) if extra else '',
                        'pct_selected_fastest':round(100*sum(v<0.001 for v in extra)/len(extra),2) if extra else '',
                        'waiting_timeout':sum(r.get('failure_reason')=='waiting_timeout' for r in group),
                        'horizon_timeout':sum(r.get('failure_reason')=='horizon_timeout' for r in group)})
    write(os.path.join(out,'strategy_diagnostics.csv'),summary,list(summary[0]))
    hospital_rows=[]
    for (strategy,ktas),cnt in sorted(cross.items()):
        for hospital,n in sorted(cnt.items()):
            hospital_rows.append({'strategy':strategy,'pre_ktas':ktas,'hospital_type':hospital,'patients':n})
    write(os.path.join(out,'pre_ktas_hospital_cross.csv'),hospital_rows,
          ['strategy','pre_ktas','hospital_type','patients'])
    # 현재 환자 CSV는 실제 적합도 함수/의료기관 수용 가능 여부를 포함하지 않으므로
    # 병원 유형만으로 임상적 적합 여부를 판정하지 않는다.
    print('[주의] 병원 적합도는 유형만으로 판정하지 않았습니다. reward.py의 적합도 기준 검증이 별도로 필요합니다.')
    if 'extra_estimated_time_sec' not in rows[0]:
        print('[안내] 기존 6단계 CSV: 최단시간 대안 비교 열이 없어 해당 통계는 비어 있습니다.')

if __name__=='__main__':
    main()
