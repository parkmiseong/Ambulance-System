# -*- coding: utf-8 -*-
"""실행: python analyze_step6.py --detail results/evaluation/comparison_detail_*.csv
환자별 로그가 있다면 --patients results/evaluation/comparison_patients_*.csv 추가.
"""
import argparse, csv, json
from collections import defaultdict, Counter
from pathlib import Path

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def num(x):
    try: return float(x)
    except (ValueError,TypeError): return 0.0
def save(path,rows):
    if not rows: return
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('저장:',path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--detail',required=True);ap.add_argument('--patients');ap.add_argument('--out',default='results/evaluation/step6_analysis');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True); rows=read(a.detail)
    by=defaultdict(list)
    for r in rows: by[r['strategy']].append(r)
    ktas=[];failure=[];types=[]
    for strategy,group in by.items():
        for k in range(1,6):
            arrived=sum(num(r[f'ktas{k}_arrived']) for r in group)
            failed=sum(num(r[f'ktas{k}_failed']) for r in group)
            ktas.append(dict(strategy=strategy,pre_ktas=k,arrived=int(arrived),failed=int(failed),arrival_rate_pct=round(100*arrived/(arrived+failed),2) if arrived+failed else ''))
        reasons=Counter();hosp=Counter()
        for r in group:
            reasons.update(json.loads(r['failure_reasons_json'] or '{}'))
            hosp.update(json.loads(r['hospital_types_json'] or '{}'))
        for reason,count in reasons.items():failure.append(dict(strategy=strategy,reason=reason,count=count))
        for typ,count in hosp.items():types.append(dict(strategy=strategy,hospital_type=typ,assigned_missions=count))
    save(out/'pre_ktas_arrival.csv',ktas);save(out/'failure_reasons.csv',failure);save(out/'hospital_type_totals.csv',types)
    if a.patients:
        patient_rows=read(a.patients);cross=defaultdict(lambda:[0,0]); detail=defaultdict(lambda:[0,0,0.0,0.0])
        for r in patient_rows:
            typ=r.get('hospital_type') or '(미배정)';k=r.get('pre_ktas') or '(미상)';key=(r['strategy'],k,typ)
            cross[key][0]+=1;cross[key][1]+=int(num(r['arrived']))
            d=detail[r['strategy']];d[0]+=1;d[1]+=int(num(r['arrived']));d[2]+=num(r['wait_sec']);d[3]+=num(r['elapsed_sec'])
        save(out/'pre_ktas_hospital_cross.csv',[dict(strategy=s,pre_ktas=k,hospital_type=t,patients=v[0],arrived=v[1],arrival_rate_pct=round(100*v[1]/v[0],2)) for (s,k,t),v in sorted(cross.items())])
        save(out/'patient_level_summary.csv',[dict(strategy=s,patients=v[0],arrived=v[1],arrival_rate_pct=round(100*v[1]/v[0],2),mean_wait_sec=round(v[2]/v[0],1),mean_elapsed_sec=round(v[3]/v[0],1)) for s,v in sorted(detail.items())])
    else: print('참고: 기존 detail CSV만으로는 Pre-KTAS × 병원유형 교차분석 불가능. 다음 실험에서 --patients 추가.')
if __name__=='__main__':main()
