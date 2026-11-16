"""Replay the reported Gangnam run with real SUMO, traffic and original DQN."""
import json
from pathlib import Path
import queue
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'CODE'))
import web_dashboard


class VerificationEvents(queue.Queue):
    def put_nowait(self, item):
        if item.get('kind') != 'frame':
            super().put_nowait(item)


def replay_reported_patient(reference, options, directory):
    import simulation
    from simulation_setup import load_experiment
    original = next(row for row in json.loads((reference/'patients_episode1.json').read_text(encoding='utf-8'))
                    if row['patient_id']=='p1027')
    env,hospitals,agent,config,_=load_experiment(options['model'])
    env.ambulance_edges=[original['edge_id']]
    config=replace(config,arrival_steps=1,admission_mode='arrival',congestion_reward_mode='balanced')
    patient={key:original[key] for key in ('patient_id','edge_id','ktas','pre_ktas')}
    patient['occurrence_time']=1.
    rows=[]
    # Replay a recorded input, with real SUMO routing, movement and admission.
    with patch.object(simulation,'generate_schedule',return_value={1:patient}):
        result=simulation.run_episode(env,hospitals,dict(id=1,prob=1.,scale=.5),42,'dqn',config,
            agent=agent,initial_occupancy_range=(.35,.5),emergency_yield=True,
            output_dir=directory/'replay',on_outcome=rows.append,quiet=True)
    assert rows[0]['arrived'] and rows[0]['hospital_id']==original['hospital_id']
    (directory/'replay.json').write_text(json.dumps(dict(result=result,patients=rows),ensure_ascii=False,indent=2),encoding='utf-8')


def verify():
    reference=ROOT/'results/web_dashboard/20261113_221013_bafb5e'
    options=json.loads((reference/'run.json').read_text(encoding='utf-8'))['options']
    directory=ROOT/'results/verification/route_safety'
    directory.mkdir(parents=True,exist_ok=True)
    replay_reported_patient(reference,options,directory)
    web_dashboard.OUTPUT=directory/'full_runs'
    events=VerificationEvents()
    web_dashboard.run_worker(web_dashboard.validate_options(options),str(ROOT/options['model']),
                             events,threading.Event(),threading.Event(),SimpleNamespace(value=100000.))
    messages=list(events.queue)
    errors=[m for m in messages if m.get('status')=='error']
    assert not errors,errors
    episodes=[m['result'] for m in messages if m.get('kind')=='episode_end']
    if not episodes:
        episodes=[m['result'] for m in messages if m.get('result') and 'generated_patients' in m['result']]
    assert episodes,'Missing completed episode'
    result=episodes[-1]
    assert result['generated_patients']==15,result['generated_patients']
    assert result['arrived']==15,result['failure_reasons']
    assert result['failed']==0 and result['censored']==0
    proof=dict(passed=True,reference=str(reference.relative_to(ROOT)),patients=result['generated_patients'],
               arrived=result['arrived'],failed=result['failed'],censored=result['censored'],
               route_safety=result['route_safety'],failure_reasons=result['failure_reasons'],
               model=options['model'],emergency_yield=options['emergency_yield'])
    (directory/'full-run-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__': verify()
