"""Calibrated CLI defaults, saved rates, and shared train/validation/eval demand."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'CODE'))
from dashboard_demand import PATIENTS_PER_HOUR, dashboard_scenario
from experiment_config import SimulationConfig, SCENARIOS, scenario_for_config, generate_schedule
import train_agent
from run_scenarios import aggregate_results


class CliDemandTests(unittest.TestCase):
    def test_cli_default_is_hourly_gangnam_and_one_hour(self):
        with patch.object(sys,'argv',['train_agent.py']), patch.object(train_agent,'train') as train:
            train_agent.main()
        config=train.call_args.args[10]
        self.assertEqual(config.arrival_steps,3600)
        self.assertEqual(config.demand_rates_per_hour,PATIENTS_PER_HOUR)

    def test_legacy_is_explicit_and_preserves_old_duration(self):
        with patch.object(sys,'argv',['train_agent.py','--demand-profile','legacy']), patch.object(train_agent,'train') as train:
            train_agent.main()
        config=train.call_args.args[10]
        self.assertEqual(config.arrival_steps,500)
        self.assertNotIn('demand_rates_per_hour',config.to_dict())

    def test_all_nine_rates_match_web_without_changing_legacy_scenarios(self):
        config=SimulationConfig(demand_rates_per_hour=PATIENTS_PER_HOUR)
        for scenario in SCENARIOS:
            level=(scenario['id']-1)//3
            web=dashboard_scenario(dict(demand=level,traffic=(scenario['id']-1)%3),SCENARIOS)
            actual=scenario_for_config(scenario,config)
            self.assertEqual(actual['prob'],web['prob'])
            self.assertEqual(actual['scale'],scenario['scale'])
            self.assertLess(actual['prob'],.001)
        self.assertEqual({s['prob'] for s in SCENARIOS},{.2,.3,.4})

    def test_saved_profile_roundtrip_preserves_exact_rates(self):
        config=SimulationConfig(demand_rates_per_hour=PATIENTS_PER_HOUR)
        restored=SimulationConfig(**json.loads(json.dumps(config.to_dict())))
        self.assertEqual(config,restored)
        self.assertEqual(scenario_for_config(SCENARIOS[4],restored)['patients_per_hour'],PATIENTS_PER_HOUR[1])

    def test_cli_and_web_generate_identical_patient_streams_for_same_rates_and_seed(self):
        config=SimulationConfig(arrival_steps=3600,demand_rates_per_hour=PATIENTS_PER_HOUR)
        cli=scenario_for_config(SCENARIOS[4],config)
        web=dashboard_scenario(dict(demand=1,traffic=1),SCENARIOS)
        patients=generate_schedule(cli,['a','b'],42,config)
        self.assertEqual(patients,generate_schedule(web,['a','b'],42,replace(config,demand_rates_per_hour=())))
        self.assertLess(len(patients),10)

    def test_validation_uses_hourly_profile(self):
        config=SimulationConfig(demand_rates_per_hour=PATIENTS_PER_HOUR)
        model=SimpleNamespace(training=True,eval=lambda:None,train=lambda _:None)
        agent=SimpleNamespace(model=model,epsilon=.5)
        with patch.object(train_agent,'run_episode',return_value=dict(generated_patients=1,reward=1.,arrived=1,golden_success=1)) as run:
            train_agent.validate_panel(None,[],agent,config,[1,5,9],[30001],10,'severity',300,'test')
        self.assertEqual([c.args[2]['prob']*3600 for c in run.call_args_list],list(PATIENTS_PER_HOUR))

    def test_invalid_profile_and_mixed_evaluation_are_rejected(self):
        for rates in ((1.,),(-1.,2.,3.),(True,2.,3.),(1.,float('nan'),3.)):
            with self.assertRaises(ValueError): SimulationConfig(demand_rates_per_hour=rates)
        with self.assertRaisesRegex(ValueError,'demand'):
            aggregate_results([dict(reward_version=4,demand_rates_per_hour=[]),
                               dict(reward_version=4,demand_rates_per_hour=list(PATIENTS_PER_HOUR))])
