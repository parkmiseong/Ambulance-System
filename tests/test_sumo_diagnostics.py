"""Native crash evidence and preservation of the first simulation exception."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
import sumo_diagnostics as diagnostics
import simulation_setup
import test_simulation_system as fixtures


class SumoDiagnosticsTests(unittest.TestCase):
    def test_normal_launcher_adds_only_file_logging_and_keeps_native_dynamics(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "run.sumocfg"
            cfg.write_text("<configuration/>")
            env = SimpleNamespace(cfg_file=cfg, update_hospital_coordinates=lambda: None,
                                  get_hospitals=lambda: [])
            with patch.object(simulation_setup, "traci") as native, \
                 patch.object(simulation_setup.sumolib, "checkBinary", return_value="sumo"):
                simulation_setup.start_sumo(env, {"scale": .5}, 10067, output_dir=directory)
            call = native.start.call_args
            command = call.args[0]
            self.assertEqual(call.kwargs, {})
            self.assertEqual(command[command.index("--seed") + 1], "10067")
            self.assertEqual(command[command.index("--scale") + 1], "0.5")
            self.assertEqual(command[command.index("--error-log") + 1], str(Path(directory).resolve() / "sumo_errors.log"))
            self.assertNotIn("--lateral-resolution", command)
            self.assertTrue((Path(directory) / "sumo_command.json").exists())

    def test_detailed_start_captures_stderr_trace_exit_code_and_last_successful_tick(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = SimpleNamespace(pid=123, poll=lambda: 3221225477)
            connection = SimpleNamespace(_process=None, _traceFile=None)
            popen = Mock(return_value=process)
            original_module = SimpleNamespace(Popen=popen)
            namespace = dict(subprocess=original_module, connection=connection)
            exec("def start(command, **kwargs):\n"
                 "    connection._process = subprocess.Popen(command, stdout=kwargs['stdout'])\n"
                 "    kwargs['stdout'].write(b'native crash evidence')\n"
                 "    connection._traceFile = open(kwargs['traceFile'], 'w', encoding='utf-8')\n"
                 "    connection._traceFile.write('import traci\\ntraci.simulationStep()\\n')\n", namespace)
            native = SimpleNamespace(start=namespace["start"], getConnection=lambda: connection,
                                     simulation=SimpleNamespace(getTime=lambda: 1297.),
                                     isLoaded=lambda: True, close=Mock(side_effect=RuntimeError("cleanup closed")))
            env = SimpleNamespace(sumo_diagnostics=True)
            diagnostics.start_sumo_logged(env, native, ["sumo", "--seed", "10067"], root)
            self.assertIs(namespace["subprocess"], original_module)
            self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
            diagnostics.diagnostic_step(env, native)
            primary = RuntimeError("Connection closed by SUMO.")
            diagnostics.record_sumo_failure(env, native, primary, {"episode": 25})
            diagnostics.close_sumo_safely(env, native, primary)
            report = json.loads((root / "sumo_failure.json").read_text(encoding="utf-8"))
            self.assertEqual(report["exit_code_hex"], "0xc0000005")
            self.assertEqual(report["last_successful_sim_time"], 1297.)
            self.assertEqual(report["last_traci_command"], "traci.simulationStep()")
            self.assertEqual(report["native_output_tail"], "native crash evidence")
            self.assertEqual(report["stdio_capture"], "stdout_and_stderr")
            self.assertEqual(report["context"]["episode"], 25)
            self.assertIn("cleanup closed", report["cleanup_error"])
            native.close.assert_called_once_with(wait=False)
            connection._traceFile.close()

    def test_normal_runs_do_not_add_time_queries_or_flush_work_per_tick(self):
        env = SimpleNamespace(_sumo_diagnostic_state=dict(detailed=False))
        native = Mock()
        diagnostics.diagnostic_step(env, native)
        native.simulation.getTime.assert_not_called()
        native.getConnection.assert_not_called()

    def test_exception_during_first_step_is_not_masked_by_cleanup_failure(self):
        with patch.object(fixtures.FakeTraCI, "simulationStep", side_effect=RuntimeError("native step failed")), \
             patch.object(fixtures.FakeTraCI, "close", side_effect=RuntimeError("cleanup failed")):
            with self.assertRaisesRegex(RuntimeError, "native step failed"):
                fixtures.SharedEngineTests().run_case()

    def test_cleanup_failure_without_an_earlier_error_is_propagated(self):
        native = SimpleNamespace(isLoaded=lambda: True, close=Mock(side_effect=RuntimeError("cleanup failed")))
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            diagnostics.close_sumo_safely(SimpleNamespace(), native)

    def test_failure_report_io_error_does_not_replace_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            env = SimpleNamespace(_sumo_diagnostic_state=dict(output=Path(directory), command=["sumo"],
                                  detailed=True, process=None, last_successful_sim_time=1.))
            with patch.object(diagnostics, "_write_json", side_effect=OSError("disk unavailable")):
                diagnostics.record_sumo_failure(env, SimpleNamespace(), RuntimeError("original"))


if __name__ == "__main__":
    unittest.main()
