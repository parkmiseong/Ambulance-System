"""SUMO crash evidence without changing simulation inputs or learning state."""
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import subprocess
import traceback


def _write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _tail(path, limit=16000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, Path(path).stat().st_size - limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _state(env):
    state = getattr(env, "_sumo_diagnostic_state", None)
    return state if isinstance(state, dict) else None


def _connection(traci_module):
    try:
        return traci_module.getConnection()
    except Exception:
        return None


def _flush_trace(traci_module):
    connection = _connection(traci_module)
    try:
        trace = getattr(connection, "_traceFile", None)
        if trace is not None:
            trace.flush()
    except Exception:
        pass


class _ProcessModuleProxy:
    """Replace only traci.start's module reference, never global Popen."""
    def __init__(self, original, state):
        self.original, self.state = original, state

    def __getattr__(self, name):
        return getattr(self.original, name)

    def Popen(self, *args, **kwargs):
        kwargs["stderr"] = subprocess.STDOUT
        process = self.original.Popen(*args, **kwargs)
        self.state["process"] = process
        return process


@contextmanager
def _capture_process(start, state):
    namespace = getattr(start, "__globals__", None)
    original = namespace.get("subprocess") if isinstance(namespace, dict) else None
    if original is None:
        state["stdio_capture"] = "stdout_only"
        yield
        return
    state["stdio_capture"] = "stdout_and_stderr"
    namespace["subprocess"] = _ProcessModuleProxy(original, state)
    try:
        yield
    finally:
        namespace["subprocess"] = original


def start_sumo_logged(env, traci_module, command, output):
    """Always keep SUMO logs; opt-in adds replayable commands and native stderr."""
    output = Path(output).resolve()
    detailed = getattr(env, "sumo_diagnostics", False) is True
    state = dict(output=output, detailed=detailed, process=None,
                 last_successful_sim_time=None, stdio=None, failure=None)
    env._sumo_diagnostic_state = state
    command = list(command) + ["--log", str(output / "sumo.log"),
                               "--error-log", str(output / "sumo_errors.log")]
    state["command"] = command
    _write_json(output / "sumo_command.json", dict(command=command, diagnostics=detailed))
    kwargs = {}
    if detailed:
        state["stdio"] = (output / "sumo_process.log").open("wb", buffering=0)
        kwargs.update(traceFile=str(output / "traci_trace.py"), traceGetters=False, stdout=state["stdio"])
    try:
        if detailed:
            with _capture_process(traci_module.start, state):
                result = traci_module.start(command, **kwargs)
        else:
            result = traci_module.start(command)
        if state["process"] is None:
            state["process"] = getattr(_connection(traci_module), "_process", None)
        if detailed:
            _flush_trace(traci_module)
        return result
    except BaseException as exc:
        record_sumo_failure(env, traci_module, exc, dict(phase="start_sumo"))
        raise


def diagnostic_step(env, traci_module):
    """Only detailed runs query the successful tick and flush buffered commands."""
    state = _state(env)
    if not state or not state["detailed"]:
        return
    state["last_successful_sim_time"] = float(traci_module.simulation.getTime())
    _flush_trace(traci_module)


def _exit_details(state):
    process = state.get("process")
    code = pid = None
    try:
        pid = process.pid
        if not isinstance(pid, int):
            pid = None
        code = process.poll()
        if code is None:
            code = process.wait(timeout=1.)
        if not isinstance(code, int):
            code = None
    except Exception:
        pass
    return dict(pid=pid, exit_code=code,
                exit_code_hex=f"0x{code & 0xffffffff:08x}" if code is not None else None)


def record_sumo_failure(env, traci_module, exc, context=None):
    """Best-effort evidence collection must never replace the primary error."""
    state = _state(env)
    if not state:
        return
    try:
        _flush_trace(traci_module)
        output = state["output"]
        trace_tail = _tail(output / "traci_trace.py")
        commands = [line for line in trace_tail.splitlines() if line.startswith("traci.")]
        report = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                      error_type=type(exc).__name__, error=str(exc),
                      traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                      command=state["command"], diagnostics=state["detailed"],
                      last_successful_sim_time=state["last_successful_sim_time"],
                      last_traci_command=commands[-1] if commands else None,
                      trace_tail=trace_tail, stdio_capture=state.get("stdio_capture"),
                      native_output_tail=_tail(output / "sumo_process.log"),
                      sumo_error_tail=_tail(output / "sumo_errors.log"),
                      context=context or {}, **_exit_details(state))
        state["failure"] = report
        _write_json(output / "sumo_failure.json", report)
    except Exception:
        pass


def close_sumo_safely(env, traci_module, primary_exception=None):
    """Keep a simulation crash primary even if TraCI cleanup also fails."""
    state = _state(env)
    cleanup_error = None
    try:
        if traci_module.isLoaded():
            if primary_exception is None:
                traci_module.close()
            else:
                # Native TraCI supports wait=False; lightweight adapters may
                # only expose close(). Inspect first rather than retrying a
                # real TypeError raised from inside connection cleanup.
                try:
                    parameters = inspect.signature(traci_module.close).parameters.values()
                    accepts_wait = any(p.name == "wait" or p.kind == inspect.Parameter.VAR_KEYWORD
                                       for p in parameters)
                except (TypeError, ValueError):
                    accepts_wait = True
                if accepts_wait:
                    traci_module.close(wait=False)
                else:
                    traci_module.close()
    except Exception as exc:
        cleanup_error = exc
        if primary_exception is None:
            record_sumo_failure(env, traci_module, exc, dict(phase="close_sumo"))
            raise
    finally:
        if state:
            stream = state.get("stdio")
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    if primary_exception is None:
                        raise
                    cleanup_error = cleanup_error or exc
                state["stdio"] = None
            if state.get("failure") is not None:
                report = state["failure"]
                report.update(_exit_details(state))
                if cleanup_error is not None:
                    report["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                try:
                    _write_json(state["output"] / "sumo_failure.json", report)
                except Exception:
                    pass
