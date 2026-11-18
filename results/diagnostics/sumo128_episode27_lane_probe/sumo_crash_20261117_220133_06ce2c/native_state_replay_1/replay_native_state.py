"""Isolated short native replay; never executes the original trace as Python."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
CASE = OUT.parent
ROOT = next(p for p in OUT.parents if (p / "CODE").is_dir())
sys.path.insert(0, str(ROOT / ".runtime/sumo-1.28.0/tools"))
import traci


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(name, data):
    path = OUT / name
    assert path.parent.resolve() == OUT
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    result_file = OUT / "result.json"
    if result_file.exists():
        raise FileExistsError("This diagnostic directory has already been used.")
    trace = CASE / "sumo/episode27/traci_trace.py"
    snapshot = CASE / "native_state_at_probe_start.xml.gz"
    original_command_file = CASE / "sumo/episode27/sumo_command.json"
    command = json.loads(original_command_file.read_text(encoding="utf-8"))["command"]
    original_command = list(command)
    state = ET.fromstring(gzip.decompress(snapshot.read_bytes()))
    state_time = float(state.get("time"))
    assert state_time == 1100.0
    assert not state.findall(".//rngState")
    allowed = {
        "traci.simulationStep": traci.simulationStep,
        "traci.edge.adaptTraveltime": traci.edge.adaptTraveltime,
        "traci.vehicle.subscribe": traci.vehicle.subscribe,
        "traci.vehicle.unsubscribe": traci.vehicle.unsubscribe,
        "traci.vehicle.isStopped": traci.vehicle.isStopped,
    }
    statements = ast.parse(trace.read_text(encoding="utf-8")).body
    saves = [i for i, node in enumerate(statements)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and ast.unparse(node.value.func) == "traci.simulation.saveState"]
    assert len(saves) == 1
    calls = []
    excluded = []
    for node in statements[saves[0] + 1:]:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise ValueError("Unexpected non-call statement after the state snapshot.")
        call = node.value
        name = ast.unparse(call.func)
        if name in ("traci.close", "traci.start"):
            excluded.append(dict(line=node.lineno, name=name))
            continue
        if name not in allowed:
            raise ValueError("Non-whitelisted trace call: " + name)
        if any(isinstance(arg, ast.Starred) for arg in call.args):
            raise ValueError("Trace argument expansion is not permitted.")
        if any(kw.arg is None for kw in call.keywords):
            raise ValueError("Trace keyword expansion is not permitted.")
        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        calls.append(dict(line=node.lineno, name=name, args=args, kwargs=kwargs))
    assert sum(call["name"] == "traci.simulationStep" for call in calls) == 25

    sources = [Path(command[0]), original_command_file, trace, snapshot]
    sources.append(Path(command[command.index("-c") + 1]))
    sources.extend(Path(p) for p in command[command.index("--route-files") + 1].split(","))
    additional_index = command.index("--additional-files") + 1
    additional = command[additional_index].split(",")
    sources.extend(Path(p) for p in additional)
    # The primary config contains only these two output options, both overridden below.
    cfg = ET.parse(sources[4]).getroot()
    assert {e.tag for e in cfg.findall("output/*")} == {"tripinfo-output", "statistic-output"}
    for relative in ("SUMO/강남/osm.net.xml.gz",
                     "results/all_scenarios_normal15_yield_repeat_v6/training/checkpoint_v6_f560a1f65da40cc8.pth",
                     "results/all_scenarios_normal15_yield_repeat_sumo128_v6/training/checkpoint_v6_f560a1f65da40cc8.pth"):
        sources.append(ROOT / relative)
    before = {str(p): digest(p) for p in sources}
    for option in ("--tripinfo-output", "--statistic-output", "--log", "--error-log"):
        index = command.index(option) + 1
        command[index] = str(OUT / Path(command[index]).name)
    copied_additional = []
    for original in additional:
        source = Path(original)
        tree = ET.parse(source)
        outputs = [element for element in tree.iter() if element.get("file") is not None]
        if outputs:
            assert source.name == "output.add.xml"
            for element in outputs:
                assert element.tag == "edgeData"
                element.set("file", str(OUT / Path(element.get("file")).name))
            copied = OUT / source.name
            assert not copied.exists()
            tree.write(copied, encoding="utf-8", xml_declaration=True)
            copied_additional.append(str(copied))
        else:
            copied_additional.append(original)
    command[additional_index] = ",".join(copied_additional)
    assert "--begin" not in command and "--load-state" not in command
    command += ["--load-state", str(snapshot), "--begin", str(int(state_time))]
    write("replay_plan.json", dict(
        diagnostic_only=True, original_command=original_command, command=command,
        state_time=state_time, rng_state_saved=False, calls=calls,
        call_counts=dict(Counter(call["name"] for call in calls)), excluded=excluded,
        source_hashes_before=before,
        limitations=["The saved state contains no RNG state.",
                     "Loading a snapshot can reconstruct caches differently from the original process.",
                     "Original trace omitted most diagnostic getters; only its explicitly recorded calls are replayed.",
                     "This diagnostic never starts model training or changes source evidence."]))
    process = None
    native = (OUT / "native_process.log").open("wb", buffering=0)
    result = dict(diagnostic_only=True, new_training_started=False, model_weights_saved=False,
                  state_time=state_time, last_successful_time=None, successful_steps=0,
                  error=None, status="starting")
    try:
        traci.start(command, stdout=native)
        process = traci.getConnection()._process
        result["pid"] = process.pid
        result["last_successful_time"] = traci.simulation.getTime()
        assert result["last_successful_time"] == state_time
        for call in calls:
            result["current_trace_line"] = call["line"]
            result["current_call"] = call["name"]
            allowed[call["name"]](*call["args"], **call["kwargs"])
            if call["name"] == "traci.simulationStep":
                result["successful_steps"] += 1
                result["last_successful_time"] = traci.simulation.getTime()
        result["status"] = "completed_without_failure"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        try:
            if traci.isLoaded():
                traci.close(wait=False)
        except Exception as exc:
            result["close_error"] = str(exc)
        if process is not None:
            try:
                result["exit_code"] = process.wait(timeout=10)
            except Exception:
                result["exit_code"] = process.poll()
            if result["exit_code"] is not None:
                result["exit_code_hex"] = f"0x{result['exit_code'] & 0xffffffff:08x}"
        native.close()
        result["source_files_unchanged"] = all(digest(p) == expected for p, expected in before.items())
        result["native_access_violation_reproduced"] = result.get("exit_code_hex") == "0xc0000005"
        result["limitations"] = ["Snapshot RNG was not saved; this is not a complete restoration of the original native state.",
                                 "A successful short replay does not establish that the original crash is fixed."]
        write("result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
