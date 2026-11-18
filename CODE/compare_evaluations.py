"""Compare completed best test evaluations using matched patients and seed-cluster CIs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from evaluation_metrics import paired_patient_comparison
from emergency_yield import yield_settings
from evaluation_audit import validate_evaluation_environment


def _records(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved = [Path(p).resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same evaluation log was supplied more than once.")
    for path in resolved:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)


def compare_runs(log_path, run_ids, reference_strategy="dqn", samples=2000,
                 allow_legacy_environment=False):
    if not run_ids or len(set(run_ids)) != len(run_ids):
        raise ValueError("Provide unique run IDs.")
    starts, patients, completed = {}, defaultdict(lambda: defaultdict(list)), set()
    for row in _records(log_path):
        if row["run_id"] not in run_ids:
            continue
        data = json.loads(row["metrics_json"])
        if row["record_type"] == "run_start":
            if row["run_id"] in starts:
                raise ValueError(f"Duplicate run_start: {row['run_id']}")
            starts[row["run_id"]] = (row, data)
        elif row["record_type"] == "run_end" and row["status"] == "completed":
            completed.add(row["run_id"])
        elif row["record_type"] == "mission":
            patients[(row["run_id"], row["strategy"])][
                (int(row["scenario_id"]), float(data["demand_scale"]))].append(
                    dict(data, scenario_id=int(row["scenario_id"])))
    for run in run_ids:
        if run not in starts:
            raise ValueError(f"Missing run_start: {run}")
        if run not in completed:
            raise ValueError(f"Evaluation has not completed: {run}")
        row, data = starts[run]
        if data.get("evaluation_split") != "test":
            raise ValueError(f"Final comparisons require explicitly recorded test evaluations: {run}")
        if data.get("model_kind") != "best" or not row["model_path"].endswith("_best.pth"):
            raise ValueError(f"Only recorded best-model evaluations can be compared: {run}")
    versions = {starts[r][1].get("reward_version") for r in run_ids
                if starts[r][1].get("reward_version") is not None}
    if len(versions) > 1:
        raise ValueError("Compare models scored with the same reward version.")
    if len({starts[r][1].get("hospital_admission_mode", "precheck") for r in run_ids}) > 1:
        raise ValueError("Compare models under the same hospital admission mode.")
    if len({starts[r][1].get("congestion_reward_mode", "penalty") for r in run_ids}) > 1:
        raise ValueError("Compare models scored with the same congestion reward mode.")
    if len({tuple(starts[r][1].get("demand_rates_per_hour", ())) for r in run_ids}) > 1:
        raise ValueError("Compare models under the same hourly demand profile.")
    if len({tuple(sorted(starts[r][1].get("emergency_yield", yield_settings(False)).items()))
            for r in run_ids}) > 1:
        raise ValueError("Compare models under the same emergency driving settings.")
    environments = [starts[run][1].get("evaluation_environment") for run in run_ids]
    fingerprints = [validate_evaluation_environment(environment)
                    for environment in environments if environment is not None]
    if len(set(fingerprints)) > 1:
        raise ValueError("Compare models under the same evaluation environment; recorded operating conditions differ.")
    environment_verified = all(environment is not None for environment in environments)
    if not environment_verified and not allow_legacy_environment:
        raise ValueError("Missing recorded evaluation environment. Re-evaluate the models, or use "
                         "--allow-legacy-environment for an explicitly unverified legacy comparison.")
    reference = (run_ids[0], reference_strategy)
    if reference not in patients:
        raise ValueError("Reference strategy has no patient records.")
    reports = []
    for label, panels in patients.items():
        if label == reference:
            continue
        if panels.keys() != patients[reference].keys():
            raise ValueError(f"Scenario/demand panels differ: {label}")
        for panel, rows in panels.items():
            result = paired_patient_comparison(patients[reference][panel], rows, samples)
            reports.append(dict(reference=list(reference), candidate=list(label),
                                scenario_id=panel[0], demand_scale=panel[1], **result))
    return dict(environment_verified=environment_verified,
                environment_fingerprint=fingerprints[0] if environment_verified else None,
                runs={run: dict(model_path=starts[run][0]["model_path"],
                                model_episode=starts[run][1].get("model_episode"),
                                evaluation_split=starts[run][1].get("evaluation_split"),
                                model_snapshot=starts[run][1].get("model_snapshot"),
                                model_sha256=(starts[run][1].get("model_snapshot") or {}).get("model_sha256"))
                      for run in run_ids}, comparisons=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--log", type=Path)
    inputs.add_argument("--logs", type=Path, nargs="+", help="Logs from different experiment roots.")
    parser.add_argument("--run-ids", required=True, help="Comma-separated IDs; first is reference.")
    parser.add_argument("--reference-strategy", default="dqn")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-legacy-environment", action="store_true",
                        help="Allow old logs without recorded runtime conditions; mark the comparison unverified.")
    args = parser.parse_args()
    try:
        report = compare_runs(args.logs or args.log or Path("results/experiment_history.csv"),
                              args.run_ids.split(","), args.reference_strategy, args.bootstrap_samples,
                              allow_legacy_environment=args.allow_legacy_environment)
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
