"""Seed-cluster uncertainty and matched-patient comparisons, in raw reward units."""
from collections import defaultdict
import numpy as np

from reward import REWARD_COMPONENT_NAMES

TEST_SEEDS = (32001, 32002, 32003)


def validate_test_seeds(seeds, manifest, loaded_episode, *, diagnostic=False):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide unique evaluation seeds.")
    if any(type(s) is not int or s < 0 or s >= 2**31 - 100000 for s in seeds):
        raise ValueError("Evaluation seeds are out of range.")
    if diagnostic:
        return
    validation = set(manifest.get("validation_seeds", ()))
    base = manifest.get("config", {}).get("seed", 42) + 10000
    reused = [s for s in seeds if s in validation or base < s <= base + loaded_episode]
    if reused:
        raise ValueError(f"Test seeds overlap training/validation: {reused}. "
                         "Use unseen seeds, or --diagnostic-seeds for explicitly labelled diagnostics.")


def _bootstrap(totals, denominators, samples=2000, seed=91017):
    if type(samples) is not int or samples < 1:
        raise ValueError("Bootstrap samples must be positive.")
    totals = np.asarray(totals, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    if totals.ndim == 1:
        totals = totals[:, None]
    count = len(denominators)
    if count < 2 or np.any(denominators <= 0):
        return None
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(samples, count))
    values = totals[indices].sum(axis=1) / denominators[indices].sum(axis=1)[:, None]
    lo, hi = np.percentile(values, [2.5, 97.5], axis=0)
    return np.stack((lo, hi), axis=1).tolist()


def episode_uncertainty(results, samples=2000):
    """One result per distinct episode seed; no patient-independent bootstrap."""
    if not results or any("seed" not in r for r in results):
        return None
    seeds = [r["seed"] for r in results]
    if len(set(seeds)) != len(seeds):
        return dict(cluster_unit="episode_seed", clusters=len(set(seeds)), confidence=.95,
                    samples=samples, intervals=None, unavailable_reason="duplicate_episode_seeds",
                    includes_training_seed_uncertainty=False)
    names = ["reward_per_patient", "arrival_rate", "golden_rate", "avg_wait"]
    totals = [[r["reward"], 100. * r["arrived"], 100. * r["golden_success"],
               r["avg_wait"] * r["generated_patients"]] for r in results]
    intervals = _bootstrap(totals, [r["generated_patients"] for r in results], samples)
    return dict(cluster_unit="episode_seed", clusters=len(results), confidence=.95,
                samples=samples, intervals=dict(zip(names, intervals)) if intervals is not None else None,
                includes_training_seed_uncertainty=False)


def _index(rows):
    indexed = {}
    for row in rows:
        key = (int(row.get("scenario_id", 1)), float(row["demand_scale"]),
               int(row["seed"]), str(row["patient_id"]))
        if key in indexed:
            raise ValueError(f"Duplicate patient record: {key}")
        indexed[key] = row
    return indexed


def paired_patient_comparison(reference, candidate, samples=2000):
    """Candidate minus reference; bootstrap complete matched seed episodes."""
    left, right = _index(reference), _index(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical complete patient panels.")
    panels = {(k[0], k[1]) for k in left}
    if len(panels) != 1:
        raise ValueError("Compare one scenario/demand panel at a time.")
    components = all("reward_components" in r for r in [*left.values(), *right.values()])
    names = ["reward_per_patient", "arrival_rate", "golden_rate", "avg_wait"]
    if components:
        names += ["component_" + k for k in REWARD_COMPONENT_NAMES]
    clusters = defaultdict(lambda: [0, np.zeros(len(names), dtype=float)])
    for key, a in left.items():
        b = right[key]
        for field in ("ktas", "occurrence_time", "edge_id"):
            if field in a and field in b and a[field] != b[field]:
                raise ValueError(f"Mismatched patient {field}: {key}")
        delta = [b["reward"] - a["reward"],
                 100. * (int(b["arrived"]) - int(a["arrived"])),
                 100. * (int(b["golden_success"]) - int(a["golden_success"])),
                 b["wait_seconds"] - a["wait_seconds"]]
        if components:
            delta += [b["reward_components"][k] - a["reward_components"][k]
                      for k in REWARD_COMPONENT_NAMES]
        cluster = clusters[key[2]]
        cluster[0] += 1
        cluster[1] += delta
    ordered = [clusters[k] for k in sorted(clusters)]
    counts = [r[0] for r in ordered]
    totals = np.asarray([r[1] for r in ordered])
    means = totals.sum(axis=0) / sum(counts)
    intervals = _bootstrap(totals, counts, samples)
    return dict(direction="candidate_minus_reference", patient_pairs=len(left),
                seed_pairs=len(clusters), cluster_unit="episode_seed",
                delta=dict(zip(names, means.tolist())),
                ci95=dict(zip(names, intervals)) if intervals is not None else None,
                reward_components_available=components,
                includes_training_seed_uncertainty=False)
