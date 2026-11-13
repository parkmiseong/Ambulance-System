"""Create a small deployment folder/ZIP without local logs, backups or environments."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SUMO_FILES = ("osm.net.xml.gz", "osm.sumocfg", "osm.passenger.trips.xml", "ambulance_routes.rou.xml",
              "output.add.xml", "hospitals.add.xml", "fire_stations.add.xml", "osm.view.xml")


def matching_manifest(model):
    tag = model.stem.removeprefix("dqn_").removesuffix("_best")
    for candidate in (model.parent.parent / "experiment.json",
                      model.parent.parent / "training" / f"experiment_{tag}.json"):
        if not candidate.is_file():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps({k:v for k,v in data.items() if k != "signature"},
            ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        if tag == f"v{data['version']}_{data['signature']}" and digest == data["signature"]:
            return candidate
    raise ValueError("The selected best model has no matching experiment manifest.")


def build_bundle(output, model=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory.")
    source_model, manifest = None, None
    if model is not None:
        source_model = Path(model).resolve()
        if not source_model.name.startswith("dqn_") or not source_model.name.endswith("_best.pth"):
            raise ValueError("Select a saved dqn_*_best.pth model.")
        manifest = matching_manifest(source_model)
    output.mkdir(parents=True)
    def copy(source, relative):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in (ROOT / "CODE").glob("*.py"):
        copy(source, Path("CODE") / source.name)
    for source in (ROOT / "web").rglob("*"):
        if source.is_file():
            copy(source, Path("web") / source.relative_to(ROOT / "web"))
    for name in ("gangnam_ems_reference.json", "서울시 응급실 위치 정보.json"):
        copy(ROOT / "DATASET" / name, Path("DATASET") / name)
    for name in SUMO_FILES:
        copy(ROOT / "SUMO" / "강남" / name, Path("SUMO") / "강남" / name)
    for name in ("Dockerfile", ".dockerignore", "compose.yaml"):
        copy(ROOT / name, name)
    for name in ("requirements.txt", "requirements-web.txt", "entrypoint.py", "Caddyfile", ".env.example"):
        copy(ROOT / "deploy" / name, Path("deploy") / name)
    copy(ROOT / "docs" / "deployment.md", "DEPLOYMENT.md")
    seed = output / "deploy" / "seed-results"
    seed.mkdir(parents=True)
    info = {"model": None, "manifest": None, "results_included": False}
    if source_model:
        digest = hashlib.sha256(source_model.read_bytes()).hexdigest()
        group = f"deployed_{digest[:12]}"
        model_relative = Path("deploy") / "seed-results" / group / "models" / source_model.name
        tag = source_model.stem.removeprefix("dqn_").removesuffix("_best")
        manifest_relative = Path("deploy") / "seed-results" / group / "training" / f"experiment_{tag}.json"
        copy(source_model, model_relative)
        copy(manifest, manifest_relative)
        info.update(model=str(model_relative.as_posix()), manifest=str(manifest_relative.as_posix()), model_sha256=digest)
    else:
        (seed / ".keep").write_text("", encoding="utf-8")
    (output / "bundle-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output.with_suffix(".zip")
    if archive.exists():
        raise ValueError("ZIP already exists; choose a new output name.")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                stream.write(source, Path(output.name) / source.relative_to(output))
    return output, archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Optional best model; otherwise start with baseline policies.")
    args = parser.parse_args()
    try:
        folder, archive = build_bundle(args.output, args.model)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Folder: {folder}\nZIP: {archive}")
