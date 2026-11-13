"""Repeat the supplied one-hour passenger demand for long dashboard episodes."""
import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

PERIOD = 3600


def prepare_background(cfg_file, directory, until, checkpoint=lambda: None):
    """Keep original inputs and append time-sorted, uniquely named later trips.

    Repeat only the project's one-hour passenger trips, including the drain
    period. Route/type definitions remain in the original input files. This
    is a periodic simulation assumption, not time-of-day traffic observations.
    """
    cfg_file, directory = Path(cfg_file), Path(directory)
    inputs = ET.parse(cfg_file).getroot().find("input/route-files")
    originals = [(cfg_file.parent / value.strip()).resolve()
                 for value in inputs.get("value", "").split(",") if value.strip()] if inputs is not None else []
    trips, sources = [], []
    for file_index, source in enumerate(originals):
        checkpoint()
        root = ET.parse(source).getroot()
        passenger_types = {element.get("id") for element in root.findall("vType")
                           if element.get("vClass", "passenger") == "passenger"}
        selected = [element for element in root.findall("trip")
                    if element.get("type", "DEFAULT_VEHTYPE") in passenger_types | {"DEFAULT_VEHTYPE"}]
        if not selected:
            continue
        for element in selected:
            depart = float(element.get("depart", "0"))
            if not 0 <= depart < PERIOD or len(element):
                raise ValueError("반복 배경 교통은 0~3600초의 정적 승용차 trip만 지원합니다.")
            trips.append((depart, file_index, element))
        sources.append(dict(file=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    trips.sort(key=lambda item: (item[0], item[1], item[2].get("id", "")))
    metadata = dict(period_seconds=PERIOD, until_seconds=until, source_trips=len(trips),
                    repeated_trips=0, sources=sources,
                    assumption="원본 1시간 승용차 패턴 반복. 시간대별 실측 교통량을 의미하지 않음.")
    paths = [str(path) for path in originals]
    if trips and until > PERIOD:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "background_repeated.trips.xml"
        with target.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
            for cycle in range(1, math.ceil(until / PERIOD)):
                checkpoint()
                for depart, file_index, original in trips:
                    if depart + cycle * PERIOD >= until:
                        break
                    element = copy.deepcopy(original)
                    element.set("id", f"dashboard_repeat_{cycle}_{file_index}_{original.get('id')}")
                    element.set("depart", f"{depart + cycle * PERIOD:.6f}")
                    stream.write(ET.tostring(element, encoding="unicode").strip() + "\n")
                    metadata["repeated_trips"] += 1
            stream.write("</routes>\n")
        paths.append(str(target.resolve()))
    return paths, metadata
