# environment.py
#!/usr/bin/env python3

import json
import os
import random
import sys
from pathlib import Path

import sumolib
import traci


class SumoMedicalEnvironment:
    def __init__(self, json_path=None):
        if "SUMO_HOME" in os.environ:
            tools_path = os.path.join(os.environ["SUMO_HOME"], "tools")
            if tools_path not in sys.path:
                sys.path.append(tools_path)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # 현재 파일 위치 기준 프로젝트 구조를 사용합니다.
        # CODE/environment.py -> 프로젝트 루트는 parents[1]
        self.project_root = Path(__file__).resolve().parents[2]
        self.sumo_dir = self.project_root / "SUMO" / "강남"
        self.net_file = self.sumo_dir / "osm.net.xml"
        self.cfg_file = self.sumo_dir / "osm.sumocfg"

        if not self.cfg_file.exists():
            sys.exit(f"SUMO 설정 파일을 찾을 수 없습니다: {self.cfg_file}")
        if not self.net_file.exists() and not self.net_file.with_suffix(".xml.gz").exists():
            sys.exit(f"SUMO 네트워크 파일을 찾을 수 없습니다: {self.net_file}")

        net_path = (
            self.net_file.with_suffix(".xml.gz")
            if self.net_file.with_suffix(".xml.gz").exists()
            else self.net_file
        )
        self.net = sumolib.net.readNet(str(net_path))
        self.valid_edges = [
            edge.getID()
            for edge in self.net.getEdges()
            if not edge.getID().startswith(":")
        ]

        if json_path is None:
            json_path = self.project_root / "DATASET" / "서울시 응급실 위치 정보.json"
        else:
            json_path = Path(json_path)
            if not json_path.is_absolute():
                json_path = self.project_root / json_path

        self.json_path = json_path
        self.hospitals = self._load_hospitals(json_path)

    def _load_hospitals(self, json_path):
        hospitals = []

        try:
            if not json_path.exists():
                raise FileNotFoundError(f"'{json_path}' 파일을 찾을 수 없습니다.")

            with open(json_path, "r", encoding="utf-8") as file:
                raw_data = json.load(file)

            for h in raw_data.get("DATA", [])[:10]:
                lat = float(h["wgs84lat"])
                lon = float(h["wgs84lon"])

                hospitals.append(
                    {
                        "id": h.get("hpid", f"H{len(hospitals) + 1}"),
                        "name": h.get("dutyname", "응급실"),
                        "sumo_x": None,
                        "sumo_y": None,
                        "lat": lat,
                        "lon": lon,
                        "type": h.get("dutyemclsname", "일반"),
                        "occupancy": random.uniform(0.2, 0.6),
                    }
                )

        except Exception as exc:
            print(f"[경고] 응급실 데이터 로딩 실패: {exc}")
            hospitals = [
                {"id": "H1", "name": "강남성모", "sumo_x": 1000.0, "sumo_y": 1000.0, "type": "권역", "occupancy": 0.4},
                {"id": "H2", "name": "강남세브란스", "sumo_x": 3000.0, "sumo_y": 2500.0, "type": "권역", "occupancy": 0.5},
                {"id": "H3", "name": "서울아산", "sumo_x": 5000.0, "sumo_y": 4000.0, "type": "지역", "occupancy": 0.3},
            ]

        return hospitals

    def update_hospital_coordinates(self):
        """SUMO가 실행된 뒤 위경도를 SUMO 좌표로 변환합니다."""
        for hospital in self.hospitals:
            lat = hospital.get("lat")
            lon = hospital.get("lon")
            if lat is None or lon is None:
                continue
            try:
                x, y = traci.simulation.convertGeo(lon, lat, fromGeo=True)
                hospital["sumo_x"] = float(x)
                hospital["sumo_y"] = float(y)
            except Exception as exc:
                print(f"[경고] 병원 좌표 변환 실패: {hospital['name']} / {exc}")

        return self.hospitals

    def get_hospitals(self):
        return self.hospitals

    def get_valid_edges(self):
        return self.valid_edges
