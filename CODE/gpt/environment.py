# -*- coding: utf-8 -*-
"""SUMO 기반 응급실 라우팅 환경.

핵심 수정:
1) SUMO 시작 전 convertGeo()를 호출하지 않음.
2) 병원 원본 위경도를 보관하고 SUMO 시작 후 update_hospital_coordinates()에서 변환.
3) ambulance_custom이 실제로 출발할 수 있는 edge 후보를 별도로 관리.
"""

import json
import os
import random
import sys
from pathlib import Path

import sumolib
import traci


class SumoMedicalEnvironment:
    def __init__(self, json_path=None):
        if "SUMO_HOME" not in os.environ:
            raise RuntimeError("SUMO_HOME 환경변수가 설정되어 있지 않습니다.")

        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

        project_root = Path(__file__).resolve().parents[2]
        sumo_dir = project_root / "SUMO" / "강남"

        self.net_file = sumo_dir / "osm.net.xml"
        self.cfg_file = sumo_dir / "osm.sumocfg"

        gz_file = sumo_dir / "osm.net.xml.gz"
        if gz_file.exists():
            self.net_file = gz_file

        if not self.cfg_file.exists():
            raise FileNotFoundError(f"SUMO 설정 파일을 찾을 수 없습니다: {self.cfg_file}")
        if not self.net_file.exists():
            raise FileNotFoundError(f"SUMO 네트워크 파일을 찾을 수 없습니다: {self.net_file}")

        self.net = sumolib.net.readNet(str(self.net_file))

        self.valid_edges = [
            e.getID()
            for e in self.net.getEdges()
            if e.getID() and not e.getID().startswith(":")
        ]

        if json_path is None:
            json_path = project_root / "DATASET" / "서울시 응급실 위치 정보.json"
        else:
            json_path = Path(json_path)
            if not json_path.is_absolute():
                json_path = project_root / json_path

        self.json_path = Path(json_path)
        self.hospitals = self._load_hospitals()
        self._ambulance_edges = None
        self._coordinates_ready = False

        print(f"유효 Edge 수 : {len(self.valid_edges)}")
        print(f"병원 수 : {len(self.hospitals)}")

    def _load_hospitals(self):
        hospitals = []

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for h in raw.get("DATA", [])[:10]:
                try:
                    lat = float(h["wgs84lat"])
                    lon = float(h["wgs84lon"])
                except (KeyError, TypeError, ValueError):
                    continue

                hospitals.append({
                    "id": h.get("hpid", f"H{len(hospitals)+1}"),
                    "name": h.get("dutyname", "응급실"),
                    "lat": lat,
                    "lon": lon,
                    "sumo_x": None,
                    "sumo_y": None,
                    "type": h.get("dutyemclsname", "응급실운영신고기관"),
                    "occupancy": random.uniform(0.20, 0.60),
                })

        except Exception as e:
            print(f"[경고] 응급실 데이터 로딩 실패: {e}")

        if not hospitals:
            hospitals = [
                {
                    "id": "H1", "name": "강남성모", "lat": None, "lon": None,
                    "sumo_x": 1000.0, "sumo_y": 1000.0,
                    "type": "권역응급의료센터", "occupancy": 0.40
                },
                {
                    "id": "H2", "name": "강남세브란스", "lat": None, "lon": None,
                    "sumo_x": 3000.0, "sumo_y": 2500.0,
                    "type": "지역응급의료센터", "occupancy": 0.50
                },
                {
                    "id": "H3", "name": "서울아산", "lat": None, "lon": None,
                    "sumo_x": 5000.0, "sumo_y": 4000.0,
                    "type": "지역응급의료기관", "occupancy": 0.30
                },
            ]

        return hospitals

    def update_hospital_coordinates(self):
        """SUMO가 연결된 뒤 병원 위경도를 SUMO 좌표로 변환."""
        if self._coordinates_ready:
            return

        for h in self.hospitals:
            if h.get("lat") is None or h.get("lon") is None:
                continue
            try:
                x, y = traci.simulation.convertGeo(
                    float(h["lon"]), float(h["lat"]), fromGeo=True
                )
                h["sumo_x"] = float(x)
                h["sumo_y"] = float(y)
            except Exception:
                h["sumo_x"] = None
                h["sumo_y"] = None

        self._coordinates_ready = True

    def get_hospitals(self):
        return self.hospitals

    def get_valid_edges(self):
        return self.valid_edges

    def get_ambulance_candidate_edges(self):
        """SUMO network permission을 이용한 구급차 출발 후보 edge.

        sumolib 버전에 따라 allows/getPermissions API가 다를 수 있으므로
        가능한 경우 emergency/authority를 확인하고, 확인 불가능한 edge는
        일단 후보로 남겨 실제 TraCI findRoute에서 최종 검증한다.
        """
        if self._ambulance_edges is not None:
            return self._ambulance_edges

        candidates = []
        for edge_id in self.valid_edges:
            try:
                edge = self.net.getEdge(edge_id)

                allowed = True
                if hasattr(edge, "allows"):
                    try:
                        allowed = bool(
                            edge.allows("emergency")
                            or edge.allows("passenger")
                            or edge.allows("authority")
                        )
                    except Exception:
                        allowed = True

                if allowed:
                    candidates.append(edge_id)
            except Exception:
                continue

        self._ambulance_edges = candidates or list(self.valid_edges)
        return self._ambulance_edges
