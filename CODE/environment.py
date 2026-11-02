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
import xml.etree.ElementTree as ET
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

        project_root = Path(__file__).resolve().parents[1]
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
        """Use only hospital IDs present in the SUMO POI file and the dataset.

        POIs are visual markers, not drivable destinations. Network reachability
        is checked separately by TraCI in train_agent.py.
        """
        poi_file = self.cfg_file.parent / "hospitals.add.xml"
        if not poi_file.exists():
            raise FileNotFoundError(f"병원 POI 파일 없음: {poi_file}")
        poi_ids = {
            poi.get("id") for poi in ET.parse(poi_file).getroot().iter("poi")
            if poi.get("type") == "hospital" and poi.get("id")
        }
        with self.json_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw.get("DATA"), list):
            raise ValueError("병원 JSON에 DATA 배열이 없습니다")

        boundary = self.net.getBoundary()
        candidates = []
        seen = set()
        excluded = {"not_in_poi": 0, "invalid_coordinates": 0,
                    "outside_network": 0, "no_nearby_edge": 0, "duplicate": 0}
        for item in raw["DATA"]:
            hid = str(item.get("hpid", ""))
            if hid not in poi_ids:
                excluded["not_in_poi"] += 1
                continue
            if hid in seen:
                excluded["duplicate"] += 1
                continue
            try:
                lat = float(item["wgs84lat"])
                lon = float(item["wgs84lon"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("invalid geographic coordinate")
                x, y = self.net.convertLonLat2XY(lon, lat)
            except (ValueError, KeyError, TypeError, OverflowError):
                excluded["invalid_coordinates"] += 1
                continue
            if not (boundary[0] <= x <= boundary[2] and boundary[1] <= y <= boundary[3]):
                excluded["outside_network"] += 1
                continue
            nearby = self.net.getNeighboringEdges(x, y, r=1200.0)
            allowed = [edge for edge, _ in nearby if not edge.getID().startswith(":")
                       and edge.allows("emergency")]
            if not allowed:
                excluded["no_nearby_edge"] += 1
                continue
            seen.add(hid)
            candidates.append({
                "id": hid,
                "name": item.get("dutyname", hid),
                "lat": lat, "lon": lon,
                "sumo_x": None, "sumo_y": None,
                "type": item.get("dutyemclsname", "응급실운영신고기관"),
                "occupancy": random.uniform(0.20, 0.60),
            })
        candidates.sort(key=lambda h: h["id"])
        # No arbitrary hospital-count cap: final eligibility is determined by
        # directed graph connectivity and TraCI verification at SUMO startup.
        print(f"[병원 데이터] POI={len(poi_ids)}개, 등록 후보={len(candidates)}개, 제외={excluded}")
        print("[병원 데이터] POI는 표시 지점이며 실제 도달성은 TraCI에서 검증합니다.")
        if not candidates:
            raise RuntimeError("병원 POI와 JSON에 공통으로 등록된 도로 접근 후보가 없습니다")
        return candidates

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
