# -*- coding: utf-8 -*-
"""SUMO 기반 응급실 라우팅 환경.

핵심 수정:
1) SUMO 시작 전 convertGeo()를 호출하지 않음.
2) 병원 원본 위경도를 보관하고 SUMO 시작 후 update_hospital_coordinates()에서 변환.
3) 병원 수를 고정하지 않고 지도 내부의 가까운 주행 가능 edge를 가진 병원만 등록.
4) ambulance_custom이 실제로 출발할 수 있는 edge 후보를 별도로 관리.
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
        """도로망 경계와 도로 접근성으로 실제 응급실 후보를 선정한다.

        주의: 이 검사는 정적 도로 접근성 검사이며, TraCI 주행 경로 검사는 별도다.
        """
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        boundary = self.net.getBoundary()  # SUMO XY: xmin, ymin, xmax, ymax
        candidates = []
        rejected = {"invalid_coordinate": 0, "outside_network": 0,
                    "no_nearby_road": 0, "duplicate_id": 0}
        seen_ids = set()
        for h in raw.get("DATA", []):
            try:
                lat, lon = float(h["wgs84lat"]), float(h["wgs84lon"])
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    raise ValueError("invalid latitude/longitude")
                x, y = self.net.convertLonLat2XY(lon, lat)
            except (KeyError, TypeError, ValueError, OverflowError):
                rejected["invalid_coordinate"] += 1
                continue

            if not (boundary[0] <= x <= boundary[2] and boundary[1] <= y <= boundary[3]):
                rejected["outside_network"] += 1
                continue

            # 도로와의 직선거리가 500m 이하인 후보만 허용한다.
            nearby = self.net.getNeighboringEdges(x, y, r=500.0)
            nearby = [(edge, dist) for edge, dist in nearby
                      if not edge.getID().startswith(":")
                      and any(lane.allows("emergency") or lane.allows("passenger")
                              for lane in edge.getLanes())]
            if not nearby:
                rejected["no_nearby_road"] += 1
                continue

            hospital_id = h.get("hpid") or f"H_{len(candidates)+1}"
            if hospital_id in seen_ids:
                rejected["duplicate_id"] += 1
                continue
            seen_ids.add(hospital_id)
            nearest_edge, nearest_dist = min(nearby, key=lambda pair: pair[1])
            candidates.append({
                "id": hospital_id,
                "name": h.get("dutyname", "응급실"),
                "lat": lat, "lon": lon,
                "sumo_x": None, "sumo_y": None,
                "type": h.get("dutyemclsname", "응급실운영신고기관"),
                "occupancy": random.uniform(0.20, 0.60),
                "nearest_edge": nearest_edge.getID(),
                "nearest_edge_distance_m": round(nearest_dist, 1),
            })

        # 병원 수를 강제로 10개로 맞추지 않고, 실제 도로 접근 후보만 사용한다.
        # 동일한 JSON/도로망에서 병원 인덱스가 재현되도록 ID로 정렬한다.
        candidates.sort(key=lambda h: str(h["id"]))
        print(f"[Step17] 전체 데이터={len(raw.get('DATA', []))}, "
              f"도로 접근 후보={len(candidates)}, 제외={rejected}")
        if not candidates:
            raise RuntimeError(
                "SUMO 도로망에서 접근 가능한 병원이 0개입니다. "
                "JSON 좌표와 도로망 범위, 탐색 반경을 확인하세요."
            )
        for i, h in enumerate(candidates, 1):
            print(f"[Step17] {i:02d} {h['name']} / {h['id']} / "
                  f"연결 후보 엣지 {h['nearest_edge']} ({h['nearest_edge_distance_m']}m)")
        print(f"[Step17] DQN 차원: 상태={3 + 4 * len(candidates)}, "
              f"행동={len(candidates)}")
        print("[Step17] 주의: 가까운 엣지 존재만으로 모든 환자 지점에서의 경로가 "
              "보장되지는 않습니다. 실제 배정 시 findRoute 검증을 유지하세요.")
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
