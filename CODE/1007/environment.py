# environment.py
#!/usr/bin/env python3

import json
import os
import random
import sys
from pathlib import Path


# ============================================================
# 1. SUMO 환경 설정
# ============================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("please declare environment variable 'SUMO_HOME'")

SUMO_TOOLS_PATH = Path(os.environ["SUMO_HOME"]) / "tools"

if not SUMO_TOOLS_PATH.exists():
    sys.exit(
        f"SUMO tools 폴더를 찾을 수 없습니다.\n"
        f"SUMO_HOME: {os.environ['SUMO_HOME']}\n"
        f"확인 경로: {SUMO_TOOLS_PATH}"
    )

if str(SUMO_TOOLS_PATH) not in sys.path:
    sys.path.append(str(SUMO_TOOLS_PATH))


import sumolib
import traci


class SumoMedicalEnvironment:
    """
    SUMO 기반 구급차/응급실 환경

    주요 역할
    ----------
    1. SUMO 프로젝트 경로 관리
    2. SUMO 네트워크 로딩
    3. 실제 도로 Edge 목록 생성
    4. 응급실 JSON 데이터 로딩
    5. SUMO 실행 후 병원 좌표 변환
    """

    def __init__(self, json_path=None):

        # ====================================================
        # 2. 프로젝트 루트 탐색
        # ====================================================

        self.project_root = self._find_project_root()

        self.sumo_dir = self.project_root / "SUMO" / "강남"

        self.net_file = self.sumo_dir / "osm.net.xml"
        self.cfg_file = self.sumo_dir / "osm.sumocfg"

        # ====================================================
        # 3. SUMO 파일 존재 여부 확인
        # ====================================================

        if not self.cfg_file.exists():
            sys.exit(
                f"\n[오류] SUMO 설정 파일을 찾을 수 없습니다.\n"
                f"확인 경로: {self.cfg_file}\n"
            )

        net_gz_file = Path(str(self.net_file) + ".gz")

        if not self.net_file.exists() and not net_gz_file.exists():
            sys.exit(
                f"\n[오류] SUMO 네트워크 파일을 찾을 수 없습니다.\n"
                f"확인 경로:\n"
                f"  {self.net_file}\n"
                f"  {net_gz_file}\n"
            )

        # ====================================================
        # 4. 사용할 네트워크 파일 결정
        # ====================================================

        if self.net_file.exists():
            self.actual_net_file = self.net_file
        else:
            self.actual_net_file = net_gz_file

        # ====================================================
        # 5. SUMO 네트워크 로딩
        # ====================================================

        try:
            self.net = sumolib.net.readNet(
                str(self.actual_net_file)
            )
        except Exception as exc:
            sys.exit(
                f"\n[오류] SUMO 네트워크를 읽을 수 없습니다.\n"
                f"파일: {self.actual_net_file}\n"
                f"원인: {exc}\n"
            )

        # ====================================================
        # 6. 실제 도로 Edge 목록 생성
        # ====================================================
        #
        # SUMO 내부 Edge:
        #   :junction_0
        #   :abc_1
        #
        # 이러한 Edge는 환자 위치로 직접 사용하는 것을
        # 피하기 위해 제외합니다.
        # ====================================================

        self.valid_edges = []

        for edge in self.net.getEdges():

            edge_id = edge.getID()

            if not edge_id:
                continue

            if edge_id.startswith(":"):
                continue

            self.valid_edges.append(edge_id)

        # ====================================================
        # 7. 병원 JSON 경로 설정
        # ====================================================

        if json_path is None:

            json_path = (
                self.project_root
                / "DATASET"
                / "서울시 응급실 위치 정보.json"
            )

        else:

            json_path = Path(json_path)

            if not json_path.is_absolute():
                json_path = self.project_root / json_path

        self.json_path = json_path

        # ====================================================
        # 8. 병원 정보 로딩
        # ====================================================

        self.hospitals = self._load_hospitals(
            self.json_path
        )

        # ====================================================
        # 9. 초기 상태 출력
        # ====================================================

        print("\n" + "=" * 60)
        print("[SumoMedicalEnvironment 초기화 완료]")
        print("=" * 60)

        print(f"프로젝트 루트 : {self.project_root}")
        print(f"SUMO 폴더     : {self.sumo_dir}")
        print(f"네트워크 파일 : {self.actual_net_file}")
        print(f"설정 파일     : {self.cfg_file}")
        print(f"병원 JSON     : {self.json_path}")
        print(f"유효 Edge 수  : {len(self.valid_edges)}")
        print(f"병원 수       : {len(self.hospitals)}")

        print("=" * 60 + "\n")

    # ========================================================
    # 프로젝트 루트 탐색
    # ========================================================

    def _find_project_root(self):
        """
        현재 environment.py 위치에서 상위 폴더를 탐색하면서

            SUMO/
            DATASET/

        폴더가 동시에 존재하는 위치를 프로젝트 루트로 사용합니다.

        현재 구조:

        ambulance system/
        ├── CODE/
        │   └── gpt/
        │       └── environment.py
        ├── SUMO/
        │   └── 강남/
        └── DATASET/
        """

        current_file = Path(__file__).resolve()

        for parent in current_file.parents:

            sumo_path = parent / "SUMO"
            dataset_path = parent / "DATASET"

            if sumo_path.exists() and dataset_path.exists():
                return parent

        # 현재 프로젝트 구조를 찾지 못한 경우
        # 기존 구조를 기준으로 fallback
        fallback_root = current_file.parents[2]

        print(
            "[경고] SUMO/DATASET 폴더를 자동으로 찾지 못했습니다.\n"
            f"기본 프로젝트 루트를 사용합니다: {fallback_root}"
        )

        return fallback_root

    # ========================================================
    # 병원 정보 로딩
    # ========================================================

    def _load_hospitals(self, json_path):
        """
        응급실 JSON 데이터를 읽어서 병원 목록을 생성합니다.

        병원 하나의 기본 구조:

        {
            "id": ...,
            "name": ...,
            "sumo_x": ...,
            "sumo_y": ...,
            "lat": ...,
            "lon": ...,
            "type": ...,
            "occupancy": ...
        }
        """

        hospitals = []

        try:

            if not json_path.exists():
                raise FileNotFoundError(
                    f"'{json_path}' 파일을 찾을 수 없습니다."
                )

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as file:

                raw_data = json.load(file)

            data = raw_data.get("DATA", [])

            if not isinstance(data, list):
                raise ValueError(
                    "JSON의 DATA 항목이 list 형식이 아닙니다."
                )

            # 현재 시스템은 최대 10개 병원 사용
            for h in data[:10]:

                # --------------------------------------------
                # 위도/경도 확인
                # --------------------------------------------

                try:
                    lat = float(h["wgs84lat"])
                    lon = float(h["wgs84lon"])
                except (KeyError, TypeError, ValueError):

                    print(
                        "[경고] 위도/경도가 없는 병원 데이터를 "
                        "건너뜁니다."
                    )

                    continue

                hospital_id = h.get(
                    "hpid",
                    f"H{len(hospitals) + 1}"
                )

                hospital_name = h.get(
                    "dutyname",
                    "응급실"
                )

                hospital_type = h.get(
                    "dutyemclsname",
                    "일반"
                )

                hospitals.append(
                    {
                        "id": hospital_id,
                        "name": hospital_name,

                        # SUMO 실행 후 convertGeo()로 채움
                        "sumo_x": None,
                        "sumo_y": None,

                        # 실제 병원 위치
                        "lat": lat,
                        "lon": lon,

                        # 응급실 유형
                        "type": hospital_type,

                        # 초기 점유율
                        "occupancy": random.uniform(
                            0.2,
                            0.6
                        ),
                    }
                )

        except Exception as exc:

            print(
                "\n[경고] 응급실 데이터 로딩 실패"
            )
            print(f"원인: {exc}")

            # --------------------------------------------
            # JSON 로딩 실패 시 최소 fallback
            # --------------------------------------------

            hospitals = [
                {
                    "id": "H1",
                    "name": "강남성모",
                    "sumo_x": 1000.0,
                    "sumo_y": 1000.0,
                    "lat": None,
                    "lon": None,
                    "type": "권역",
                    "occupancy": 0.4,
                },
                {
                    "id": "H2",
                    "name": "강남세브란스",
                    "sumo_x": 3000.0,
                    "sumo_y": 2500.0,
                    "lat": None,
                    "lon": None,
                    "type": "권역",
                    "occupancy": 0.5,
                },
                {
                    "id": "H3",
                    "name": "서울아산",
                    "sumo_x": 5000.0,
                    "sumo_y": 4000.0,
                    "lat": None,
                    "lon": None,
                    "type": "지역",
                    "occupancy": 0.3,
                },
            ]

        return hospitals

    # ========================================================
    # 병원 좌표 변환
    # ========================================================

    def update_hospital_coordinates(self):
        """
        SUMO가 실행된 이후 호출해야 합니다.

        실제 병원의 WGS84 위경도:
            longitude / latitude

        를 SUMO 좌표:
            x / y

        로 변환합니다.

        중요:
        traci.simulation.convertGeo()는 SUMO 연결이
        활성화된 상태에서 호출해야 합니다.
        """

        if not traci.isLoaded():
            print(
                "[경고] SUMO가 실행되지 않은 상태입니다. "
                "병원 좌표 변환을 수행하지 않습니다."
            )

            return self.hospitals

        for hospital in self.hospitals:

            lat = hospital.get("lat")
            lon = hospital.get("lon")

            # 위경도가 없는 fallback 병원
            if lat is None or lon is None:
                continue

            try:

                x, y = traci.simulation.convertGeo(
                    lon,
                    lat,
                    fromGeo=True
                )

                hospital["sumo_x"] = float(x)
                hospital["sumo_y"] = float(y)

            except Exception as exc:

                print(
                    f"[경고] 병원 좌표 변환 실패: "
                    f"{hospital.get('name', '알 수 없음')} / "
                    f"{exc}"
                )

        return self.hospitals

    # ========================================================
    # 병원 정보 반환
    # ========================================================

    def get_hospitals(self):
        """
        현재 병원 정보를 반환합니다.
        """

        return self.hospitals

    # ========================================================
    # 유효 Edge 반환
    # ========================================================

    def get_valid_edges(self):
        """
        실제 도로 Edge ID 목록을 반환합니다.

        ':'로 시작하는 SUMO 내부 Edge는 제외되어 있습니다.
        """

        return self.valid_edges
