# environment.py  ── 개선 버전
#!/usr/bin/env python3
"""
[수정 사항]
  1. traci.start() 전에 convertGeo 호출 → sumolib 좌표 변환으로 교체
  2. 병원 데이터에 capability / equipment 필드 추가
  3. EMG 등급 코드 → capability(1/2/3) 매핑 명시
"""
import os
import sys
import json
import random
from pathlib import Path
import sumolib

# EMG 등급 코드 → 치료역량(1=지역, 2=센터, 3=권역)
EMG_LEVEL = {
    'G001':3,'G002':3,'G003':3,'G004':3,'G005':3,  # 권역응급의료센터
    'G006':2,'G007':2,'G008':2,                     # 지역응급의료센터
    'G009':1,'G010':1,                              # 응급실운영신고기관
}
EQ_BY_LEVEL = {
    3: ["CT","MRI","수술실","중환자실","혈관조영실","혈액검사기"],
    2: ["CT","수술실","중환자실","혈액검사기"],
    1: ["혈액검사기"],
}
CAP_BY_LEVEL = {3: 80, 2: 40, 1: 20}


class SumoMedicalEnvironment:
    def __init__(self, json_path='DATASET/서울시 응급실 위치 정보.json'):
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        PROJECT_ROOT  = Path(__file__).resolve().parents[1]
        SUMO_DIR      = PROJECT_ROOT / "SUMO" / "강남"
        self.net_file = SUMO_DIR / "osm.net.xml"
        self.cfg_file = SUMO_DIR / "osm.sumocfg"

        if not self.cfg_file.exists():
            sys.exit(f"SUMO 설정 파일 없음: {self.cfg_file}")

        net_path  = (self.net_file.with_suffix(".xml.gz")
                     if self.net_file.with_suffix(".xml.gz").exists()
                     else self.net_file)
        self.net  = sumolib.net.readNet(str(net_path))
        self.valid_edges = [e.getID() for e in self.net.getEdges()
                            if not e.getID().startswith(":")]

        self.hospitals = self._load_hospitals(json_path)

    # ── [수정 ①②③] 병원 로딩 ──
    def _load_hospitals(self, json_path):
        hospitals = []
        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError(json_path)

            with open(json_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            for h in raw.get('DATA', [])[:10]:
                lat = float(h['wgs84lat'])
                lon = float(h['wgs84lon'])

                # [수정 ①] sumolib으로 좌표 변환 (traci 불필요)
                try:
                    sx, sy = self.net.convertLonLat2XY(lon, lat)
                except Exception:
                    sx, sy = random.uniform(0, 5000), random.uniform(0, 5000)

                lv  = EMG_LEVEL.get(h.get('dutyemcls', 'G009'), 1)
                hospitals.append({
                    'id'        : h['hpid'],
                    'name'      : h['dutyname'],
                    'sumo_x'    : sx,
                    'sumo_y'    : sy,
                    'type'      : h.get('dutyemclsname', '일반'),
                    # [수정 ②] capability / equipment / capacity 추가
                    'capability': lv,
                    'equipment' : EQ_BY_LEVEL[lv],
                    'capacity'  : CAP_BY_LEVEL[lv],
                    'occupancy' : random.uniform(0.2, 0.6),
                    'occupied'  : 0,
                })

        except Exception as e:
            print(f"[경고] 응급실 데이터 로딩 실패: {e}  → 기본 가상 병원 사용")
            hospitals = [
                {'id':'H1','name':'강남성모',      'sumo_x':1000,'sumo_y':1000,
                 'type':'권역응급의료센터','capability':3,'capacity':80,
                 'equipment':EQ_BY_LEVEL[3],'occupancy':0.4,'occupied':0},
                {'id':'H2','name':'강남세브란스',   'sumo_x':3000,'sumo_y':2500,
                 'type':'권역응급의료센터','capability':3,'capacity':80,
                 'equipment':EQ_BY_LEVEL[3],'occupancy':0.5,'occupied':0},
                {'id':'H3','name':'서울아산(지역)', 'sumo_x':5000,'sumo_y':4000,
                 'type':'지역응급의료센터','capability':2,'capacity':40,
                 'equipment':EQ_BY_LEVEL[2],'occupancy':0.3,'occupied':0},
            ]
        return hospitals

    def get_hospitals(self):    return self.hospitals
    def get_valid_edges(self):  return self.valid_edges
