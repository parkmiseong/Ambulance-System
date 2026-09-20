#environment.py
#!/usr/init/env python3
import os
import sys
import json
import random
from pathlib import Path
import sumolib
import traci

class SumoMedicalEnvironment:
    def __init__(self, json_path='DATASET/서울시 응급실 위치 정보.json'):
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        SUMO_DIR = PROJECT_ROOT / "SUMO" / "강남"
        self.net_file = SUMO_DIR / "osm.net.xml"
        self.cfg_file = SUMO_DIR / "osm.sumocfg"
        
        if not self.cfg_file.exists():
            sys.exit(f"SUMO 설정 파일을 찾을 수 없습니다: {self.cfg_file}")
        
        net_path = self.net_file.with_suffix(".xml.gz") if self.net_file.with_suffix(".xml.gz").exists() else self.net_file
        self.net = sumolib.net.readNet(str(net_path))
        self.valid_edges = [e.getID() for e in self.net.getEdges() if not e.getID().startswith(":")]
        
        self.hospitals = self._load_hospitals(json_path)

    def _load_hospitals(self, json_path):
        hospitals = []
        try:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"'{json_path}' 파일을 찾을 수 없습니다. 기본 가상 병원 데이터를 사용합니다.")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            for h in raw_data.get('DATA', [])[:10]:
                lat, lon = float(h['wgs84lat']), float(h['wgs84lon'])
                try:
                    sx, sy = traci.simulation.convertGeo(lon, lat, fromGeo=True)
                except:
                    sx, sy = random.uniform(0, 5000), random.uniform(0, 5000)
                    
                hospitals.append({
                    'id': h['hpid'],
                    'name': h['dutyname'],
                    'sumo_x': sx,
                    'sumo_y': sy,
                    'type': h.get('dutyemclsname', '일반'),
                    'occupancy': random.uniform(0.2, 0.6)
                })
        except Exception as e:
            print(f"[경고] 응급실 데이터 로딩 실패: {e}")
            hospitals = [
                {'id': 'H1', 'name': '강남성모', 'sumo_x': 1000, 'sumo_y': 1000, 'type': '권역', 'occupancy': 0.4},
                {'id': 'H2', 'name': '강남세브란스', 'sumo_x': 3000, 'sumo_y': 2500, 'type': '권역', 'occupancy': 0.5},
                {'id': 'H3', 'name': '서울아산', 'sumo_x': 5000, 'sumo_y': 4000, 'type': '지역', 'occupancy': 0.3},
            ]
        return hospitals

    def get_hospitals(self):
        return self.hospitals

    def get_valid_edges(self):
        return self.valid_edges