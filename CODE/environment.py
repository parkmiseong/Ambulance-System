#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import random
import numpy as np
import sumolib
import traci

class SumoEnvironment:
    def __init__(self):
        if 'SUMO_HOME' in os.environ:
            sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        SUMO_DIR = PROJECT_ROOT / "SUMO" / "강남"
        self.net_file = SUMO_DIR / "osm.net.xml"
        self.cfg_file = SUMO_DIR / "osm.sumocfg"
        
        # 파일 경로 확인
        if not self.cfg_file.exists():
            sys.exit(f"SUMO 설정 파일을 찾을 수 없습니다: {self.cfg_file}")
        
        net_path = self.net_file.with_suffix(".xml.gz") if self.net_file.with_suffix(".xml.gz").exists() else self.net_file
        if not net_path.exists():
            sys.exit(f"SUMO 네트워크 파일을 찾을 수 없습니다: {net_path}")
            
        self.net = sumolib.net.readNet(str(net_path))
        self.valid_edges = [e.getID() for e in self.net.getEdges() if not e.getID().startswith(":")]
        
    def start_simulation(self, gui=False):
        sumo_binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
        sumo_config = [sumo_binary, "-c", str(self.cfg_file), "--scale", "0.5", "--no-warnings"]
        traci.start(sumo_config)
        print("SUMO 시뮬레이션이 시작되었습니다.")

    def close_simulation(self):
        try:
            traci.close()
            print("SUMO 시뮬레이션이 종료되었습니다.")
        except:
            pass

    def get_valid_edges(self):
        return self.valid_edges

    def get_net(self):
        return self.net