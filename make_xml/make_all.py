import gzip
import json
import os
import xml.etree.ElementTree as ET
import pandas as pd
from pyproj import Transformer
import sumolib

def get_absolutely_nearest_edge(net, x, y):
    radius = 100
    while True:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            return sorted(edges, key=lambda e: e[1])[0][0].getID()
        radius += 500

def main():
    gz_filename = 'SUMO/2026-08-13-10-48-20/osm.net.xml.gz'
    net_filename = 'osm.net.xml'

    # 1. 네트워크 파일 준비
    if not os.path.exists(net_filename) and os.path.exists(gz_filename):
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(net_filename, 'wb') as f_out:
                f_out.write(f_in.read())

    tree = ET.parse(net_filename)
    root = tree.getroot()
    location = root.find('location')
    if location is not None and not location.get('projParameter'):
        location.set('projParameter', "+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0.0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs")
        tree.write(net_filename, encoding='utf-8', xml_declaration=True)

    net = sumolib.net.readNet(net_filename)

    # 2. 소방서(구급기지) 파일 생성 (파란색: color="0,0,255") 및 구급차 라우팅 생성
    df_119 = pd.read_excel('DATASET/서울시 소방서,안전센터,구조대 위치정보.xlsx')
    transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    df_119['lon'], df_119['lat'] = transformer.transform(df_119['X좌표'].values, df_119['Y좌표'].values)

    # 소방서 POI 파일 (파란색)
    with open('fire_stations.add.xml', 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
        for idx, row in df_119.iterrows():
            f.write(f'    <poi id="{row["서ㆍ센터ID"]}" type="fire_station" name="{row["서ㆍ센터명"]}" layer="10" lon="{row["lon"]}" lat="{row["lat"]}" width="30" height="30" color="0,0,255"/>\n')
        f.write('</additional>')
    print("fire_stations.add.xml (파란색) 생성 완료!")

    # 구급차 라우팅 파일
    with open('ambulance_routes.rou.xml', 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('    <vType id="ambulance" vClass="emergency" guiShape="emergency" maxSpeed="35.0" color="255,0,0">\n')
        f.write('        <param key="has.bluelight.device" value="true"/>\n')
        f.write('    </vType>\n\n')
        
        for idx, row in df_119.iterrows():
            x, y = net.convertLonLat2XY(row['lon'], row['lat'])
            closest_edge = get_absolutely_nearest_edge(net, x, y)
            amb_id = f"amb_{row['서ㆍ센터ID']}"
            
            f.write(f'    <vehicle id="{amb_id}" type="ambulance" depart="0.00">\n')
            f.write(f'        <route edges="{closest_edge}"/>\n')
            f.write(f'        <stop lane="{closest_edge}_0" endPos="10" duration="99999"/>\n')
            f.write('    </vehicle>\n')
        f.write('</routes>')
    print("ambulance_routes.rou.xml 생성 완료!")

    # 3. 응급실(병원) POI 파일 생성 (빨간색: color="255,0,0")
    try:
        with open('DATASET/서울시 응급실 위치 정보.json', 'r', encoding='utf-8') as jf:
            er_json = json.load(jf)
            
        with open('hospitals.add.xml', 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
            for h in er_json.get('DATA', []):
                hpid = h.get('hpid')
                name = h.get('dutyname')
                lon = h.get('wgs84lon')
                lat = h.get('wgs84lat')
                if lon and lat:
                    f.write(f'    <poi id="{hpid}" type="hospital" name="{name}" layer="10" lon="{lon}" lat="{lat}" width="30" height="30" color="255,0,0"/>\n')
            f.write('</additional>')
        print("hospitals.add.xml (빨간색) 생성 완료!")
    except Exception as e:
        print("응급실 JSON 파일 처리 중 에러:", e)

if __name__ == "__main__":
    main()