import gzip
import os
import xml.etree.ElementTree as ET
import pandas as pd
from pyproj import Transformer
import sumolib

def get_valid_nearest_edge(net, x, y):
    """
    반경을 넓혀가며 탐색하되, 네트워크에 실제로 존재하는(hasEdge) 
    유효한 Edge ID만 확실하게 반환합니다.
    """
    radius = 100
    while True:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            # 거리 순으로 정렬 후 실제 존재하는 엣지인지 확인
            sorted_edges = sorted(edges, key=lambda e: e[1])
            for edge, dist in sorted_edges:
                edge_id = edge.getID()
                if net.hasEdge(edge_id) and not edge_id.startswith(':'):
                    return edge_id
        radius += 300

def main():
    net_filename = 'SUMO/강남/osm.net.xml'
    gz_filename = 'SUMO/강남/osm.net.xml.gz'

    if not os.path.exists(net_filename) and os.path.exists(gz_filename):
        with gzip.open(gz_filename, 'rb') as f_in:
            with open(net_filename, 'wb') as f_out:
                f_out.write(f_in.read())

    net = sumolib.net.readNet(net_filename)

    df_119 = pd.read_excel('DATASET/서울시 소방서,안전센터,구조대 위치정보.xlsx')
    transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    df_119['lon'], df_119['lat'] = transformer.transform(df_119['X좌표'].values, df_119['Y좌표'].values)

    with open('ambulance_routes.rou.xml', 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        
        f.write('    <vType id="ambulance" vClass="emergency" guiShape="emergency" maxSpeed="35.0" color="255,0,0">\n')
        f.write('        <param key="has.bluelight.device" value="true"/>\n')
        f.write('    </vType>\n\n')
        
        for idx, row in df_119.iterrows():
            lon, lat = row['lon'], row['lat']
            x, y = net.convertLonLat2XY(lon, lat)
            
            closest_edge = get_valid_nearest_edge(net, x, y)
            amb_id = f"amb_{row['서ㆍ센터ID']}"
            
            if closest_edge:
                f.write(f'    <vehicle id="{amb_id}" type="ambulance" depart="0.00">\n')
                f.write(f'        <route edges="{closest_edge}"/>\n')
                f.write(f'        <stop lane="{closest_edge}_0" endPos="10" duration="99999"/>\n')
                f.write('    </vehicle>\n')
                print(f"[{row['서ㆍ센터명']}] 검증 완료 -> Edge: {closest_edge}")

        f.write('</routes>')
        
    print("\n유효한 Edge로만 구성된 ambulance_routes.rou.xml 재생성 완료!")

if __name__ == "__main__":
    main()