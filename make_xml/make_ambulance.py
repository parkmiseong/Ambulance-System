import gzip
import os
import pandas as pd
from pyproj import Transformer
import sumolib

def get_valid_nearest_edge(net, x, y):
    """
    반경을 넓혀가며 탐색하되, 네트워크에 실제로 존재하는 
    유효한 Edge ID만 확실하게 반환합니다.
    """
    radius = 100
    while True:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e[1])
            for edge, dist in sorted_edges:
                edge_id = edge.getID()
                if net.hasEdge(edge_id) and not edge_id.startswith(':'):
                    return edge_id
        radius += 300

def get_connected_route_edges(net, start_edge_id, num_edges=10):
    """
    출발 엣지로부터 연결된 여러 개의 도로 엣지 시퀀스를 생성합니다.
    """
    edge_list = [start_edge_id]
    curr_edge = net.getEdge(start_edge_id)
    for _ in range(num_edges - 1):
        to_node = curr_edge.getToNode()
        out_edges = to_node.getOutgoing()
        valid_out = [e for e in out_edges if not e.getID().startswith(':')]
        if not valid_out:
            break
        next_edge = valid_out[0]
        edge_list.append(next_edge.getID())
        curr_edge = next_edge
    return " ".join(edge_list)

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
                # 단일 엣지가 아닌 연결된 다중 엣지 시퀀스 생성 (기본 10개 엣지 연결)
                route_edges = get_connected_route_edges(net, closest_edge, num_edges=10)
                
                f.write(f'    <vehicle id="{amb_id}" type="ambulance" depart="0.00">\n')
                f.write(f'        <route edges="{route_edges}"/>\n')
                f.write('    </vehicle>\n')
                print(f"[{row['서ㆍ센터명']}] 검증 완료 -> 연결된 Edge 수: {len(route_edges.split())}개")

        f.write('</routes>')
        
    print("\n연결된 다중 엣지로 구성된 ambulance_routes.rou.xml 재생성 완료!")

if __name__ == "__main__":
    main()