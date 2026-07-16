import pandas as pd
from pyproj import Transformer

# 데이터 로드
df_119 = pd.read_excel('DATASET/서울시 소방서,안전센터,구조대 위치정보.xlsx')

# 좌표계 변환 (EPSG:5186 -> WGS84)
transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326")
df_119['lat'], df_119['lon'] = transformer.transform(df_119['Y좌표'].values, df_119['X좌표'].values)

# XML 파일 생성 (영문 ID 사용)
with open('fire_stations.add.xml', 'w', encoding='utf-8') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
    for idx, row in df_119.iterrows():
        f.write(f'    <poi id="{row["서ㆍ센터ID"]}" type="fire_station" name="{row["서ㆍ센터명"]}" layer="10" lon="{row["lon"]}" lat="{row["lat"]}" width="30" height="30" color="0,0,255"/>\n')
    f.write('</additional>')
print("fire_stations.add.xml 생성 완료!")