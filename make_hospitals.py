import pandas as pd

# 1. 병원 데이터 CSV 파일 읽기 (한글 인코딩 처리)
df = pd.read_csv('DATASET\seoul_emergency.csv', encoding='cp949')

# 2. SUMO용 additional XML 포맷 구성
xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
xml_content += '<additional>\n'

for idx, row in df.iterrows():
    hosp_id = row['org_id']
    name = row['org_name']
    lon = row['lng']
    lat = row['lat']
    hosp_type = row['emg_cd_nm'] # 예: 권역응급의료센터, 지역응급의료센터
    
    # 3. 병원 등급에 따른 시각적 색상 구분
    if '권역' in str(hosp_type):
        color = '255,0,0'       # 권역센터: 빨간색
    elif '지역' in str(hosp_type):
        color = '255,165,0'     # 지역센터: 주황색
    else:
        color = '0,255,0'       # 기타: 초록색

    # lon, lat 속성을 사용하면 SUMO가 자동으로 맵 내부 X,Y 좌표로 투영합니다.
    xml_content += f'    <poi id="{hosp_id}_{name}" type="hospital" color="{color}" layer="10" lon="{lon}" lat="{lat}" width="30" height="30"/>\n'

xml_content += '</additional>'

# 4. 최종 XML 파일 저장
with open('hospitals.add.xml', 'w', encoding='utf-8') as f:
    f.write(xml_content)

print("hospitals.add.xml 파일 생성 완료.")