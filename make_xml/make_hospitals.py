import json

with open('DATASET/서울시 응급실 위치 정보.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    hospital_list = raw_data['DATA']

with open('DATASET/hospitals.add.xml', 'w', encoding='utf-8') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n<additional>\n')
    for h in hospital_list:
        # ID는 오직 HPID(영문+숫자)만 사용하여 오류 방지
        h_id = h['hpid']
        name = h['dutyname']
        lat = h['wgs84lat']
        lon = h['wgs84lon']
        
        # SUMO POI 생성 (한글은 name 속성에 넣어서 툴팁 등에서 확인 가능)
        f.write(f'    <poi id="{h_id}" type="hospital" name="{name}" layer="10" lon="{lon}" lat="{lat}" width="30" height="30" color="255,0,0"/>\n')
    f.write('</additional>')
print("hospitals.add.xml 생성 완료")