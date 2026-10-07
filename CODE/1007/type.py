import json

# 원본 파일명 설정
input_filename = "C:/Users/emaet/OneDrive/바탕 화면/STUDY/VSCODE/ambulance system/DATASET/서울시 응급실 위치 정보.json"

# JSON 파일 읽어오기
with open(input_filename, 'r', encoding='utf-8') as file:
    source_data = json.load(file)

# 원본 파일의 DESCRIPTION(메타데이터)과 DATA 추출
description_meta = source_data.get("DESCRIPTION", {})
hospital_list = source_data.get("DATA", [])

# dutyemclsname별로 데이터를 그룹화할 딕셔너리
grouped_hospitals = {}

# 병원 데이터를 순회하며 분류
for hospital in hospital_list:
    # dutyemclsname 값이 없는 경우 '미분류'로 처리
    cls_name = hospital.get("dutyemclsname", "미분류")
    
    if cls_name not in grouped_hospitals:
        grouped_hospitals[cls_name] = []
        
    grouped_hospitals[cls_name].append(hospital)

# 그룹화된 데이터를 각각 새로운 JSON 파일로 저장
for cls_name, hospitals in grouped_hospitals.items():
    output_filename = f"{cls_name}.json"
    
    # 원본과 동일한 JSON 구조(DESCRIPTION + DATA) 생성
    output_data = {
        "DESCRIPTION": description_meta,
        "DATA": hospitals
    }
    
    # 새 파일 쓰기
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        json.dump(output_data, out_file, ensure_ascii=False, indent=4)

print("병원 분류 및 개별 파일 저장이 완료되었습니다.")