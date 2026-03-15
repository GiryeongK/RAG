import requests
import zipfile
import io
import os
import json
from datetime import datetime
from dotenv import load_dotenv

def download_dart_xml(api_key, corp_code, corp_name="삼성전자"):
    today_str = datetime.today().strftime('%Y%m%d')
    target_year = "2023" # 우리가 진짜로 찾고자 하는 사업보고서 연도
    
    # 1. 공시검색 API
    search_url = 'https://opendart.fss.or.kr/api/list.json'
    search_params = {
        'crtfc_key': api_key,
        'corp_code': corp_code,
        'bgn_de': f'{target_year}0101', 
        'end_de': today_str,
        'pblntf_ty': 'A',          
        'pblntf_detail_ty': 'A001' 
    }

    print(f"1. [{corp_name}] 사업보고서 목록 조회 중...")
    response = requests.get(search_url, params=search_params)
    data = response.json()

    if data.get('status') != '000':
        print(f"조회 실패: 에러 코드 {data.get('status')} - {data.get('message')}")
        return
        
    # 💡 핵심 수정: 무조건 [0]을 가져오는 대신, 목록을 뒤져서 타겟 연도와 일치하는 보고서를 찾음
    target_report = None
    for report in data['list']:
        # DART 사업보고서 이름은 보통 "사업보고서 (2023.12)" 형태를 띱니다.
        if target_year in report['report_nm']:
            target_report = report
            break # 찾았으면 즉시 탐색 중단
            
    if not target_report:
        print(f"에러: 목록에서 {target_year}년도 사업보고서를 찾을 수 없습니다.")
        return

    rcept_no = target_report['rcept_no']
    report_nm = target_report['report_nm']
    print(f"조회 성공. 접수번호: {rcept_no} / 보고서명: {report_nm}")

    # 2. 원본 XML 파일 다운로드 및 압축 해제
    print("2. 원본 XML 파일 다운로드 및 압축 해제 중...")
    doc_url = 'https://opendart.fss.or.kr/api/document.xml'
    doc_params = {
        'crtfc_key': api_key,
        'rcept_no': rcept_no
    }
    doc_response = requests.get(doc_url, params=doc_params)
    
    extract_folder = f"report_xml_{corp_code}_{target_year}" 
    os.makedirs(extract_folder, exist_ok=True)
    
    with zipfile.ZipFile(io.BytesIO(doc_response.content)) as z:
        z.extractall(extract_folder)
        xml_files = [f for f in z.namelist() if f.endswith('.xml')]
        
    # 3. 메타데이터 생성 및 저장
    metadata = {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "report_year": target_year,
        "report_type": "사업보고서",
        "rcept_no": rcept_no,
        "source_file": xml_files[0] if xml_files else "unknown.xml"
    }
    
    meta_path = os.path.join(extract_folder, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"완료. 추출된 XML 파일: {xml_files}")
    print(f"💡 메타데이터 생성 완료: {meta_path}")
    print(f"파일 저장 경로: ./{extract_folder}/")

load_dotenv()

# 실행 영역
API_KEY = os.getenv("DART_API_KEY")
SAMSUNG_CORP_CODE = '00126380'

download_dart_xml(API_KEY, SAMSUNG_CORP_CODE, "삼성전자")