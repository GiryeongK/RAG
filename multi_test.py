import requests
import zipfile
import io
import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv

def download_dart_xml(api_key, corp_code, corp_name, target_year):
    today_str = datetime.today().strftime('%Y%m%d')
    search_url = 'https://opendart.fss.or.kr/api/list.json'
    search_params = {
        'crtfc_key': api_key,
        'corp_code': corp_code,
        'bgn_de': f'{target_year}0101', 
        'end_de': today_str,
        'pblntf_ty': 'A',          
        'pblntf_detail_ty': 'A001' 
    }

    print(f"[{corp_name}] {target_year}년도 사업보고서 목록 조회 중...")
    try:
        response = requests.get(search_url, params=search_params)
        data = response.json()
    except Exception as e:
        print(f" -> API 통신 에러: {e}")
        return False

    if data.get('status') != '000':
        print(f" -> 조회 실패 또는 데이터 없음 (에러 코드: {data.get('status')})")
        return False
        
    target_report = None
    for report in data.get('list', []):
        if target_year in report['report_nm']:
            target_report = report
            break 
            
    if not target_report:
        print(f" -> 에러: {target_year}년도 사업보고서 누락 (미공시 상태일 수 있음)")
        return False

    rcept_no = target_report['rcept_no']
    print(f" -> 접수번호 확인: {rcept_no}. XML 다운로드 진행...")

    doc_url = 'https://opendart.fss.or.kr/api/document.xml'
    doc_params = {'crtfc_key': api_key, 'rcept_no': rcept_no}
    
    try:
        doc_response = requests.get(doc_url, params=doc_params)
        extract_folder = os.path.join("사업보고서", corp_name, target_year)
        os.makedirs(extract_folder, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(doc_response.content)) as z:
            z.extractall(extract_folder)
            xml_files = [f for f in z.namelist() if f.endswith('.xml')]
            
        # 💡 핵심 수정: 압축 해제된 XML 파일 중 진짜 사업보고서 본문 찾기
        target_xml = None
        for xml_file in xml_files:
            file_path = os.path.join(extract_folder, xml_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                clean_content = re.sub(r'\s+', '', content)
                # 문서 내에 사업의 내용 태그가 존재하는지 확인
                if '사업의내용</title>' in clean_content or '사업의내용</TITLE>' in clean_content:
                    target_xml = xml_file
                    break
        
        # 만약 찾지 못했다면, 파일 용량이 가장 큰 것을 본문으로 간주 (감사보고서는 용량이 작음)
        if not target_xml and xml_files:
            target_xml = max(xml_files, key=lambda x: os.path.getsize(os.path.join(extract_folder, x)))
            
        metadata = {
            "corp_code": corp_code,
            "corp_name": corp_name,
            "report_year": target_year,
            "report_type": "사업보고서",
            "rcept_no": rcept_no,
            "source_file": target_xml if target_xml else "unknown.xml"
        }
        
        meta_path = os.path.join(extract_folder, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f" -> 완료. 지정된 본문 파일: {target_xml}")
        return True
    except Exception as e:
        print(f" -> 압축 해제 및 저장 에러: {e}")
        return False

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("DART_API_KEY")
    if not API_KEY:
        print("에러: .env 파일에 DART_API_KEY가 없습니다.")
        exit()

    TARGET_COMPANIES = {'00126380': '삼성전자', '00164779': 'SK하이닉스'}
    TARGET_YEARS = ['2023', '2024', '2025']

    for code, name in TARGET_COMPANIES.items():
        for year in TARGET_YEARS:
            download_dart_xml(API_KEY, code, name, year)
            time.sleep(2)