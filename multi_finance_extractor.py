import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

def extract_dart_finance(api_key, corp_code, corp_name, bsns_year):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        'crtfc_key': api_key,
        'corp_code': corp_code,
        'bsns_year': bsns_year,
        'reprt_code': '11011', 
        'fs_div': 'CFS'        
    }
    
    print(f"[{corp_name}] {bsns_year}년 연결재무제표 데이터 요청 중...")
    try:
        response = requests.get(url, params=params)
        data = response.json()
    except Exception as e:
        print(f" -> API 통신 에러: {e}")
        return None
    
    if data.get('status') != '000':
        print(f" -> 조회 에러 또는 데이터 없음 (상태 코드: {data.get('status')})")
        return None
        
    df = pd.DataFrame(data['list'])
    
    target_columns = {
        'bsns_year': '사업연도',
        'sj_nm': '재무제표명',   
        'account_nm': '계정명',  
        'thstrm_amount': '당기금액'
    }
    
    df_clean = df[[col for col in target_columns.keys() if col in df.columns]].copy()
    df_clean.rename(columns=target_columns, inplace=True)
    df_clean['당기금액'] = pd.to_numeric(df_clean['당기금액'], errors='coerce')
    df_clean = df_clean.dropna(subset=['당기금액'])
    
    # 향후 SQL 데이터베이스에서 기업 및 연도를 식별하기 위한 핵심 컬럼 추가
    df_clean.insert(0, '회사명', corp_name)
    df_clean.insert(0, '회사코드', corp_code)
    
    print(f" -> 추출 성공: 유효 데이터 {len(df_clean)}행")
    return df_clean

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("DART_API_KEY")
    if not API_KEY:
        print("에러: .env 파일에 DART_API_KEY가 없습니다.")
        exit()

    TARGET_COMPANIES = {'00126380': '삼성전자', '00164779': 'SK하이닉스'}
    TARGET_YEARS = ['2023', '2024', '2025']
    
    all_finance_data = []

    for code, name in TARGET_COMPANIES.items():
        for year in TARGET_YEARS:
            df = extract_dart_finance(API_KEY, code, name, year)
            if df is not None:
                all_finance_data.append(df)
            time.sleep(2) # DART API 트래픽 제한 방지

    if all_finance_data:
        final_df = pd.concat(all_finance_data, ignore_index=True)
        save_path = "finance_data_multi.csv"
        final_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n[최종 완료] 총 {len(final_df)}행 통합 CSV 저장: {save_path}")
    else:
        print("\n추출된 재무 데이터가 없습니다.")