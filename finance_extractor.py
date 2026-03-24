import requests
import pandas as pd
import os
from dotenv import load_dotenv

def extract_dart_finance(api_key, corp_code, bsns_year):
    # DART 단일회사 전체 재무제표 API 엔드포인트
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    
    # API 요청 파라미터
    params = {
        'crtfc_key': api_key,
        'corp_code': corp_code,
        'bsns_year': bsns_year,
        'reprt_code': '11011', # 11011: 사업보고서 (1분기: 11013, 반기: 11012, 3분기: 11014)
        'fs_div': 'CFS'        # CFS: 연결재무제표, OFS: 별도재무제표
    }
    
    print(f"[{corp_code}] {bsns_year}년 연결재무제표 데이터 요청 중...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("API 요청 실패: 서버 응답 오류")
        return None
        
    data = response.json()
    
    if data.get('status') != '000':
        print(f"조회 에러: {data.get('message')}")
        return None
        
    # JSON 데이터를 Pandas DataFrame으로 변환
    df = pd.DataFrame(data['list'])
    
    # SQL DB 적재에 필요한 핵심 컬럼 추출 및 한글명 맵핑
    target_columns = {
        'bsns_year': '사업연도',
        'sj_nm': '재무제표명',   # 예: 재무상태표, 손익계산서, 자본변동표, 현금흐름표
        'account_nm': '계정명',  # 예: 자산총계, 매출액, 영업이익
        'thstrm_amount': '당기금액'
    }
    
    # 필요한 컬럼만 필터링
    df_clean = df[[col for col in target_columns.keys() if col in df.columns]].copy()
    df_clean.rename(columns=target_columns, inplace=True)
    
    # 💡 핵심 전처리: 텍스트로 들어온 금액 데이터를 SQL에서 연산 가능한 숫자형(Float)으로 강제 변환
    df_clean['당기금액'] = pd.to_numeric(df_clean['당기금액'], errors='coerce')
    
    # 결측치(NaN)가 발생한 행(금액이 기재되지 않은 빈 계정) 제거
    df_clean = df_clean.dropna(subset=['당기금액'])
    
    print("\n[추출 완료] 재무제표 데이터 샘플 (상위 5개):")
    print(df_clean.head())
    print(f"\n총 추출된 유효 계정 수: {len(df_clean)}개")
    
    return df_clean

if __name__ == "__main__":
    load_dotenv()
    API_KEY = os.getenv("DART_API_KEY")
    SAMSUNG_CORP_CODE = '00126380'
    TARGET_YEAR = '2023'
    
    if not API_KEY:
        print("에러: .env 파일에 DART_API_KEY가 설정되지 않았습니다.")
    else:
        df_finance = extract_dart_finance(API_KEY, SAMSUNG_CORP_CODE, TARGET_YEAR)
        
        if df_finance is not None:
            # 추출된 데이터를 눈으로 확인하기 위해 임시 CSV 파일로 저장
            save_path = f"finance_data_{SAMSUNG_CORP_CODE}_{TARGET_YEAR}.csv"
            df_finance.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"데이터 정제 및 CSV 임시 저장 완료: {save_path}")