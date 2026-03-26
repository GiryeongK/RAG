import sqlite3
import pandas as pd
import os

# 1. 정제된 CSV 데이터 로드
csv_path = "finance_data_00126380_2023.csv"

if not os.path.exists(csv_path):
    print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
    exit()
    
df = pd.read_csv(csv_path)

# 2. SQLite DB 연결 (파일이 없으면 로컬에 새로 생성됨)
db_path = "samsung_finance.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 3. 기존 테이블이 있다면 삭제 (중복 방지 및 초기화 목적)
cursor.execute("DROP TABLE IF EXISTS finance_2023")

# 4. DataFrame을 SQL 테이블로 밀어넣기
# pandas의 to_sql을 사용하면 테이블 스키마 생성과 데이터 삽입(INSERT)을 한 번에 처리합니다.
df.to_sql('finance_2023', conn, if_exists='replace', index=False)
print(f"[{db_path}] DB 생성 및 'finance_2023' 테이블 적재 완료.")

# 5. [검증] SQL 쿼리 테스트 (매출액 및 영업이익 조회)
# 실제 데이터 분석 시 활용하게 될 기초적인 형태의 쿼리입니다.
test_query = """
SELECT 사업연도, 재무제표명, 계정명, 당기금액 
FROM finance_2023 
WHERE 계정명 LIKE '%매출%' OR 계정명 LIKE '%영업이익%'
LIMIT 10;
"""

print("\n[SQL 테스트 쿼리 실행 결과: 핵심 수치 확인]")
test_df = pd.read_sql_query(test_query, conn)
print(test_df)

# 6. DB 연결 종료
conn.close()