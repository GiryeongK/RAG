import sqlite3
import pandas as pd
import os

# 1. 정제된 통합 CSV 데이터 로드
csv_path = "finance_data_multi.csv"

if not os.path.exists(csv_path):
    print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
    exit()
    
df = pd.read_csv(csv_path)

# 2. 다중 기업용 SQLite DB 연결 (파일이 없으면 새로 생성)
db_path = "multi_finance.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 3. 기존 테이블이 있다면 삭제 (중복 적재 방지)
table_name = "finance_data"
cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

# 4. DataFrame을 SQL 테이블로 밀어넣기
df.to_sql(table_name, conn, if_exists='replace', index=False)
print(f"[{db_path}] DB 생성 및 '{table_name}' 테이블 적재 완료.")

# 5. [검증] SQL 쿼리 테스트 (다중 기업 매출액 비교 조회)
# 향후 LLM SQL 에이전트가 작성하게 될 다중 조건 쿼리의 형태입니다.
test_query = f"""
SELECT 회사명, 사업연도, 계정명, 당기금액 
FROM {table_name}
WHERE 계정명 LIKE '%매출액%' 
ORDER BY 사업연도 DESC, 회사명 ASC
LIMIT 10;
"""

print("\n[SQL 테스트 쿼리 실행 결과: 다중 기업 매출액 조회]")
try:
    test_df = pd.read_sql_query(test_query, conn)
    # 가독성을 위해 '당기금액'을 조 단위로 환산하여 보여주는 컬럼 임시 추가
    test_df['당기금액(조원)'] = (test_df['당기금액'] / 1000000000000).round(2).astype(str) + '조'
    print(test_df.to_string(index=False))
except Exception as e:
    print(f"테스트 쿼리 실행 중 오류 발생: {e}")

# 6. DB 연결 종료
conn.close()