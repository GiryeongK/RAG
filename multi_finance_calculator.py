import sqlite3

def analyze_financial_trend(db_path, corp_name, account_nm, periods_str):
    periods = [p.strip() for p in periods_str.split(',')]
    print(f"      -> [다중 기간 연산 도구 실행]: {corp_name} {account_nm} ({', '.join(periods)})")
    
    # 1. 팩트 기반: DART 표준 계정명 동의어 사전 (포괄 검색 오류 원천 차단)
    account_synonyms = {
        "매출액": ["매출액", "영업수익", "수익(매출액)"],
        "영업이익": ["영업이익", "영업이익(손실)"],
        "당기순이익": ["당기순이익", "당기순이익(손실)", "연결당기순이익"],
        "자산총계": ["자산총계", "자산 총계"],
        "부채총계": ["부채총계", "부채 총계"],
        "자본총계": ["자본총계", "자본 총계"]
    }
    
    # 에이전트가 던진 계정명(예: 매출액)을 기반으로 동의어 리스트 추출 (없으면 원래 단어 배열)
    target_accounts = account_synonyms.get(account_nm, [account_nm])
    
    # SQL IN 구문을 위한 플레이스홀더 생성 (예: ?, ?, ?)
    placeholders = ','.join('?' * len(target_accounts))
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        results = []
        for period in periods:
            try:
                period_val = int(period)
                # LIKE 검색 폐기 -> IN 구문으로 정확한 일치 검색 (매출원가 혼입 방지)
                query = f"SELECT 당기금액 FROM finance_data WHERE 회사명 = ? AND 사업연도 = ? AND 계정명 IN ({placeholders})"
                cursor.execute(query, [corp_name, period_val] + target_accounts)
            except ValueError:
                query = f"SELECT 당기금액 FROM finance_data WHERE 회사명 = ? AND 사업연도 = ? AND 계정명 IN ({placeholders})"
                cursor.execute(query, [corp_name, period] + target_accounts)
                
            row = cursor.fetchone()
            if row:
                results.append({"period": period, "value": row[0]})
            else:
                results.append({"period": period, "value": None})
                
        conn.close()
        
        # (이하 금액 포매팅 및 증감률 연산 로직은 기존과 동일)
        def format_money(val):
            if val is None: return "데이터 없음"
            if isinstance(val, (float, int)) and abs(val) >= 100000000:
                jo = int(abs(val) // 1000000000000)
                eok = int((abs(val) % 1000000000000) // 100000000)
                val_str = f"{jo}조 {eok}억원" if jo > 0 else f"{eok}억원"
                return "-" + val_str if val < 0 else val_str
            return str(val)

        output_lines = [f"[{corp_name} {account_nm} 추세 분석]"]
        for i in range(len(results)):
            curr_period = results[i]["period"]
            curr_val = results[i]["value"]
            curr_str = format_money(curr_val)
            
            if i == 0 or curr_val is None or results[i-1]["value"] is None:
                output_lines.append(f"- {curr_period}년: {curr_str}")
            else:
                prev_val = results[i-1]["value"]
                if prev_val == 0:
                    growth_str = "계산 불가(이전 값 0)"
                else:
                    rate = ((curr_val - prev_val) / prev_val) * 100
                    growth_str = f"{rate:+.1f}%"
                output_lines.append(f"- {curr_period}년: {curr_str} (전기 대비 {growth_str})")
                
        return "\n".join(output_lines)

    except Exception as e:
        return f"연산 도구 내부 오류: {e}"