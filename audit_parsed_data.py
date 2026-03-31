import os
import json
import re

base_dir = "사업보고서"

def audit_parsed_data():
    if not os.path.exists(base_dir):
        print(f"에러: '{base_dir}' 폴더가 없습니다.")
        return

    total_files = 0
    total_tables = 0
    total_texts = 0
    
    dirty_tables = 0
    dirty_texts = 0
    
    sample_tables = []

    # 🚀 [업데이트됨] 파서가 완벽히 제거했어야 하는 '불순물' 정규식 패턴들 (극한의 엄격성 적용)
    dirty_patterns = [
        r'style\s*=', 
        r'width\s*=', 
        r'height\s*=',
        r'bgcolor\s*=',
        r'class\s*=',     # 최상위 table 태그의 class 속성 잔존 여부
        r'border\s*=',    # 최상위 table 태그의 border 속성 잔존 여부
        r'<span', 
        r'</span',
        r'<p', 
        r'</p',
        r'<div', 
        r'</div',
        r'<br',           # br 또는 br/ 태그
        r'<colgroup',     # colgroup 태그
        r'<col',          # 빈 col 태그
        r'<a\s',          # 하이퍼링크 a 태그
        r'</a>',
        r'\\xa0',         # 특수 공백
        r'&nbsp;',
        r'[\n\r\t]'       # 표 내부에 하드코딩된 줄바꿈 및 탭 문자가 살아있는지 검사
    ]

    print("🔍 파싱 정제 결과 검사(Audit) 시작...\n" + "-"*50)

    for root, dirs, files in os.walk(base_dir):
        if "parsed_business_content.json" in files:
            total_files += 1
            file_path = os.path.join(root, "parsed_business_content.json")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for section in data:
                corp_name = section.get('metadata', {}).get('corp_name', 'Unknown')
                year = section.get('metadata', {}).get('report_year', 'Unknown')
                
                for block in section.get('blocks', []):
                    # 1. 표(Table) 검사
                    if block['type'] == 'table':
                        total_tables += 1
                        html = block['table_html']
                        
                        # 불순물 패턴이 하나라도 매칭되는지 확인
                        is_dirty = False
                        for p in dirty_patterns:
                            if re.search(p, html, re.IGNORECASE):
                                is_dirty = True
                                break
                                
                        if is_dirty:
                            dirty_tables += 1
                            
                        # 깨끗하게 정제된 표 샘플 수집 (최대 3개)
                        if len(sample_tables) < 3 and not is_dirty:
                            sample_tables.append({
                                'corp': corp_name,
                                'year': year,
                                'html': html[:800] + "\n...(중략)" if len(html) > 800 else html
                            })
                            
                    # 2. 텍스트 검사
                    elif block['type'] == 'text':
                        total_texts += 1
                        text = block['content']
                        # 텍스트 블록 내 특수 공백 및 불필요한 탭 잔존 여부 검사
                        if '\xa0' in text or '&nbsp;' in text or '\t' in text:
                            dirty_texts += 1

    print(f"✅ 검사 완료! 총 {total_files}개의 파싱된 JSON 파일을 확인했습니다.\n")
    print(f"[📊 전체 블록 통계]")
    print(f" - 전체 텍스트 블록: {total_texts:,} 개")
    print(f" - 전체 표(Table) 블록: {total_tables:,} 개")
    
    print(f"\n[🧹 정제(Cleaning) 무결성 검증]")
    print(f" - 불순물(속성 찌꺼기, br, col, \\n 등)이 남은 표: {dirty_tables} 개 (목표: 0개)")
    print(f" - 특수공백(\\xa0) 등이 남은 텍스트: {dirty_texts} 개 (목표: 0개)")
    
    if dirty_tables == 0 and dirty_texts == 0:
        print("   👉 결과: 수학적 한계치까지 완벽하게 정제되었습니다. (토큰 낭비 요소 0%)")
    else:
        print("   ⚠️ 결과: 일부 불순물이 남아있습니다. 파서 로직을 재확인하십시오.")

    print(f"\n[👀 정제된 표(Table) HTML 뼈대 구조 확인 (샘플 3개)]")
    for i, sample in enumerate(sample_tables):
        print(f"\n--- 샘플 {i+1} ({sample['corp']} {sample['year']}년) ---")
        print(sample['html'])
        print("-" * 60)

if __name__ == "__main__":
    audit_parsed_data()