import json
import os
from bs4 import BeautifulSoup, NavigableString
import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

# 💡 1. 핵심 알고리즘: HTML 표를 마크다운으로 변환 (격자 복제 로직)
def parse_html_table_to_markdown(table_tag):
    rows = table_tag.find_all('tr')
    if not rows:
        return ""

    matrix = {} # 2차원 좌표계 (엑셀 시트 역할)
    max_col = 0

    for r_idx, row in enumerate(rows):
        c_idx = 0
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            # 누군가 병합으로 내 자리를 차지하고 있다면 다음 칸으로 이동
            while (r_idx, c_idx) in matrix:
                c_idx += 1

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))

            # 줄바꿈 및 파이프(|) 기호 등 마크다운 충돌 문자 정제
            cell_text = cell.get_text(strip=True).replace('\n', ' ').replace('|', '&#124;')

            # 격자에 데이터 채우기 (병합된 만큼 복제)
            for r in range(rowspan):
                for c in range(colspan):
                    matrix[(r_idx + r, c_idx + c)] = cell_text

            c_idx += colspan
            if c_idx > max_col:
                max_col = c_idx

    # 완성된 격자를 마크다운 문자열로 조립
    md_lines = []
    max_row = len(rows)

    for r in range(max_row):
        row_data = [matrix.get((r, c), "") for c in range(max_col)]
        md_lines.append("| " + " | ".join(row_data) + " |")
        
        # 첫 줄(헤더) 아래에 마크다운 구분선 추가
        if r == 0:
            md_lines.append("|" + "|".join(["---"] * max_col) + "|")

    # 표 위아래로 줄바꿈을 넣어 일반 텍스트와 분리되게 함
    return "\n" + "\n".join(md_lines) + "\n"

# --- 메인 파이프라인 시작 ---
extract_folder = "report_xml_00126380_2023" 
meta_path = os.path.join(extract_folder, "metadata.json")

with open(meta_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

xml_file_path = os.path.join(extract_folder, metadata["source_file"])
print(f"[{metadata['corp_name']}] {metadata['report_year']}년도 파싱 시작 (표 마크다운 변환 포함)...")

with open(xml_file_path, 'r', encoding='utf-8') as f:
    xml_data = f.read()

soup = BeautifulSoup(xml_data, 'lxml')

start_title = None
for title in soup.find_all('title'):
    if 'II. 사업의 내용' in title.text:
        start_title = title
        break

if not start_title:
    print("에러: 'II. 사업의 내용'을 찾을 수 없습니다.")
else:
    parsed_sections = []
    current_section_name = "도입부"
    current_content = []
    processed_tables = set() # 이미 변환한 표를 기억하는 메모리

    for element in start_title.next_elements:
        if getattr(element, 'name', None) == 'title' and 'III. 재무' in element.text:
            break
            
        # 1. 소제목(목차) 처리
        if getattr(element, 'name', None) == 'title':
            clean_title = element.text.strip()
            if clean_title:
                if current_content:
                    parsed_sections.append({
                        "section_sub": current_section_name,
                        "text": "\n".join(current_content),
                        "metadata": metadata
                    })
                current_section_name = clean_title
                current_content = []
            continue

        # 💡 2. 표(Table) 발견 시 마크다운으로 변환하여 바구니에 담기
        if getattr(element, 'name', None) == 'table':
            # 같은 표를 두 번 읽지 않도록 방어
            if id(element) not in processed_tables:
                
                # 💡 핵심 방어 로직 추가: 내부에 table 태그가 또 있다면 레이아웃용 껍데기이므로 패스
                if element.find('table'):
                    continue
                    
                md_table = parse_html_table_to_markdown(element)

        # 3. 일반 텍스트 처리
        if isinstance(element, NavigableString):
            # 💡 핵심: 부모 태그가 <table>인데 그게 이미 위에서 마크다운으로 변환된 표라면? -> 글자만 따로 또 뽑지 말고 건너뛰기!
            parent_table = element.find_parent('table')
            if parent_table and id(parent_table) in processed_tables:
                continue
                
            clean_str = element.strip()
            if clean_str: 
                current_content.append(clean_str)

    # 마지막 남은 바구니 비우기
    if current_content:
        parsed_sections.append({
            "section_sub": current_section_name,
            "text": "\n".join(current_content),
            "metadata": metadata
        })

    save_path = os.path.join(extract_folder, "parsed_business_content.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_sections, f, ensure_ascii=False, indent=2)
        
    print(f"\n완료! 표 마크다운 변환 완료 및 총 {len(parsed_sections)}개 목차로 분할 성공.")
    print(f"저장 경로: {save_path}")