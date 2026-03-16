import json
import os
from bs4 import BeautifulSoup, NavigableString
import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

# 1. HTML 표를 마크다운으로 변환 (격자 복제 로직 - 유지)
def parse_html_table_to_markdown(table_tag):
    rows = table_tag.find_all('tr')
    if not rows:
        return ""

    matrix = {}
    max_col = 0

    for r_idx, row in enumerate(rows):
        c_idx = 0
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            while (r_idx, c_idx) in matrix:
                c_idx += 1

            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            cell_text = cell.get_text(strip=True).replace('\n', ' ').replace('|', '&#124;')

            for r in range(rowspan):
                for c in range(colspan):
                    matrix[(r_idx + r, c_idx + c)] = cell_text

            c_idx += colspan
            if c_idx > max_col:
                max_col = c_idx

    md_lines = []
    max_row = len(rows)

    for r in range(max_row):
        row_data = [matrix.get((r, c), "") for c in range(max_col)]
        md_lines.append("| " + " | ".join(row_data) + " |")
        if r == 0:
            md_lines.append("|" + "|".join(["---"] * max_col) + "|")

    return "\n" + "\n".join(md_lines) + "\n"

# --- 메인 파이프라인 ---
extract_folder = "report_xml_00126380_2023" 
meta_path = os.path.join(extract_folder, "metadata.json")

with open(meta_path, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

xml_file_path = os.path.join(extract_folder, metadata["source_file"])
print(f"[{metadata['corp_name']}] {metadata['report_year']}년도 파싱 시작 (블록 분리 구조 적용)...")

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
    current_blocks = []  # 💡 텍스트와 표를 순서대로 담을 배열
    text_buffer = []     # 💡 표가 나오기 전까지 텍스트를 임시로 모아두는 버퍼
    processed_tables = set()

    for element in start_title.next_elements:
        if getattr(element, 'name', None) == 'title' and 'III. 재무' in element.text:
            break
            
        # 1. 소제목(목차) 처리
        if getattr(element, 'name', None) == 'title':
            clean_title = element.text.strip()
            if clean_title:
                if text_buffer:
                    current_blocks.append({"type": "text", "content": "\n".join(text_buffer)})
                    text_buffer = []
                
                if current_blocks:
                    parsed_sections.append({
                        "section_main": start_title.text.strip(), # 💡 추가: 중간 목차(대분류) 저장
                        "section_sub": current_section_name,
                        "blocks": current_blocks,
                        "metadata": metadata
                    })
                
                current_section_name = clean_title
                current_blocks = []
            continue

        # 2. 표(Table) 발견 시 마크다운으로 변환하여 바구니에 담기
        if getattr(element, 'name', None) == 'table':
            if id(element) not in processed_tables:
                # 중첩 껍데기 표 무시
                if element.find('table'):
                    continue
                
                # 💡 [신규 방어 로직] 레이아웃용 껍데기 표 필터링 (조건 고도화)
                table_text = element.get_text(strip=True)
                trs = element.find_all('tr')
                
                # 표의 행이 1~2줄이면서, '단위'가 포함되어 있거나 특정 기호로 시작하는 경우 껍데기로 간주
                is_layout_table = False
                if len(trs) <= 2:
                    if '단위' in table_text or table_text.startswith(('※', '주)', '*', '[')):
                        is_layout_table = True
                
                if is_layout_table or len(table_text) < 20:
                    text_buffer.append(table_text)
                    processed_tables.add(id(element))
                    continue
                    
                # 진짜 데이터 표만 마크다운으로 변환
                md_table = parse_html_table_to_markdown(element)
                
                pre_table_text = "\n".join(text_buffer)
                current_blocks.append({
                    "type": "table",
                    "pre_text": pre_table_text,
                    "markdown": md_table
                })
                text_buffer = []
                processed_tables.add(id(element))
                
        # 3. 일반 텍스트 처리
        if isinstance(element, NavigableString):
            parent_table = element.find_parent('table')
            if parent_table and id(parent_table) in processed_tables:
                continue
                
            clean_str = element.strip()
            if clean_str: 
                text_buffer.append(clean_str)

    # 마지막 남은 찌꺼기 처리
    if text_buffer:
        current_blocks.append({"type": "text", "content": "\n".join(text_buffer)})
    if current_blocks:
        parsed_sections.append({
            "section_main": start_title.text.strip(), # 💡 추가: 중간 목차(대분류) 저장
            "section_sub": current_section_name,
            "blocks": current_blocks,
            "metadata": metadata
        })

    save_path = os.path.join(extract_folder, "parsed_business_content.json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_sections, f, ensure_ascii=False, indent=2)
        
    print(f"\n완료! 블록 분리 파싱 완료. 저장 경로: {save_path}")