import json
import os
import re
from bs4 import BeautifulSoup, NavigableString
import warnings
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)

base_dir = "사업보고서"

if not os.path.exists(base_dir):
    print(f"에러: '{base_dir}' 폴더를 찾을 수 없습니다.")
    exit()

print("다중 사업보고서 파싱 파이프라인 시작...\n" + "-"*50)

for root, dirs, files in os.walk(base_dir):
    if "metadata.json" in files:
        meta_path = os.path.join(root, "metadata.json")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        xml_file_path = os.path.join(root, metadata["source_file"])
        
        if not os.path.exists(xml_file_path):
            print(f"에러: [{metadata['corp_name']}] {metadata['report_year']} 원본 XML 파일이 없습니다.")
            continue

        print(f"[{metadata['corp_name']}] {metadata['report_year']}년도 파싱 시작...")

        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_data = f.read()

        soup = BeautifulSoup(xml_data, 'lxml')

        # 💡 정규식을 사용해 \xa0, \n, \t 등 모든 형태의 공백을 완벽히 제거 (질문자님 원본 로직)
        start_title = None
        for title in soup.find_all('title'):
            clean_text = re.sub(r'\s+', '', title.text)
            if '사업의내용' in clean_text and ('II' in clean_text or 'Ⅱ' in clean_text or clean_text.startswith('사업의내용')):
                start_title = title
                break

        if not start_title:
            sample_titles = [t.text.strip() for t in soup.find_all('title')][:10]
            print(f" -> 에러: '사업의 내용'을 찾을 수 없습니다.")
            print(f"    [디버깅용 실제 목차 샘플]: {sample_titles}")
            continue
            
        parsed_sections = []
        current_section_name = "도입부"
        current_blocks = []
        text_buffer = []
        processed_tables = set()

        for element in start_title.next_elements:
            # [종료 조건 검사]
            if getattr(element, 'name', None) == 'title':
                clean_element_text = re.sub(r'\s+', '', element.text)
                if 'III.재무' in clean_element_text or 'Ⅲ.재무' in clean_element_text or '재무에관한사항' in clean_element_text:
                    break
                
            # [소제목 분리 로직]
            if getattr(element, 'name', None) == 'title':
                clean_title = element.text.strip()
                if clean_title:
                    if text_buffer:
                        current_blocks.append({"type": "text", "content": "\n".join(text_buffer)})
                        text_buffer = []
                    
                    if current_blocks:
                        parsed_sections.append({
                            "section_main": start_title.text.strip(),
                            "section_sub": current_section_name,
                            "blocks": current_blocks,
                            "metadata": metadata
                        })
                    
                    current_section_name = clean_title
                    current_blocks = []
                continue

            # [표(Table) 처리 및 극한의 토큰 최적화 구간]
            if getattr(element, 'name', None) == 'table':
                if id(element) not in processed_tables:
                    if element.find('table'): # 중첩 표 무시
                        continue
                    
                    table_text = element.get_text(strip=True)
                    trs = element.find_all('tr')
                    
                    is_layout_table = False
                    if len(trs) <= 2:
                        if '단위' in table_text or table_text.startswith(('※', '주)', '*', '[')):
                            is_layout_table = True
                    
                    if is_layout_table or len(table_text) < 20:
                        text_buffer.append(table_text)
                        processed_tables.add(id(element))
                        continue
                        
                    # 🚀 [추가된 100% 무결점 토큰 다이어트 로직] 🚀
                    
                    # 1. 아예 불필요한 레이아웃 태그 삭제 (colgroup, col)
                    for cg in element.find_all(['colgroup', 'col']):
                        cg.decompose()
                        
                    # 2. br 태그는 단어가 붙지 않도록 띄어쓰기로 치환
                    for br in element.find_all('br'):
                        br.replace_with(' ')

                    # 3. 필수 속성(rowspan, colspan) 외 모든 속성 삭제 (element 자신 포함)
                    allowed_attrs = ['rowspan', 'colspan']
                    for tag in [element] + element.find_all(True):
                        attrs = list(tag.attrs.keys()) 
                        for attr in attrs:
                            if attr not in allowed_attrs:
                                del tag[attr]
                                
                    # 4. 무의미한 래퍼 태그 해제 (a 태그 추가)
                    for wrapper_tag in element.find_all(['span', 'p', 'div', 'b', 'i', 'u', 'em', 'strong', 'a']):
                        wrapper_tag.unwrap()
                        
                    # 5. HTML 문자열 변환 및 모든 공백/줄바꿈 완벽 압축
                    html_table = str(element)
                    html_table = html_table.replace('\xa0', ' ')
                    # \n, \r, \t 등 모든 내부 줄바꿈과 탭을 스페이스로 변환
                    html_table = re.sub(r'[\r\n\t]+', ' ', html_table)
                    html_table = re.sub(r'\s{2,}', ' ', html_table)
                    html_table = re.sub(r'>\s+<', '><', html_table)
                    html_table = html_table.strip()
                    # --------------------------------------
                    
                    pre_table_text = "\n".join(text_buffer)
                    current_blocks.append({
                        "type": "table",
                        "pre_text": pre_table_text,
                        "table_html": html_table
                    })
                    text_buffer = []
                    processed_tables.add(id(element))
                    
            # [일반 텍스트 처리 구간]
            if isinstance(element, NavigableString):
                parent_table = element.find_parent('table')
                if parent_table and id(parent_table) in processed_tables:
                    continue
                    
                clean_str = element.strip().replace('\xa0', ' ')
                clean_str = re.sub(r'[\r\n\t]+', ' ', clean_str) # 텍스트 내부 줄바꿈 방지 추가
                clean_str = re.sub(r'\s+', ' ', clean_str)
                
                if clean_str: 
                    text_buffer.append(clean_str)

        if text_buffer:
            current_blocks.append({"type": "text", "content": "\n".join(text_buffer)})
        if current_blocks:
            parsed_sections.append({
                "section_main": start_title.text.strip(),
                "section_sub": current_section_name,
                "blocks": current_blocks,
                "metadata": metadata
            })

        save_path = os.path.join(root, "parsed_business_content.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_sections, f, ensure_ascii=False, indent=2)
            
        print(f" -> 완료. 저장 위치: {save_path}")

print("-" * 50)
print("✨ [모든 기업/연도 파싱 완료]")