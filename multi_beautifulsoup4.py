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

        # 💡 정규식을 사용해 \xa0, \n, \t 등 모든 형태의 공백을 완벽히 제거
        start_title = None
        for title in soup.find_all('title'):
            clean_text = re.sub(r'\s+', '', title.text)
            if '사업의내용' in clean_text and ('II' in clean_text or 'Ⅱ' in clean_text or clean_text.startswith('사업의내용')):
                start_title = title
                break

        if not start_title:
            # 실패할 경우 실제 문서에 존재하는 상위 10개 목차를 강제로 출력하여 눈으로 확인
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
            if getattr(element, 'name', None) == 'title':
                clean_element_text = re.sub(r'\s+', '', element.text)
                # III, Ⅲ, 또는 그냥 '재무에관한사항' 등 유연하게 탈출 조건 설정
                if 'III.재무' in clean_element_text or 'Ⅲ.재무' in clean_element_text or '재무에관한사항' in clean_element_text:
                    break
                
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

            if getattr(element, 'name', None) == 'table':
                if id(element) not in processed_tables:
                    if element.find('table'):
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
                        
                    html_table = str(element)
                    
                    pre_table_text = "\n".join(text_buffer)
                    current_blocks.append({
                        "type": "table",
                        "pre_text": pre_table_text,
                        "table_html": html_table
                    })
                    text_buffer = []
                    processed_tables.add(id(element))
                    
            if isinstance(element, NavigableString):
                parent_table = element.find_parent('table')
                if parent_table and id(parent_table) in processed_tables:
                    continue
                    
                clean_str = element.strip()
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