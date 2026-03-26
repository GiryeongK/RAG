import json
import os
import re
import time
from tqdm import tqdm
from mlx_vlm import load, generate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 다중 파싱 데이터 일괄 로드
base_dir = "사업보고서"
all_parsed_sections = []

print(f"1. '{base_dir}' 폴더 내의 모든 파싱 데이터를 불러옵니다...")
for root, dirs, files in os.walk(base_dir):
    if "parsed_business_content.json" in files:
        file_path = os.path.join(root, "parsed_business_content.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            parsed_sections = json.load(f)
            # 데이터를 하나의 거대한 리스트로 병합
            all_parsed_sections.extend(parsed_sections)

print(f" -> 총 {len(all_parsed_sections)}개의 섹션 데이터가 로드되었습니다.")

# 2. MLX 모델 로드
model_id = "mlx-community/Qwen3.5-9B-8bit"
print(f"\n2. MLX 모델 로드 중... ({model_id})")
model, processor = load(model_id)

# ==========================================
# 🚀 [Phase 1] 표 메타데이터 룰베이스 추출 & 소제목 요약
# ==========================================
print("\n3. 표 데이터 정제 및 소제목 요약 시작 (시간이 다소 소요됩니다)...")
section_summaries = {}

invalid_endings = (
    '다.', '다', '요.', '요', '음.', '음', '함.', '함', '은', '는', '이', '가', '을', '를', 
    '에', '에게', '에서', '로', '으로', '과', '와', '며', '고', '부터', '까지', '해,', '위해,',
    '면', '서', '며,', '고,', '니다.', '니다', '습니다.', '습니다', '습',
    '바랍니다.', '바랍니다', '입니다.', '입니다', '작성하였습니다.', '작성하였습니다', '있습니다.', '있습니다',
    '대비 약', '생산라인별', '환산 기준', '감안하여,'
)

# 진행률 표시줄(tqdm) 적용
for section in tqdm(all_parsed_sections, desc="섹션 전처리 및 LLM 요약"):
    base_meta = section.get('metadata', {})
    # 기업명과 연도를 포함한 고유 키 생성 (예: 삼성전자_2023_소제목)
    unique_sec_key = f"{base_meta.get('corp_name')}_{base_meta.get('report_year')}_{section.get('section_sub', '소제목 없음')}"
    
    full_text_for_summary = ""

    for block in section.get('blocks', []):
        if block['type'] == 'text':
            full_text_for_summary += block['content'] + "\n"
        
        elif block['type'] == 'table':
            pre_text = block.get('pre_text', '')
            
            if pre_text:
                full_text_for_summary += pre_text + "\n"
            
            if len(pre_text) > 10:
                ext_title = "제목 없음"
                ext_unit = "단위 없음"
                
                unit_match = re.search(r'[\(\[\{]\s*단위\s*[:：]?\s*(.*?)[\)\]\}]', pre_text)
                if unit_match:
                    ext_unit = unit_match.group(1).strip()
                
                lines = [line.strip() for line in pre_text.split('\n') if line.strip()]
                for line in reversed(lines):
                    clean_line = re.sub(r'[\(\[\{]\s*단위\s*[:：]?\s*(.*?)[\)\]\}]', '', line).strip()
                    clean_line = re.sub(r'^(다음은|아래는|당사의|다음표는|표 제목:|\*|※|-)\s*', '', clean_line).strip()
                    
                    if any(clean_line.endswith(ending) for ending in invalid_endings):
                        continue
                        
                    if re.match(r'^[\d\s\.,%조원천개]+$', clean_line) or clean_line.endswith(','):
                        continue
                        
                    if 0 < len(clean_line) < 35:
                        ext_title = clean_line
                        break

                extracted_meta = f"표 제목: {ext_title}\n단위: {ext_unit}"
                block['assembled_table'] = f"{extracted_meta}\n{block.get('table_html', '')}"
            else:
                block['assembled_table'] = block.get('table_html', '')
            
            full_text_for_summary += block['assembled_table'] + "\n"

    if len(full_text_for_summary) > 100:
        messages_sec = [
            {"role": "system", "content": "당신은 요약 전문 AI입니다. 분석이나 변명 없이 오직 핵심 내용만 300자 이내의 한국어로 요약하세요."},
            {"role": "user", "content": f"다음 문서를 요약하세요.\n\n[문서]\n{full_text_for_summary[:4000]}"}
        ]
        prompt_sec = processor.apply_chat_template(messages_sec, tokenize=False, add_generation_prompt=True)
        prompt_sec += "<요약>\n"
        
        response_sec = generate(model, processor, prompt=prompt_sec, max_tokens=400, temp=0.1, verbose=False)
        raw_output = response_sec.text.strip()
        
        tag_match = re.search(r'(.*?)(?:</요약>|$)', raw_output, re.DOTALL)
        
        if tag_match:
            clean_sec = tag_match.group(1).replace('`', '').strip()
            clean_sec = re.sub(r'^.*?(현황 요약입니다|요약입니다|요약한 내용입니다)[\.\:\n\s]*', '', clean_sec)
            clean_sec = clean_sec.strip()
            
            if len(clean_sec) < 5 or clean_sec.lower() == 'and':
                clean_sec = "요약 실패 (내용 부족)"
        else:
            clean_sec = "요약 추출 실패"
            
        section_summaries[unique_sec_key] = clean_sec
    else:
        section_summaries[unique_sec_key] = full_text_for_summary.strip()

# ==========================================
# 🚀 [Phase 2] 투트랙 분할 및 최종 조립
# ==========================================
print("\n4. 투트랙(Two-Track) 청킹 및 룰베이스 최종 조립 진행...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

super_chunks = []
global_chunk_idx = 0

for section in all_parsed_sections:
    base_meta = section.get('metadata', {}).copy()
    
    section_main = section.get('section_main', base_meta.get('section_main', 'II. 사업의 내용'))
    section_sub = section.get('section_sub', '소제목 없음')
    
    unique_sec_key = f"{base_meta.get('corp_name')}_{base_meta.get('report_year')}_{section_sub}"
    current_sec_summary = section_summaries.get(unique_sec_key, "")
    
    base_meta.update({
        "section_main": section_main, 
        "section_sub": section_sub,
        "section_summary": current_sec_summary
    })
    
    breadcrumb = f"[경로: {base_meta.get('corp_name', '알수없음')} > {base_meta.get('report_year', '알수없음')} 사업보고서 > {section_main} > {section_sub}]"
    
    for block in section.get('blocks', []):
        if block['type'] == 'text':
            chunks = text_splitter.split_text(block['content'])
            for chunk_text in chunks:
                final_content = f"{breadcrumb}\n[본문]:\n{chunk_text}"
                super_chunks.append({
                    "chunk_id": f"{base_meta.get('corp_code', '0000')}_{base_meta.get('report_year', '0000')}_{global_chunk_idx:05d}",
                    "content": final_content,
                    "metadata": base_meta
                })
                global_chunk_idx += 1
                
        elif block['type'] == 'table':
            pre_text = block.get('pre_text', '').strip()
            table_content = block.get('assembled_table', '')
            
            content_body = ""
            if pre_text:
                content_body += f"[표 설명 텍스트]:\n{pre_text}\n\n"
            content_body += f"[표 데이터]:\n{table_content}"
            
            final_content = f"{breadcrumb}\n{content_body}"
            
            super_chunks.append({
                "chunk_id": f"{base_meta.get('corp_code', '0000')}_{base_meta.get('report_year', '0000')}_{global_chunk_idx:05d}_table",
                "content": final_content,
                "metadata": base_meta
            })
            global_chunk_idx += 1

# ==========================================
# 🚀 [최종 저장]
# ==========================================
save_path = "multi_hybrid_chunks_final.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(super_chunks, f, ensure_ascii=False, indent=2)

print(f"\n✨ [최종 완료] {len(super_chunks)}개의 다중 하이브리드 청킹 완료!")
print(f"저장 경로: {save_path}")