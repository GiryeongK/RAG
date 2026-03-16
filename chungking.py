import json
import os
import re
import time
from tqdm import tqdm
from mlx_vlm import load, generate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 데이터 로드
file_path = "report_xml_00126380_2023/parsed_business_content.json"
print(f"1. '{file_path}' 데이터를 불러옵니다...")
with open(file_path, 'r', encoding='utf-8') as f:
    parsed_sections = json.load(f)

# 2. MLX 모델 로드
model_id = "mlx-community/Qwen3.5-9B-8bit"
print(f"\n2. MLX 모델 로드 중... ({model_id})")
model, processor = load(model_id)

# ==========================================
# 🚀 [Phase 1] 표 메타데이터 룰베이스 추출 & 소제목 요약
# ==========================================
print("\n3. 표 데이터 정제(금지어 룰베이스) 및 소제목 요약(LLM Force Prefix) 시작...")
section_summaries = {}

# 💡 [수정] 표 제목 금지어 리스트에 '부터', '습', '해,' 등 엣지 케이스 추가
invalid_endings = (
    '다.', '다', '요.', '요', '음.', '음', '함.', '함', '은', '는', '이', '가', '을', '를', 
    '에', '에게', '에서', '로', '으로', '과', '와', '며', '고', '부터', '까지', '해,', '위해,',
    '면', '서', '며,', '고,', '니다.', '니다', '습니다.', '습니다', '습',
    '바랍니다.', '바랍니다', '입니다.', '입니다', '작성하였습니다.', '작성하였습니다', '있습니다.', '있습니다'
)

for section in tqdm(parsed_sections, desc="섹션 전처리"):
    section_sub = section.get('section_sub', '소제목 없음')
    full_text_for_summary = ""

    for block in section.get('blocks', []):
        if block['type'] == 'text':
            full_text_for_summary += block['content'] + "\n"
        
        elif block['type'] == 'table':
            pre_text = block.get('pre_text', '')
            if len(pre_text) > 10:
                ext_title = "제목 없음"
                ext_unit = "단위 없음"
                
                # 1. 단위 추출
                unit_match = re.search(r'[\(\[\{]\s*단위\s*[:：]?\s*(.*?)[\)\]\}]', pre_text)
                if unit_match:
                    ext_unit = unit_match.group(1).strip()
                
                # 2. 표 제목 추출 (금지어 기반 완벽 차단)
                lines = [line.strip() for line in pre_text.split('\n') if line.strip()]
                for line in reversed(lines):
                    clean_line = re.sub(r'[\(\[\{]\s*단위\s*[:：]?\s*(.*?)[\)\]\}]', '', line).strip()
                    clean_line = re.sub(r'^(다음은|아래는|당사의|다음표는|표 제목:|\*|※|-)\s*', '', clean_line).strip()
                    
                    # 금지된 조사나 서술어로 끝나면 무조건 스킵
                    if any(clean_line.endswith(ending) for ending in invalid_endings):
                        continue
                        
                    # 35자 이내의 명사형 텍스트만 채택
                    if 0 < len(clean_line) < 35:
                        ext_title = clean_line
                        break

                extracted_meta = f"표 제목: {ext_title}\n단위: {ext_unit}"
                block['assembled_table'] = f"{extracted_meta}\n{block.get('markdown', '')}"
            else:
                block['assembled_table'] = block.get('markdown', '')
            
            full_text_for_summary += block['assembled_table'] + "\n"

    # [소제목 요약]
    if len(full_text_for_summary) > 100:
        messages_sec = [
            {"role": "system", "content": "당신은 요약 전문 AI입니다. 분석이나 변명 없이 오직 핵심 내용만 300자 이내의 한국어로 요약하세요."},
            {"role": "user", "content": f"다음 문서를 요약하세요.\n\n[문서]\n{full_text_for_summary[:4000]}"} # 메모리 부하 방지를 위해 4000자로 컷
        ]
        prompt_sec = processor.apply_chat_template(messages_sec, tokenize=False, add_generation_prompt=True)
        
        # 💡 [핵심 수정 2] 모델이 딴소리를 못하도록 출력 시작점을 <요약>으로 강제 고정 (Force Prefix)
        prompt_sec += "<요약>\n"
        
        response_sec = generate(model, processor, prompt=prompt_sec, max_tokens=400, temp=0.1, verbose=False)
        raw_output = response_sec.text.strip()
        
        # 강제 시작점인 <요약> 이후부터, </요약>이 나오거나 끝날 때까지 긁어옴
        tag_match = re.search(r'(.*?)(?:</요약>|$)', raw_output, re.DOTALL)
        
        if tag_match:
            clean_sec = tag_match.group(1).replace('`', '').strip()
            # 만약 요약 내용이 너무 짧거나 비정상적(and)이면 방어
            if len(clean_sec) < 5 or clean_sec.lower() == 'and':
                clean_sec = "요약 실패 (내용 부족)"
        else:
            clean_sec = "요약 추출 실패"
            
        section_summaries[section_sub] = clean_sec
        time.sleep(1)
    else:
        section_summaries[section_sub] = full_text_for_summary

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

for section in parsed_sections:
    base_meta = section.get('metadata', {}).copy()
    
    section_main = section.get('section_main', base_meta.get('section_main', 'II. 사업의 내용'))
    section_sub = section.get('section_sub', '소제목 없음')
    
    base_meta.update({"section_main": section_main, "section_sub": section_sub})
    
    current_sec_summary = section_summaries.get(section_sub, "")
    breadcrumb = f"[경로: {base_meta.get('corp_name', '알수없음')} > {base_meta.get('report_year', '알수없음')} 사업보고서 > {section_main} > {section_sub}]"
    
    for block in section.get('blocks', []):
        if block['type'] == 'text':
            chunks = text_splitter.split_text(block['content'])
            for chunk_text in chunks:
                final_content = f"{breadcrumb}\n[섹션 핵심]: {current_sec_summary}\n\n[본문]:\n{chunk_text}"
                super_chunks.append({
                    "chunk_id": f"{base_meta.get('corp_code', '0000')}_{base_meta.get('report_year', '0000')}_{global_chunk_idx:04d}",
                    "content": final_content,
                    "metadata": base_meta
                })
                global_chunk_idx += 1
                
        elif block['type'] == 'table':
            table_content = block.get('assembled_table', '')
            final_content = f"{breadcrumb}\n[섹션 핵심]: {current_sec_summary}\n\n[표 데이터]:\n{table_content}"
            super_chunks.append({
                "chunk_id": f"{base_meta.get('corp_code', '0000')}_{base_meta.get('report_year', '0000')}_{global_chunk_idx:04d}_table",
                "content": final_content,
                "metadata": base_meta
            })
            global_chunk_idx += 1

# ==========================================
# 🚀 [최종 저장]
# ==========================================
save_path = "report_xml_00126380_2023/samsung_hybrid_chunks_final.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(super_chunks, f, ensure_ascii=False, indent=2)

print(f"\n✨ [최종 완료] 무거운 LLM 연산을 덜어낸 하이브리드 청킹이 완료되었습니다!")
print(f"저장 경로: {save_path}")