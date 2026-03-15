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
# 🚀 [Phase 1] 표 메타데이터 핀셋 추출 & 소제목 요약 (프롬프트 완전 통제 적용)
# ==========================================
print("\n3. 표 메타데이터 추출 및 소제목 요약 (1회성 가벼운 LLM 호출) 시작...")
section_summaries = {}

for section in tqdm(parsed_sections, desc="섹션 전처리"):
    section_sub = section['section_sub']
    full_text_for_summary = ""

    for block in section['blocks']:
        if block['type'] == 'text':
            full_text_for_summary += block['content'] + "\n"
        
        elif block['type'] == 'table':
            pre_text = block['pre_text']
            if len(pre_text) > 10:
                # 💡 해결 1: 괄호 및 영어 예시 완벽 제거. 철저한 단답형 요구.
                messages = [
                    {"role": "system", "content": "당신은 데이터 추출기입니다. 반드시 한국어로 대답하며, '표 제목:'과 '단위:' 두 줄만 출력해야 합니다. 서론이나 설명은 절대 금지합니다."},
                    {"role": "user", "content": f"다음 텍스트에서 표의 제목과 단위를 찾아 아래 형식에 맞춰 작성하세요.\n\n[텍스트]:\n{pre_text[-500:]}\n\n[출력 형식]\n표 제목: 실제표제목\n단위: 실제단위"}
                ]
                prompt_table = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                response = generate(model, processor, prompt=prompt_table, max_tokens=100, verbose=False)
                clean_text = response.text.strip()
                
                title_match = re.search(r'표 제목:\s*(.*)', clean_text)
                unit_match = re.search(r'단위:\s*(.*)', clean_text)
                
                ext_title = title_match.group(1).strip() if title_match else "제목 없음"
                ext_unit = unit_match.group(1).strip() if unit_match else "단위 없음"
                
                extracted_meta = f"표 제목: {ext_title}\n단위: {ext_unit}"
                block['assembled_table'] = f"{extracted_meta}\n{block['markdown']}"
                
                time.sleep(1) 
            else:
                block['assembled_table'] = block['markdown']
            
            full_text_for_summary += block['assembled_table'] + "\n"

    # [소제목 요약]
    if len(full_text_for_summary) > 100:
        # 💡 해결 2: Thinking Process 원천 차단 및 한국어 출력 강제
        messages_sec = [
            {"role": "system", "content": "당신은 한국인 금융 분석가입니다. 반드시 '한국어'로만 작성하십시오. 'Thinking Process', '분석' 등의 사고 과정이나 서론을 절대 출력하지 마십시오. 오직 최종 요약 결과물만 즉시 출력하십시오."},
            {"role": "user", "content": f"아래 [원문]을 읽고 핵심 숫자와 전략을 300자 이내의 한국어로 요약하십시오.\n\n[원문]:\n{full_text_for_summary[:10000]}"}
        ]
        prompt_sec = processor.apply_chat_template(messages_sec, tokenize=False, add_generation_prompt=True)
        
        response_sec = generate(model, processor, prompt=prompt_sec, max_tokens=500, verbose=False)
        clean_sec = response_sec.text.strip()
        
        # 💡 해결 3: 모델이 지시를 어기고 영어를 출력할 경우를 대비한 파이썬 강제 절단 로직
        if "Thinking Process" in clean_sec or "Analyze" in clean_sec:
            # 영어 분석 과정이 포함되었다면 마지막 한글 부분(실제 요약)만 강제로 추출
            clean_sec = re.sub(r'.*?(당사는|이 부문은|이 기업은|종합하면|요약하면)', r'\1', clean_sec, flags=re.DOTALL)
            
        clean_sec = re.split(r'(요약\(|요약:|\[원문\])', clean_sec)[0].strip()
        
        section_summaries[section_sub] = clean_sec
        time.sleep(1) 
    else:
        section_summaries[section_sub] = full_text_for_summary

# ==========================================
# 🚀 [Phase 2] 투트랙 분할 및 최종 조립 (하드코딩 결합)
# ==========================================
print("\n4. 투트랙(Two-Track) 청킹 및 룰베이스 최종 조립 진행...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

super_chunks = []
global_chunk_idx = 0

for section in parsed_sections:
    section_sub = section['section_sub']
    base_meta = section['metadata'].copy()
    base_meta.update({"section_sub": section_sub})
    
    current_sec_summary = section_summaries.get(section_sub, "")
    breadcrumb = f"[경로: {base_meta['corp_name']} > {base_meta['report_year']} 사업보고서 > {section_sub}]"
    
    for block in section['blocks']:
        # 트랙 1: 일반 텍스트 분할
        if block['type'] == 'text':
            chunks = text_splitter.split_text(block['content'])
            for chunk_text in chunks:
                final_content = f"{breadcrumb}\n[섹션 핵심]: {current_sec_summary}\n\n[본문]:\n{chunk_text}"
                super_chunks.append({
                    "chunk_id": f"{base_meta['corp_code']}_{base_meta['report_year']}_{global_chunk_idx:04d}",
                    "content": final_content,
                    "metadata": base_meta
                })
                global_chunk_idx += 1
                
        # 트랙 2: 표 데이터 원본 유지
        elif block['type'] == 'table':
            table_content = block['assembled_table']
            final_content = f"{breadcrumb}\n[섹션 핵심]: {current_sec_summary}\n\n[표 데이터]:\n{table_content}"
            super_chunks.append({
                "chunk_id": f"{base_meta['corp_code']}_{base_meta['report_year']}_{global_chunk_idx:04d}_table",
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