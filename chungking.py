import json
import os
from tqdm import tqdm
from mlx_vlm import load, generate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 앞 단계에서 만든 구조화된 JSON(마크다운 표 포함) 읽어오기
file_path = "report_xml_00126380_2023/parsed_business_content.json"
print(f"1. '{file_path}' 데이터를 불러옵니다...")

with open(file_path, 'r', encoding='utf-8') as f:
    parsed_sections = json.load(f)

# 2. MLX 모델 로드 (M5 Pro 가속)
model_id = "mlx-community/Qwen3.5-9B-8bit"
print(f"\n2. MLX 모델 로드 중... ({model_id})")
model, processor = load(model_id)

# ==========================================
# 🚀 [Phase 1] 소제목별 500자 요약 (Map)
# ==========================================
print("\n3. [계층형 요약 1단계] 각 소제목(목차)별 핵심 500자 요약을 시작합니다...")
section_summaries = {}

for section in tqdm(parsed_sections, desc="소제목 요약 진행"):
    section_sub = section['section_sub']
    section_text = section['text']
    
    # 텍스트가 너무 짧으면 요약 생략
    if len(section_text) < 100:
        section_summaries[section_sub] = section_text
        continue
        
    prompt = f"""당신은 금융 분석 전문가입니다. 다음은 사업보고서의 '{section_sub}' 목차 내용입니다. 
이 내용에 등장하는 핵심 비즈니스 정보, 주요 숫자(매출, 비중 등), 전략을 500자 이내로 요약하세요.

[원문]:
{section_text[:15000]} # 메모리 초과 방지를 위해 최대 15,000자까지만 읽음

요약(500자 이내):"""

    response = generate(model, processor, prompt=prompt, max_tokens=500, verbose=False)
    section_summaries[section_sub] = response.text.strip()

# ==========================================
# 🚀 [Phase 2] 전체 문서 마스터 요약 (Reduce)
# ==========================================
print("\n4. [계층형 요약 2단계] 3.8만 자를 아우르는 '전체 마스터 요약본'을 생성합니다...")
combined_summaries = "\n\n".join([f"[{k}] 요약:\n{v}" for k, v in section_summaries.items()])

master_prompt = f"""당신은 최고 재무 책임자(CFO)입니다. 다음은 사업보고서 각 목차별 요약본 모음입니다. 
이 기업의 전체적인 사업 현황, 주력 제품, 리스크 및 전략을 아우르는 1000자 이내의 '전체 총괄 요약본(Master Summary)'을 작성하세요.

[각 목차별 요약 모음]:
{combined_summaries}

전체 총괄 요약(1000자 이내):"""

master_response = generate(model, processor, prompt=master_prompt, max_tokens=1000, verbose=False)
master_summary = master_response.text.strip()
print("\n[성공] 마스터 요약본이 완성되었습니다!")

# ==========================================
# 🚀 [Phase 3] 텍스트 청킹 (Chunking)
# ==========================================
print("\n5. 텍스트 분할(Chunking)을 시작합니다. (오버랩 200자 적용)")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, # 회원님 요청대로 오버랩 증가 (문맥 단절 최소화)
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []
global_chunk_idx = 0

for section in parsed_sections:
    section_sub = section['section_sub']
    chunks_in_section = text_splitter.split_text(section['text'])
    
    for text_chunk in chunks_in_section:
        all_chunks.append({
            "chunk_id": f"{section['metadata']['corp_code']}_{section['metadata']['report_year']}_{global_chunk_idx:04d}",
            "section_sub": section_sub,
            "text": text_chunk,
            "metadata": section['metadata']
        })
        global_chunk_idx += 1

# ==========================================
# 🚀 [Phase 4] 문맥 주입 (Contextual Retrieval)
# ==========================================
super_chunks = []
print(f"\n6. 총 {len(all_chunks)}개 청크에 [마스터 요약 + 소제목 요약 + 앞뒤 문맥]을 주입합니다...")

for i, chunk_data in enumerate(tqdm(all_chunks, desc="초지능형 청크 생성")):
    prev_id = all_chunks[i-1]['chunk_id'] if i > 0 else None
    next_id = all_chunks[i+1]['chunk_id'] if i < len(all_chunks) - 1 else None
    
    prev_text = all_chunks[i-1]['text'] if i > 0 else "문서의 시작 부분입니다."
    next_text = all_chunks[i+1]['text'] if i < len(all_chunks) - 1 else "문서의 끝 부분입니다."
    
    current_text = chunk_data['text']
    current_section = chunk_data['section_sub']
    
    # 회원님 아이디어의 결정체: 완벽한 프롬프트 구조
    prompt = f"""당신은 AI 데이터 엔지니어입니다. 아래 정보를 바탕으로 [현재 청크]가 어떤 맥락을 가지는지 200자 이내로 요약하세요.

[전체 문서 마스터 요약]: {master_summary}
[현재 목차({current_section}) 핵심 요약]: {section_summaries.get(current_section, '')}

---
[이전 내용]: {prev_text[-300:]} # 앞 내용의 끝부분 300자만 참조
[현재 청크]: {current_text}
[다음 내용]: {next_text[:300]} # 뒷 내용의 첫부분 300자만 참조

청크 맥락 요약(200자 이내):"""

    response = generate(model, processor, prompt=prompt, max_tokens=200, verbose=False)
    context_summary = response.text.strip()
    
    # 메타데이터에 마스터 요약과 소제목 요약까지 모두 저장 (나중에 검색용으로 엄청난 위력 발휘)
    final_metadata = chunk_data['metadata'].copy()
    final_metadata.update({
        "section_sub": current_section,
        "prev_chunk_id": prev_id,
        "next_chunk_id": next_id,
        "master_summary": master_summary,
        "section_summary": section_summaries.get(current_section, '')
    })
    
    combined_content = f"[속한 목차]: {current_section}\n[맥락]: {context_summary}\n\n[원문]:\n{current_text}"
    
    super_chunks.append({
        "chunk_id": chunk_data['chunk_id'],
        "content": combined_content,
        "metadata": final_metadata
    })

# ==========================================
# 🚀 [최종 저장]
# ==========================================
save_path = "report_xml_00126380_2023/samsung_super_chunks_final.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(super_chunks, f, ensure_ascii=False, indent=2)

print(f"\n✨ [최종 완료] '나만의 블룸버그 터미널'을 위한 초지능형 데이터셋이 완성되었습니다!")
print(f"저장 경로: {save_path}")