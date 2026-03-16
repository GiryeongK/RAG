import json
import os
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from mlx_vlm import load, generate

json_path = "report_xml_00126380_2023/samsung_hybrid_chunks_final.json"
persist_directory = "./chroma_db_samsung"

# ==========================================
# 1. 시스템 컴포넌트 로드 (DB, 검색기, LLM)
# ==========================================
print("1. 임베딩 및 Vector DB 로드 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'mps'},
    encode_kwargs={'normalize_embeddings': True}
)

if not os.path.exists(persist_directory):
    raise FileNotFoundError("Vector DB가 없습니다. vector_db.py를 먼저 실행하여 DB를 생성하세요.")

vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 20})

with open(json_path, 'r', encoding='utf-8') as f:
    chunks_data = json.load(f)
documents = [Document(page_content=item["content"], metadata={**{k: str(v) for k, v in item.get("metadata", {}).items()}, "chunk_id": str(item["chunk_id"])}) for item in chunks_data]

print("2. Sparse Retriever (BM25) 설정 중...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 20

print("3. Re-ranker 로드 중...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="mps")

print("4. LLM (Qwen3.5-9B) 로드 중 (MLX)...")
model_id = "mlx-community/Qwen3.5-9B-8bit"
model, processor = load(model_id)

# ==========================================
# 2. LLM 에이전트 핵심 함수 정의
# ==========================================
def decompose_query_with_llm(user_query):
    """LLM을 사용하여 복합 질문을 독립적인 검색어들로 분해 (일반화된 Few-Shot 적용)"""
    prompt_messages = [
        {"role": "system", "content": "당신은 키워드 추출기입니다. 사용자의 복합 질문을 독립적인 핵심 검색어로만 변환하세요. 설명이나 'Thinking Process'는 절대 출력하지 마세요."},
        # 💡 특정 도메인에 종속되지 않은 일반적인 비즈니스 질문 예시 주입
        {"role": "user", "content": "2022년 가전제품 매출액이랑 주요 수출 국가는 어디야?"},
        {"role": "assistant", "content": "가전제품 매출액, 주요 수출 국가"},
        {"role": "user", "content": "반도체 부문 영업이익 적자 규모와 그 원인에 대해 설명해 줘."},
        {"role": "assistant", "content": "반도체 부문 영업이익 적자 규모, 반도체 영업이익 적자 원인"},
        {"role": "user", "content": f"{user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    # temp를 0.0으로 고정하여 창의성을 제거하고 패턴만 따르도록 강제
    response = generate(model, processor, prompt=prompt, max_tokens=50, temp=0.0, verbose=False) 
    
    raw_text = response.text.strip()
    clean_queries = [q.strip() for q in raw_text.split(',')]
    return clean_queries if clean_queries else [user_query]

def generate_final_answer(user_query, retrieved_contexts):
    """검색된 문서를 바탕으로 최종 답변 생성 (일반화된 Few-Shot 적용)"""
    context_text = "\n\n".join([f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(retrieved_contexts)])
    
    prompt_messages = [
        {"role": "system", "content": "당신은 한국어로만 대답하는 요약 봇입니다. 주어진 문서의 팩트만 결합하여 간결하게 대답하세요. 'Thinking Process' 등 영어 사고 과정 출력은 엄격히 금지됩니다."},
        # 💡 범용적인 문서 요약 예시 주입
        {"role": "user", "content": "[참고 문서]\n문서 1: 2022년 가전 부문 매출은 10조 원이다.\n문서 2: 당사의 주요 가전 수출국은 미국과 독일이다.\n\n[사용자 질문]: 가전 매출과 수출국 알려줘"},
        {"role": "assistant", "content": "2022년 가전 부문 매출은 10조 원입니다. 주요 수출국은 미국과 독일입니다."},
        {"role": "user", "content": f"[참고 문서]\n{context_text}\n\n[사용자 질문]: {user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, processor, prompt=prompt, max_tokens=500, temp=0.0, verbose=False)
    
    final_text = response.text.strip()
    # 엣지 케이스: 모델이 지시를 어기고 영어를 출력할 경우 강제 컷오프
    if "Thinking Process:" in final_text:
        final_text = final_text.split("Thinking Process:")[0].strip()
        
    return final_text

# ==========================================
# 3. 🚀 메인 실행 파이프라인
# ==========================================
user_query = "2023년 스마트폰 시장 점유율은 몇 프로야? 그리고 어떤 제품들을 출시했어?"
print(f"\n[사용자 원본 질문]: {user_query}")
print("-" * 50)

# 단계 1: LLM을 이용한 다중 질의 분해
print("-> [Step 1] LLM 질의 분해 중...")
sub_queries = decompose_query_with_llm(user_query)
print(f"-> 분해된 검색어: {sub_queries}")

# 단계 2: 분해된 질의별 병렬 검색 및 문서 병합
print("\n-> [Step 2] 하이브리드 병렬 검색 진행...")
unique_docs = {}
for q in sub_queries:
    d_docs = dense_retriever.invoke(q)
    s_docs = bm25_retriever.invoke(q)
    for doc in d_docs + s_docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id and chunk_id not in unique_docs:
            unique_docs[chunk_id] = doc

combined_docs = list(unique_docs.values())
print(f"-> 검색된 고유 문서 총 {len(combined_docs)}개")

# 단계 3: 원본 질문 기준으로 Re-ranking
print("\n-> [Step 3] Re-ranking 진행...")
cross_inp = [[user_query, doc.page_content] for doc in combined_docs]
scores = reranker.predict(cross_inp)
scored_docs = zip(combined_docs, scores)
sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

# LLM에게 던져줄 Top-5 문서만 추출
top_k_docs = [doc for doc, score in sorted_docs[:5]]

# 단계 4: LLM 최종 답변 생성
print("\n-> [Step 4] LLM 최종 답변 생성 중...\n")
final_answer = generate_final_answer(user_query, top_k_docs)

print("=" * 50)
print(f"[최종 답변]\n{final_answer}")
print("=" * 50)