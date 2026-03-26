import json
import os
import gc
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from mlx_vlm import load as mlx_load, generate as mlx_generate

json_path = "report_xml_00126380_2023/samsung_hybrid_chunks_final.json"
persist_directory = "./chroma_db_samsung"
model_id = "mlx-community/Qwen3.5-9B-8bit"

def clear_memory():
    """RAM 및 Mac GPU(MPS) 캐시를 강제로 비우는 함수"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==========================================
# 1. LLM 에이전트 함수 정의
# ==========================================
def route_query(user_query, model, processor):
    prompt_messages = [
        {"role": "system", "content": "사용자의 질문이 명확한 재무 수치(매출액, 영업이익 등)를 묻는다면 'SQL', 사업의 내용, 전략, 시장 전망 등을 묻는다면 'RAG'라고만 답하세요."},
        {"role": "user", "content": f"{user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<route>"
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=10, temp=0.0, verbose=False)
    route_result = response.text.split("</route>")[0].strip().upper()
    return "SQL" if "SQL" in route_result else "RAG"

def decompose_query_with_llm(user_query, model, processor):
    prompt_messages = [
        {"role": "system", "content": "사용자의 질문을 쉼표로 구분된 핵심 검색어로만 변환하세요. 단, 각 검색어는 독립적으로 의미를 갖도록 핵심 주체(예: 스마트폰, 2023년)를 반드시 포함해야 합니다. 다른 말은 절대 금지합니다."},
        {"role": "user", "content": "2022년 가전제품 매출액이랑 주요 수출 국가는 어디야?"},
        {"role": "assistant", "content": "2022년 가전제품 매출액, 가전제품 주요 수출 국가"},
        {"role": "user", "content": f"{user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<result>"
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=100, temp=0.0, verbose=False)
    clean_text = response.text.split("</result>")[0].strip()
    clean_queries = [q.strip() for q in clean_text.split(',') if q.strip()]
    return clean_queries if clean_queries else [user_query]

def generate_final_answer(user_query, retrieved_contexts, model, processor):
    unique_summaries = {}
    for doc in retrieved_contexts:
        sub = doc.metadata.get("section_sub", "알수없음")
        summary = doc.metadata.get("section_summary", "")
        if sub not in unique_summaries and summary:
            unique_summaries[sub] = summary

    context_text = "[관련 섹션 배경지식 요약]\n"
    if unique_summaries:
        for sub, summary in unique_summaries.items():
            context_text += f"- {sub}: {summary}\n"
    else:
        context_text += "배경지식 없음\n"

    context_text += "\n[검색된 본문 팩트 데이터]\n"
    for i, doc in enumerate(retrieved_contexts):
        context_text += f"[문서 {i+1} (ID: {doc.metadata.get('chunk_id')})]\n{doc.page_content}\n\n"
    
    prompt_messages = [
        # 💡 핵심 수정: 지어내기 및 무한 루프 방지를 위한 강력한 제약 조건 추가
        {"role": "system", "content": "당신은 객관적인 데이터 분석가입니다. 주어진 팩트 데이터만 사용하여 한국어로 답변하세요. 문서에 질문에 대한 명확한 전체 목록이나 데이터가 없다면, 억지로 유추하거나 지어내지 말고 반드시 '제공된 문서에는 해당 정보가 부족합니다'라고만 답하세요. 같은 단어를 반복하지 마세요."},
        {"role": "user", "content": f"{context_text}\n[사용자 질문]: {user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<answer>"
    
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=1024, temp=0.1, verbose=False)
    return response.text.split("</answer>")[0].strip()

# ==========================================
# 2. 🚀 메인 파이프라인 (메모리 격리 실행)
# ==========================================
if __name__ == "__main__":
    user_query = "2023년 스마트폰 시장 점유율은 몇 프로야? 그리고 어떤 제품들을 출시했어?"
    print(f"\n[사용자 원본 질문]: {user_query}")
    print("-" * 50)

    # ---------------------------------------------------------
    # [Step 1] LLM 로드 및 라우팅/질의 분해
    # ---------------------------------------------------------
    print("-> [Step 1] LLM 로드 중 (라우팅 및 분해용)...")
    model, processor = mlx_load(model_id)
    
    route = route_query(user_query, model, processor)
    print(f"-> 라우터 판별 결과: {route}")
    
    if route == "SQL":
        print("\n[안내] SQL 쿼리가 필요한 질문입니다. 현재 SQL DB 연동이 개발되지 않아 프로세스를 종료합니다.")
        del model, processor
        clear_memory()
        exit()
        
    print("-> [Step 1-1] RAG 파이프라인 진행. LLM 질의 분해 중...")
    sub_queries = decompose_query_with_llm(user_query, model, processor)
    print(f"-> 분해된 검색어: {sub_queries}")
    
    del model, processor
    clear_memory()

    # ---------------------------------------------------------
    # [Step 2] 임베딩/검색 모델 로드 및 정보 검색
    # ---------------------------------------------------------
    print("\n-> [Step 2] 검색 모델 로드 및 하이브리드 검색 진행...")
    if not os.path.exists(persist_directory):
        raise FileNotFoundError("Vector DB가 없습니다. vector_db.py를 먼저 실행하세요.")

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': 'mps'}, encode_kwargs={'normalize_embeddings': True})
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    dense_retriever = vector_db.as_retriever(search_kwargs={"k": 20})

    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    documents = [Document(page_content=item["content"], metadata={**{k: str(v) for k, v in item.get("metadata", {}).items()}, "chunk_id": str(item["chunk_id"])}) for item in chunks_data]
    
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 20
    
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="mps")

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

    print("-> [Step 2-1] Re-ranking 진행...")
    cross_inp = [[user_query, doc.page_content] for doc in combined_docs]
    scores = reranker.predict(cross_inp)
    scored_docs = zip(combined_docs, scores)
    
    # 💡 추출 문서 개수 8개로 유지
    top_k_docs = [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:8]]

    del embeddings, vector_db, dense_retriever, bm25_retriever, reranker
    clear_memory()

    # ---------------------------------------------------------
    # [Step 3] LLM 재로드 및 최종 답변 생성
    # ---------------------------------------------------------
    print("\n-> [Step 3] LLM 재로드 및 최종 답변 생성 중...")
    model, processor = mlx_load(model_id)
    
    final_answer = generate_final_answer(user_query, top_k_docs, model, processor)
    
    print("\n[최종 답변]\n")
    print(final_answer)