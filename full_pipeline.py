import json
import os
import gc
import torch
import sqlite3
import re
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from mlx_vlm import load as mlx_load, generate as mlx_generate

json_path = "report_xml_00126380_2023/samsung_hybrid_chunks_final.json"
persist_directory = "./chroma_db_samsung"
model_id = "mlx-community/Qwen3.5-9B-8bit"
sql_db_path = "samsung_finance.db"

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
        {"role": "system", "content": """당신은 질문을 분류하는 엄격한 라우터입니다. 규칙은 단 하나입니다.
질문에 '영업이익', '매출액', '자산', '부채', '자본', '재무상태' 등 재무 수치를 묻는 단어가 포함되어 있으면 무조건 'SQL'이라고만 답하세요.
그 외의 모든 질문(시장 점유율, 출시 제품, 전망, 사업 내용 등)은 무조건 'RAG'라고만 답하세요.
부연 설명은 절대 금지합니다.

[예시]
질문: 2023년 영업이익은 얼마야? -> SQL
질문: 스마트폰 시장 점유율은? -> RAG"""},
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

def get_sql_data(user_query, model, processor):
    """SQL 쿼리를 생성하고 실행하여 답변 생성이 아닌 순수 DB 텍스트 데이터만 반환합니다."""
    schema_info = """
    [SQLite 스키마 정보]
    테이블명: finance_2023
    컬럼:
    - 사업연도 (INTEGER): 예) 2023
    - 재무제표명 (TEXT): 예) 손익계산서, 재무상태표
    - 계정명 (TEXT): 예) 매출액, 영업이익, 매출원가, 자산총계
    - 당기금액 (FLOAT): 원 단위 (예: 6566976000000.0)
    """
    
    sql_messages = [
        {"role": "system", "content": f"당신은 데이터 분석가입니다. 스키마를 바탕으로 정확한 SQLite 쿼리를 작성하세요. 마크다운 ```sql 과 ``` 안에 쿼리문만 작성하세요.\n{schema_info}"},
        {"role": "user", "content": user_query}
    ]
    prompt = processor.apply_chat_template(sql_messages, tokenize=False, add_generation_prompt=True)
    
    print("\n   -> [SQL 에이전트] 쿼리 생성 중...")
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=200, temp=0.0, verbose=False)
    raw_output = response.text
    
    query_match = re.search(r'```sql\n(.*?)\n```', raw_output, re.DOTALL)
    if query_match:
        sql_query = query_match.group(1).strip()
    else:
        sql_query = raw_output.replace('```sql', '').replace('```', '').strip()

    print(f"   -> 실행 쿼리: {sql_query}")
    try:
        conn = sqlite3.connect(sql_db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        records = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not records:
            return f"[질문: {user_query}] 조회된 DB 데이터가 없습니다."
        else:
            db_result_text = f"[질문: {user_query}]\n조회된 컬럼: {', '.join(column_names)}\n결과 데이터:\n"
            for row in records:
                formatted_row = []
                for val in row:
                    if isinstance(val, (float, int)) and val >= 100000000:
                        jo = int(val // 1000000000000)
                        eok = int((val % 1000000000000) // 100000000)
                        val_str = f"{jo}조 {eok}억원" if jo > 0 else f"{eok}억원"
                        formatted_row.append(val_str)
                    else:
                        formatted_row.append(str(val))
                db_result_text += f"{tuple(formatted_row)}\n"
            return db_result_text
    except Exception as e:
         return f"[질문: {user_query}] SQL 실행 오류 발생: {e}"

def generate_integrated_answer(user_query, db_data_list, retrieved_contexts, model, processor):
    """수집된 모든 DB 데이터와 RAG 문서를 하나로 통합하여 최종 답변을 생성합니다."""
    context_text = ""
    
    if db_data_list:
        context_text += "[[재무 DB 조회 데이터]]\n"
        for data in db_data_list:
            context_text += f"{data}\n\n"
            
    if retrieved_contexts:
        context_text += "[[사업 내용 검색 문서 데이터]]\n"
        for i, doc in enumerate(retrieved_contexts):
            context_text += f"[문서 {i+1} (ID: {doc.metadata.get('chunk_id')})]\n{doc.page_content}\n\n"

    # 💡 핵심 수정: 분석 과정, 서론, 번호 매기기 등을 강제 차단하고 최종 정답만 출력하도록 압박
    prompt_messages = [
        {"role": "system", "content": "당신은 제공된 데이터를 바탕으로 정답만 짧게 말하는 AI입니다. [[재무 DB 조회 데이터]]와 [[사업 내용 검색 문서 데이터]]를 종합하여 질문에 답변하세요.\n\n절대 지켜야 할 규칙:\n1. '제공된 데이터를 분석합니다', '답변을 구성합니다' 같은 서론을 절대 쓰지 마세요.\n2. 데이터를 분석하는 과정이나 이유를 설명하지 마세요.\n3. 오직 최종 정답만 1~2문장의 깔끔한 한국어 평문으로 바로 출력하세요."},
        {"role": "user", "content": f"{context_text}\n[최종 사용자 질문]: {user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<answer>"
    
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=300, temp=0.1, verbose=False)
    return response.text.split("</answer>")[0].strip()

# ==========================================
# 2. 🚀 메인 파이프라인 (메모리 격리 및 통합 구조)
# ==========================================
if __name__ == "__main__":
    user_query = "2023년 스마트폰 시장 점유율은 몇 프로야? 그리고 2023년 삼성전자 영업이익은 얼마야?"
    print(f"\n[사용자 원본 질문]: {user_query}")
    print("-" * 50)

    # ---------------------------------------------------------
    # [Step 1] LLM 로드 및 라우팅/질의 분해/SQL 데이터 수집
    # ---------------------------------------------------------
    print("-> [Step 1] LLM 로드 중 (분해, 판단 및 SQL 쿼리용)...")
    model, processor = mlx_load(model_id)
    
    print("-> [Step 1-1] 질문 분해 중...")
    sub_queries = decompose_query_with_llm(user_query, model, processor)
    print(f"-> 분해된 검색어: {sub_queries}")
    
    rag_queries = []
    collected_db_data = []
    
    print("\n-> [Step 1-2] 라우터 의도 판별 및 DB 데이터 추출 중...")
    for q in sub_queries:
        route = route_query(q, model, processor)
        print(f"   [{q}] -> 판별 결과: {route}")
        
        if route == "SQL":
            db_text = get_sql_data(q, model, processor)
            collected_db_data.append(db_text)
        else:
            rag_queries.append(q)

    # SQL 쿼리 생성이 끝났으므로 LLM을 메모리에서 해제합니다.
    del model, processor
    clear_memory()

    # ---------------------------------------------------------
    # [Step 2] 임베딩/검색 모델 로드 및 정보 검색 (RAG 데이터 수집)
    # ---------------------------------------------------------
    top_k_docs = []
    if rag_queries:
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
        for q in rag_queries:
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
        
        top_k_docs = [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:8]]

        del embeddings, vector_db, dense_retriever, bm25_retriever, reranker
        clear_memory()

    # ---------------------------------------------------------
    # [Step 3] LLM 재로드 및 최종 단일 답변 합성
    # ---------------------------------------------------------
    if top_k_docs or collected_db_data:
        print("\n-> [Step 3] LLM 재로드 및 추출된 전체 데이터 통합 답변 생성 중...")
        model, processor = mlx_load(model_id)
        
        integrated_answer = generate_integrated_answer(user_query, collected_db_data, top_k_docs, model, processor)

        print("\n==================================================")
        print("[최종 통합 분석 결과]")
        print("==================================================\n")
        print(integrated_answer)
        
        del model, processor
        clear_memory()
    else:
        print("\n[최종 통합 결과] 추출된 정보가 없습니다.")