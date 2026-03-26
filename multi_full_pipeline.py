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

# 다중 문서 환경으로 설정값 변경
json_path = "multi_hybrid_chunks_final.json"
persist_directory = "./chroma_db_multi"
sql_db_path = "multi_finance.db"
model_id = "mlx-community/Qwen3.5-9B-8bit"

def clear_memory():
    """RAM 및 Mac GPU(MPS) 캐시 강제 비움 (OOM 방지)"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==========================================
# 1. LLM 에이전트 파이프라인 정의
# ==========================================
def decompose_query_with_llm(user_query, model, processor):
    prompt_messages = [
        {"role": "system", "content": "사용자의 복합 질문을 독립적인 의미를 갖는 짧은 구문(Phrase)들로 분해하여 쉼표로 구분하세요. 단일 단어로 무의미하게 쪼개는 것을 절대 금지합니다. 반드시 '연도, 기업명, 검색 목적'이 하나의 구문에 유지되어야 합니다.\n[예시]\n질문: 2024년 삼성전자와 SK하이닉스의 매출액을 각각 알려줘. 그리고 두 회사의 연구개발 조직 구성은 어떻게 달라?\n출력: 2024년 삼성전자 매출액, 2024년 SK하이닉스 매출액, 삼성전자와 SK하이닉스 연구개발 조직 구성 비교"},
        {"role": "user", "content": f"{user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<result>"
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=100, temp=0.0, verbose=False)
    clean_text = response.text.split("</result>")[0].strip()
    clean_queries = [q.strip() for q in clean_text.split(',') if q.strip()]
    return clean_queries if clean_queries else [user_query]

def route_query(user_query, model, processor):
    prompt_messages = [
        {"role": "system", "content": "질문에 '영업이익', '매출액', '자산', '부채', '자본', '재무상태' 등 재무 수치를 묻는 단어가 포함되어 있으면 무조건 'SQL'이라고만 답하세요. 그 외의 모든 질문(시장 점유율, 출시 제품, 조직, 사업 내용 등)은 무조건 'RAG'라고만 답하세요. 부연 설명 금지."},
        {"role": "user", "content": f"{user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<route>"
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=100, temp=0.0, verbose=False)
    route_result = response.text.split("</route>")[0].strip().upper()
    return "SQL" if "SQL" in route_result else "RAG"

def get_sql_data(user_query, model, processor):
    schema_info = """
    [SQLite 스키마 정보]
    테이블명: finance_data
    컬럼:
    - 회사명 (TEXT): 예) '삼성전자', 'SK하이닉스'
    - 사업연도 (INTEGER): 예) 2023, 2024, 2025
    - 재무제표명 (TEXT): 예) 손익계산서, 재무상태표
    - 계정명 (TEXT): 예) 매출액, 영업이익, 자산총계
    - 당기금액 (FLOAT): 원 단위 (예: 6566976000000.0)
    """
    
    sql_messages = [
        {"role": "system", "content": f"당신은 데이터 분석가입니다. 스키마를 바탕으로 정확한 SQLite 쿼리를 작성하세요. 마크다운 ```sql 과 ``` 안에 쿼리문만 작성하세요. 쿼리는 다중 기업 비교를 고려하여 작성하며, LIKE '%계정명%' 형태로 유연하게 검색하세요.\n{schema_info}"},
        {"role": "user", "content": user_query}
    ]
    prompt = processor.apply_chat_template(sql_messages, tokenize=False, add_generation_prompt=True)
    
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=200, temp=0.0, verbose=False)
    raw_output = response.text
    
    query_match = re.search(r'```sql\n(.*?)\n```', raw_output, re.DOTALL)
    sql_query = query_match.group(1).strip() if query_match else raw_output.replace('```sql', '').replace('```', '').strip()

    try:
        conn = sqlite3.connect(sql_db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        records = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not records:
            return f"[질문: {user_query}]\n조회된 DB 데이터가 없습니다. (실행 쿼리: {sql_query})"
        else:
            db_result_text = f"[[DB 추출 결과]]\n- 질문: {user_query}\n- 컬럼: {', '.join(column_names)}\n결과 데이터:\n"
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
         return f"[질문: {user_query}] SQL 실행 오류 발생: {e}\n생성된 쿼리: {sql_query}"

def generate_integrated_answer(user_query, db_data_list, retrieved_contexts, model, processor):
    context_text = ""
    
    if db_data_list:
        context_text += "[[재무 DB 조회 데이터]]\n"
        for data in db_data_list:
            context_text += f"{data}\n\n"
            
    if retrieved_contexts:
        context_text += "[[사업 내용 검색 문서 데이터]]\n"
        for i, doc in enumerate(retrieved_contexts):
            context_text += f"[문서 {i+1} (ID: {doc.metadata.get('chunk_id')})]\n{doc.page_content}\n\n"

    # 사용자 요청에 따른 강력한 제약사항 (팩트 기반, 지어내기 금지)
    prompt_messages = [
        {"role": "system", "content": "당신은 제공된 데이터를 바탕으로 정답만 짧게 말하는 AI입니다. 문서에 없는 내용은 절대 지어내지 말고 '정보 부족'이라고 답하세요. 데이터를 분석하는 과정이나 이유를 설명하지 말고 최종 정답만 출력하세요."},
        {"role": "user", "content": f"{context_text}\n[최종 사용자 질문]: {user_query}"}
    ]
    prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt += "<answer>"
    
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=600, temp=0.1, verbose=False)
    return response.text.split("</answer>")[0].strip()

# ==========================================
# 2. 메인 실행 블록 (메모리 격리 구조)
# ==========================================
if __name__ == "__main__":
    # 다중 문서 환경을 검증하기 위한 복합 질문 세팅
    user_query = "2024년 삼성전자와 SK하이닉스의 매출액을 각각 알려줘. 23년 대비 얼마나 증가했어? 24년 영업이익은 얼마야? 그리고 두 회사의 연구개발(R&D) 조직 구성은 어떻게 달라?"
    print(f"\n[사용자 원본 질문]: {user_query}")
    print("-" * 50)

    # ---------------------------------------------------------
    # [Step 1] LLM 로드 및 분해/라우팅/SQL 연산
    # ---------------------------------------------------------
    print("-> [Step 1] LLM 로드 중 (의도 판별 및 SQL 쿼리용)...")
    model, processor = mlx_load(model_id)
    
    sub_queries = decompose_query_with_llm(user_query, model, processor)
    print(f"-> 분해된 질의: {sub_queries}")
    
    rag_queries = []
    collected_db_data = []
    
    for q in sub_queries:
        route = route_query(q, model, processor)
        print(f"   [{q}] -> 라우팅: {route}")
        
        if route == "SQL":
            db_text = get_sql_data(q, model, processor)
            collected_db_data.append(db_text)
        else:
            rag_queries.append(q)

    # 메모리 환원
    del model, processor
    clear_memory()

    # ---------------------------------------------------------
    # [Step 2] 하이브리드 검색 (RAG)
    # ---------------------------------------------------------
    top_k_docs = []
    if rag_queries:
        print("\n-> [Step 2] 임베딩/리랭커 모델 로드 및 문서 검색...")
        if not os.path.exists(persist_directory):
            raise FileNotFoundError("Vector DB 폴더가 없습니다.")

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
        print(f"-> 검색된 고유 문서 수: {len(combined_docs)}개")

        cross_inp = [[user_query, doc.page_content] for doc in combined_docs]
        scores = reranker.predict(cross_inp)
        scored_docs = zip(combined_docs, scores)
        
        # 다중 비교 질문이므로 맥락 보존을 위해 상위 10개 문서를 추출 (기존 8개)
        top_k_docs = [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:10]]

        # 메모리 환원
        del embeddings, vector_db, dense_retriever, bm25_retriever, reranker
        clear_memory()

    # ---------------------------------------------------------
    # [Step 3] LLM 재로드 및 최종 답변 생성
    # ---------------------------------------------------------
    if top_k_docs or collected_db_data:
        print("\n-> [Step 3] LLM 재로드 및 데이터 통합 답변 생성 중...")
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