import json
import os
import gc
import torch
import sqlite3
import re
import networkx as nx
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from mlx_vlm import load as mlx_load, generate as mlx_generate
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# 전역 설정
json_path = "multi_hybrid_chunks_final.json"
persist_directory = "./chroma_db_multi"
sql_db_path = "multi_finance.db"
model_id = "mlx-community/Qwen3.5-9B-8bit"

def clear_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# ==========================================
# 1. 지식 그래프 (Knowledge Graph) 구축
# ==========================================
def build_financial_knowledge_graph():
    G = nx.DiGraph()
    edges = [
        ("부채", "차입금", "포함"),
        ("부채", "사채", "포함"),
        ("부채", "매입채무", "포함"),
        ("자금 조달", "차입금", "관련계정"),
        ("자금 조달", "사채", "관련계정"),
        ("자금 조달", "유상증자", "관련계정"),
        ("CAPEX", "설비투자", "동의어"),
        ("CAPEX", "유형자산", "관련계정"),
        ("주당순이익", "주당이익", "동의어"),
        ("EPS", "주당이익", "동의어"),
        ("당기순이익", "당기이익", "동의어")
    ]
    for u, v, relation in edges:
        G.add_edge(u, v, relation=relation)
    return G

finance_kg = build_financial_knowledge_graph()

# ==========================================
# 2. 도구(Tools) 정의
# ==========================================
def query_knowledge_graph(keyword):
    print(f"      -> [Graph DB 검색 실행]: {keyword}")
    related_concepts = []
    if keyword in finance_kg:
        for neighbor in finance_kg.neighbors(keyword):
            relation = finance_kg[keyword][neighbor]['relation']
            related_concepts.append(f"[{relation}] {neighbor}")
    else:
        for node in finance_kg.nodes():
            if node in keyword or keyword in node:
                for neighbor in finance_kg.neighbors(node):
                    relation = finance_kg[node][neighbor]['relation']
                    related_concepts.append(f"({node}의 {relation}) {neighbor}")

    if not related_concepts:
        return f"'{keyword}'에 대한 재무 개념 매핑 정보가 없습니다. 원래 단어로 query_finance_db를 검색해보세요."
    
    hint = "\n(안내: 이 정확한 명칭을 사용하여 query_finance_db를 호출하면 실제 재무 수치와 금액을 확인할 수 있습니다.)"
    return "연관된 재무/회계 정확한 명칭: " + ", ".join(list(set(related_concepts))) + hint

def query_finance_db(target_info, model, processor):
    schema_info = "테이블명: finance_data\n컬럼: 회사명(TEXT), 사업연도(INTEGER), 재무제표명(TEXT), 계정명(TEXT), 당기금액(FLOAT)"
    
    sql_messages = [
        {"role": "system", "content": f"당신은 사용자의 질문을 SQLite 쿼리로 번역해주는 데이터베이스 전문가입니다.\n사용자가 데이터를 빠르게 확인할 수 있도록, 불필요한 설명은 생략하고 즉시 실행 가능한 ```sql 쿼리문만 제공해 주는 것이 당신의 훌륭한 역할입니다.\n\n{schema_info}"},
        {"role": "user", "content": "추출 목표: 2024년 삼성전자 영업이익"},
        {"role": "assistant", "content": "```sql\nSELECT 회사명, 사업연도, 계정명, 당기금액 FROM finance_data WHERE 회사명 = '삼성전자' AND 사업연도 = 2024 AND 계정명 LIKE '%영업이익%';\n```"},
        {"role": "user", "content": "추출 목표: 2025년 SK하이닉스 매출액 및 주당이익"},
        {"role": "assistant", "content": "```sql\nSELECT 회사명, 사업연도, 계정명, 당기금액 FROM finance_data WHERE 회사명 = 'SK하이닉스' AND 사업연도 = 2025 AND (계정명 LIKE '%매출액%' OR 계정명 LIKE '%주당이익%');\n```"},
        {"role": "user", "content": f"추출 목표: {target_info}"}
    ]
    prompt = processor.apply_chat_template(sql_messages, tokenize=False, add_generation_prompt=True)
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=150, temp=0.0, verbose=False)
    
    query_match = re.search(r'```sql\n?(.*?)\n?```', response.text, re.DOTALL | re.IGNORECASE)
    if query_match:
        sql_query = query_match.group(1).strip()
    else:
        sql_match = re.search(r'SELECT.*?;', response.text, re.DOTALL | re.IGNORECASE)
        sql_query = sql_match.group(0).strip() if sql_match else response.text.strip()
        
    print(f"      -> [SQL 자동 실행]: {sql_query.replace(chr(10), ' ')}")
    
    try:
        conn = sqlite3.connect(sql_db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        records = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not records:
            return "조회된 DB 데이터가 없습니다. (계정명을 query_knowledge_graph로 먼저 확인해보는 것을 추천합니다.)"
        
        result_str = f"컬럼: {', '.join(column_names)}\n"
        for row in records:
            formatted_row = []
            for val in row:
                if isinstance(val, (float, int)) and val >= 100000000:
                    jo = int(val // 1000000000000)
                    eok = int((val % 1000000000000) // 100000000)
                    val_str = f"{jo}조 {eok}억원" if jo > 0 else f"{eok}억원"
                    if val < 0: val_str = "-" + val_str
                    formatted_row.append(val_str)
                else:
                    formatted_row.append(str(val))
            result_str += f"{tuple(formatted_row)}\n"
        return result_str
    except Exception as e:
        return f"SQL 오류: {e} (검색어를 더 단순하게 수정해보세요)"

def search_business_report(query, dense_retriever, bm25_retriever, reranker):
    print(f"      -> [RAG 검색 실행]: {query}")
    d_docs = dense_retriever.invoke(query)
    s_docs = bm25_retriever.invoke(query)
    
    unique_docs = {}
    for doc in d_docs + s_docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id and chunk_id not in unique_docs:
            unique_docs[chunk_id] = doc
            
    combined_docs = list(unique_docs.values())
    if not combined_docs:
        return "검색된 문서가 없습니다."
        
    cross_inp = [[query, doc.page_content] for doc in combined_docs]
    scores = reranker.predict(cross_inp)
    scored_docs = zip(combined_docs, scores)
    top_k_docs = [doc for doc, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)[:5]]
    
    result_str = ""
    for i, doc in enumerate(top_k_docs):
        result_str += f"[문서 {i+1}] {doc.page_content}\n"
    return result_str

# ==========================================
# 3. 에이전트 루프 (Agent Loop)
# ==========================================
def run_agent(user_query, model, processor, dense_retriever, bm25_retriever, reranker):
    # 💡 역할극 제거 및 SOP(작업 표준 절차) 명문화
    system_prompt = """당신은 사용자의 질문을 분석하여, 사전에 구축된 하이브리드 데이터베이스(RAG 기반 텍스트 DB, SQL 기반 정형 DB)에서 정확한 정보를 추출하고 통합하는 데이터 에이전트입니다. 
정확하고 신뢰성 있는 정보 제공을 위해, 당신의 사전 지식보다는 도구(Tool)를 통해 직접 조회된 팩트(Observation)를 우선하여 답변을 구성해 주세요.

[사용 가능한 도구]
1. query_knowledge_graph("검색어"): 일상적/추상적 재무 용어(예: 주당순이익)를 실제 DB에 저장된 공식 계정명(예: 주당이익)으로 변환해주는 매핑 도구입니다. (수치 데이터 없음)
2. query_finance_db("검색어"): 공식 계정명을 사용하여 SQL DB에서 실제 재무 수치(당기금액)를 추출하는 도구입니다.
3. search_business_report("검색어"): 사업보고서 내 텍스트 및 주석 정보를 검색하는 RAG 도구입니다.

[작업 표준 절차 (SOP) - 권장 진행 순서]
1. 용어 확인: 필요한 경우 [CALL: query_knowledge_graph("검색어")]를 호출하여 DB 검색에 사용할 정확한 계정명을 확인합니다.
2. 데이터 검색: 확인된 계정명 또는 키워드를 바탕으로 [CALL: query_finance_db("검색어")] 또는 [CALL: search_business_report("검색어")]를 호출하여 실제 데이터를 요청합니다.
3. 결과 대기: 도구를 호출(CALL)한 후에는 시스템이 'Observation'으로 실제 검색 결과를 반환할 때까지 텍스트 생성을 멈추고 대기합니다.
4. 최종 도출: 시스템이 반환한 데이터가 모두 수집되면, 해당 데이터를 조합하여 [FINAL: 최종 답변] 형식으로 마크다운 표와 함께 결과를 작성합니다.

[환경 맥락]
우리 DB에는 2024년 실적뿐만 아니라 2025년 등의 미래 목표치/추정치도 '당기금액'으로 저장되어 있습니다. 연도에 구애받지 말고 절차대로 DB를 편안하게 조회해 주시면 됩니다."""
    
    conversation_history = f"User: {user_query}\n"
    max_iterations = 6 
    
    for i in range(max_iterations):
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_history}
        ]
        prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        response = mlx_generate(model, processor, prompt=prompt, max_tokens=1500, temp=0.0, verbose=False)
        agent_output = response.text.strip()
        
        if "Observation:" in agent_output:
            agent_output = agent_output.split("Observation:")[0].strip()
            
        print(f"\n[Agent Thought & Action]\n{agent_output}")
        
        if "[FINAL:" in agent_output:
            final_idx = agent_output.rfind("[FINAL:")
            final_answer = agent_output[final_idx + 7 :].rstrip("]")
            return final_answer.strip()
            
        elif "[CALL:" in agent_output:
            calls = re.findall(r'\[CALL:\s*([a-zA-Z_]+)\(["\'](.*?)["\']\)\]', agent_output)
            
            if not calls:
                conversation_history += f"Agent: {agent_output}\nObservation: 호출 형식이 맞지 않습니다. [CALL: 도구이름(\"검색어\")] 형식을 지켜서 다시 진행해주세요.\n"
                continue
            
            unique_calls = []
            seen = set()
            for call in calls:
                if call not in seen:
                    unique_calls.append(call)
                    seen.add(call)
            
            observations = []
            for tool_name, tool_arg in unique_calls[:3]: 
                if tool_name == "query_knowledge_graph":
                    obs = query_knowledge_graph(tool_arg)
                elif tool_name == "query_finance_db":
                    obs = query_finance_db(tool_arg, model, processor)
                elif tool_name == "search_business_report":
                    obs = search_business_report(tool_arg, dense_retriever, bm25_retriever, reranker)
                else:
                    obs = f"오류: {tool_name}은(는) 없는 도구입니다."
                observations.append(f"[{tool_name}(\"{tool_arg}\") 결과]:\n{obs}")
                
            combined_observation = "\n\n".join(observations)
            conversation_history += f"Agent: {agent_output}\nObservation:\n{combined_observation}\n"
        else:
            conversation_history += f"Agent: {agent_output}\nObservation: 데이터 추출이 더 필요하면 도구를 호출하고, 수집이 완료되었으면 SOP 4단계에 따라 [FINAL: 최종 답변] 형식으로 출력하여 답변을 마무리해주세요.\n"
            
    return "최대 탐색 횟수를 초과하여 답변 도출을 중단했습니다."

# ==========================================
# 4. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    print("-> [System] 모델 및 데이터베이스 초기화 (1회만 실행)...")
    model, processor = mlx_load(model_id)
    
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
    
    user_query = "삼성전자의 2025년 매출액, 영업이익, 주당순이익은 얼마야?"
    print(f"\n[사용자 질의]: {user_query}")
    print("-" * 50)
    
    final_result = run_agent(user_query, model, processor, dense_retriever, bm25_retriever, reranker)
    
    print("\n==================================================")
    print("[최종 팩트 출력]")
    print("==================================================\n")
    print(final_result)