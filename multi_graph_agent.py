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
    """재무 도메인 온톨로지(개념망) 구축"""
    G = nx.DiGraph()
    
    # 노드 및 엣지 정의 (개념 간의 관계)
    edges = [
        ("부채", "차입금", "포함"),
        ("부채", "사채", "포함"),
        ("부채", "매입채무", "포함"),
        ("자금 조달", "차입금", "관련계정"),
        ("자금 조달", "사채", "관련계정"),
        ("자금 조달", "유상증자", "관련계정"),
        ("조달처", "금융기관", "주요출처"),
        ("조달처", "사채발행", "주요출처"),
        ("원재료", "매입액", "관련지표"),
        ("원재료", "매입채무", "관련계정")
    ]
    for u, v, relation in edges:
        G.add_edge(u, v, relation=relation)
    return G

finance_kg = build_financial_knowledge_graph()

# ==========================================
# 2. 도구(Tools) 정의
# ==========================================
def query_knowledge_graph(keyword):
    """지식 그래프에서 관련 재무 개념을 탐색하는 도구"""
    print(f"      -> [Graph DB 검색 실행]: {keyword}")
    related_concepts = []
    
    # 정확한 키워드 매칭
    if keyword in finance_kg:
        for neighbor in finance_kg.neighbors(keyword):
            relation = finance_kg[keyword][neighbor]['relation']
            related_concepts.append(f"[{relation}] {neighbor}")
            
    # 부분 매칭 (예: '부채 조달' -> '부채', '자금 조달' 등 탐색)
    else:
        for node in finance_kg.nodes():
            if node in keyword or keyword in node:
                for neighbor in finance_kg.neighbors(node):
                    relation = finance_kg[node][neighbor]['relation']
                    related_concepts.append(f"({node}의 {relation}) {neighbor}")

    if not related_concepts:
        return f"'{keyword}'에 대한 재무 개념 매핑 정보가 없습니다. 일반 검색을 진행하세요."
    
    return "연관된 재무/회계 정확한 명칭: " + ", ".join(list(set(related_concepts)))

def query_finance_db(target_info, model, processor):
    schema_info = "테이블명: finance_data\n컬럼: 회사명(TEXT), 사업연도(INTEGER), 재무제표명(TEXT), 계정명(TEXT), 당기금액(FLOAT)"
    sql_messages = [
        {"role": "system", "content": f"당신은 SQLite 쿼리 생성기입니다. 부연 설명 없이 오직 SQL 쿼리만 출력하세요. 작성된 쿼리는 ```sql 과 ``` 안에 넣으세요. 증감률 등은 비교 연도를 모두 조회하세요.\n{schema_info}"},
        {"role": "user", "content": "추출 목표: 2024년 삼성전자 영업이익"},
        {"role": "assistant", "content": "```sql\nSELECT 회사명, 사업연도, 계정명, 당기금액 FROM finance_data WHERE 회사명 = '삼성전자' AND 사업연도 = 2024 AND 계정명 LIKE '%영업이익%';\n```"},
        {"role": "user", "content": f"추출 목표: {target_info}"}
    ]
    prompt = processor.apply_chat_template(sql_messages, tokenize=False, add_generation_prompt=True)
    response = mlx_generate(model, processor, prompt=prompt, max_tokens=200, temp=0.0, verbose=False)
    
    query_match = re.search(r'```sql\n?(.*?)\n?```', response.text, re.DOTALL | re.IGNORECASE)
    sql_query = query_match.group(1).strip() if query_match else response.text.replace('```sql', '').replace('```', '').strip()
    
    print(f"      -> [SQL 자동 실행]: {sql_query.replace(chr(10), ' ')}")
    
    try:
        conn = sqlite3.connect(sql_db_path)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        records = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not records:
            return "조회된 DB 데이터가 없습니다."
        
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
        return f"SQL 오류: {e}"

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
    system_prompt = """당신은 팩트와 교차 검증을 중시하는 전문 재무 분석 AI 에이전트입니다.
사용할 수 있는 도구(Tool)는 세 가지입니다.

1. query_knowledge_graph(검색어): 추상적인 단어(예: 자금 조달, 원재료 현금)를 검색하여 정확한 회계 계정명(예: 차입금, 사채)을 찾을 때 최우선으로 사용합니다.
2. query_finance_db(검색어): DB에 명시된 재무 수치 추출용.
3. search_business_report(검색어): 주석 및 사업보고서 텍스트 추출용.

[행동 규칙]
- 도구 호출 형식: [CALL: 도구이름("검색어")] (한 번에 여러 도구 동시 호출 가능)
- 모호한 재무 개념이 포함된 질문은 반드시 query_knowledge_graph를 먼저 호출하여 정확한 키워드를 파악한 후, 그 키워드로 DB나 RAG를 검색하십시오.
- RAG 검색 시 해당 문서의 '기준 연도'를 철저히 검증하십시오. 사용자가 요구한 연도의 데이터가 아니라면 지어내지 말고 '해당 연도 정보 없음'으로 처리하십시오.
- 충분한 정보를 모아 팩트 기반의 답변이 가능할 때만 [FINAL: 최종 답변]을 출력하세요.
- 출력 속도 최적화를 위해 [Agent Thought]는 반드시 한국어로 3문장 이내로 극도로 짧게 작성하십시오. 길게 생각하지 마십시오.
- query_finance_db 호출 시, 여러 계정명(예: 차입금, 유상증자)을 띄어쓰기로 묶어서 한 번에 호출하면 DB에서 절대 찾을 수 없습니다. 반드시 [CALL: query_finance_db("삼성전자 2024 차입금")], [CALL: query_finance_db("삼성전자 2024 유상증자")] 처럼 각각 분리하여 개별 호출하십시오.

[진행 과정]"""
    
    conversation_history = f"User: {user_query}\n"
    max_iterations = 8 
    
    for i in range(max_iterations):
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation_history}
        ]
        prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        response = mlx_generate(model, processor, prompt=prompt, max_tokens=1500, temp=0.0, verbose=False)
        agent_output = response.text.strip()
        
        print(f"\n[Agent Thought]\n{agent_output}")
        
        if "[FINAL:" in agent_output:
            final_answer = agent_output[agent_output.find("[FINAL:") + 7 :].rstrip("]")
            return final_answer.strip()
            
        elif "[CALL:" in agent_output:
            calls = re.findall(r'\[CALL:\s*([a-zA-Z_]+)\(["\'](.*?)["\']\)\]', agent_output)
            
            if not calls:
                conversation_history += f"Agent: {agent_output}\nObservation: 도구 호출 파싱 오류. 형식을 지켜주세요.\n"
                continue
            
            observations = []
            for tool_name, tool_arg in calls:
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
            conversation_history += f"Agent: {agent_output}\nObservation: 규칙 위반. [CALL: ...] 또는 [FINAL: ...] 형식을 사용하십시오.\n"
            
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
    
    user_query = "24년 삼성전자와 SK하이닉스의 자금 조달 규모는 23년 대비 각각 얼마나 증감했어? 그리고 확보한 자금을 바탕으로 각 회사가 24년 사업보고서에서 밝힌 주요 시설투자(CAPEX) 목적은 뭐야?"
    print(f"\n[사용자 질의]: {user_query}")
    print("-" * 50)
    
    final_result = run_agent(user_query, model, processor, dense_retriever, bm25_retriever, reranker)
    
    print("\n==================================================")
    print("[최종 팩트 출력]")
    print("==================================================\n")
    print(final_result)