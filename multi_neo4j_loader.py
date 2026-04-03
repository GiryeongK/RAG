import json
import re
import os
import ast
from dotenv import load_dotenv
from neo4j import GraphDatabase
from mlx_vlm import load, generate

# ==========================================
# 1. 환경 변수 및 Neo4j DB 연결 설정
# ==========================================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_URI or not NEO4J_PASSWORD:
    print("에러: .env 파일에 NEO4J_URI 또는 NEO4J_PASSWORD가 설정되지 않았습니다.")
    exit()

print("Neo4j AuraDB 클라우드 연결 시도 중...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("✅ Neo4j 연결 성공")
except Exception as e:
    print(f"❌ Neo4j 연결 실패: {e}")
    exit()

# ==========================================
# 2. 로컬 LLM 로드
# ==========================================
model_id = "mlx-community/Qwen3.5-9B-8bit"
print(f"\n모델 로드 중... ({model_id})")
model, processor = load(model_id)

def extract_graph_data(text_chunk, corp_name, year):
    # 강제 제약을 배제하고 역할 부여 및 예시 모방을 자연스럽게 유도
    system_prompt = f"""당신은 텍스트에서 중심 개체와 관계를 분석하여 구조화된 JSON 데이터로 깔끔하게 정리해 주는 전문가입니다.
항상 아래의 [출력 예시]와 동일한 형태의 JSON 객체로 답변을 제공합니다.

[분석 기준]
1. Node label: Company, Product, Market, Technology, Competitor, Subsidiary 중에서 가장 적절한 것을 선택합니다.
2. Edge type: PRODUCES, COMPETES_WITH, OPERATES_IN, INVESTS_IN, OWNS, DEVELOPS 중에서 선택합니다.
3. 텍스트에 등장하는 '당사', '동사', '연결실체'는 '{corp_name}'(으)로 기록합니다.
4. 추출할 대상이 텍스트에 없다면 빈 노드와 엣지를 가진 JSON을 작성하여 반환합니다.

[출력 예시]
```json
{{
  "nodes": [
    {{"id": "{corp_name}", "label": "Company"}},
    {{"id": "메모리반도체", "label": "Product"}}
  ],
  "edges": [
    {{"source": "{corp_name}", "target": "메모리반도체", "type": "PRODUCES"}}
  ]
}}
```"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"다음 텍스트를 분석하여 JSON으로 정리해 주세요.\n\n텍스트: {text_chunk}"}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response = generate(model, processor, prompt=prompt, max_tokens=1500, temp=0.0, verbose=False)
    
    raw_output = response if isinstance(response, str) else getattr(response, 'text', str(response))
    raw_output = raw_output.strip()
    
    # 디버깅 출력 (출력량이 너무 길면 콘솔이 지저분해지므로 앞부분만 확인)
    print(f"      [LLM 출력 원문 미리보기]: {raw_output[:150]}...") 
    
    start_idx = raw_output.find('{')
    end_idx = raw_output.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = raw_output[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_str)
            except Exception as e:
                print(f"      ❌ [파싱 실패 사유]: {e}")
                # 파싱 실패 시 원문 전체를 출력하여 어디서 잘렸는지 확인
                print(f"      ❌ [에러 원문 전체]:\n{raw_output}")
                return {"nodes": [], "edges": []}
    else:
        return {"nodes": [], "edges": []}

# ==========================================
# 4. Neo4j 클라우드 적재 로직 (MERGE 쿼리)
# ==========================================
def insert_into_neo4j(graph_data):
    def tx_logic(tx, data):
        for node in data.get("nodes", []):
            label = node.get("label", "Unknown")
            node_id = node.get("id", "").strip()
            if not node_id: continue
            
            query = f"MERGE (n:{label} {{id: $id}})"
            tx.run(query, id=node_id)
            
        for edge in data.get("edges", []):
            source = edge.get("source", "").strip()
            target = edge.get("target", "").strip()
            rel_type = edge.get("type", "RELATED_TO").upper()
            rel_type = re.sub(r'[^A-Z_]', '', rel_type)
            
            if not source or not target or not rel_type: continue
            
            query = f"""
            MATCH (a {{id: $source}})
            MATCH (b {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            """
            tx.run(query, source=source, target=target)

    with driver.session() as session:
        session.execute_write(tx_logic, graph_data)

# ==========================================
# 5. 메인 실행 블록
# ==========================================
if __name__ == "__main__":
    chunk_file = "multi_hybrid_chunks_final.json"
    
    if not os.path.exists(chunk_file):
        print(f"에러: {chunk_file} 파일이 없습니다.")
        driver.close()
        exit()

    with open(chunk_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"\n총 {len(chunks)}개 청크 중 상위 10개로 1차 적재 테스트를 진행합니다.")
    
    for i, chunk in enumerate(chunks[:10]):
        metadata = chunk.get("metadata", {})
        corp_name = metadata.get("corp_name", "Unknown")
        year = metadata.get("report_year", "Unknown")
        text_content = chunk.get("content", "")
        
        # 🚨 [디버깅]: 어떤 텍스트가 들어가고 있는지 힌트 출력
        clean_hint = text_content[:60].replace('\n', ' ')
        print(f"\n[{i+1}/10] {corp_name} {year}년도 분석 중... (입력 텍스트: {clean_hint}...)")
        
        if len(text_content) < 50:
            print("  -> 텍스트가 너무 짧아 스킵합니다.")
            continue
            
        extracted_data = extract_graph_data(text_content, corp_name, year)
        
        if extracted_data.get("nodes"):
            insert_into_neo4j(extracted_data)
            print(f"  -> 성공: 노드 {len(extracted_data['nodes'])}개, 관계 {len(extracted_data['edges'])}개 적재.")
        else:
            print("  -> 추출된 그래프 데이터가 없습니다.")

    driver.close()
    print("\n[테스트 종료] 데이터베이스 연결이 안전하게 해제되었습니다.")