import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

json_path = "report_xml_00126380_2023/samsung_hybrid_chunks_final.json"
persist_directory = "./chroma_db_samsung"

# 1. 임베딩 모델 로드 (공통)
print("1. 임베딩 모델 로드 중 (BAAI/bge-m3)...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': 'mps'},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 2  # 💡 [핵심 방어 1] OOM 방지: 한 번에 연산하는 임베딩 배치 수를 2개로 극단적 축소
    }
)

# 2. DB 존재 여부에 따른 분기 처리 (최초 1회만 임베딩)
if os.path.exists(persist_directory):
    print("-> 기존 Vector DB 폴더를 발견했습니다. 연산을 생략하고 즉시 로드합니다.")
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    documents = []
    for item in chunks_data:
        clean_metadata = {k: str(v) for k, v in item.get("metadata", {}).items()}
        clean_metadata["chunk_id"] = str(item["chunk_id"])
        documents.append(Document(page_content=item["content"], metadata=clean_metadata))
else:
    print("-> Vector DB가 없습니다. 최초 텍스트 임베딩 및 DB 생성을 시작합니다...")
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    documents = []
    for item in chunks_data:
        clean_metadata = {k: str(v) for k, v in item.get("metadata", {}).items()}
        clean_metadata["chunk_id"] = str(item["chunk_id"])
        documents.append(Document(page_content=item["content"], metadata=clean_metadata))
        
    # 💡 [핵심 방어 2] 391개를 한 번에 던지지 않고 DB 껍데기를 먼저 생성
    vector_db = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    # 문서를 10개씩 잘라서 순차적으로 DB에 밀어 넣음 (메모리 스파이크 차단)
    chunk_size = 10
    total_docs = len(documents)
    print(f"-> 총 {total_docs}개의 문서를 {chunk_size}개 단위로 나누어 적재합니다.")
    
    for i in range(0, total_docs, chunk_size):
        batch_docs = documents[i : i + chunk_size]
        vector_db.add_documents(batch_docs)
        print(f"   [적재 중] {min(i + chunk_size, total_docs)} / {total_docs} 완료")
        
    print("-> DB 최초 적재 완료.")

# 3. 검색기(Retriever) 세팅
dense_retriever = vector_db.as_retriever(search_kwargs={"k": 30})

print("2. Sparse Retriever 설정 중 (BM25)...")
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 30

# 4. Re-ranking 모델 로드
print("3. Re-ranker 모델 로드 중 (BAAI/bge-reranker-v2-m3)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="mps")

# ==========================================
# 🚀 [테스트] 검색 (Retrieval) 실행
# ==========================================
query = "2023년에 어떤 스마트폰 신제품들을 출시했어?"

print(f"\n[질문]: {query}")
print("-" * 50)

# 1차 검색: 하이브리드 병합
print("-> 1차 하이브리드 검색 진행 (Dense 30개 + Sparse 30개 결합)...")
dense_docs = dense_retriever.invoke(query)
sparse_docs = bm25_retriever.invoke(query)

unique_docs = {}
for doc in dense_docs + sparse_docs:
    chunk_id = doc.metadata.get("chunk_id")
    if chunk_id and chunk_id not in unique_docs:
        unique_docs[chunk_id] = doc

combined_docs = list(unique_docs.values())
print(f"-> 1차 검색 완료: 총 {len(combined_docs)}개 문서 추출됨 (중복 제거)")

# 2차 검색: Cross-Encoder를 이용한 Re-ranking
print("-> 2차 Re-ranking 연산 중...")
cross_inp = [[query, doc.page_content] for doc in combined_docs]
scores = reranker.predict(cross_inp)

scored_docs = zip(combined_docs, scores)
sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

print("\n[최종 Re-ranking 결과 상위 5개]")
for i, (doc, score) in enumerate(sorted_docs[:5]):
    print(f"\n[순위 {i+1} | 적합도 점수: {score:.4f}]")
    print(f"경로: {doc.metadata.get('section_main')} > {doc.metadata.get('section_sub')} (ID: {doc.metadata.get('chunk_id')})")
    print(f"내용 앞부분:\n{doc.page_content[:200]}...") 
    print("-" * 50)