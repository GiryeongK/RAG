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
    encode_kwargs={'normalize_embeddings': True}
)

# 2. DB 존재 여부에 따른 분기 처리 (최초 1회만 임베딩)
if os.path.exists(persist_directory):
    print("-> 기존 Vector DB 폴더를 발견했습니다. 연산을 생략하고 즉시 로드합니다.")
    # 💡 무거운 임베딩 과정(from_documents) 생략하고 저장된 DB만 연결
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # BM25(키워드 검색) 구동을 위해 텍스트만 가볍게 메모리에 로드
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
        
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("-> DB 최초 적재 완료.")

# 3. 검색기(Retriever) 세팅
# 💡 복합 질문의 정보 누락을 막기 위해 1차 검색 풀을 30개로 확장
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

# chunk_id 기반 완벽한 중복 제거
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

# 점수 기준으로 내림차순 정렬
scored_docs = zip(combined_docs, scores)
sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

# 기존 [:3] 에서 [:10] 으로 변경하여 10위까지 확인
print("\n[최종 Re-ranking 결과 상위 10개]")
for i, (doc, score) in enumerate(sorted_docs[:10]):
    print(f"\n[순위 {i+1} | 적합도 점수: {score:.4f}]")
    print(f"경로: {doc.metadata.get('section_main')} > {doc.metadata.get('section_sub')} (ID: {doc.metadata.get('chunk_id')})")
    # 내용이 기니까 제목과 핵심만 짧게 출력
    print(f"내용 앞부분:\n{doc.page_content[:150]}...") 
    print("-" * 50)