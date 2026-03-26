import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
# 최신 langchain 패키지 권고에 따라 수정
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import warnings

# Langchain의 자잘한 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class VectorDBManager:
    def __init__(self, json_path="report_xml_00126380_2023/samsung_hybrid_chunks_final.json", db_dir="./chroma_db_samsung"):
        self.json_path = json_path
        self.db_dir = db_dir
        
        # 1. 임베딩 모델 로드
        print("\n[VDB] 임베딩 모델 로드 중 (BAAI/bge-m3)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'mps'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 2  # OOM 방지
            }
        )
        
        # 2. Re-ranker 로드
        print("[VDB] Re-ranker 모델 로드 중 (BAAI/bge-reranker-v2-m3)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="mps")
        
        # 3. DB 로드 및 초기화
        self._initialize_db()

    def _initialize_db(self):
        """DB 존재 여부에 따라 로드하거나 새로 생성합니다."""
        if os.path.exists(self.db_dir):
            print("[VDB] 기존 Vector DB 폴더를 발견했습니다. 연산을 생략하고 즉시 로드합니다.")
            self.vector_db = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            
            # Sparse 검색용 문서는 로드해야 함
            with open(self.json_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.documents = self._create_documents(chunks_data)
            
        else:
            print("[VDB] Vector DB가 없습니다. 최초 텍스트 임베딩 및 DB 생성을 시작합니다...")
            with open(self.json_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.documents = self._create_documents(chunks_data)
            self.vector_db = Chroma(embedding_function=self.embeddings, persist_directory=self.db_dir)
            
            # 10개 단위로 나누어 적재
            chunk_size = 10
            total_docs = len(self.documents)
            print(f"[VDB] 총 {total_docs}개의 문서를 {chunk_size}개 단위로 나누어 적재합니다.")
            
            for i in range(0, total_docs, chunk_size):
                batch_docs = self.documents[i : i + chunk_size]
                self.vector_db.add_documents(batch_docs)
                print(f"   [적재 중] {min(i + chunk_size, total_docs)} / {total_docs} 완료")
                
            print("[VDB] DB 최초 적재 완료.")
            
        # 검색기 세팅
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": 30})
        print("[VDB] Sparse Retriever 설정 중 (BM25)...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30

    def _create_documents(self, chunks_data):
        """JSON 데이터를 Langchain Document 객체로 변환합니다."""
        docs = []
        for item in chunks_data:
            clean_metadata = {k: str(v) for k, v in item.get("metadata", {}).items()}
            clean_metadata["chunk_id"] = str(item["chunk_id"])
            docs.append(Document(page_content=item["content"], metadata=clean_metadata))
        return docs

    def search_and_rerank(self, query, top_k=5, dense_weight=0.5):
        """
        주어진 쿼리에 대해 하이브리드 검색 및 Re-ranking을 수행하여 최상위 문서를 반환합니다.
        (이 메서드가 rag_pipeline에서 호출됩니다.)
        """
        # 1차 검색
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        unique_docs = {}
        for doc in dense_docs + sparse_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in unique_docs:
                unique_docs[chunk_id] = doc

        combined_docs = list(unique_docs.values())
        
        # 2차 검색 (Re-ranking)
        cross_inp = [[query, doc.page_content] for doc in combined_docs]
        scores = self.reranker.predict(cross_inp)

        scored_docs = zip(combined_docs, scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        # 상위 top_k 개 문서만 반환
        return [doc for doc, score in sorted_docs[:top_k]]

# ==========================================
# 단독 실행 시 테스트 영역
# ==========================================
if __name__ == "__main__":
    manager = VectorDBManager()
    query = "2023년에 어떤 스마트폰 신제품들을 출시했어?"
    print(f"\n[단독 테스트 질문]: {query}")
    
    results = manager.search_and_rerank(query, top_k=5)
    
    print("\n[최종 검색 결과 상위 5개]")
    for i, doc in enumerate(results):
        print(f"\n[순위 {i+1}] 경로: {doc.metadata.get('section_main')} > {doc.metadata.get('section_sub')}")
        print(f"내용 앞부분:\n{doc.page_content[:200]}...") 
        print("-" * 50)