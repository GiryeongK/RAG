import json
import os
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
import warnings

# Langchain의 자잘한 경고 무시
warnings.filterwarnings('ignore', category=UserWarning)

class MultiVectorDBManager:
    def __init__(self, json_path="multi_hybrid_chunks_final.json", db_dir="./chroma_db_multi"):
        self.json_path = json_path
        self.db_dir = db_dir
        
        # 1. 임베딩 모델 로드
        print("\n[VDB] 임베딩 모델 로드 중 (BAAI/bge-m3)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'mps'},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 2  # Mac 환경 OOM 방지용 배치 사이즈
            }
        )
        
        # 2. Re-ranker 로드
        print("[VDB] Re-ranker 모델 로드 중 (BAAI/bge-reranker-v2-m3)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=1024, device="mps")
        
        # 3. DB 로드 및 초기화
        self._initialize_db()

    def _initialize_db(self):
        if os.path.exists(self.db_dir):
            print(f"[VDB] 기존 Vector DB 폴더({self.db_dir})를 발견했습니다. 연산을 생략하고 즉시 로드합니다.")
            self.vector_db = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            
            with open(self.json_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.documents = self._create_documents(chunks_data)
            
        else:
            print("[VDB] Vector DB가 없습니다. 최초 텍스트 임베딩 및 DB 생성을 시작합니다...")
            if not os.path.exists(self.json_path):
                raise FileNotFoundError(f"에러: {self.json_path} 파일이 존재하지 않습니다.")

            with open(self.json_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            self.documents = self._create_documents(chunks_data)
            self.vector_db = Chroma(embedding_function=self.embeddings, persist_directory=self.db_dir)
            
            # OOM 방지를 위해 10개 단위로 나누어 적재
            chunk_size = 10
            total_docs = len(self.documents)
            print(f"[VDB] 총 {total_docs}개의 문서를 {chunk_size}개 단위로 나누어 임베딩/적재합니다.")
            
            for i in range(0, total_docs, chunk_size):
                batch_docs = self.documents[i : i + chunk_size]
                self.vector_db.add_documents(batch_docs)
                
                # 50개 단위로 진행률 출력
                if (i + chunk_size) % 50 == 0 or (i + chunk_size) >= total_docs:
                    print(f"   [적재 중] {min(i + chunk_size, total_docs)} / {total_docs} 완료")
                
            print("[VDB] DB 최초 임베딩 적재 완료.")
            
        # 검색기 세팅
        self.dense_retriever = self.vector_db.as_retriever(search_kwargs={"k": 30})
        print("[VDB] Sparse Retriever 설정 중 (BM25)...")
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 30

    def _create_documents(self, chunks_data):
        docs = []
        for item in chunks_data:
            clean_metadata = {k: str(v) for k, v in item.get("metadata", {}).items()}
            clean_metadata["chunk_id"] = str(item["chunk_id"])
            docs.append(Document(page_content=item["content"], metadata=clean_metadata))
        return docs

    def search_and_rerank(self, query, top_k=5):
        # 1차 검색 (Dense + Sparse 하이브리드)
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)

        # 중복 제거
        unique_docs = {}
        for doc in dense_docs + sparse_docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in unique_docs:
                unique_docs[chunk_id] = doc

        combined_docs = list(unique_docs.values())
        
        # 2차 검색 (Cross-Encoder Re-ranking)
        cross_inp = [[query, doc.page_content] for doc in combined_docs]
        scores = self.reranker.predict(cross_inp)

        scored_docs = zip(combined_docs, scores)
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in sorted_docs[:top_k]]

# ==========================================
# 단독 실행 시 다중 기업 검색 테스트
# ==========================================
if __name__ == "__main__":
    manager = MultiVectorDBManager()
    
    # 비교 검색 테스트 질의
    query = "삼성전자와 SK하이닉스의 주요 연구개발 조직이나 성과를 비교해줘."
    print(f"\n[단독 테스트 질문]: {query}")
    
    results = manager.search_and_rerank(query, top_k=5)
    
    print("\n[최종 하이브리드 검색 결과 상위 5개]")
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\n[순위 {i+1}] 기업: {meta.get('corp_name')} / 연도: {meta.get('report_year')} / 섹션: {meta.get('section_sub')}")
        # 본문 일부만 출력
        print(f"내용 앞부분:\n{doc.page_content[:200]}...") 
        print("-" * 50)