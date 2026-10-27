import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest

class HybridRetriever:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

        # Sparse BM25 Setup
        tokenized_corpus = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense ChromaDB Setup
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="doc_store")
        
        embeddings = self.embed_model.encode(chunks).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    def retrieve_and_rerank(self, query: str, top_k: int = 4) -> list[str]:
        # 1. Vector Search
        query_emb = self.embed_model.encode([query]).tolist()
        dense_results = self.collection.query(
            query_embeddings=query_emb, 
            n_results=min(10, len(self.chunks))
        )
        dense_chunks = dense_results["documents"][0] if dense_results["documents"] else []

        # 2. BM25 Search
        bm25_chunks = self.bm25.get_top_n(
            query.lower().split(), 
            self.chunks, 
            n=min(10, len(self.chunks))
        )

        # 3. Merge & Deduplicate
        candidate_pool = list(set(dense_chunks + bm25_chunks))

        # 4. Cross-Encoder Re-rank
        passages = [{"id": idx, "text": chunk} for idx, chunk in enumerate(candidate_pool)]
        rerank_req = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(rerank_req)

        return [item["text"] for item in ranked[:top_k]]