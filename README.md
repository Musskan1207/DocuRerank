# 📄 DocuRerank: Enterprise Hybrid-Search RAG Engine

A production-oriented Retrieval-Augmented Generation (RAG) system engineered to overcome semantic and lexical retrieval failures on enterprise documents. DocuRerank pairs dense vector retrieval with BM25 sparse keyword matching, passed through a cross-encoder re-ranking stage, and synthesized via high-throughput Groq LPU endpoints.

---

## 🏗️ Architecture Pipeline

```text
[ User Query ]
       │
       ├──► [ BM25 Sparse Index ] ──────────┐ (Lexical Candidates)
       │                                     ▼
       └──► [ Dense ChromaDB Index ] ──────► [ Reciprocal Rank Fusion / Candidate Pool ]
                                             │
                                             ▼
                                  [ FlashRank Cross-Encoder ]
                                  (Joint Query-Document Scoring)
                                             │
                                             ▼
                                    [ Top-K Contexts ]
                                             │
                                             ▼
                                  [ Groq LPU: Llama 3.x ]
                                             │
                                             ▼
                                  [ Grounded Response + Citations ]
