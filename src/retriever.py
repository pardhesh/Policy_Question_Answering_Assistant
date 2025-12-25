import re
from typing import List, Dict

class HybridRetriever:
    def __init__(self, vector_store, embedding_model, score_threshold=0.30):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.score_threshold = score_threshold

    def _keyword_overlap_score(self, query: str, text: str) -> int:
        query_tokens = set(re.findall(r"\w+", query.lower()))
        text_tokens = set(re.findall(r"\w+", text.lower()))
        return len(query_tokens.intersection(text_tokens))

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.embed_texts([query])

        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2
        )

        # Apply similarity threshold
        filtered_results = [
            doc for doc in dense_results
            if doc.get("score", 0) >= self.score_threshold
        ]

        if not filtered_results:
            return []

        # Keyword-based re-ranking
        reranked = sorted(
            filtered_results,
            key=lambda doc: self._keyword_overlap_score(query, doc["text"]),
            reverse=True
        )

        return reranked[:top_k]
