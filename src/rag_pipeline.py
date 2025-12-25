import os
from typing import List

from groq import Groq

from src.load_data import load_documents
from src.chunking import chunk_documents
from src.embedding import EmbeddingModel
from src.vector_store import FAISSVectorStore
from src.retriever import HybridRetriever
from src.prompts import (
    IMPROVED_SYSTEM_PROMPT,
    IMPROVED_USER_PROMPT
)


class RAGPipeline:
    def __init__(
        self,
        data_dir: str = "data",
        model_name: str = "llama-3.1-8b-instant",
        score_threshold: float = 0.30
    ):
        self.data_dir = data_dir
        self.model_name = model_name
        self.score_threshold = score_threshold

        # Initialize LLM client (Groq)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.llm = Groq(api_key=api_key)
        
        # Build vector store ONCE
        self.embedding_model = EmbeddingModel()
        self.vector_store = self._build_vector_store()
        self.retriever = HybridRetriever(
            self.vector_store,
            self.embedding_model,
            score_threshold=self.score_threshold
        )

    def _build_vector_store(self) -> FAISSVectorStore:
        """
        One-time indexing step:
        Load documents → chunk → embed → store in FAISS
        """
        documents = load_documents(self.data_dir)
        chunks = chunk_documents(documents)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        vector_store = FAISSVectorStore(embedding_dim=embeddings.shape[1])
        vector_store.add(embeddings, chunks)

        return vector_store

    def _format_context(self, retrieved_docs: List[dict]) -> str:
        """
        Format retrieved chunks with source + chunk index
        """
        formatted_chunks = []
        for idx, doc in enumerate(retrieved_docs):
            formatted_chunks.append(
                f"[Source: {doc['source']}, chunk {idx}]\n{doc['text']}"
            )

        return "\n\n".join(formatted_chunks)

    def answer_question(self, question: str) -> str:
        """
        Main query-time method:
        Retrieve → threshold → LLM OR refuse
        """
        if len(question.strip()) < 3:
            return "No relevant policy information was found in the provided documents."
        retrieved_docs = self.retriever.retrieve(question)

        # Threshold-based refusal (NO LLM CALL)
        if not retrieved_docs:
            return "No relevant policy information was found in the provided documents."

        context = self._format_context(retrieved_docs)

        user_prompt = IMPROVED_USER_PROMPT.format(
            context=context,
            question=question
        )

        messages = [
            {"role": "system", "content": IMPROVED_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )

        return response.choices[0].message.content.strip()
