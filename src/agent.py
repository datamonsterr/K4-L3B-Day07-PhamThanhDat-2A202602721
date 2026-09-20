from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Knowledge base is empty — no context to answer from."

        # retrieve top-k relevant chunks
        results = self.store.search(question, top_k=top_k)

        # build prompt with retrieved chunks
        prompt = self._build_prompt(question, [r["content"] for r in results])
        return self._llm_fn(prompt)

    def _build_prompt(self, question: str, chunks: list[str]) -> str:
        context = "\n\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(chunks, start=1))
        return (
            "You are a knowledge base assistant. Answer the question using ONLY "
            "the numbered context chunks below, citing [n] for every claim. "
            "If the context does not contain the answer, say you could not find it.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )

    def _llm_fn(self, prompt: str) -> str:
        return self.llm_fn(prompt)
