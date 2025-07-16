from typing import List, Dict, Any
from ..core.abstractions.base_rag import BaseRAG

class MockRAGService(BaseRAG):
    """A mock RAG service."""
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        print(f"--- MockRAGService Retrieving ---")
        print(f"Query: {query}, Top K: {top_k}")
        return [{"source": "mock_source", "content": "This is a mock document."}]