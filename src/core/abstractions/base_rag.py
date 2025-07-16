from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRAG(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Retrieves relevant documents for a given query."""
        pass