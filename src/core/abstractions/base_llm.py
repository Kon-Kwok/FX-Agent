from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, prompt: str, config: Dict[str, Any]) -> Any:
        """Invokes the language model with a given prompt and configuration."""
        pass