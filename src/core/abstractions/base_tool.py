from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Executes the tool with given arguments."""
        pass