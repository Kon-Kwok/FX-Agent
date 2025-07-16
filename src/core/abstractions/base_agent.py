from abc import ABC, abstractmethod
from ..workflow_state import AppState

class BaseAgent(ABC):
    @abstractmethod
    def run(self, state: AppState) -> AppState:
        """Runs the agent's logic and returns the updated state."""
        pass