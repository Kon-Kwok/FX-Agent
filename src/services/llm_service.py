from typing import Any, Dict
from ..core.abstractions.base_llm import BaseLLM

class MockLLMService(BaseLLM):
    """A mock LLM service for development and testing."""
    def invoke(self, prompt: str, config: Dict[str, Any]) -> Any:
        print(f"--- MockLLMService Invoked ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Config: {config}")
        # Return a dummy response structure
        return {"plan": "This is a mock plan from MockLLMService."}