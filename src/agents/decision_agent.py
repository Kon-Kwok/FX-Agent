from ..core.abstractions.base_agent import BaseAgent
from ..core.workflow_state import AppState
from ..services.rag_service import MockRAGService
from ..services.llm_service import MockLLMService

class DecisionAgent(BaseAgent):
    def __init__(self, rag_service: MockRAGService, llm_service: MockLLMService):
        self._rag_service = rag_service
        self._llm_service = llm_service

    def run(self, state: AppState) -> AppState:
        print("--- Decision Agent Running ---")
        
        # Use RAG to retrieve relevant info
        retrieved_docs = self._rag_service.retrieve(query=state.user_request, top_k=3)
        
        # Use LLM to generate commentary and select features
        prompt = f"Based on the user request '{state.user_request}' and the following data, generate market commentary and select features for forecasting. Data: {state.perception_data}, Docs: {retrieved_docs}"
        
        response = self._llm_service.invoke(prompt, config={"model": "decision_llm"})
        
        # Mock response parsing
        state.market_commentary = "This is a mock market commentary."
        state.features_for_forecasting = ["feature1", "feature2"]
        state.next_step = "forecasting_agent"
        
        print(f"Market commentary: {state.market_commentary}")
        print(f"Features selected: {state.features_for_forecasting}")
        return state