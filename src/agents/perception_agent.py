from ..core.abstractions.base_agent import BaseAgent
from ..core.workflow_state import AppState
from ..services.tool_service import ToolService

class PerceptionAgent(BaseAgent):
    def __init__(self, tool_service: ToolService):
        self._tool_service = tool_service

    def run(self, state: AppState) -> AppState:
        print("--- Perception Agent Running ---")
        
        # Mock execution based on the plan
        plan = state.plan
        if not plan or "plan" not in plan:
             raise ValueError("Plan is missing or invalid.")

        print(f"Executing plan: {plan['plan']}")
        
        # Mock tool execution
        # In a real scenario, you would parse the plan and call tools dynamically
        structured_data = self._tool_service.execute_tool("structured_data_fetcher")
        unstructured_data = self._tool_service.execute_tool("unstructured_data_scraper")

        state.perception_data = {
            "structured": structured_data,
            "unstructured": unstructured_data
        }
        state.next_step = "decision_agent"
        
        print(f"Data collected: {state.perception_data}")
        return state
