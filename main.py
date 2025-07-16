import json
from src.core.workflow_state import AppState
from src.core.graph import create_workflow

# Import services
from src.services.llm_service import MockLLMService
from src.services.tool_service import ToolService
from src.services.rag_service import MockRAGService
from src.services.forecasting_service import MockForecastingService

# Import agents
from src.agents.planning_agent import PlanningAgent
from src.agents.perception_agent import PerceptionAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.forecasting_agent import ForecastingAgent

# Import original tools to register them with the ToolService
from src.tools.structured_data_fetcher import StructuredDataFetcher
from src.tools.unstructured_data_scraper import UnstructuredDataScraper

def main():
    """
    The main entry point for the application.
    Initializes services, agents, and the workflow, then runs it.
    """
    # 1. Initialize Services
    llm_service = MockLLMService()
    rag_service = MockRAGService()
    forecasting_service = MockForecastingService()
    
    # Initialize and register tools with the ToolService
    tool_service = ToolService()
    tool_service.register_tool(StructuredDataFetcher())
    tool_service.register_tool(UnstructuredDataScraper())

    # 2. Initialize Agents (with dependency injection)
    planning_agent = PlanningAgent(llm_service=llm_service)
    perception_agent = PerceptionAgent(tool_service=tool_service)
    decision_agent = DecisionAgent(rag_service=rag_service, llm_service=llm_service)
    forecasting_agent = ForecastingAgent(forecasting_service=forecasting_service)

    # 3. Create Agent Map for the Workflow
    # The keys are the names used in `state.next_step`
    agent_map = {
        "planning_agent": planning_agent.run,
        "perception_agent": perception_agent.run,
        "decision_agent": decision_agent.run,
        "forecasting_agent": forecasting_agent.run,
    }

    # 4. Create and run the workflow
    app = create_workflow(agent_map=agent_map, entry_point="planning_agent")
    
    # 5. Define the initial state
    initial_state = AppState(user_request="What is the forecast for EUR/USD next week?")

    # 6. Invoke the workflow
    final_state = app.invoke(initial_state)

    # 7. Print the final result
    print("\n--- Workflow Finished ---")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    main()