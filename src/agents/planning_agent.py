# FX-Agent/src/agents/planning_agent.py

import json
import textwrap
from typing import List, Dict, Any

def planning_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a plan to fulfill the user's request by selecting appropriate tools.

    Args:
        state (Dict[str, Any]): The current state of the workflow.
                               It must contain a 'user_request' key.

    Returns:
        Dict[str, Any]: The updated state with the 'plan' key populated.
    """
    print("---PLANNING AGENT---")
    
    # 1. Define the available tools
    available_tools = [
        {
            "tool_name": "fetch_structured_data",
            "description": "Fetches historical time-series data (e.g., exchange rates, interest rates) for specified currencies.",
            "parameters": ["currency_pair", "start_date", "end_date", "interval"]
        },
        {
            "tool_name": "scrape_unstructured_data",
            "description": "Scrapes news articles and reports related to a given query from the web.",
            "parameters": ["query"]
        },
        {
            "tool_name": "run_forecasting_model",
            "description": "Runs a forecasting model (e.g., LSTM, Transformer) on the prepared data.",
            "parameters": ["model_name", "processed_data"]
        }
    ]

    # 2. Build a dynamic prompt for the LLM
    user_request = state.get("user_request")
    if not user_request:
        raise ValueError("State must include a 'user_request'.")

    tools_json_str = json.dumps(available_tools, indent=2)
    
    prompt_template = textwrap.dedent("""
    You are an expert financial planning agent. Your goal is to create a step-by-step plan to fulfill the user's request using the available tools.

    **User Request:**
    {user_request}

    **Available Tools:**
    {tools_json}

    **Instructions:**
    Based on the user request, create a JSON array of steps. Each step must be a dictionary with "tool_name" and "arguments" keys. The "arguments" must be a dictionary of parameters required by the tool.
    
    - For `fetch_structured_data`, determine a reasonable date range (e.g., 5 years of historical data for a 30-day forecast).
    - For `scrape_unstructured_data`, create a concise query relevant to the user's request.
    - The `run_forecasting_model` step should logically follow data collection and processing steps.
    
    Return ONLY the JSON plan.
    """).strip()

    prompt = prompt_template.format(
        user_request=user_request, 
        tools_json=tools_json_str
    )
    
    print(f"\nGenerated Prompt for LLM:\n{prompt}")

    # 3. Implement the plan generation logic (using a simulated LLM call)
    print("\nGenerating plan using LLM simulator...")
    llm_response_json = call_llm_simulator(prompt)
    
    # 4. Update the state with the generated plan
    try:
        plan = json.loads(llm_response_json)
        state["plan"] = plan
        print(f"\nSuccessfully generated plan with {len(plan)} steps.")
        print(json.dumps(plan, indent=2))
    except json.JSONDecodeError:
        print("\nError: Failed to decode the plan from the LLM response.")
        state["plan"] = []

    return state

def call_llm_simulator(prompt: str) -> str:
    """
    Simulates a call to a Large Language Model (LLM).
    
    This function returns a hardcoded JSON string representing a plausible plan
    for a sample user request, mimicking the output of a real LLM.

    Args:
        prompt (str): The prompt that would be sent to the LLM.

    Returns:
        str: A JSON string representing the execution plan.
    """
    # This hardcoded response simulates the LLM's output for the request:
    # "Forecast USD/CNY exchange rate for the next 30 days"
    
    simulated_json_response = """
    [
      {
        "tool_name": "fetch_structured_data",
        "arguments": {
          "currency_pair": "USD/CNY",
          "start_date": "2020-01-01",
          "end_date": "2024-12-31",
          "interval": "daily"
        }
      },
      {
        "tool_name": "scrape_unstructured_data",
        "arguments": {
          "query": "USD CNY exchange rate news and economic events"
        }
      },
      {
        "tool_name": "run_forecasting_model",
        "arguments": {
          "model_name": "Transformer",
          "processed_data": "output_of_previous_steps"
        }
      }
    ]
    """
    return textwrap.dedent(simulated_json_response).strip()

# Example Usage (for demonstration purposes)
if __name__ == '__main__':
    # Initialize a sample state
    initial_state = {
        "user_request": "Forecast USD/CNY exchange rate for the next 30 days",
        "plan": [],
        "data": {}
    }

    # Run the planning agent
    updated_state = planning_agent(initial_state)

    print("\n--- Final State ---")
    print(json.dumps(updated_state, indent=2))