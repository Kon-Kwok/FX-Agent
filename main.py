import json
from src.core.graph import app

def main():
    """
    Executes the primary workflow of the forecasting agent.

    This function initializes the agent with a user-defined request,
    invokes the processing graph, and prints the final state of the
    workflow upon completion.
    """
    # Define the user request for the forecasting task
    user_request = "Forecast USD/CNY exchange rate for the next 30 days based on historical data and recent news."

    # Prepare the inputs for the workflow graph
    inputs = {"user_request": user_request}

    print("🚀 Initiating the forecasting workflow...")

    # Invoke the workflow with the prepared inputs
    final_state = app.invoke(inputs)

    print("\n✅ Workflow completed. Displaying final state:")
    # Pretty-print the final state for clear inspection
    print(json.dumps(final_state, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()