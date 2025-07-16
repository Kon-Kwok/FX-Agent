from langgraph.graph import StateGraph, END
from .workflow_state import AppState

def create_workflow(agent_map: dict, entry_point: str):
    """
    Creates a dynamic, router-based workflow using langgraph.

    Args:
        agent_map (dict): A mapping of agent names to their handler functions.
        entry_point (str): The name of the first agent to run.

    Returns:
        A compiled langgraph application.
    """
    workflow = StateGraph(AppState)

    # Add all agents as nodes in the graph
    for agent_name, agent_handler in agent_map.items():
        workflow.add_node(agent_name, agent_handler)

    # Set the entry point for the workflow
    workflow.set_entry_point(entry_point)

    # The central router logic
    def route(state: AppState):
        """
        Determines the next step in the workflow based on the current state.
        If `next_step` is not set or is 'END', the workflow finishes.
        """
        next_step = state.next_step
        if not next_step or next_step == "END":
            return END
        if next_step in agent_map:
            return next_step
        # Handle cases where the next step is invalid
        raise ValueError(f"Invalid next step specified: {next_step}")

    # Add a conditional edge from every node to the router
    # This makes the router the central decision point after each step
    for agent_name in agent_map.keys():
        workflow.add_conditional_edges(
            agent_name,
            route,
            # The router will decide which node to go to next
        )

    # Compile the graph into a runnable application
    app = workflow.compile()
    return app