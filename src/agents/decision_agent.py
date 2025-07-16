import json
import os
import textwrap
from typing import Dict, Any, List

# Assuming KnowledgeBaseHandler and Retriever are located at the following paths.
# You may need to adjust these imports based on your project structure.
from src.rag.knowledge_base_handler import KnowledgeBaseHandler
from src.rag.retriever import Retriever


def _call_llm_simulator_for_decision(prompt: str) -> str:
    """
    An internal function that simulates a call to an LLM for decision-making.
    It returns a hardcoded JSON string mimicking the LLM's output.

    Args:
        prompt (str): The formatted prompt to be sent to the LLM.

    Returns:
        str: A JSON-formatted string containing 'critique' and 'suggested_features'.
    """
    print("\n--- LLM Simulator for Decision ---")
    print("Received Prompt:\n", prompt)
    
    # Hardcoded LLM response
    critique = (
        "Based on the retrieved news, the market sentiment appears to be cautiously optimistic. "
        "The potential for interest rate hikes by the Fed could strengthen the USD, but this is "
        "counterbalanced by concerns over China's economic slowdown, which might reduce demand. "
        "The key drivers appear to be central bank policies and overarching market sentiment. "
        "Risks include unexpected geopolitical events or a sharper-than-expected decline in global trade."
    )
    
    response_data = {
        "critique": critique,
        "suggested_features": [
            "historical_price", 
            "interest_rate_indicator", 
            "news_sentiment_score"
        ]
    }
    
    print(f"Simulated LLM Response:\n{json.dumps(response_data, indent=2)}")
    print("---------------------------------")
    
    return json.dumps(response_data)


def decision_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes collected data using a RAG pipeline to generate a critique and decide on features for forecasting.

    Args:
        state (Dict[str, Any]): The current state of the workflow.

    Returns:
        Dict[str, Any]: The updated state with the decision.
    """
    print("--- Decision Agent: RAG-Powered Analysis ---")

    # 1. Initialize the RAG pipeline
    # Use a temporary index path to avoid polluting the main environment
    temp_index_path = "temp_decision_agent_faiss_index"
    kb_handler = KnowledgeBaseHandler(index_path=temp_index_path)
    retriever = Retriever(kb_handler=kb_handler)

    # 2. Get unstructured data from state and build/load the knowledge base
    unstructured_data = state.get('data', {}).get('unstructured_data', [])
    if not unstructured_data:
        print("Warning: No unstructured data found. Using default features.")
        state['data']['critique'] = "No news context available to generate a critique."
        state['data']['selected_features'] = ["historical_price", "volume"]
        return state

    print(f"Building knowledge base from {len(unstructured_data)} articles...")
    kb_handler.build_index(unstructured_data, force_rebuild=True)

    # 3. Perform retrieval
    user_request = state.get('user_request', "USD/CNY exchange rate forecast analysis")
    print(f"Retrieving context for query: '{user_request}'")
    retrieved_context_list = retriever.retrieve(user_request, top_k=3)
    retrieved_context = "\n\n".join(retrieved_context_list)
    
    if not retrieved_context:
        print("Warning: Could not retrieve any context. Using default features.")
        state['data']['critique'] = "Failed to retrieve relevant news context."
        state['data']['selected_features'] = ["historical_price", "volume"]
        return state

    # 4. Construct the LLM prompt
    prompt_template = textwrap.dedent("""
    You are an expert financial analyst. Your task is to provide a critical analysis of the current market situation based on the provided news context and suggest relevant features for a financial forecasting model.

    **Task:**
    1.  Analyze the provided context from recent news to identify key economic drivers, potential risks, and overall market sentiment concerning the user's request.
    2.  Based on your analysis, provide a concise 'critique' of the situation.
    3.  Suggest a list of 'features' that would be most effective for a model forecasting the target variable.

    **Context from News:**
    {retrieved_context}

    **User Request:**
    "{user_request}"

    **Output Format:**
    Please provide your response as a single JSON object with two keys:
    - "critique": A string containing your expert analysis.
    - "suggested_features": A list of strings representing the feature names.

    **JSON Output:**
    """).strip()

    formatted_prompt = prompt_template.format(
        retrieved_context=retrieved_context,
        user_request=user_request
    )

    # 5. Call the (simulated) LLM and parse the response
    llm_response_str = _call_llm_simulator_for_decision(formatted_prompt)
    
    try:
        llm_response_json = json.loads(llm_response_str)
        critique = llm_response_json.get("critique", "Critique not available.")
        suggested_features = llm_response_json.get("suggested_features", [])
    except json.JSONDecodeError:
        print("Error: Failed to parse LLM response. Using default values.")
        critique = "Failed to parse analysis from LLM."
        suggested_features = ["historical_price", "volume"]

    # 6. Update the state with the decision
    # In a real application, 'selected_features' might replace 'suggested_features'
    state['data']['critique'] = critique
    state['data']['selected_features'] = suggested_features
    
    print(f"\nCritique Generated:\n{textwrap.fill(critique, width=80)}")
    print(f"\nFinal Selected Features: {suggested_features}")
    print("--- Decision Agent: Analysis Complete ---")

    return state


if __name__ == '__main__':
    # Create a mock state for demonstration purposes
    mock_state = {
        "user_request": "Analyze the forecast for USD/CNY exchange rate.",
        "data": {
            "unstructured_data": [
                {"title": "Fed Signals Potential Rate Hikes", "description": "The US Federal Reserve indicated that interest rate hikes might be necessary to combat rising inflation, potentially strengthening the USD."},
                {"title": "China's Manufacturing PMI Disappoints", "description": "Recent data shows a slowdown in China's manufacturing sector, raising concerns about its economic growth and potentially weakening the CNY."},
                {"title": "Global Trade Tensions Ease Slightly", "description": "Positive talks between major economies have led to a slight easing of global trade tensions, improving overall market sentiment."},
                {"title": "ECB Maintains Neutral Stance", "description": "The European Central Bank has decided to keep its monetary policy unchanged, citing balanced risks to the economic outlook."}
            ]
        }
    }

    # Call the decision agent
    final_state = decision_agent(mock_state)

    print("\n--- Final State after Decision Agent ---")
    print(f"Critique: {final_state.get('data', {}).get('critique')}")
    print(f"Selected Features: {final_state.get('data', {}).get('selected_features')}")
    print("------------------------------------")

    # Clean up the temporary index files created during the demonstration
    temp_index_path = "temp_decision_agent_faiss_index"
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)
        print(f"\nCleaned up temporary index file: {temp_index_path}")
    if os.path.exists(f"{temp_index_path}.pkl"):
        os.remove(f"{temp_index_path}.pkl")
        print(f"Cleaned up temporary pickle file: {temp_index_path}.pkl")