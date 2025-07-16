from typing import TypedDict, List, Optional
import pandas as pd

class FXAgentState(TypedDict):
    """
    Represents the shared state of the FX-Agent workflow.
    This TypedDict is used by LangGraph to pass data between nodes.
    """
    initial_prompt: str
    structured_data: Optional[pd.DataFrame]
    unstructured_data: Optional[List[str]]
    candidate_features: Optional[pd.DataFrame]
    approved_features: Optional[List[str]]
    final_report: Optional[str]