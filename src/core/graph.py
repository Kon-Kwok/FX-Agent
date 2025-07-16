from langgraph.graph import StatefulGraph, END
from src.core.workflow_state import WorkflowState
from src.agents.planning_agent import planning_agent
from src.agents.perception_agent import perception_agent
from src.agents.decision_agent import decision_agent
from src.agents.forecasting_agent import forecasting_agent

# 实例化工作流
workflow = StatefulGraph(WorkflowState)

# 定义节点
workflow.add_node("planner", planning_agent)
workflow.add_node("perceiver", perception_agent)
workflow.add_node("decider", decision_agent)
workflow.add_node("forecaster", forecasting_agent)

# 定义边
workflow.set_entry_point("planner")
workflow.add_edge("planner", "perceiver")
workflow.add_edge("perceiver", "decider")
workflow.add_edge("decider", "forecaster")
workflow.add_edge("forecaster", END)

# 编译工作流
app = workflow.compile()