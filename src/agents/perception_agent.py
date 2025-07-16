from typing import Dict, List, Any

def _execute_fetch_structured_data(arguments: Dict) -> str:
    """
    模拟从数据库或 API 获取结构化数据的工具。

    Args:
        arguments: 包含工具所需参数的字典，例如查询条件。

    Returns:
        一个代表模拟 DataFrame 的 JSON 字符串。
    """
    print(f"--- Mocking structured data fetch with arguments: {arguments} ---")
    # 模拟返回一个 DataFrame 的 JSON 格式
    simulated_json_data = '[{"date": "2024-07-15", "price": 150.0, "volume": 10000}, {"date": "2024-07-16", "price": 152.5, "volume": 12000}]'
    return simulated_json_data

def _execute_scrape_unstructured_data(arguments: Dict) -> List[str]:
    """
    模拟从网页或文档中抓取非结构化数据的工具。

    Args:
        arguments: 包含工具所需参数的字典，例如 URL 或搜索关键词。

    Returns:
        一个包含模拟新闻摘要的列表。
    """
    print(f"--- Mocking unstructured data scraping with arguments: {arguments} ---")
    # 模拟返回几条新闻摘要
    simulated_news = [
        "Global markets rally on positive economic news.",
        "Tech giant announces breakthrough in AI research.",
        "Central bank hints at potential interest rate changes."
    ]
    return simulated_news

def perception_agent(state: Dict) -> Dict:
    """
    根据计划执行数据收集任务的代理。

    Args:
        state: 当前的工作流状态，必须包含 'plan' 键。

    Returns:
        更新后的工作流状态，其中包含收集到的数据。
    """
    print("--- Entering Perception Agent ---")
    plan = state.get("plan", [])
    if not plan:
        print("No plan found in state. Skipping perception.")
        return state

    # 初始化 state['data']
    if 'data' not in state:
        state['data'] = {}

    for step in plan:
        tool_name = step.get("tool_name")
        arguments = step.get("arguments", {})

        if tool_name == "fetch_structured_data":
            print("Executing step: fetch_structured_data...")
            try:
                structured_data_result = _execute_fetch_structured_data(arguments)
                state['data']['structured_data'] = structured_data_result
                print("Successfully collected structured data.")
            except Exception as e:
                print(f"Error executing fetch_structured_data: {e}")

        elif tool_name == "scrape_unstructured_data":
            print("Executing step: scrape_unstructured_data...")
            try:
                unstructured_data_result = _execute_scrape_unstructured_data(arguments)
                state['data']['unstructured_data'] = unstructured_data_result
                print("Successfully collected unstructured data.")
            except Exception as e:
                print(f"Error executing scrape_unstructured_data: {e}")

    print("--- Exiting Perception Agent ---")
    return state
