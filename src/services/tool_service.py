from typing import Any, Dict, List
from ..core.abstractions.base_tool import BaseTool

class ToolService:
    """A service to register and execute tools."""
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool
        print(f"Tool '{tool.name}' registered.")

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        tool = self._tools[tool_name]
        return tool.execute(**kwargs)

    @property
    def available_tools(self) -> List[Dict[str, str]]:
        return [{"name": name, "description": tool.description} for name, tool in self._tools.items()]