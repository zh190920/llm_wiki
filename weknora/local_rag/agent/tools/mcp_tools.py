"""
MCP 工具包装器 - 将 MCP 工具包装为本地 Tool 实例
借鉴 WeKnora 的 MCP 工具集成设计
"""
import json
import logging
from typing import Any, Dict, Optional

from agent.mcp_client import MCPClient
from agent.tool_registry import Tool

logger = logging.getLogger(__name__)


class MCPToolWrapper(Tool):
    """
    MCP 工具包装器

    将远程 MCP 工具包装为本地 Tool 实例：
    - 动态注册 MCP 服务器上的工具
    - 工具输出截断（16KB 最大）
    - 错误处理与重试提示
    """

    MAX_OUTPUT_SIZE = 16 * 1024  # 16KB

    def __init__(self, mcp_client: MCPClient, tool_info: dict):
        """
        初始化 MCP 工具包装器

        Args:
            mcp_client: MCP 客户端实例
            tool_info: 工具信息（来自 MCP 服务器发现）
        """
        self._mcp_client = mcp_client
        self._tool_info = tool_info
        self._original_name = tool_info.get("name", "unknown")

    @property
    def name(self) -> str:
        return f"mcp_{self._mcp_client.server_name}_{self._original_name}"

    @property
    def description(self) -> str:
        base_desc = self._tool_info.get("description", "")
        return (
            f"[MCP:{self._mcp_client.server_name}] {base_desc}"
            if base_desc
            else f"[MCP:{self._mcp_client.server_name}] 远程工具: {self._original_name}"
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return self._tool_info.get("inputSchema", {
            "type": "object",
            "properties": {},
        })

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """执行 MCP 工具调用"""
        try:
            result = await self._mcp_client.call_tool(
                self._original_name, arguments
            )

            # 输出截断
            if len(result) > self.MAX_OUTPUT_SIZE:
                original_len = len(result)
                result = result[:self.MAX_OUTPUT_SIZE]
                result += f"\n\n[输出已截断：原始长度 {original_len} 字符，截断至 {self.MAX_OUTPUT_SIZE} 字符]"
                logger.warning(
                    f"MCP 工具 {self.name} 输出截断: {original_len} → {self.MAX_OUTPUT_SIZE}"
                )

            return result

        except Exception as e:
            error_msg = f"MCP 工具执行错误: {str(e)}"
            logger.error(error_msg)

            # 提供重试提示
            if not self._mcp_client.is_connected:
                error_msg += "\n提示：MCP 服务器连接已断开，系统将尝试在下次调用时自动重连。"

            return error_msg

    async def cleanup(self):
        """清理资源"""
        pass  # MCP 客户端的清理由外部管理


async def register_mcp_tools(
    tool_registry,
    mcp_servers: list,
) -> list:
    """
    从 MCP 服务器注册工具到工具注册表

    Args:
        tool_registry: 工具注册表
        mcp_servers: MCP 服务器配置列表

    Returns:
        成功注册的 MCP 客户端列表（用于后续清理）
    """
    mcp_clients = []

    for server_config in mcp_servers:
        server_name = server_config.get("name", "unknown")
        server_url = server_config.get("url", "")
        transport = server_config.get("transport", "sse")

        if not server_url:
            logger.warning(f"MCP 服务器 '{server_name}' 未配置 URL，跳过")
            continue

        try:
            # 创建客户端并连接
            client = MCPClient(
                server_name=server_name,
                server_url=server_url,
                transport=transport,
            )

            connected = await client.connect()
            if not connected:
                logger.warning(f"MCP 服务器 '{server_name}' 连接失败，跳过")
                continue

            # 注册每个工具
            for tool_info in client._tools:
                wrapper = MCPToolWrapper(client, tool_info)
                registered = tool_registry.register(wrapper)
                if registered:
                    logger.info(f"注册 MCP 工具: {wrapper.name}")

            mcp_clients.append(client)

        except Exception as e:
            logger.error(f"注册 MCP 服务器 '{server_name}' 工具失败: {e}")

    return mcp_clients
