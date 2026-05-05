"""
MCP 客户端 - 连接 MCP 服务器并发现和调用工具
借鉴 WeKnora 的 MCP 集成设计，支持 SSE/HTTP 传输
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class MCPClient:
    """
    MCP 客户端

    功能：
    - 连接 MCP 服务器（SSE/HTTP 传输）
    - 发现可用工具
    - 转换 MCP 工具 Schema 为 OpenAI function calling 格式
    - 执行 MCP 工具调用
    - 自动重连
    """

    def __init__(
        self,
        server_name: str,
        server_url: str,
        transport: str = "sse",
        reconnect_attempts: int = 3,
        reconnect_delay: float = 5.0,
    ):
        """
        初始化 MCP 客户端

        Args:
            server_name: 服务器名称
            server_url: 服务器 URL
            transport: 传输协议 (sse / http)
            reconnect_attempts: 重连尝试次数
            reconnect_delay: 重连延迟（秒）
        """
        self.server_name = server_name
        self.server_url = server_url.rstrip("/")
        self.transport = transport
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay

        self._client = httpx.AsyncClient(timeout=30.0)
        self._tools: List[dict] = []
        self._connected = False

    async def connect(self) -> bool:
        """连接到 MCP 服务器并发现工具"""
        for attempt in range(self._reconnect_attempts):
            try:
                # 尝试发现工具
                tools = await self._discover_tools()
                if tools is not None:
                    self._tools = tools
                    self._connected = True
                    logger.info(
                        f"MCP 服务器 '{self.server_name}' 连接成功，"
                        f"发现 {len(self._tools)} 个工具"
                    )
                    return True
            except Exception as e:
                logger.warning(
                    f"MCP 连接尝试 {attempt + 1}/{self._reconnect_attempts} 失败: {e}"
                )
                if attempt < self._reconnect_attempts - 1:
                    await asyncio.sleep(self._reconnect_delay)

        logger.error(f"MCP 服务器 '{self.server_name}' 连接失败")
        return False

    async def _discover_tools(self) -> Optional[List[dict]]:
        """从 MCP 服务器发现可用工具"""
        if self.transport == "sse":
            # SSE 传输：通过 /tools 端点获取工具列表
            url = f"{self.server_url}/tools"
        else:
            # HTTP 传输：通过 JSON-RPC
            url = f"{self.server_url}/mcp"

        try:
            if self.transport == "http":
                # JSON-RPC 方式
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/list",
                    "id": 1,
                }
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("result", {}).get("tools", [])
            else:
                # SSE/REST 方式
                response = await self._client.get(url)
                response.raise_for_status()
                data = response.json()
                return data.get("tools", [])
        except Exception as e:
            logger.debug(f"工具发现失败: {e}")
            return None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具"""
        if not self._connected:
            # 尝试重连
            if not await self.connect():
                return f"错误：无法连接到 MCP 服务器 '{self.server_name}'"

        try:
            if self.transport == "http":
                # JSON-RPC 方式
                url = f"{self.server_url}/mcp"
                payload = {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": tool_name,
                        "arguments": arguments,
                    },
                    "id": 2,
                }
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                result = data.get("result", {})
                # 提取文本内容
                content = result.get("content", [])
                if content:
                    texts = [
                        c.get("text", "") for c in content
                        if c.get("type") == "text"
                    ]
                    return "\n".join(texts) if texts else json.dumps(result, ensure_ascii=False)
                return json.dumps(result, ensure_ascii=False)
            else:
                # SSE/REST 方式
                url = f"{self.server_url}/tools/{tool_name}/call"
                response = await self._client.post(url, json=arguments)
                response.raise_for_status()
                data = response.json()
                return json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"MCP 工具调用失败 (HTTP {e.response.status_code}): {e}")
            return f"错误：MCP 工具调用失败 (HTTP {e.response.status_code})"
        except Exception as e:
            logger.error(f"MCP 工具调用异常: {e}")
            self._connected = False  # 标记为断连，下次尝试重连
            return f"错误：MCP 工具调用异常 - {str(e)}"

    def get_openai_tools_schema(self) -> List[Dict[str, Any]]:
        """将 MCP 工具转换为 OpenAI function calling 格式"""
        openai_tools = []
        for tool in self._tools:
            schema = {
                "type": "function",
                "function": {
                    "name": f"mcp_{self.server_name}_{tool.get('name', '')}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                }
            }
            openai_tools.append(schema)
        return openai_tools

    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return [
            f"mcp_{self.server_name}_{t.get('name', '')}"
            for t in self._tools
        ]

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def close(self):
        """关闭连接"""
        await self._client.aclose()
        self._connected = False
