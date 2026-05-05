"""
会话管理器 - 管理对话会话的创建、检索、持久化和清理
借鉴 WeKnora 的会话管理设计，支持会话级别的对话历史管理
"""
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Session:
    """会话对象"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.conversation_id = conversation_id or self.session_id
        self.messages: List[dict] = []
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.metadata: Dict = {}

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.updated_at = time.time()

    def get_history(self, max_turns: int = 3) -> List[dict]:
        """获取对话历史（限制轮次）"""
        max_messages = max_turns * 2
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages[-max_messages:]
        ]

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """从字典反序列化"""
        session = cls(
            session_id=data.get("session_id"),
            conversation_id=data.get("conversation_id"),
        )
        session.messages = data.get("messages", [])
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", time.time())
        session.metadata = data.get("metadata", {})
        return session


class SessionManager:
    """
    会话管理器

    功能：
    - 创建和检索会话
    - 追加对话历史
    - 持久化到磁盘（JSON 文件）
    - 自动清理过期会话
    """

    def __init__(
        self,
        workspace: str = "./data/sessions",
        ttl_seconds: float = 3600 * 24 * 7,  # 默认7天过期
    ):
        self._workspace = Path(workspace)
        self._workspace.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Session] = {}

        # 启动时加载已有会话
        self._load_sessions()

    def create_session(self, conversation_id: Optional[str] = None) -> Session:
        """创建新会话"""
        session = Session(conversation_id=conversation_id)
        self._sessions[session.session_id] = session
        self._save_session(session)
        logger.info(f"创建会话: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)

    def get_or_create_session(self, conversation_id: Optional[str] = None) -> Session:
        """获取或创建会话"""
        if conversation_id and conversation_id in self._sessions:
            return self._sessions[conversation_id]
        # 尝试按 conversation_id 查找
        if conversation_id:
            for session in self._sessions.values():
                if session.conversation_id == conversation_id:
                    return session
        return self.create_session(conversation_id=conversation_id)

    def append_message(
        self, session_id: str, role: str, content: str
    ) -> bool:
        """追加消息到会话"""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.add_message(role, content)
        self._save_session(session)
        return True

    def cleanup_expired(self) -> int:
        """清理过期会话"""
        now = time.time()
        expired_ids = []
        for sid, session in self._sessions.items():
            if now - session.updated_at > self._ttl:
                expired_ids.append(sid)

        for sid in expired_ids:
            del self._sessions[sid]
            # 删除持久化文件
            session_file = self._workspace / f"{sid}.json"
            if session_file.exists():
                session_file.unlink()

        if expired_ids:
            logger.info(f"清理 {len(expired_ids)} 个过期会话")

        return len(expired_ids)

    def _save_session(self, session: Session):
        """持久化会话到磁盘"""
        try:
            session_file = self._workspace / f"{session.session_id}.json"
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存会话失败: {e}")

    def _load_sessions(self):
        """从磁盘加载会话"""
        count = 0
        for session_file in self._workspace.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                session = Session.from_dict(data)
                self._sessions[session.session_id] = session
                count += 1
            except Exception as e:
                logger.warning(f"加载会话文件失败 {session_file}: {e}")

        if count > 0:
            logger.info(f"加载 {count} 个会话")

        # 清理过期会话
        self.cleanup_expired()
