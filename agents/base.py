"""
Agent 基础设施：BaseAgent + MessageBus + 状态管理
"""

import json
import time
import threading
import traceback
from enum import Enum
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"      # 被 Critic / RiskManager 打回
    WAITING = "waiting"


@dataclass
class Message:
    sender: str
    receiver: str
    msg_type: str            # "data", "result", "review", "reject", "approve", "risk_alert"
    content: Any
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class AgentLog:
    agent: str
    status: str
    action: str
    detail: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class MessageBus:
    """中央消息总线 — 所有 Agent 通过它通信，前端通过它获取状态"""

    def __init__(self, workspace: str = "workspace"):
        self._messages: list[Message] = []
        self._logs: list[AgentLog] = []
        self._agent_status: dict[str, AgentStatus] = {}
        self._shared: dict[str, Any] = {}        # 共享数据区
        self._lock = threading.Lock()
        self._subscribers: list[Callable] = []
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ── 状态 ──
    def set_status(self, agent: str, status: AgentStatus):
        with self._lock:
            self._agent_status[agent] = status
        self._notify()

    def get_status(self, agent: str) -> AgentStatus:
        return self._agent_status.get(agent, AgentStatus.IDLE)

    def get_all_status(self) -> dict:
        with self._lock:
            return dict(self._agent_status)

    # ── 消息 ──
    def send(self, msg: Message):
        with self._lock:
            self._messages.append(msg)
        self._notify()

    def get_messages(self, receiver: str = None) -> list[Message]:
        with self._lock:
            if receiver:
                return [m for m in self._messages if m.receiver == receiver]
            return list(self._messages)

    # ── 日志 ──
    def log(self, agent: str, status: str, action: str, detail: str = ""):
        entry = AgentLog(agent=agent, status=status, action=action, detail=detail)
        with self._lock:
            self._logs.append(entry)
        self._notify()

    def get_logs(self, limit: int = 200) -> list[dict]:
        with self._lock:
            return [asdict(l) for l in self._logs[-limit:]]

    # ── 共享数据 ──
    def put(self, key: str, value: Any):
        with self._lock:
            self._shared[key] = value

    def get(self, key: str, default=None) -> Any:
        with self._lock:
            return self._shared.get(key, default)

    # ── 订阅（给 WebSocket 用）──
    def subscribe(self, fn: Callable):
        self._subscribers.append(fn)

    def _notify(self):
        for fn in self._subscribers:
            try:
                fn()
            except Exception:
                pass

    # ── 快照（给前端用）──
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "agents": {k: v.value for k, v in self._agent_status.items()},
                "logs": [asdict(l) for l in self._logs[-100:]],
                "messages": [asdict(m) for m in self._messages[-50:]],
                "shared_keys": list(self._shared.keys()),
            }


class BaseAgent:
    """所有 Agent 的基类"""

    name: str = "base"
    role: str = "未定义角色"
    color: str = "#888"

    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.bus.set_status(self.name, AgentStatus.IDLE)

    def log(self, action: str, detail: str = "", status: str = "info"):
        self.bus.log(self.name, status, action, detail)

    def run(self, **kwargs):
        """主执行入口"""
        self.bus.set_status(self.name, AgentStatus.RUNNING)
        self.log("启动", f"{self.role} 开始工作")
        try:
            result = self.execute(**kwargs)
            self.bus.set_status(self.name, AgentStatus.SUCCESS)
            self.log("完成", f"{self.role} 工作完成", status="success")
            return result
        except Exception as e:
            self.bus.set_status(self.name, AgentStatus.FAILED)
            self.log("失败", f"{str(e)}\n{traceback.format_exc()}", status="error")
            raise

    def execute(self, **kwargs) -> Any:
        """子类实现具体逻辑"""
        raise NotImplementedError
