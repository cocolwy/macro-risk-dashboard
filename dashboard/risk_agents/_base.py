"""
轻量导入: 只从 agents/base.py 抽取 BaseAgent + MessageBus, 不触发整包 __init__。
"""
import importlib.util
from pathlib import Path

_BASE_FILE = Path(__file__).resolve().parent.parent.parent / "agents" / "base.py"
_spec = importlib.util.spec_from_file_location("agents_base", _BASE_FILE)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BaseAgent = _mod.BaseAgent
MessageBus = _mod.MessageBus
AgentStatus = _mod.AgentStatus
