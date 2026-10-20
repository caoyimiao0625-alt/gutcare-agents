"""Agent 注册表。

一个 Agent 一个模块、一套提示词、一套输入输出契约——
即使共享同一个基础模型，职责、工具、失败语义也必须不同，
否则只是“换了名字的同一段提示词”，不构成实质分工。
"""
from __future__ import annotations

from .advice import AdviceAgent
from .base import BaseAgent
from .conversation import ConversationAgent
from .diet import DietAgent
from .medical import MedicalAgent
from .stool import StoolAgent
from .trend import TrendAgent

AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "medical": MedicalAgent,
    "diet": DietAgent,
    "stool": StoolAgent,
    "trend": TrendAgent,
    "advice": AdviceAgent,
    "conversation": ConversationAgent,
}

__all__ = [
    "AGENT_CLASSES",
    "AdviceAgent",
    "BaseAgent",
    "ConversationAgent",
    "DietAgent",
    "MedicalAgent",
    "StoolAgent",
    "TrendAgent",
]
