"""Agent 基类：统一结果构造、用量记录与降级约定。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..config import Settings
from ..contract import (
    AgentName,
    AgentResult,
    TaskRequest,
    Status,
    new_id,
    now_iso,
    record_version,
)
from ..errors import AgentError
from ..llm import LLMClient
from ..prompts import load_prompt, prompt_meta
from ..store import Store

AGENT_VERSION = "0.2.0"


class BaseAgent(ABC):
    name: str = AgentName.ORCHESTRATOR
    #: 提示词文件名（不含版本与扩展名），见 prompts/README.md
    prompt_key: str = ""
    prompt_version: str = "v1"
    #: 该 Agent 是否依赖模型（False 表示纯程序计算，模型故障时仍可用）
    requires_model: bool = True

    def __init__(self, settings: Settings, client: LLMClient, store: Store | None = None) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self._prompt_meta: dict[str, Any] = {}
        self.system_prompt: str = ""
        if self.prompt_key:
            self._apply_prompt(self.prompt_key)

    def _apply_prompt(self, key: str) -> None:
        """加载提示词并记录版本与摘要；缺文件直接抛错（部署错误要早暴露）。"""
        self._prompt_meta = prompt_meta(key, self.prompt_version, root=self.settings.prompts_dir)
        self.system_prompt = load_prompt(key, self.prompt_version, root=self.settings.prompts_dir)

    # ------------------------------------------------------------------
    def new_result(self, request: TaskRequest, *, model: str = "") -> AgentResult:
        records = request.confirmed_records()
        return AgentResult(
            task_id=request.task_id,
            agent=self.name,
            patient_id=request.patient_id,
            agent_version=AGENT_VERSION,
            prompt_version=self.prompt_version,
            prompt_sha256=str(self._prompt_meta.get("sha256", "")),
            model_id=model or (self.settings.text_model if self.requires_model else "deterministic"),
            input_record_ids=[str(r.get("record_id")) for r in records],
            input_versions={str(r.get("record_id")): record_version(r) for r in records},
            input_snapshot_hash=request.snapshot_hash(),
            started_at=now_iso(),
        )

    def log_usage(self, result: AgentResult, usage: dict[str, Any], *, cached: bool = False) -> None:
        if not self.store:
            return
        self.store.log_usage(
            task_id=result.task_id,
            agent=self.name,
            model_id=str(usage.get("model", result.model_id)),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cached=cached or bool(usage.get("cached")),
            latency_ms=int(usage.get("latency_ms", 0) or 0),
        )

    # ------------------------------------------------------------------
    def degrade_on_error(self, result: AgentResult, error: AgentError) -> AgentResult:
        result.error = error.to_dict()
        result.finish(Status.FAILED if not result.observations else Status.PARTIAL)
        if not result.observations:
            result.add_uncertainty(error.code, error.message)
        return result

    @abstractmethod
    def run(self, request: TaskRequest) -> AgentResult:
        """执行一次任务，必须返回结构化结果，不允许抛裸异常。"""


def next_obs_id(prefix: str = "obs") -> str:
    return new_id(prefix)


def resolve_input_path(value: str, fixture_root: Any) -> Any:
    """请求里的文件路径可能是相对路径（相对测试包根目录），这里统一解析。"""
    from pathlib import Path

    path = Path(str(value))
    if not path.is_absolute():
        path = Path(fixture_root) / path
    return path


def file_evidence_id(file_entry: dict[str, Any]) -> str:
    return str(file_entry.get("file_id") or file_entry.get("asset_id") or "")
