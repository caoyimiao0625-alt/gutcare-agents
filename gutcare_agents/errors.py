"""错误码与异常。

错误码在联调时是稳定契约的一部分（见《开发边界》§6.1：稳定 `code`），
因此单独集中定义，禁止在各处手写字符串。
"""
from __future__ import annotations

from typing import Any


class ErrorCode:
    # —— 请求/权限（由后端保证的部分，智能体侧仍做防御性校验）——
    INVALID_REQUEST = "invalid_request"
    PATIENT_SCOPE_VIOLATION = "patient_scope_violation"
    UNAUTHORIZED = "unauthorized"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"

    # —— 输入数据 ——
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_MEDIA_TYPE = "unsupported_media_type"
    FILE_TOO_LARGE = "file_too_large"
    IMAGE_UNREADABLE = "image_unreadable"
    IMAGE_LOW_QUALITY = "image_low_quality"
    INPUT_DATA_INSUFFICIENT = "input_data_insufficient"

    # —— 模型与流程 ——
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_RATE_LIMITED = "model_rate_limited"
    MODEL_INVALID_OUTPUT = "model_invalid_output"
    AGENT_FAILED = "agent_failed"
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_SKIPPED = "agent_skipped"
    DEADLINE_EXCEEDED = "deadline_exceeded"

    # —— 输出校验 ——
    EVIDENCE_UNRESOLVED = "evidence_unresolved"
    UNSAFE_CONTENT_BLOCKED = "unsafe_content_blocked"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    OUT_OF_SCOPE = "out_of_scope_not_advice"

    # —— 其他 ——
    NOT_IMPLEMENTED = "not_implemented"
    INTERNAL_ERROR = "internal_error"


#: 可重试错误（仅瞬时故障；输出结构错误只允许一次修复重试，另行处理）
RETRYABLE = frozenset(
    {
        ErrorCode.MODEL_UNAVAILABLE,
        ErrorCode.MODEL_TIMEOUT,
        ErrorCode.MODEL_RATE_LIMITED,
        ErrorCode.AGENT_TIMEOUT,
    }
)


class AgentError(Exception):
    """智能体内部错误，携带稳定错误码与可选的诊断信息。"""

    def __init__(
        self,
        code: str,
        message: str = "",
        *,
        agent: str = "",
        retryable: bool | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message or code
        self.agent = agent
        self.retryable = RETRYABLE.__contains__(code) if retryable is None else retryable
        self.detail = detail or {}
        super().__init__(f"[{code}] {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "agent": self.agent,
            "retryable": self.retryable,
            "detail": self.detail,
        }
