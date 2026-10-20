"""统一数据契约。

所有 Agent 的输入输出都走这里定义的结构，禁止“只返回一段自然语言”。
字段来源：《协作机制说明》§8、《开发边界》§6.2、《详细功能设计》§15.3。
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import CONTRACT_VERSION

BJT = timezone(timedelta(hours=8))


# --------------------------------------------------------------------------
# 枚举（用字符串常量而非 Enum，便于 JSON 序列化与跨语言对齐）
# --------------------------------------------------------------------------
class AgentName:
    ORCHESTRATOR = "orchestrator"
    MEDICAL = "medical"
    DIET = "diet"
    STOOL = "stool"
    TREND = "trend"
    ADVICE = "advice"
    CONVERSATION = "conversation"
    VALIDATOR = "validator"  # 校验组件，不是大模型 Agent


class TaskKind:
    """对外暴露的任务类型（一个任务类型 ≠ 一个部署单元）。"""

    REPORT_EXTRACT = "report_extract"      # 医疗报告字段抽取（Medical）
    REPORT_TEXT_EXTRACT = "report_text_extract"  # 视觉不可用时的降级：从 OCR/转写文本抽取
    REPORT_OCR = "report_ocr"              # 可见文字转写（Medical 子能力）
    ADVICE_DOC_EXTRACT = "advice_doc_extract"  # 模拟医嘱文书抽取（Medical）
    MEAL_RECOGNIZE = "meal_recognize"      # 食物图片识菜（Diet）
    STOOL_RECORD = "stool_record"          # 排便结构化记录校验（不依赖模型）
    STOOL_IMAGE_RECOGNIZE = "stool_image_recognize"  # 排便图片质量与 Bristol 候选
    TREND_ANALYZE = "trend_analyze"        # 趋势与关联（Trend）
    ADVICE_GENERATE = "advice_generate"    # 建议生成（Advice）
    ANALYSIS = "analysis"                  # 综合分析（Orchestrator 编排）
    ASSISTANT_CHAT = "assistant_chat"      # 对话助手（Conversation）
    SAFETY_CHECK = "safety_check"          # 即时危险征象规则（同步、非模型）


class Status:
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"
    INSUFFICIENT = "insufficient_data"
    TIMEOUT = "timeout"


#: 任务级状态（对外，与后端业务任务状态映射）
class TaskStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient_data"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObsKind:
    """观察的类型必须显式区分，避免把模型推测写成患者事实。"""

    FACT = "fact"                    # 输入记录里明确存在的内容
    PATIENT_REPORT = "patient_report"  # 患者自述/自报
    DERIVED_STAT = "derived_stat"    # 程序确定性计算
    MODEL_INFERENCE = "model_inference"  # 模型推断
    KNOWLEDGE = "knowledge"          # 审核知识库内容


CONFIDENCE = ("high", "medium", "low")
SEVERITY = ("info", "warn", "error")


def now_iso() -> str:
    return datetime.now(BJT).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 输入
# --------------------------------------------------------------------------
@dataclass
class TimeRange:
    start: str | None = None  # ISO8601 带 +08:00
    end: str | None = None

    def contains(self, occurred_at: str) -> bool:
        if self.start and occurred_at < self.start:
            return False
        if self.end and occurred_at > self.end:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}


@dataclass
class PatientProfile:
    """业务后端确认后的档案快照（只读）。"""

    patient_id: str
    display_name: str = ""
    age: int | None = None
    diagnosis_status: str = "unknown"  # confirmed / unconfirmed / unknown
    diagnosis: str | None = None
    profile_version: int = 1
    restrictions: list[str] = field(default_factory=list)

    @property
    def has_confirmed_diagnosis(self) -> bool:
        return self.diagnosis_status == "confirmed" and bool(self.diagnosis)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_version(record: dict[str, Any]) -> str:
    """记录版本标识：用于缓存键与历史可比性。"""
    return str(record.get("version", 1))


@dataclass
class TaskRequest:
    """智能体接口的请求体。

    `patient_id` 为后端内部受控标识，不是 APP 传入的可信身份；
    智能体侧仍会校验所有记录的 patient_id 与之一致（防御性校验）。
    """

    task_id: str
    kind: str
    patient_id: str
    time_range: TimeRange = field(default_factory=TimeRange)
    records: list[dict[str, Any]] = field(default_factory=list)
    profile: PatientProfile | None = None
    files: list[dict[str, Any]] = field(default_factory=list)  # {file_id,path,mime,kind}
    confirmed_results: list[dict[str, Any]] = field(default_factory=list)
    domain_results: list[dict[str, Any]] = field(default_factory=list)  # 本次运行产生的候选结果，仅内部传递
    safety_alerts: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    received_at: str = field(default_factory=now_iso)

    # ---- 便捷访问 ----
    def confirmed_records(self) -> list[dict[str, Any]]:
        """只保留 confirmed 且属于本患者的记录。

        `pending_confirmation` / `deleted` / `superseded` 一律排除
        （见测试包 docs/字段与验收.md），缺失不能被解释为“没发生”。
        """
        out = []
        for r in self.records:
            if r.get("patient_id") != self.patient_id:
                continue
            if r.get("status", "confirmed") != "confirmed":
                continue
            out.append(r)
        return out

    def snapshot_hash(self) -> str:
        """输入快照指纹：幂等去重、缓存命中与“依据是否更新”判断都用它。"""
        payload = {
            "kind": self.kind,
            "patient_id": self.patient_id,
            "range": self.time_range.to_dict(),
            "records": sorted(
                (
                    {
                        "record_id": r.get("record_id"),
                        "version": record_version(r),
                        "status": r.get("status", "confirmed"),
                    }
                    for r in self.records
                ),
                key=lambda x: str(x["record_id"]),
            ),
            "profile_version": self.profile.profile_version if self.profile else None,
            "files": sorted(str(f.get("file_id")) for f in self.files),
            "confirmed_results": sha256_of(self.confirmed_results),
            "query": self.query,
            "description": self.options.get("description"),
        }
        return sha256_of(payload)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["profile"] = self.profile.to_dict() if self.profile else None
        d["snapshot_hash"] = self.snapshot_hash()
        return d


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
@dataclass
class SourceRef:
    """证据引用：必须能解析回输入记录、原报告位置或知识文档。"""

    ref_type: str  # record | report_page | image_region | knowledge | profile
    ref_id: str
    locator: str = ""  # 例如 "p1/line3"、"区域(0.31,0.44,0.52,0.60)"
    quote: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    obs_id: str
    kind: str
    statement: str
    source_record_ids: list[str] = field(default_factory=list)
    source_refs: list[SourceRef] = field(default_factory=list)
    confidence: str = "medium"
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_refs"] = [r.to_dict() if isinstance(r, SourceRef) else r for r in self.source_refs]
        return d


@dataclass
class Uncertainty:
    code: str
    message: str
    affects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """输出校验发现。error 级会降级/移除不合规内容。"""

    code: str
    severity: str
    message: str
    target: str = ""  # observation / advice item 的 id
    action: str = ""  # dropped | downgraded | flagged

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    """单一 Agent 的结构化输出。"""

    task_id: str
    agent: str
    patient_id: str
    status: str = Status.OK
    agent_version: str = "0.1.0"
    prompt_version: str = "v1"
    prompt_sha256: str = ""
    model_id: str = "mock"
    input_record_ids: list[str] = field(default_factory=list)
    input_versions: dict[str, str] = field(default_factory=dict)
    input_snapshot_hash: str = ""
    observations: list[Observation] = field(default_factory=list)
    uncertainties: list[Uncertainty] = field(default_factory=list)
    needs_confirmation: bool = False
    confirmation_fields: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # Agent 专属结构化结果
    findings: list[Finding] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    started_at: str = field(default_factory=now_iso)
    ended_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # 消融实验可在进程内查看校验前的候选；接口、任务库和报告绝不暴露它。
        d["data"].pop("raw_advice_items", None)
        d["observations"] = [o.to_dict() if isinstance(o, Observation) else o for o in self.observations]
        d["uncertainties"] = [u.to_dict() if isinstance(u, Uncertainty) else u for u in self.uncertainties]
        d["findings"] = [f.to_dict() if isinstance(f, Finding) else f for f in self.findings]
        d["contract_version"] = CONTRACT_VERSION
        return d

    # ---- 构造辅助 ----
    def add_obs(
        self,
        kind: str,
        statement: str,
        *,
        record_ids: list[str] | None = None,
        refs: list[SourceRef] | None = None,
        confidence: str = "medium",
        value: Any = None,
        obs_id: str = "",
    ) -> Observation:
        obs = Observation(
            obs_id=obs_id or f"{self.agent}_obs_{len(self.observations) + 1:03d}",
            kind=kind,
            statement=statement,
            source_record_ids=list(record_ids or []),
            source_refs=list(refs or []),
            confidence=confidence,
            value=value,
        )
        self.observations.append(obs)
        return obs

    def add_uncertainty(self, code: str, message: str, affects: list[str] | None = None) -> None:
        self.uncertainties.append(Uncertainty(code=code, message=message, affects=list(affects or [])))

    def add_finding(self, code: str, severity: str, message: str, *, target: str = "", action: str = "") -> None:
        self.findings.append(Finding(code=code, severity=severity, message=message, target=target, action=action))

    def finish(self, status: str | None = None) -> "AgentResult":
        if status:
            self.status = status
        self.ended_at = now_iso()
        return self


@dataclass
class StageRecord:
    """编排阶段的执行记录：现场演示时用来展示“谁在什么时候做了什么”。"""

    stage: str
    agent: str
    status: str = Status.SKIPPED
    attempt: int = 0
    started_at: str = ""
    ended_at: str = ""
    error_code: str = ""
    reused: bool = False
    result_ref: str = ""  # 指向存储中的结果
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRun:
    """任务级结果（对外返回体）。"""

    task_id: str
    kind: str
    patient_id: str
    status: str = TaskStatus.QUEUED
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    input_snapshot_hash: str = ""
    stages: list[StageRecord] = field(default_factory=list)
    agent_results: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)  # 面向患者的综合结果
    findings: list[Finding] = field(default_factory=list)
    error: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.to_dict() if isinstance(s, StageRecord) else s for s in self.stages]
        d["agent_results"] = {
            k: (v.to_dict() if isinstance(v, AgentResult) else v) for k, v in self.agent_results.items()
        }
        d["findings"] = [f.to_dict() if isinstance(f, Finding) else f for f in self.findings]
        d["contract_version"] = CONTRACT_VERSION
        return d


# --------------------------------------------------------------------------
# 反序列化：缓存复用与回放时按原结构还原，避免“缓存直接返回脏 dict”
# --------------------------------------------------------------------------
def observation_from_dict(raw: dict[str, Any]) -> Observation:
    refs = [
        r
        if isinstance(r, SourceRef)
        else SourceRef(**{k: v for k, v in r.items() if k in SourceRef.__dataclass_fields__})
        for r in (raw.get("source_refs") or [])
    ]
    return Observation(
        obs_id=str(raw.get("obs_id") or ""),
        kind=str(raw.get("kind") or ObsKind.MODEL_INFERENCE),
        statement=str(raw.get("statement") or ""),
        source_record_ids=[str(x) for x in (raw.get("source_record_ids") or [])],
        source_refs=refs,
        confidence=str(raw.get("confidence") or "medium"),
        value=raw.get("value"),
    )


def agent_result_from_dict(raw: dict[str, Any]) -> AgentResult:
    """把缓存的 dict 还原为 AgentResult，保证断言、序列化与类型行为一致。"""
    return AgentResult(
        task_id=str(raw.get("task_id") or ""),
        agent=str(raw.get("agent") or ""),
        patient_id=str(raw.get("patient_id") or ""),
        status=str(raw.get("status") or Status.OK),
        agent_version=str(raw.get("agent_version") or ""),
        prompt_version=str(raw.get("prompt_version") or ""),
        prompt_sha256=str(raw.get("prompt_sha256") or ""),
        model_id=str(raw.get("model_id") or ""),
        input_record_ids=[str(x) for x in (raw.get("input_record_ids") or [])],
        input_versions={str(k): str(v) for k, v in (raw.get("input_versions") or {}).items()},
        input_snapshot_hash=str(raw.get("input_snapshot_hash") or ""),
        observations=[observation_from_dict(o) for o in (raw.get("observations") or [])],
        uncertainties=[
            Uncertainty(**{k: v for k, v in u.items() if k in Uncertainty.__dataclass_fields__})
            for u in (raw.get("uncertainties") or [])
        ],
        needs_confirmation=bool(raw.get("needs_confirmation")),
        confirmation_fields=[str(x) for x in (raw.get("confirmation_fields") or [])],
        data=dict(raw.get("data") or {}),
        findings=[
            Finding(**{k: v for k, v in f.items() if k in Finding.__dataclass_fields__})
            for f in (raw.get("findings") or [])
        ],
        usage=dict(raw.get("usage") or {}),
        error=raw.get("error"),
        started_at=str(raw.get("started_at") or ""),
        ended_at=str(raw.get("ended_at") or ""),
    )
