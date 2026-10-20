"""输出校验（确定性，不依赖模型）。

这是《协作机制说明》§4.2 第 5 阶段“输出复核”的实现，也是演示时最容易
被评委认可的一环：不合规的内容不是“提醒模型注意”，而是**直接从结果里删掉**
并留下一条可追溯的校验记录。

三层校验：
1. 结构：必填字段与类型（见各 Agent 的 `spec`）
2. 证据：每条观察/建议必须能解析回输入记录或知识条目
3. 安全：禁用表述（加量、确诊、因果、绝对化）与数值落地（数字必须来自事实表）
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .contract import AgentResult, Finding, ObsKind, Status

# ----------------------------------------------------------------------
# 禁用表述
# ----------------------------------------------------------------------
FORBIDDEN_PATTERNS: list[tuple[str, str, str, str]] = [
    # (code, regex, message, severity)
    (
        "dosage_change",
        r"(加倍|加量|减量|加剂量|减剂量|自行停药|停药|换药|调整(?:剂量|用药)|改为\s*\d+\s*(?:mg|片|粒))",
        "出现自行调整药物或剂量的表述；系统不得提供处方决策",
        "error",
    ),
    (
        "diagnosis_claim",
        r"(确诊为|诊断为|可以确诊|证明(?:患有|存在)|确定为(?:某|该)?病|就是(?:某|该)?病)",
        "出现确立诊断的表述；系统不得给出独立诊断",
        "error",
    ),
    (
        "causal_claim",
        r"(?:导致|引起(?!注意|关注|重视)|造成|诱发)",
        "出现因果断言；时间邻近不能写成因果关系",
        "error",
    ),
    (
        "absolute_food_claim",
        r"(一定|必然|肯定|绝对)(?:会|是|有害|致病|过敏|不能吃)",
        "出现绝对化饮食结论；个体差异大，不得这样表述",
        "error",
    ),
    (
        "energy_estimate",
        r"\d+(?:\.\d+)?\s*(?:kcal|千卡|大卡|卡路里|kJ|千焦)",
        "出现精确热量或能量估算；单张图片不足以支持这类数值",
        "error",
    ),
    (
        "photo_diagnosis",
        r"(照片|图片|图像)(?:可以|能|已经)(?:确诊|诊断|判断为)",
        "出现依据图像诊断的表述",
        "error",
    ),
    (
        "emergency_downplay",
        r"(不必就医|无需就医|不用去医院|可以先观察几天再说)",
        "出现延误就医的表述",
        "error",
    ),
]

#: 数值落地检查关注的单位（避免把日期、编号误判为结论中的数字）
_NUMERIC_UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(次|天|日|条|分钟|小时|%|％|分|周)"
)
_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
#: 日期写法（叙述里几乎必然出现，不能被当成“结论里的数字”）
_DATE_PATTERNS = (
    re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"),
    re.compile(r"\d{1,2}月\d{1,2}日"),
)


def _strip_dates(text: str) -> str:
    cleaned = text
    for pattern in _DATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    dropped_observations: list[str] = field(default_factory=list)
    dropped_advice: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def errors(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "dropped_observations": self.dropped_observations,
            "dropped_advice": self.dropped_advice,
            "degraded": self.degraded,
        }


_NEGATION_TOKENS = ("不能", "不得", "不要", "禁止", "避免", "无法", "未", "不", "没有", "并非", "尚不")
_CLAUSE_BREAKS = "。；;！？\n"


def _negated(text: str, start: int) -> bool:
    """判断命中处是否处于**同一分句**的否定语境。

    早期用固定 12 字符窗口，长句里“目前无法确定是 A 还是 B 等其他因素导致……”会被误判成
    因果断言，把一条合规建议删掉。这里改成按分句判断：只要本分句内出现否定词，
    就认为这是“不能据此认定因果”这类安全表述。
    """
    cut = max(text.rfind(ch, 0, start) for ch in _CLAUSE_BREAKS)
    prefix = text[cut + 1 : start]
    return any(token in prefix for token in _NEGATION_TOKENS)


#: 这些表述可能出现“安全否定句”（例如“不要自行停药”），需要结合语境判断
NEGATABLE_CODES = {
    "causal_claim",
    "dosage_change",
    "diagnosis_claim",
    "absolute_food_claim",
    "emergency_downplay",
}

_REF_NUMBER = re.compile(r"\d+")


def normalize_ref_id(value: str) -> str:
    """引用编号的宽容写法：模型常把 trend_obs_011 写成 trend_obs_11。

    只做“去前导零 + 转小写”的规范化，不做模糊匹配——避免把不存在的编号硬凑成存在的。
    """
    text = str(value).strip().lower()
    return _REF_NUMBER.sub(lambda m: str(int(m.group(0))), text)


def build_ref_index(allowed: Iterable[str]) -> tuple[set[str], dict[str, str]]:
    exact = {str(x) for x in allowed}
    normalized = {normalize_ref_id(x): x for x in exact}
    return exact, normalized


def resolve_ref(raw: str, exact: set[str], normalized: dict[str, str]) -> str | None:
    text = str(raw).strip()
    if text in exact:
        return text
    return normalized.get(normalize_ref_id(text))


def resolve_refs(
    raw_ids: Iterable[str], exact: set[str], normalized: dict[str, str]
) -> tuple[list[str], list[str]]:
    """返回 (规范化后的有效编号, 无法解析的编号)。"""
    resolved: list[str] = []
    unknown: list[str] = []
    for item in raw_ids:
        hit = resolve_ref(item, exact, normalized)
        if hit is None:
            unknown.append(str(item))
        elif hit not in resolved:
            resolved.append(hit)
    return resolved, unknown


def scan_text(text: str) -> list[tuple[str, str, str]]:
    """返回 [(code, matched, message)]。"""
    if not text:
        return []
    hits = []
    for code, pattern, message, _severity in FORBIDDEN_PATTERNS:
        for match in re.finditer(pattern, text):
            if code in NEGATABLE_CODES and _negated(text, match.start()):
                continue
            hits.append((code, match.group(0), message))
            break
    return hits


def collect_numbers(text: str) -> list[tuple[str, str]]:
    """抽取“带单位的数字”，用于与事实表比对。"""
    return [(m.group(1), m.group(2)) for m in _NUMERIC_UNIT_PATTERN.finditer(text or "")]


def facts_number_set(facts: Any) -> set[str]:
    """事实表中出现过的所有数字（含字符串形式）。"""
    numbers: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, (int, float)):
            numbers.add(f"{node:g}" if isinstance(node, float) else str(node))
        elif isinstance(node, str):
            for token in _NUMBER_PATTERN.findall(node):
                numbers.add(token)

    walk(facts)
    return numbers


def check_numeric_grounding(text: str, facts: Any) -> list[tuple[str, str]]:
    """结论里的“带单位数字”必须出现在事实表里；否则视为无依据的数字。

    日期不算结论数字：`2026-08-31`、`8月31日`、`31日` 这类写法先剔除，
    否则“8 月 31 日无记录”会被误判成编造了数字 31。
    """
    if not text or facts is None:
        return []
    allowed = facts_number_set(facts)
    bad = []
    for value, unit in collect_numbers(_strip_dates(text)):
        normalized = f"{float(value):g}"
        if value in allowed or normalized in allowed:
            continue
        # 单独的“N 日/N 号”（N≤31）按日期处理，不当成结论数字
        if unit in ("日", "号"):
            try:
                if 1 <= float(value) <= 31:
                    continue
            except ValueError:
                pass
        bad.append((value + unit, unit))
    return bad


# ----------------------------------------------------------------------
# 主校验流程
# ----------------------------------------------------------------------
def validate_agent_result(
    result: AgentResult,
    *,
    allowed_record_ids: Iterable[str],
    knowledge_ids: Iterable[str] = (),
    facts: Any = None,
    require_citation: bool = True,
) -> ValidationReport:
    report = ValidationReport()
    allowed, allowed_norm = build_ref_index(allowed_record_ids)
    knowledge, knowledge_norm = build_ref_index(knowledge_ids)

    kept_obs = []
    for obs in result.observations:
        obs_id = obs.obs_id

        # 1) 证据必须可解析
        if obs.kind in (ObsKind.FACT, ObsKind.PATIENT_REPORT, ObsKind.DERIVED_STAT, ObsKind.MODEL_INFERENCE):
            ids, unresolved = resolve_refs(obs.source_record_ids or [], allowed, allowed_norm)
            if require_citation and not ids and not unresolved:
                report.findings.append(
                    Finding(
                        code="evidence_missing",
                        severity="error",
                        message=f"观察缺少来源记录：{obs.statement[:40]}",
                        target=obs_id,
                        action="dropped",
                    )
                )
                report.dropped_observations.append(obs_id)
                continue
            if unresolved:
                report.findings.append(
                    Finding(
                        code="evidence_unresolved",
                        severity="error",
                        message=f"观察引用了本次输入之外的记录：{unresolved[:3]}",
                        target=obs_id,
                        action="dropped",
                    )
                )
                report.dropped_observations.append(obs_id)
                continue
            # 统一成规范编号（模型可能把 trend_obs_011 写成 trend_obs_11）
            obs.source_record_ids = ids
        elif obs.kind == ObsKind.KNOWLEDGE:
            ref_ids = [r.ref_id for r in obs.source_refs if getattr(r, "ref_type", "") == "knowledge"]
            if require_citation and not ref_ids:
                report.findings.append(
                    Finding(
                        code="knowledge_ref_missing",
                        severity="error",
                        message="知识性观察缺少知识条目引用",
                        target=obs_id,
                        action="dropped",
                    )
                )
                report.dropped_observations.append(obs_id)
                continue
            resolved_k, unknown_k = resolve_refs(ref_ids, knowledge, knowledge_norm)
            if knowledge and unknown_k:
                report.findings.append(
                    Finding(
                        code="knowledge_ref_unresolved",
                        severity="error",
                        message=f"引用了不存在的知识条目：{unknown_k[:3]}",
                        target=obs_id,
                        action="dropped",
                    )
                )
                report.dropped_observations.append(obs_id)
                continue

        # 2) 禁用表述
        hits = scan_text(obs.statement)
        if hits:
            codes = ", ".join(sorted({h[0] for h in hits}))
            report.findings.append(
                Finding(
                    code="unsafe_content_blocked",
                    severity="error",
                    message=f"观察含禁用表述({codes})：{hits[0][1]}",
                    target=obs_id,
                    action="dropped",
                )
            )
            report.dropped_observations.append(obs_id)
            continue

        # 3) 数值落地（只对模型叙述型观察检查）
        if facts is not None and obs.kind in (ObsKind.MODEL_INFERENCE, ObsKind.DERIVED_STAT):
            bad = check_numeric_grounding(obs.statement, facts)
            if bad:
                report.findings.append(
                    Finding(
                        code="ungrounded_claim",
                        severity="warn",
                        message=f"结论中的数字未出现在事实表：{[b[0] for b in bad][:3]}",
                        target=obs_id,
                        action="flagged",
                    )
                )

        kept_obs.append(obs)

    if len(kept_obs) != len(result.observations):
        result.observations = kept_obs
        report.degraded = True
        if result.status == Status.OK:
            result.status = Status.PARTIAL

    # 摘要与观察一样会直接展示给患者，不能绕过观察条目的安全和数字校验。
    summary = result.data.get("narrative_summary")
    if isinstance(summary, str) and summary.strip():
        unsafe = scan_text(summary)
        ungrounded = check_numeric_grounding(summary, facts) if facts is not None else []
        if unsafe or ungrounded:
            result.data["narrative_summary"] = ""
            report.findings.append(
                Finding(
                    code="unsafe_content_blocked" if unsafe else "ungrounded_claim",
                    severity="error",
                    message="模型摘要含禁用表述或无事实依据的数字，已移除",
                    target="narrative_summary",
                    action="dropped",
                )
            )
            report.degraded = True
            if result.status == Status.OK:
                result.status = Status.PARTIAL

    # 4) 需要确认的字段必须指向存在的内容
    if result.needs_confirmation and not result.confirmation_fields:
        report.findings.append(
            Finding(
                code="confirmation_fields_missing",
                severity="warn",
                message="标记为需确认但未给出确认字段清单",
            )
        )

    for finding in report.findings:
        result.findings.append(finding)
    return report


def validate_advice_items(
    result: AgentResult,
    *,
    obs_ids: Iterable[str],
    knowledge_ids: Iterable[str],
    facts: Any = None,
) -> ValidationReport:
    """建议条目必须带可解析证据，否则删除该条目（宁可少说，不可无据）。"""
    report = ValidationReport()
    valid_obs, obs_norm = build_ref_index(obs_ids)
    valid_knowledge, knowledge_norm = build_ref_index(knowledge_ids)
    items = list(result.data.get("advice_items") or [])
    kept = []
    for item in items:
        item_id = str(item.get("advice_id") or f"advice_{len(kept) + 1}")
        text = " ".join(
            str(item.get(key, "")) for key in ("title", "body", "applicability", "uncertainty")
        )
        evidence, unknown = resolve_refs(item.get("evidence_obs_ids") or [], valid_obs, obs_norm)
        knowledge_refs, unknown_k = resolve_refs(item.get("knowledge_ids") or [], valid_knowledge, knowledge_norm)

        if not evidence and not knowledge_refs and not unknown and not unknown_k:
            report.findings.append(
                Finding(
                    code="advice_evidence_missing",
                    severity="error",
                    message=f"建议条目无任何依据：{text[:40]}",
                    target=item_id,
                    action="dropped",
                )
            )
            report.dropped_advice.append(item_id)
            continue
        if unknown or unknown_k:
            report.findings.append(
                Finding(
                    code="advice_evidence_unresolved",
                    severity="error",
                    message=f"建议引用了不存在的依据：{(unknown + unknown_k)[:3]}",
                    target=item_id,
                    action="dropped",
                )
            )
            report.dropped_advice.append(item_id)
            continue
        # 写回规范编号（模型可能少写前导零）
        item["evidence_obs_ids"] = evidence
        item["knowledge_ids"] = knowledge_refs

        hits = scan_text(text)
        if hits:
            report.findings.append(
                Finding(
                    code="unsafe_content_blocked",
                    severity="error",
                    message=f"建议含禁用表述({hits[0][0]})：{hits[0][1]}",
                    target=item_id,
                    action="dropped",
                )
            )
            report.dropped_advice.append(item_id)
            continue

        if facts is not None:
            bad = check_numeric_grounding(text, facts)
            if bad:
                report.findings.append(
                    Finding(
                        code="ungrounded_claim",
                        severity="error",
                        message=f"建议中的数字无法在事实表中找到：{[b[0] for b in bad][:3]}",
                        target=item_id,
                        action="dropped",
                    )
                )
                report.dropped_advice.append(item_id)
                continue

        kept.append(item)

    if len(kept) != len(items):
        result.data["advice_items"] = kept
        result.status = Status.PARTIAL if result.status == Status.OK else result.status
        report.degraded = True
    for finding in report.findings:
        result.findings.append(finding)
    return report


# ----------------------------------------------------------------------
# 结构校验（零依赖的极简实现）
# ----------------------------------------------------------------------
def check_required(payload: dict[str, Any], spec: dict[str, type | tuple[type, ...]], *, where: str) -> list[str]:
    problems: list[str] = []
    for key, expected in spec.items():
        if key not in payload:
            problems.append(f"{where}.{key} 缺失")
            continue
        value = payload[key]
        if value is None:
            continue
        if not isinstance(value, expected):  # type: ignore[arg-type]
            problems.append(f"{where}.{key} 类型应为 {expected}，实际 {type(value).__name__}")
    return problems


def require_non_empty_list(payload: dict[str, Any], key: str, *, where: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        return [f"{where}.{key} 必须是非空列表"]
    return []


def dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)
