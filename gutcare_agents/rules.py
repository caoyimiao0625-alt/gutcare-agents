"""独立安全规则引擎。

它与大模型完全解耦：纯 Python 求值、同步执行、毫秒级返回。
`APP 与业务后端开发方` 在生产环境负责同一套规则的工程执行；
智能体侧保留这条路径用于两件事：
1. 综合分析前的前置检查（已有风险提示不得被综合建议覆盖）；
2. 演示与自测时证明“模型不可用，危险征象提示仍然工作”。

注意：规则内容与阈值由医学专业人员审核后才能用于真实用户。本仓库的
`knowledge/safety_rules.json` 明确标记为 `unverified_demo`。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import now_iso

#: 关键词识别只做“候选信号”，最终是否成立由结构化字段或患者确认决定。
KEYWORD_GROUPS: dict[str, tuple[str, ...]] = {
    "stool.has_black_tarry": ("柏油", "黑便", "黑色大便", "黑色便", "沥青样", "tarry"),
    "text.vomit_blood": ("呕血", "吐血", "咖啡色呕吐", "呕吐物呈咖啡"),
    "text.fever": ("发热", "发烧", "体温 38", "体温38", "高热"),
    "text.weight_loss": ("体重下降", "体重减轻", "消瘦", "掉秤"),
    "text.dehydration": ("无法进水", "不能喝水", "喝不进水", "尿量减少", "尿量明显减少", "尿少"),
    "symptom.mentions_bleeding": ("便血", "鲜血便", "血便", "大便带血", "排便带血"),
}

LEVEL_LABEL = {3: "紧急", 2: "及时就医评估", 1: "一般关注"}


@dataclass
class RuleHit:
    rule_id: str
    rule_version: str
    severity_level: int
    title: str
    message: str
    recommended_action: str
    rationale: str
    review_status: str
    trigger_fields: dict[str, Any] = field(default_factory=dict)
    trigger_record_ids: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    @property
    def level_label(self) -> str:
        return LEVEL_LABEL.get(self.severity_level, "未知")

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        d = asdict(self)
        d["level_label"] = self.level_label
        return d


class RuleEngine:
    def __init__(self, rules_path: Path | str) -> None:
        self.rules_path = Path(rules_path)
        payload = json.loads(self.rules_path.read_text(encoding="utf-8"))
        self.rule_set_version: str = payload.get("rule_set_version", "unknown")
        self.review_status: str = payload.get("review_status", "unknown")
        self.disclaimer: str = payload.get("disclaimer", "")
        self.levels: dict[str, str] = payload.get("levels", {})
        self.rules: list[dict[str, Any]] = payload.get("rules", [])
        self._keyword_rules = [r for r in self.rules if self._uses_keywords(r.get("when", {}))]

    # ------------------------------------------------------------------
    def evaluate(self, context: dict[str, Any], *, stage: str = "record_submit") -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            applies = rule.get("applies_to")
            if applies and stage not in applies:
                continue
            if not self._match(rule.get("when", {}), context):
                continue
            hits.append(self._to_hit(rule, context))
        hits.sort(key=lambda h: (-h.severity_level, h.rule_id))
        return hits

    def max_level(self, hits: list[RuleHit]) -> int:
        return max((h.severity_level for h in hits), default=0)

    # ------------------------------------------------------------------
    @staticmethod
    def _uses_keywords(cond: dict[str, Any]) -> bool:
        if "field" in cond:
            return str(cond["field"]) in KEYWORD_GROUPS
        return any(RuleEngine._uses_keywords(c) for c in (cond.get("all") or []) + (cond.get("any") or []))

    def _match(self, cond: dict[str, Any], ctx: dict[str, Any]) -> bool:
        if not cond:
            return False
        if "all" in cond:
            return all(self._match(c, ctx) for c in cond["all"])
        if "any" in cond:
            return any(self._match(c, ctx) for c in cond["any"])
        if "not" in cond:
            return not self._match(cond["not"], ctx)

        field_name = str(cond.get("field", ""))
        op = str(cond.get("op", "eq"))
        expected = cond.get("value")
        actual = ctx.get(field_name)
        if actual is None:
            return False
        try:
            if op == "eq":
                return actual == expected
            if op == "neq":
                return actual != expected
            if op == "gte":
                return float(actual) >= float(expected)
            if op == "lte":
                return float(actual) <= float(expected)
            if op == "gt":
                return float(actual) > float(expected)
            if op == "lt":
                return float(actual) < float(expected)
            if op == "in":
                return actual in (expected or [])
            if op == "contains_any":
                return any(str(token) in str(actual) for token in (expected or []))
        except (TypeError, ValueError):
            return False
        return False

    def _to_hit(self, rule: dict[str, Any], ctx: dict[str, Any]) -> RuleHit:
        trigger_fields = {
            key: ctx.get(key)
            for key in ctx
            if key.startswith(("stool.", "symptom.", "text.", "patient."))
        }
        keyword_based = self._uses_keywords(rule.get("when", {}))
        return RuleHit(
            rule_id=str(rule.get("rule_id")),
            rule_version=str(rule.get("version", "1")),
            severity_level=int(rule.get("severity_level", 1)),
            title=str(rule.get("title", "")),
            message=str(rule.get("message", "")),
            recommended_action=str(rule.get("recommended_action", "")),
            rationale=str(rule.get("rationale", "")),
            review_status=str(rule.get("review_status", self.review_status)),
            trigger_fields=trigger_fields,
            trigger_record_ids=list(ctx.get("_trigger_record_ids") or []),
            # 关键词命中的规则一律要求患者确认，避免把自由文本当成已确认事实
            needs_confirmation=keyword_based,
        )


# ----------------------------------------------------------------------
# 上下文构建
# ----------------------------------------------------------------------
def build_context(
    records: list[dict[str, Any]],
    profile: dict[str, Any] | None = None,
    extra_text: list[str] | None = None,
) -> dict[str, Any]:
    """把本次提交的记录压成规则可判定的扁平上下文。"""
    symptoms = [r for r in records if r.get("kind") == "symptom"]
    stools = [r for r in records if r.get("kind") == "stool"]

    severities: list[int] = []
    durations: list[float] = []
    for r in symptoms:
        data = r.get("data") or {}
        sev = data.get("severity")
        if isinstance(sev, (int, float)):
            severities.append(int(sev))
        dur = data.get("duration_hours")
        if isinstance(dur, (int, float)):
            durations.append(float(dur))
        else:
            minutes = data.get("duration_minutes")
            if isinstance(minutes, (int, float)):
                durations.append(float(minutes) / 60.0)

    texts: list[str] = []
    for r in records:
        data = r.get("data") or {}
        for key in ("note", "remark", "description", "name", "color_text", "appearance"):
            value = data.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
        for tag in data.get("accompanying") or []:
            if isinstance(tag, str):
                texts.append(tag)
    for text in extra_text or []:
        texts.append(text)
    merged_text = " ".join(texts)

    context: dict[str, Any] = {
        "patient.diagnosis_confirmed": bool(
            (profile or {}).get("diagnosis_status") == "confirmed" and (profile or {}).get("diagnosis")
        ),
        "record_count": len(records),
        "symptom.count": len(symptoms),
        "symptom.max_severity": max(severities) if severities else None,
        "symptom.duration_hours_max": max(durations) if durations else None,
        "symptom.names": [str((r.get("data") or {}).get("name") or "") for r in symptoms],
        "stool.count": len(stools),
        "stool.blood_reported": any((r.get("data") or {}).get("blood_reported") is True for r in stools),
        "stool.black_tarry_reported": any(
            (r.get("data") or {}).get("black_tarry") is True for r in stools
        ),
        "_trigger_record_ids": [str(r.get("record_id")) for r in records if r.get("record_id")],
        "_text": merged_text,
    }
    for field_name, keywords in KEYWORD_GROUPS.items():
        context[field_name] = any(k.lower() in merged_text.lower() for k in keywords)
    # 结构化字段优先于关键词：显式记录黑色柏油样便时提升为成立
    if context["stool.black_tarry_reported"]:
        context["stool.has_black_tarry"] = True
    return context


def make_alert(
    *,
    patient_id: str,
    hit: RuleHit,
    dedupe_scope: str = "",
) -> dict[str, Any]:
    material = f"{patient_id}|{hit.rule_id}|{','.join(sorted(hit.trigger_record_ids))}|{dedupe_scope}"
    dedupe_key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return {
        "alert_id": f"alert_{dedupe_key[:12]}",
        "patient_id": patient_id,
        "dedupe_key": dedupe_key,
        "severity_level": hit.severity_level,
        "level_label": hit.level_label,
        "rule_id": hit.rule_id,
        "rule_version": hit.rule_version,
        "review_status": hit.review_status,
        "message": hit.message,
        "recommended_action": hit.recommended_action,
        "rationale": hit.rationale,
        "trigger_record_ids": hit.trigger_record_ids,
        "trigger_fields": hit.trigger_fields,
        "needs_confirmation": hit.needs_confirmation,
        "status": "unread",
        "created_at": now_iso(),
        "disclaimer": "本提示由软件测试规则生成，不构成诊断，也不替代医务人员判断。",
    }
