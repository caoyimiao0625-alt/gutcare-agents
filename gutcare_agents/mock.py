"""Mock 模式的结构化输出。

原则：**不伪造能力**。没有真实模型时，mock 只回填“真实可得”的内容
（事实表衍生叙述、结构化校验结果），识图类任务明确返回“未执行真实识别”。
这样联调与断网演示不会把假结果当成真结果，也避免给评审留下“编造”的印象。

真实演示使用 replay 模式：由 live 调用录制下来的响应文件驱动。
"""
from __future__ import annotations

from typing import Any


def mock_json(agent: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    handler = {
        "trend": _trend,
        "advice": _advice,
        "diet": _diet,
        "medical": _medical,
        "conversation": _conversation,
        "stool": _stool,
    }.get(agent)
    if handler is None:
        return {"mock": True, "note": f"未定义 mock 输出的 agent: {agent}"}
    payload = handler(context)
    payload["_mock"] = True
    return payload


def mock_text(agent: str, user: str, images: list[Any] | None = None, context: dict[str, Any] | None = None) -> str:
    """LLMClient 在 mock 模式下调用；返回 JSON 字符串以复用同一解析路径。"""
    import json

    return json.dumps(mock_json(agent, context), ensure_ascii=False)


# ----------------------------------------------------------------------
def _trend(ctx: dict[str, Any]) -> dict[str, Any]:
    facts = ctx.get("facts") or {}
    ids = ctx.get("record_ids") or []
    by_kind = facts.get("by_kind") or {}
    # mock 不复述确定性观察，避免与事实表内容重复；真实模型负责“解释”而非“重复统计”
    uncertainties = []
    for reason in (facts.get("sufficiency") or {}).get("reasons") or []:
        uncertainties.append({"code": "insufficient_data", "message": reason})
    if facts.get("missing_days"):
        uncertainties.append(
            {
                "code": "missing_days",
                "message": f"有 {len(facts['missing_days'])} 天没有记录；无记录不等于没有发生，也不等于数值为零。",
            }
        )
    return {
        "narrative_summary": (
            f"（mock 模式）本次共 {facts.get('total_records', len(ids))} 条记录、覆盖 "
            f"{facts.get('days_with_records', 0)} 天，饮食 {by_kind.get('diet', 0)} 条、"
            f"排便 {by_kind.get('stool', 0)} 条、症状 {by_kind.get('symptom', 0)} 条。"
            "本段为程序统计结果的复述，未调用真实模型进行解释。"
        ),
        "observations": [],
        "uncertainties": uncertainties,
    }


def _advice(ctx: dict[str, Any]) -> dict[str, Any]:
    facts = ctx.get("facts") or {}
    enough = bool((facts.get("sufficiency") or {}).get("enough_for_advice"))
    obs_ids = list(ctx.get("obs_ids") or [])
    knowledge_ids = list(ctx.get("knowledge_ids") or [])
    if not enough or not obs_ids:
        return {
            "advice_items": [],
            "degraded_reason": "数据不足或缺少已确认医疗背景，本次不生成个性化建议",
        }
    items = [
        {
            "advice_id": "advice_001",
            "category": "record_management",
            "title": "继续按天记录饮食、排便与症状",
            "body": "保持每天记录餐次时间、排便类型和症状出现时间，便于后续对比同一时间段的变化。",
            "applicability": "已确认档案与近期记录范围内适用",
            "uncertainty": "记录天数有限，暂不能说明长期规律",
            "evidence_obs_ids": [obs_ids[0]],
            "knowledge_ids": knowledge_ids[:1],
        }
    ]
    return {"advice_items": items, "degraded_reason": ""}


def _diet(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "recognizable": False,
        "foods": [],
        "portion_note": "",
        "uncertainties": [
            {
                "code": "mock_mode",
                "message": "mock 模式未执行真实图像识别；请使用 live 或 replay 模式获得识别结果",
            }
        ],
    }


def _medical(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_type": "unreadable",
        "exam_items": [],
        "conclusion": "",
        "suggestions": [],
        "diagnosis_candidates": [],
        "uncertainties": [
            {
                "code": "mock_mode",
                "message": "mock 模式未执行真实报告抽取；请使用 live 或 replay 模式",
            }
        ],
    }


def _stool(ctx: dict[str, Any]) -> dict[str, Any]:
    return {"quality": {"usable": False, "issue": "mock 模式未执行真实图像识别"}, "recognizable": False, "bristol_type": None, "confidence": "low", "appearance": "", "evidence": ""}


def _conversation(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": "当前为 mock 模式，未调用真实模型；可先使用已有分析结果进行页面演示。",
        "citations": [],
        "needs_more_info": True,
    }
