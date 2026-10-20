"""评测脚本：用测试包衡量“能自动量的部分”。

诚实原则：
- 只有三类可以自动比对：Trend 的记录筛选与计数（business_answers）、
  模拟医嘱字段抽取（advice_answers）、安全规则的触发/不触发；
- 报告抽取与食物识别没有逐字/字段级金标准，本脚本只做结构合规与
  “不编造”检查，准确性需要人工核对，不能在这里宣称。

运行：
    python eval/run_eval.py --mode mock           # 只跑确定性部分
    python eval/run_eval.py --mode live --limit 5 # 真实模型，小样本
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings
from gutcare_agents.contract import TaskKind
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.knowledge_base import KnowledgeBase
from gutcare_agents.llm import LLMClient
from gutcare_agents.orchestrator import run_single
from gutcare_agents.rules import RuleEngine, build_context
from gutcare_agents.stats import facts_for_request
from gutcare_agents.store import Store


def evaluate_trend(settings, fixtures: Fixtures, reference: dict[str, Any]) -> dict[str, Any]:
    exact = 0
    total = 0
    id_precision: list[float] = []
    id_recall: list[float] = []
    count_mismatch: list[str] = []
    started = time.time()
    for sample_id, expected in reference.items():
        if not sample_id.startswith("TREND"):
            continue
        request = fixtures.build_request(sample_id)
        facts, records = facts_for_request(
            request,
            min_days_coverage=settings.min_days_coverage,
            min_records_for_trend=settings.min_records_for_trend,
        )
        got = {str(r["record_id"]) for r in records}
        want = set(expected["eligible_record_ids"])
        total += 1
        if got == want and facts.by_kind == expected["record_counts"]:
            exact += 1
        elif facts.by_kind != expected["record_counts"]:
            count_mismatch.append(sample_id)
        inter = len(got & want)
        id_precision.append(inter / len(got) if got else 0.0)
        id_recall.append(inter / len(want) if want else 0.0)
    return {
        "cases": total,
        "exact_match": exact,
        "exact_match_rate": round(exact / max(1, total), 4),
        "record_id_precision": round(statistics.fmean(id_precision), 4) if id_precision else 0.0,
        "record_id_recall": round(statistics.fmean(id_recall), 4) if id_recall else 0.0,
        "count_mismatch_cases": count_mismatch[:5],
        "elapsed_s": round(time.time() - started, 2),
    }


def evaluate_safety(settings, fixtures: Fixtures) -> dict[str, Any]:
    engine = RuleEngine(settings.rules_path)
    positives = 0
    positives_hit = 0
    negatives = 0
    negatives_hit = 0
    max_levels: list[int] = []
    for record in fixtures.seed["records"]:
        if record.get("status", "confirmed") != "confirmed":
            continue
        profile = fixtures.profile(record["patient_id"]).to_dict()
        context = build_context([record], profile)
        hits = engine.evaluate(context, stage="record_submit")
        level = engine.max_level(hits)
        if record["kind"] == "stool" and (record.get("data") or {}).get("blood_reported") is True:
            positives += 1
            max_levels.append(level)
            if any(h.rule_id == "R-BLOOD-STOOL-EXPLICIT" for h in hits):
                positives_hit += 1
        elif record["kind"] in ("stool", "symptom"):
            negatives += 1
            if level > 0:
                negatives_hit += 1
    return {
        "positive_cases": positives,
        "positive_hit": positives_hit,
        "positive_recall": round(positives_hit / max(1, positives), 4),
        "negative_cases": negatives,
        "false_positive": negatives_hit,
        "false_positive_rate": round(negatives_hit / max(1, negatives), 4),
        "sample_max_levels": sorted(set(max_levels)),
        "note": "只覆盖“患者主动报告便血”这一类可结构化判定的阳性；其余规则靠人工用例测试",
    }


def evaluate_advice_docs(settings, fixtures: Fixtures, limit: int) -> dict[str, Any]:
    if settings.mode == "mock":
        return {"status": "skipped", "reason": "mock 模式没有真实视觉抽取结果"}
    reference = {
        item["sample_id"]: item
        for item in json.loads(
            (settings.fixture_root / "reference_answers/advice_answers.json").read_text(encoding="utf-8")
        )
    }
    items = [i for i in fixtures.catalog("advice_document")][:limit]
    results = []
    for item in items:
        request = fixtures.build_request(item["sample_id"])
        result = run_single(settings, request)
        expected = reference[item["sample_id"]]
        got = [str(x) for x in (result.data or {}).get("items") or []]
        want = [str(x) for x in expected.get("expected_items") or []]
        matched = sum(1 for w in want if any(w.strip() in g or g.strip() in w for g in got))
        results.append(
            {
                "sample_id": item["sample_id"],
                "status": result.status,
                "got_items": len(got),
                "expected_items": len(want),
                "item_recall": round(matched / max(1, len(want)), 3),
                "marker_detected": bool((result.data or {}).get("warning_marker_present")),
                "inferred_forbidden": bool((result.data or {}).get("medications")),
                "redactions": len((result.data or {}).get("redactions") or []),
            }
        )
    # 只统计真正跑出结果的样例：replay 模式没有录制、或模型失败时，
    # 把它们算进平均值会让数字毫无意义（0 召回不是模型差，而是没跑到）。
    valid = [r for r in results if r["status"] in ("ok", "partial")]
    recalls = [r["item_recall"] for r in valid]
    return {
        "status": "done",
        "cases": len(valid),
        "skipped_cases": [r["sample_id"] for r in results if r not in valid],
        "mean_item_recall": round(statistics.fmean(recalls), 3) if recalls else None,
        "marker_detection_rate": round(sum(r["marker_detected"] for r in valid) / max(1, len(valid)), 3)
        if valid
        else None,
        "results": results,
        "note": "expected_items 是逐字原文；“条目召回”只说明抽取完整度，不等于医学理解正确；"
        "失败/未录制的样例不计入平均",
    }


def evaluate_structure(settings, fixtures: Fixtures, reference: dict[str, Any], limit: int) -> dict[str, Any]:
    """结构合规：每个 Trend 结果都必须有可解析来源，且状态合法。"""
    legal = {"ok", "partial", "failed", "skipped", "insufficient_data"}
    checked = 0
    missing_source = 0
    illegal_status = 0
    for sample_id in list(reference.keys())[:limit]:
        if not sample_id.startswith("TREND"):
            continue
        request = fixtures.build_request(sample_id)
        result = run_single(settings, request)
        checked += 1
        if result.status not in legal:
            illegal_status += 1
        for obs in result.observations:
            if not obs.source_record_ids:
                missing_source += 1
    return {
        "checked": checked,
        "illegal_status": illegal_status,
        "observations_without_source": missing_source,
        "pass": illegal_status == 0 and missing_source == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live", "replay"])
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default=str(ROOT / "eval" / "report.json"))
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.mode:
        overrides["mode"] = args.mode
    settings = load_settings(**overrides)
    fixtures = Fixtures(settings.fixture_root)
    reference = {
        item["sample_id"]: item
        for item in json.loads(
            (settings.fixture_root / "reference_answers/business_answers.json").read_text(encoding="utf-8")
        )
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": settings.mode,
        "model": settings.vision_model,
        "fixture_root": str(settings.fixture_root),
        "trend": evaluate_trend(settings, fixtures, reference),
        "safety": evaluate_safety(settings, fixtures),
        "structure": evaluate_structure(settings, fixtures, reference, args.limit),
        "advice_document": evaluate_advice_docs(settings, fixtures, args.limit),
        "not_measured": [
            "报告字段抽取准确性（无字段级金标准，需人工核对）",
            "食物识别准确性（来源配料不是可见食材金标准）",
            "OCR 字符错误率（参考转写为 pending_manual_transcription）",
            "排便图像分类（本轮未实现图像，测试包也没有真实粪便图像）",
            "临床有效性（不参与评测，也不做任何声称）",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"模式：{settings.mode}  模型：{settings.vision_model}")
    print(f"Trend 记录筛选：{report['trend']['exact_match']}/{report['trend']['cases']} 完全一致"
          f"（id 精确率 {report['trend']['record_id_precision']}，召回 {report['trend']['record_id_recall']}）")
    print(f"安全规则：阳性召回 {report['safety']['positive_recall']}"
          f"（{report['safety']['positive_hit']}/{report['safety']['positive_cases']}），"
          f"假阳性 {report['safety']['false_positive']}/{report['safety']['negative_cases']}")
    print(f"结构合规：{'通过' if report['structure']['pass'] else '不通过'}"
          f"（检查 {report['structure']['checked']} 例）")
    print(f"医嘱抽取：{report['advice_document'].get('status')}")
    print(f"报告已写入：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
