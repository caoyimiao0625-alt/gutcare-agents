"""消融实验：Baseline（单模型一把梭）vs Proposed（多智能体流水线）。

对齐条件（尽量公平）：
- 同一基础模型、同一批病例、同一时间窗；
- **两边都给到相同的确定性事实表与相同的检索知识**，避免用“削弱 baseline”来制造优势；
- 唯一差别是架构：baseline 一次调用直接产出建议；proposed 走
  Medical/Diet/Stool → Trend → Advice → 输出校验。

评估维度（对齐设计方案 §7.1）：
- 证据完整率：建议条目带可解析依据的比例
- 引用失效率：引用了不存在编号的比例
- 不支持结论率：含“带单位数字但事实表里没有”的条目比例
- 违规表述率：命中禁用词（剂量/诊断/因果/绝对化）的条目比例
- 耗时与 token 成本

诚实说明：mock 模式无法做质量对比，脚本会明确写 `skipped`，
不生成任何“看起来很好”的假数字。

用法：
    python eval/ablation.py --mode live --cases 5
    python eval/ablation.py --mode replay --cases 5
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
from gutcare_agents.contract import AgentResult, Status
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.knowledge_base import KnowledgeBase
from gutcare_agents.llm import LLMClient
from gutcare_agents.orchestrator import Orchestrator
from gutcare_agents.stats import facts_for_request
from gutcare_agents.store import Store
from gutcare_agents.validation import check_numeric_grounding, scan_text

BASELINE_SYSTEM = """你是健康管理建议助手。根据患者的档案、事实表与近期记录，直接给出个性化建议。

要求：
1. 每条建议必须给出依据的 record_id（只能从给定记录编号中选择）；
2. 不得给出药物剂量调整、停药换药建议，不得给出诊断结论；
3. 不得使用因果断言（导致/引起/造成/诱发）与绝对化表述（一定/必然/肯定）；
4. 数字只能来自事实表；
5. 数据不足时减少建议数量并说明原因。
6. 中文输出。

只输出 JSON：
{"advice_items": [{"title": "标题", "body": "2—3 句可执行建议",
  "evidence_record_ids": ["SYN001-diet-1"], "uncertainty": "不确定性说明"}],
 "degraded_reason": ""}
"""


def knowledge_block(kb: KnowledgeBase, query: str) -> tuple[str, list[str]]:
    hits = kb.search(query, top_k=3)
    text = "\n\n".join(
        f"[{doc.doc_id}] {doc.title}（来源：{doc.source or '未标注'}；审核状态：{doc.review_status}）\n{doc.text}"
        for doc, _score in hits
    )
    return (text or "（无检索结果）"), [doc.doc_id for doc, _ in hits]


def run_baseline(
    settings, client: LLMClient, request, facts, kb: KnowledgeBase
) -> dict[str, Any]:
    records = [
        {
            "record_id": r.get("record_id"),
            "kind": r.get("kind"),
            "occurred_at": r.get("occurred_at"),
            "data": r.get("data"),
        }
        for r in request.confirmed_records()
    ]
    query = f"{request.profile.diagnosis if request.profile else ''} 排便 症状 饮食 记录 建议"
    k_text, k_ids = knowledge_block(kb, query)
    user = (
        "## 患者档案\n"
        f"{json.dumps(request.profile.to_dict() if request.profile else {}, ensure_ascii=False)}\n\n"
        "## 事实表（程序计算，数字只能引用这里）\n"
        f"{json.dumps(facts.to_dict(), ensure_ascii=False)}\n\n"
        "## 记录（可引用的 record_id）\n"
        f"{json.dumps(records, ensure_ascii=False)}\n\n"
        "## 可用知识条目\n"
        f"{k_text}\n\n"
        "## 任务\n给出 1—3 条可执行建议，按 schema 输出 JSON。"
    )
    started = time.monotonic()
    payload, response = client.complete_json(
        agent="baseline",
        system=BASELINE_SYSTEM,
        user=user,
        context={"facts": facts.to_dict(), "record_ids": [r["record_id"] for r in records]},
    )
    latency = int((time.monotonic() - started) * 1000)
    items = []
    for idx, item in enumerate(payload.get("advice_items") or [], start=1):
        if not isinstance(item, dict):
            continue
        items.append(
            {
                "advice_id": f"baseline_{idx:03d}",
                "title": str(item.get("title") or ""),
                "body": str(item.get("body") or ""),
                "uncertainty": str(item.get("uncertainty") or ""),
                "evidence": [str(x) for x in (item.get("evidence_record_ids") or [])],
                "knowledge_ids": k_ids,
            }
        )
    return {
        "items": items,
        "degraded_reason": str(payload.get("degraded_reason") or ""),
        "latency_ms": latency,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "model": response.model,
    }


def score_items(
    items: list[dict[str, Any]],
    *,
    allowed_evidence: set[str],
    facts: dict[str, Any],
    evidence_key: str,
) -> dict[str, Any]:
    if not items:
        return {
            "items": 0,
            "evidence_completeness": None,
            "unresolved_citation_rate": None,
            "unsupported_claim_rate": None,
            "forbidden_rate": None,
        }
    evidence_ok = 0
    unresolved = 0
    unsupported = 0
    forbidden = 0
    for item in items:
        ev = [str(x) for x in (item.get(evidence_key) or [])]
        if ev:
            if any(x in allowed_evidence for x in ev):
                evidence_ok += 1
            if any(x not in allowed_evidence for x in ev):
                unresolved += 1
        text = " ".join(str(item.get(k, "")) for k in ("title", "body", "uncertainty"))
        if check_numeric_grounding(text, facts):
            unsupported += 1
        if scan_text(text):
            forbidden += 1
    n = len(items)
    return {
        "items": n,
        "evidence_completeness": round(evidence_ok / n, 3),
        "unresolved_citation_rate": round(unresolved / n, 3),
        "unsupported_claim_rate": round(unsupported / n, 3),
        "forbidden_rate": round(forbidden / n, 3),
    }


def run_case(settings, fixtures: Fixtures, store: Store, orchestrator: Orchestrator, patient: str) -> dict[str, Any]:
    request = fixtures.analysis_request(patient, days=7)
    request.options["force_refresh"] = True
    facts, _records = facts_for_request(request)
    allowed_records = {str(r["record_id"]) for r in request.confirmed_records()}
    client = LLMClient(settings)
    kb = orchestrator.knowledge

    baseline = run_baseline(settings, client, request, facts, kb)
    baseline_score = score_items(
        baseline["items"], allowed_evidence=allowed_records, facts=facts.to_dict(), evidence_key="evidence"
    )

    started = time.monotonic()
    run = orchestrator.run(request)
    proposed_latency = int((time.monotonic() - started) * 1000)
    advice = run.agent_results.get("advice")
    raw_items: list[dict[str, Any]] = []
    shipped: list[dict[str, Any]] = []
    obs_ids: set[str] = set()
    knowledge_ids: set[str] = set(kb.doc_ids)
    if isinstance(advice, AgentResult):
        raw_items = list(advice.data.get("raw_advice_items") or advice.data.get("advice_items") or [])
        shipped = list(advice.data.get("advice_items") or [])
    for result in run.agent_results.values():
        if isinstance(result, AgentResult):
            obs_ids.update(o.obs_id for o in result.observations)
    allowed_proposed = obs_ids | knowledge_ids
    tokens = {"prompt_tokens": baseline["prompt_tokens"], "completion_tokens": baseline["completion_tokens"],
              "proposed_prompt_tokens": 0, "proposed_completion_tokens": 0}
    for result in run.agent_results.values():
        if isinstance(result, AgentResult):
            tokens["proposed_prompt_tokens"] += int((result.usage or {}).get("prompt_tokens", 0) or 0)
            tokens["proposed_completion_tokens"] += int((result.usage or {}).get("completion_tokens", 0) or 0)

    return {
        "patient_id": patient,
        "baseline": {
            **baseline_score,
            "latency_ms": baseline["latency_ms"],
            "tokens": tokens["prompt_tokens"] + tokens["completion_tokens"],
            "degraded_reason": baseline["degraded_reason"],
        },
        "proposed": {
            "raw": score_items(raw_items, allowed_evidence=allowed_proposed, facts=facts.to_dict(), evidence_key="evidence_obs_ids"),
            "shipped": score_items(shipped, allowed_evidence=allowed_proposed, facts=facts.to_dict(), evidence_key="evidence_obs_ids"),
            "latency_ms": proposed_latency,
            "tokens": tokens["proposed_prompt_tokens"] + tokens["proposed_completion_tokens"],
            "task_status": run.status,
            "stages": [f"{s.stage}:{s.status}" for s in run.stages],
            "dropped_by_validator": sum(
                1 for f in run.report.get("validation", {}).get("findings", []) if f.get("action") == "dropped"
            ),
            "observations": len(run.report.get("observations") or []),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live", "replay"], default="live")
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--sleep-s", type=float, default=3.0, help="病例之间暂停，避开平台限流")
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="仅用于冒烟测试代码路径：允许 mock 跑通流程，但结果不具备任何比较意义",
    )
    parser.add_argument("--out", default=str(ROOT / "eval" / "ablation.json"))
    args = parser.parse_args()

    settings = load_settings(mode=args.mode)
    fixtures = Fixtures(settings.fixture_root)
    store = Store(settings.db_path)
    orchestrator = Orchestrator(settings, store=store)

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": settings.mode,
        "model": settings.text_model,
        "cases": args.cases,
        "alignment": [
            "同一基础模型、同一批病例与时间窗",
            "两边都提供相同的确定性事实表与相同的检索知识",
            "唯一差别是架构：单次调用 vs 多 Agent + 输出校验",
        ],
    }

    if settings.mode == "mock" and not args.allow_mock:
        report["status"] = "skipped"
        report["reason"] = "mock 模式输出为占位内容，不能用于模型质量对比；请用 live 或 replay 运行"
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已跳过：mock 模式不能做质量消融。请用 --mode live（或 replay）。")
        print(f"报告已写入：{args.out}")
        return 0
    if settings.mode == "mock":
        report["status"] = "smoke_only"
        report["reason"] = "mock 冒烟运行：只验证代码路径，不得引用任何指标"
        print("⚠ mock 冒烟运行：指标无意义，仅验证脚本可跑通")

    results = []
    for idx in range(args.cases):
        patient = f"SYN{idx + 1:03d}"
        try:
            if idx:
                time.sleep(args.sleep_s)
            results.append(run_case(settings, fixtures, store, orchestrator, patient))
            print(f"  完成 {patient}")
        except Exception as exc:  # 单例失败不中断整批
            results.append({"patient_id": patient, "error": str(exc)})
            print(f"  失败 {patient}: {exc}")

    ok = [r for r in results if "error" not in r]

    def mean(path_keys: list[str], which: str) -> float | None:
        values = []
        for item in ok:
            node: Any = item[which]
            for key in path_keys:
                node = node.get(key) if isinstance(node, dict) else None
            if isinstance(node, (int, float)):
                values.append(float(node))
        return round(statistics.fmean(values), 3) if values else None

    report["status"] = "done"
    report["results"] = results
    report["summary"] = {
        "ok_cases": len(ok),
        "failed_cases": len(results) - len(ok),
        "baseline_evidence_completeness": mean(["evidence_completeness"], "baseline"),
        "proposed_evidence_completeness": mean(["shipped", "evidence_completeness"], "proposed"),
        "baseline_unsupported_claim_rate": mean(["unsupported_claim_rate"], "baseline"),
        "proposed_unsupported_claim_rate": mean(["shipped", "unsupported_claim_rate"], "proposed"),
        "baseline_forbidden_rate": mean(["forbidden_rate"], "baseline"),
        "proposed_forbidden_rate": mean(["shipped", "forbidden_rate"], "proposed"),
        "baseline_unresolved_citation_rate": mean(["unresolved_citation_rate"], "baseline"),
        "proposed_unresolved_citation_rate": mean(["shipped", "unresolved_citation_rate"], "proposed"),
        "baseline_latency_ms": mean(["latency_ms"], "baseline"),
        "proposed_latency_ms": mean(["latency_ms"], "proposed"),
        "baseline_tokens": mean(["tokens"], "baseline"),
        "proposed_tokens": mean(["tokens"], "proposed"),
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    s = report["summary"]
    print("\n指标                         Baseline      Proposed(校验后)")
    print(f"证据完整率                   {s['baseline_evidence_completeness']!s:<13} {s['proposed_evidence_completeness']}")
    print(f"不支持结论率                 {s['baseline_unsupported_claim_rate']!s:<13} {s['proposed_unsupported_claim_rate']}")
    print(f"违规表述率                   {s['baseline_forbidden_rate']!s:<13} {s['proposed_forbidden_rate']}")
    print(f"引用失效率                   {s['baseline_unresolved_citation_rate']!s:<13} {s['proposed_unresolved_citation_rate']}")
    print(f"耗时(ms)                     {s['baseline_latency_ms']!s:<13} {s['proposed_latency_ms']}")
    print(f"token 合计                   {s['baseline_tokens']!s:<13} {s['proposed_tokens']}")
    print(f"\n报告已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
