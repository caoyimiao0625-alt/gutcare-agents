"""一键演示：把 8—10 分钟实机演示的动作按顺序真跑一遍，并把每一步输出落盘。

用途：
1. 录演示视频前先跑一遍，确认每个动作都能出结果（不靠现场赌运气）；
2. 现场演示时照着终端输出念（`docs/演示脚本.md` 有对应话术）；
3. 把每一步的 JSON 存下来，作为“真实运行记录”随材料提交。

默认直接用库调用（不需要先起服务）；也可以 `--base-url` 对服务发 HTTP 请求。

用法：
    python tools/demo_run.py                       # mock 模式，输出到 data/demo/<时间戳>/
    python tools/demo_run.py --mode replay         # 断网演示（需已有录制）
    python tools/demo_run.py --out data/demo/1017
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.orchestrator import Orchestrator, run_single
from gutcare_agents.rules import RuleEngine, build_context, make_alert
from gutcare_agents.store import Store

STEPS: list[dict[str, Any]] = []


def section(title: str, hint: str = "") -> None:
    print(f"\n{'=' * 72}\n{title}")
    if hint:
        print(f"（{hint}）")
    print("=" * 72)


def record(step: str, payload: dict[str, Any], lines: list[str]) -> None:
    for line in lines:
        print(f"  · {line}")
    STEPS.append({"step": step, "payload": payload, "lines": lines})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "live", "replay"], default="mock")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    settings = load_settings(mode=args.mode)
    fixtures = Fixtures(settings.fixture_root)
    store = Store(settings.db_path)
    orchestrator = Orchestrator(settings, store=store)
    engine = RuleEngine(settings.rules_path)
    STEPS.clear()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "data" / "demo" / stamp
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    print(f"GutCare 演示序列 · 模式 {settings.mode} · 模型 {settings.vision_model}")
    print(f"输出目录：{out_dir}")

    # 1) 报告抽取
    section("① 报告字段抽取（Medical）", "候选值必须人工确认；遮挡处留空")
    result = run_single(settings, fixtures.build_request("MED002"))
    (out_dir / "01_report_extract.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record(
        "report_extract",
        result.to_dict(),
        [
            f"状态 {result.status}，模型 {result.model_id}，提示词 {result.prompt_version}",
            f"待确认字段：{', '.join(result.confirmation_fields) or '无'}",
            f"观察 {len(result.observations)} 条，不确定性 {len(result.uncertainties)} 条",
            *( [f"未完成原因：{u.message}" for u in result.uncertainties[:2]] ),
        ],
    )

    # 2) 识菜
    section("② 单餐识别（Diet）", "只认可见食物；不给热量")
    result = run_single(settings, fixtures.build_request("FOOD001"))
    (out_dir / "02_meal.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    foods = (result.data or {}).get("foods") or []
    record(
        "meal_recognize",
        result.to_dict(),
        [
            f"状态 {result.status}，识别到 {len(foods)} 项",
            *( [f"食物：{f.get('name')}（{f.get('category')}）" for f in foods[:5]] or ["未识别（这是允许的结果，不编造）"] ),
        ],
    )

    # 3) 危险征象（不调模型）
    section("③ 危险征象规则（不调用模型）", "黑色柏油样便 → L3 紧急提示")
    stool_record = {
        "record_id": "demo-black",
        "patient_id": "SYN001",
        "kind": "stool",
        "occurred_at": "2026-10-16T08:00:00+08:00",
        "status": "confirmed",
        "version": 1,
        "data": {"bristol_type": 6, "note": "黑色柏油样大便"},
    }
    hits = engine.evaluate(build_context([stool_record], {"diagnosis_status": "confirmed", "diagnosis": "IBS"}))
    alerts = [make_alert(patient_id="SYN001", hit=hit) for hit in hits]
    (out_dir / "03_safety.json").write_text(json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8")
    record(
        "safety_check",
        {"alerts": alerts},
        [
            f"最高级别 L{engine.max_level(hits)}，规则版本 {engine.rule_set_version}（{engine.review_status}）",
            *( [f"{a['rule_id']} → {a['message'][:40]}…（需确认={a['needs_confirmation']}）" for a in alerts[:2]] ),
        ],
    )

    # 4) 综合分析
    section("④ 综合分析（多 Agent 编排）", "阶段时间线 + 观察 + 建议 + 依据")
    request = fixtures.analysis_request("SYN001", days=7)
    request.options["force_refresh"] = True
    run = orchestrator.run(request)
    (out_dir / "04_analysis.json").write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    advice = (run.report or {}).get("advice") or []
    with_evidence = [a for a in advice if (a.get("evidence_obs_ids") or a.get("knowledge_ids"))]
    record(
        "analysis",
        run.to_dict(),
        [
            f"任务状态 {run.status}",
            "阶段：" + " → ".join(f"{s.stage}({s.status})" for s in run.stages),
            f"观察 {len((run.report or {}).get('observations') or [])} 条，建议 {len(advice)} 条（带依据 {len(with_evidence)} 条）",
            *( [f"建议：{a.get('title')}｜依据 {a.get('evidence_obs_ids')} {a.get('knowledge_ids')}" for a in advice[:2]] ),
            f"缺口说明：{((run.report or {}).get('advice_status') or {}).get('degraded_reason') or '无'}",
        ],
    )

    # 5) 结果复用（同一文件不重复解析）
    section("⑤ 结果复用（改了别的记录，不必重跑报告解析）", "同一文件命中缓存")
    request2 = fixtures.build_request("MED002")
    request2.kind = "analysis"
    request2.task_id = "demo-reuse-1"
    request2.records = []
    request2.options["force_refresh"] = True
    first = orchestrator.run(request2)
    request3 = fixtures.build_request("MED002")
    request3.kind = "analysis"
    request3.task_id = "demo-reuse-2"
    request3.records = []
    request3.options["force_refresh"] = True
    second = orchestrator.run(request3)
    reused = [s for s in second.stages if s.reused]
    (out_dir / "05_reuse.json").write_text(json.dumps(second.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    record(
        "reuse",
        {"first": first.task_id, "second": second.task_id, "reused_stages": [s.stage for s in reused]},
        [
            f"第一次领域识别：{[s.stage for s in first.stages if s.stage.startswith('domain')]}",
            f"第二次复用阶段：{[s.stage for s in reused] or '（无，说明未命中缓存）'}",
            f"缓存条目：{store.cache_stats()}",
        ],
    )

    # 6) 降级：故障注入 + 未确诊
    section("⑥ 异常与降级（模型故障 / 未确诊）", "部分成功与“不给疾病专属建议”")
    faulted = fixtures.analysis_request("SYN002", days=7)
    faulted.options["simulate_faults"] = ["advice"]
    run_fault = orchestrator.run(faulted)
    insufficient = fixtures.analysis_request("SYN009", days=365)
    insufficient.options["force_refresh"] = True
    run_insufficient = orchestrator.run(insufficient)
    (out_dir / "06_degraded.json").write_text(
        json.dumps(
            {"fault": run_fault.to_dict(), "unconfirmed": run_insufficient.to_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    record(
        "degraded",
        {"fault": run_fault.to_dict(), "unconfirmed": run_insufficient.to_dict()},
        [
            f"注入 advice 故障 → 任务 {run_fault.status}，建议状态 {((run_fault.report or {}).get('advice_status') or {}).get('status')}，观察仍在 {len((run_fault.report or {}).get('observations') or [])} 条",
            f"未确诊患者 SYN009 → 建议 {len((run_insufficient.report or {}).get('advice') or [])} 条，原因：{((run_insufficient.report or {}).get('advice_status') or {}).get('degraded_reason')}",
        ],
    )

    elapsed = time.monotonic() - started
    # 落盘文字记录
    transcript = [f"# 演示运行记录（{settings.mode}）", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    for item in STEPS:
        transcript.append(f"## {item['step']}")
        transcript.extend(f"- {line}" for line in item["lines"])
        transcript.append("")
    transcript.append(f"总耗时：{elapsed:.1f}s")
    (out_dir / "transcript.md").write_text("\n".join(transcript), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print(f"演示序列完成：{len(STEPS)} 步，{elapsed:.1f}s")
    print(f"输出：{out_dir}（每步 JSON + transcript.md）")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
