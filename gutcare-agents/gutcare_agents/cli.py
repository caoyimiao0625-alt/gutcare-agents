"""命令行入口。

用途：
1. 逐 Agent 调试与演示（不依赖 APP / 后端）；
2. 录制 replay 缓存（live 跑一遍，之后断网可复现）；
3. 评测：批量跑测试包并输出 JSON 报告。

示例：
  python -m gutcare_agents cases --category longitudinal
  python -m gutcare_agents run --case TREND001
  python -m gutcare_agents analyze --patient SYN001 --days 7
  python -m gutcare_agents safety --case SYM001
  python -m gutcare_agents serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Settings, load_settings
from .contract import AgentName, TaskKind, TaskRequest, new_id
from .fixtures import Fixtures
from .orchestrator import Orchestrator, run_single
from .rules import RuleEngine, build_context, make_alert
from .storage_bootstrap import get_store


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "mode", None):
        overrides["mode"] = args.mode
    if getattr(args, "fixture_root", None):
        overrides["fixture_root"] = Path(args.fixture_root)
    return load_settings(**overrides)


def cmd_cases(args: argparse.Namespace) -> int:
    settings = _settings(args)
    fixtures = Fixtures(settings.fixture_root)
    catalog = fixtures.catalog()
    counts = fixtures.categories()
    print(f"测试包：{settings.fixture_root}")
    print(f"样例总数：{len(catalog)}")
    for category, count in sorted(counts.items()):
        print(f"  {category:16s} {count:4d}")
    if args.category:
        for item in fixtures.catalog(args.category):
            print(f"    {item['sample_id']:12s} {item['input_path']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    fixtures = Fixtures(settings.fixture_root)
    request = fixtures.build_request(
        args.case,
        options={"force_refresh": args.force},
        query=args.query or "",
    )
    result = run_single(settings, request)
    payload = result.to_dict()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.out}")
    if args.view == "json":
        _print(payload)
    else:
        _print(_agent_summary(result))
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    settings = _settings(args)
    fixtures = Fixtures(settings.fixture_root)
    if getattr(args, "source", "json") == "sqlite":
        request = fixtures.analysis_request_from_db(
            args.patient, days=args.days, start=args.start, end=args.end
        )
    else:
        request = fixtures.analysis_request(
            args.patient,
            days=args.days,
            task_id=new_id("analysis"),
            start=args.start,
            end=args.end,
        )
    request.options["force_refresh"] = args.force
    if args.fault:
        request.options["simulate_faults"] = list(args.fault)
    if args.delay_ms:
        request.options["simulate_delay_ms"] = args.delay_ms
    orchestrator = Orchestrator(settings, store=get_store(settings))
    run = orchestrator.run(request)
    payload = run.to_dict()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.out}")
    if args.view == "json":
        _print(payload)
    else:
        _print(_run_summary(run))
    return 0


def cmd_safety(args: argparse.Namespace) -> int:
    settings = _settings(args)
    engine = RuleEngine(settings.rules_path)
    if args.case:
        fixtures = Fixtures(settings.fixture_root)
        request = fixtures.build_request(args.case)
        records = request.records
        profile = request.profile.to_dict() if request.profile else None
        patient_id = request.patient_id
    else:
        if not args.json:
            print("需要 --case 或 --json", file=sys.stderr)
            return 2
        payload = json.loads(args.json)
        records = payload.get("records") or []
        profile = payload.get("profile")
        patient_id = payload.get("patient_id") or "unknown"
    context = build_context(records, profile)
    hits = engine.evaluate(context, stage="record_submit")
    alerts = [make_alert(patient_id=patient_id, hit=hit) for hit in hits]
    _print(
        {
            "rule_set_version": engine.rule_set_version,
            "review_status": engine.review_status,
            "disclaimer": engine.disclaimer,
            "max_severity_level": engine.max_level(hits),
            "alerts": alerts,
            "context": {k: v for k, v in context.items() if not k.startswith("_")},
        }
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import serve

    settings = _settings(args)
    if args.port:
        settings.http_port = args.port
    if args.host:
        settings.http_host = args.host
    return serve(settings)


def cmd_show(args: argparse.Namespace) -> int:
    settings = _settings(args)
    store = get_store(settings)
    payload = store.get_task(args.task_id)
    if not payload:
        print(f"未找到任务 {args.task_id}", file=sys.stderr)
        return 1
    _print(payload)
    return 0


def _agent_summary(result: Any) -> dict[str, Any]:
    data = result.to_dict()
    return {
        "agent": data["agent"],
        "status": data["status"],
        "model": data["model_id"],
        "observations": [o["statement"] for o in data["observations"]],
        "uncertainties": [u["message"] for u in data["uncertainties"]],
        "findings": [f["message"] for f in data["findings"]],
        "data": {k: v for k, v in data["data"].items() if k not in ("facts",)},
    }


def _run_summary(run: Any) -> dict[str, Any]:
    report = run.report
    return {
        "task_id": run.task_id,
        "status": run.status,
        "stages": [
            {"stage": s.stage, "agent": s.agent, "status": s.status, "note": s.note} for s in run.stages
        ],
        "window": report.get("window"),
        "data_overview": report.get("data_overview"),
        "narrative_summary": report.get("narrative_summary"),
        "advice": [
            {"category": a.get("category_label"), "title": a.get("title"), "body": a.get("body"),
             "evidence": a.get("evidence_obs_ids"), "knowledge": a.get("knowledge_ids")}
            for a in report.get("advice", [])
        ],
        "risks": [
            {"level": a.get("severity_level"), "rule": a.get("rule_id"), "message": a.get("message")}
            for a in report.get("risks", [])
        ],
        "insufficient": report.get("insufficient"),
        "degraded": report.get("degraded"),
        "validation": report.get("validation"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gutcare_agents", description="GutCare 多智能体系统 CLI")
    parser.add_argument("--mode", choices=["live", "mock", "replay"], help="覆盖 GCA_MODE")
    parser.add_argument("--fixture-root", help="测试包根目录")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cases = sub.add_parser("cases", help="列出测试包样例")
    p_cases.add_argument("--category")
    p_cases.set_defaults(func=cmd_cases)

    p_run = sub.add_parser("run", help="运行单条样例（单 Agent）")
    p_run.add_argument("--case", required=True)
    p_run.add_argument("--query", default="")
    p_run.add_argument("--out")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--view", choices=["summary", "json"], default="summary")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="运行综合分析（多 Agent 编排）")
    p_analyze.add_argument("--patient", required=True)
    p_analyze.add_argument(
        "--source",
        choices=["json", "sqlite"],
        default="json",
        help="数据来源：json=测试包 seed 文件，sqlite=测试包 fixture.sqlite（只读，模拟后端数据库接入）",
    )
    p_analyze.add_argument("--days", type=int, default=7)
    p_analyze.add_argument("--start")
    p_analyze.add_argument("--end")
    p_analyze.add_argument("--out")
    p_analyze.add_argument("--force", action="store_true")
    p_analyze.add_argument(
        "--fault",
        action="append",
        choices=["medical", "diet", "stool", "trend", "advice", "all"],
        help="注入故障用于演示异常场景（可重复）",
    )
    p_analyze.add_argument("--delay-ms", type=int, default=0, help="注入故障前的延迟（模拟超时）")
    p_analyze.add_argument("--view", choices=["summary", "json"], default="summary")
    p_analyze.set_defaults(func=cmd_analyze)

    p_safety = sub.add_parser("safety", help="运行独立安全规则")
    p_safety.add_argument("--case")
    p_safety.add_argument("--json")
    p_safety.set_defaults(func=cmd_safety)

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务与演示控制台")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.set_defaults(func=cmd_serve)

    p_show = sub.add_parser("show", help="查看已保存的任务")
    p_show.add_argument("task_id")
    p_show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
