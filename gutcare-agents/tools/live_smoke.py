"""真实模型冒烟 + 录制：把"拿到 Key 之后要做的第一件事"做成一条命令。

它回答三个问题：
1. 模型接口通不通（鉴权 / 地址 / 模型名）；
2. **视觉输入能不能用**（这是本仓库唯一未验证的能力）；
3. 关键病例的真实响应有没有录进 `replays/`（断网演示依赖它）。

用法（Key 只放环境变量，不落盘）：
    $env:GCA_API_KEY="sk-..."
    $env:GCA_VISION_MODEL="deepseek-chat"
    python tools/live_smoke.py
    python tools/live_smoke.py --cases MED002 FOOD001 TREND001 --skip-vision

退出码：
    0  全部成功（视觉可用，录制已完成）
    2  文本可用但视觉不可用（需要换视觉型号或走文本降级）
    3  接口/鉴权不可用（先解决账号）
    4  参数或环境问题
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings
from gutcare_agents.errors import AgentError
from gutcare_agents.fixtures import Fixtures
from gutcare_agents.llm import LLMClient
from gutcare_agents.orchestrator import Orchestrator, run_single
from gutcare_agents.store import Store

DEFAULT_CASES = ["MED002", "ADVICE001", "FOOD001", "TREND001", "TREND002", "TREND003"]


def http_probe(base_url: str, api_key: str, timeout: float = 20.0) -> tuple[bool, str]:
    """用 /models 探测地址与鉴权（不消耗生成额度）。"""
    request = urllib.request.Request(f"{base_url}/models")
    request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            return True, f"HTTP {resp.status}（{body[:120]}）"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:160]}"
    except urllib.error.URLError as exc:
        return False, f"网络不可达：{exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"探测失败：{exc}"


def replay_count(directory: Path) -> int:
    return sum(1 for _ in directory.rglob("*.json")) if directory.is_dir() else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None, help=f"要录制的样例，默认 {DEFAULT_CASES}")
    parser.add_argument("--skip-vision", action="store_true", help="只测文本，不测视觉")
    parser.add_argument("--out", default="", help="报告输出目录，默认 data/live_smoke/<时间戳>")
    args = parser.parse_args()

    settings = load_settings(mode="live")
    if not settings.api_key:
        print("缺少 GCA_API_KEY：请先设置环境变量（不要写进文件）。", file=sys.stderr)
        return 4
    if not settings.fixture_root.is_dir():
        print(f"缺少测试包：{settings.fixture_root}（可用 GCA_FIXTURE_ROOT 指定）", file=sys.stderr)
        return 4

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else ROOT / "data" / "live_smoke" / stamp
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "base_url": settings.base_url,
        "text_model": settings.text_model,
        "vision_model": settings.vision_model,
        "steps": [],
        "replays_before": replay_count(settings.replay_dir),
    }
    failures: list[str] = []

    def note(step: str, ok: bool, detail: str) -> None:
        report["steps"].append({"step": step, "ok": ok, "detail": detail})
        print(f"  [{'OK  ' if ok else 'FAIL'}] {step}  {detail}")
        if not ok:
            failures.append(step)

    print(f"GutCare 真实模型冒烟 · {settings.base_url}")
    print(f"文本模型 {settings.text_model} / 视觉模型 {settings.vision_model}")
    print(f"报告目录 {out_dir}\n")

    # 1) 接口与鉴权
    ok, detail = http_probe(settings.base_url, settings.api_key)
    note("接口与鉴权 /models", ok, detail)
    if not ok:
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n接口不可用：先解决地址/Key/额度，再重跑本脚本。")
        return 3

    # 2) 文本生成
    client = LLMClient(settings)
    try:
        response = client.complete(agent="smoke.text", system="你是连通性测试助手。", user="只回复两个字：正常", max_tokens=16)
        note("文本生成", bool(response.text), f"model={response.model} 输出={response.text[:40]!r} tokens={response.prompt_tokens}/{response.completion_tokens}")
    except AgentError as exc:
        note("文本生成", False, f"{exc.code}: {exc.message}")
        (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n文本调用失败：检查模型名与额度。")
        return 3

    # 3) 视觉输入（决定 Medical/Diet 是真识别还是降级）
    vision_ok = False
    if args.skip_vision:
        note("视觉输入", True, "按要求跳过（--skip-vision）")
        vision_ok = True
    else:
        fixtures = Fixtures(settings.fixture_root)
        store = Store(settings.db_path)
        for attempt in range(2):
            try:
                result = run_single(settings, fixtures.build_request("MED002"))
                if result.error and result.error.get("code") == "unsupported_media_type":
                    note("视觉输入", False, f"模型拒绝图像：{result.error.get('message')}")
                    break
                if result.status in ("ok", "partial") and (result.data or {}).get("exam_items"):
                    note(
                        "视觉输入",
                        True,
                        f"报告抽取成功：{len(result.data['exam_items'])} 个检查项，状态 {result.status}"
                        + (f"（第 {attempt + 1} 次尝试）" if attempt else ""),
                    )
                    vision_ok = True
                    break
                if result.status in ("ok", "partial", "insufficient_data"):
                    note("视觉输入", True, f"接口接受了图像（返回 {result.status}，可能图片不清晰或字段为空）")
                    vision_ok = True
                    break
                detail = f"状态 {result.status}，错误 {(result.error or {}).get('code')}"
                if (result.error or {}).get("retryable") and attempt == 0:
                    print(f"  [重试] {detail}（视觉模型首次可能较慢）")
                    continue
                note("视觉输入", False, detail)
                break
            except AgentError as exc:
                if exc.retryable and attempt == 0:
                    print(f"  [重试] {exc.code}（视觉模型首次可能较慢）")
                    continue
                note("视觉输入", False, f"{exc.code}: {exc.message}")
                break

    # 4) 录制关键病例
    fixtures = Fixtures(settings.fixture_root)
    store = Store(settings.db_path)
    orchestrator = Orchestrator(settings, store=store)
    recorded: list[dict] = []
    for sample_id in (args.cases or DEFAULT_CASES):
        before = replay_count(settings.replay_dir)
        try:
            if sample_id.startswith("SYN"):
                request = fixtures.analysis_request(sample_id, days=7)
                # 关键：必须强制重跑，否则同输入快照会命中之前的 mock/旧任务，
                # 看起来“录制成功”，实际一个模型响应都没录到。
                request.options["force_refresh"] = True
                run = orchestrator.run(request)
                after = replay_count(settings.replay_dir)
                detail = (
                    f"分析 {sample_id} → {run.status}"
                    f"（建议 {len(run.report.get('advice') or [])} 条，新增录制 {after - before} 个）"
                )
            else:
                result = run_single(settings, fixtures.build_request(sample_id))
                after = replay_count(settings.replay_dir)
                detail = f"{sample_id} → {result.status}（model={result.model_id}，新增录制 {after - before} 个）"
            recorded.append({"sample": sample_id, "ok": True, "detail": detail})
            print(f"  [OK  ] 录制 {detail}")
        except AgentError as exc:
            recorded.append({"sample": sample_id, "ok": False, "detail": f"{exc.code}: {exc.message}"})
            print(f"  [FAIL] 录制 {sample_id}: {exc.code}")
    report["recorded"] = recorded

    report["replays_after"] = replay_count(settings.replay_dir)
    report["replays_added"] = report["replays_after"] - report["replays_before"]
    report["vision_ok"] = vision_ok
    report["ok"] = vision_ok and not failures

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 真实模型冒烟报告",
        "",
        f"- 时间：{report['generated_at']}",
        f"- 地址：{report['base_url']}",
        f"- 文本模型：{report['text_model']}　视觉模型：{report['vision_model']}",
        f"- 视觉可用：{'是' if vision_ok else '否'}",
        f"- 录制新增：{report['replays_added']} 个响应（总计 {report['replays_after']}）",
        "",
        "## 步骤",
        *[f"- {'✅' if s['ok'] else '❌'} {s['step']}：{s['detail']}" for s in report["steps"]],
        "",
        "## 录制明细",
        *[f"- {'✅' if r['ok'] else '❌'} {r['detail']}" for r in recorded],
        "",
        "## 下一步",
        "1. 视觉可用 → `GCA_MODE=replay python -m gutcare_agents serve`，断网演示真实结果；",
        "2. 视觉不可用 → 换支持图像的模型（改 GCA_VISION_MODEL），或确认走文本降级；",
        "3. `python eval/ablation.py --mode replay --cases 5` 出消融表（需先录制分析类病例）。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n录制新增 {report['replays_added']} 个响应，总计 {report['replays_after']} 个")
    print(f"报告：{out_dir / 'report.md'}")
    if not vision_ok:
        print("\n视觉不可用：换模型或确认降级策略（不影响文本类 Agent）。")
        return 2
    if failures:
        print("\n有步骤失败，详见报告。")
        return 3
    print("\n全部通过：可以进入提示词调优与消融实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
