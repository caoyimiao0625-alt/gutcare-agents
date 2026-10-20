"""文档一致性检查：文档里提到的仓库路径必须真实存在。

为什么需要：比赛材料被判分时，文档写了却找不到文件（或改名后没同步）是很低级的失分点。
本脚本扫描所有 Markdown 里的“仓库内路径”写法，逐个校验存在性；发现缺失即退出码 1，
可以放进 CI 或提交前检查。

用法：
    python tools/check_docs.py
    python tools/check_docs.py --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_DIRS = ["docs", "prompts", "knowledge", ".github"]
SCAN_FILES = ["README.md"]

#: 反引号里的路径候选
PATH_PATTERNS = [
    re.compile(r"^[\w./-]+\.(?:md|py|json|yaml|yml|ps1|html|zip|txt|sqlite3|toml|cfg|ini)$"),
    re.compile(r"^(?:tools|docs|tests|gutcare_agents|prompts|knowledge|console|eval|replays|\.github)/[\w./-]*$"),
]
SKIP_PREFIXES = ("E:", "/", "http", "data/", "dist/", "C:")
SKIP_CONTAINS = ("<", ">", "*", "|", " ", "{", "}", "$", "…", "YYYY", "xxx")

#: 仓库之外的引用（团队其他文档、测试包内文件），不参与本仓库检查
EXTERNAL_TOKENS = {
    "设计方案_App版.md",
    "详细功能设计.md",
    "设计方案.md",
    "技术方案.md",
    "数据库设计方案.md",
    "GutCare智能体独立任务与协作机制说明.md",
    "APP及业务后端与智能体系统开发边界(1).md",
    "GutCare智能体独立开发路线与模型选型.md",
    "docs/字段与验收.md",
    "docs/v2_to_v3_changes.json",
    "inputs/test_catalog.json",
    "inputs/seed_database.json",
    "reference_answers/business_answers.json",
    "reference_answers/advice_answers.json",
    # 后端仓库（gutcare-agent）里的路径：本仓库不包含，文档里作为外部引用出现
    "contracts/agent/agent-v1.yaml",
    "docs/ADR/0012-f08-frozen-analysis-task-workflow.md",
    "docs/ADR/0005-智能体通过内部API集成.md",
    "backend/src/gutcare/analysis/api.py",
}

#: 运行时生成、不在版本控制里的产物
GENERATED_TOKENS = {
    "transcript.md",
    "PACKAGE_MANIFEST.json",
}

_BASENAME_CACHE: dict[str, str] | None = None


def basename_index() -> dict[str, str]:
    """仓库内文件名 → 相对路径，用于解析文档里只写文件名的情况（如 `stats.py`）。"""
    global _BASENAME_CACHE
    if _BASENAME_CACHE is None:
        index: dict[str, str] = {}
        for item in ROOT.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(ROOT)
            if any(part in {".git", "data", "__pycache__", "dist"} for part in rel.parts):
                continue
            index.setdefault(item.name, rel.as_posix())
        _BASENAME_CACHE = index
    return _BASENAME_CACHE


def resolves(candidate: str) -> bool:
    if (ROOT / candidate).exists():
        return True
    if "/" in candidate:
        # 形如 agents/diet.py 的片段，按路径后缀匹配
        for rel in basename_index().values():
            if rel.endswith(candidate) or rel.endswith(candidate.split("/", 1)[-1]):
                return True
        return False
    return candidate in basename_index()


def iter_markdown() -> list[Path]:
    files = [ROOT / name for name in SCAN_FILES if (ROOT / name).is_file()]
    for directory in SCAN_DIRS:
        base = ROOT / directory
        if base.is_dir():
            files.extend(sorted(base.rglob("*.md")))
    return files


def check() -> tuple[int, list[tuple[Path, int, str]]]:
    missing: list[tuple[Path, int, str]] = []
    checked = 0
    for path in iter_markdown():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for token in re.findall(r"`([^`]+)`", line):
                token = token.strip()
                if not token or token.startswith(SKIP_PREFIXES):
                    continue
                if any(marker in token for marker in SKIP_CONTAINS):
                    continue
                if not any(pattern.match(token) for pattern in PATH_PATTERNS):
                    continue
                # 允许带行号或锚点：path.md#L12 / path.py:24
                candidate = re.split(r"[#:]", token, maxsplit=1)[0]
                if not candidate:
                    continue
                if candidate in EXTERNAL_TOKENS or candidate in GENERATED_TOKENS:
                    continue
                if candidate.endswith("answers.json") or candidate.startswith("reference_answers/"):
                    continue
                checked += 1
                if not resolves(candidate):
                    missing.append((path.relative_to(ROOT), lineno, token))
    return checked, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checked, missing = check()
    if args.verbose:
        print(f"检查了 {checked} 处仓库路径引用，涉及 {len(iter_markdown())} 个 Markdown 文件")
    if missing:
        print(f"发现 {len(missing)} 处引用指向不存在的路径：")
        for path, lineno, token in missing:
            print(f"  {path}:{lineno}  {token}")
        return 1
    print(f"文档路径检查通过（{checked} 处引用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
