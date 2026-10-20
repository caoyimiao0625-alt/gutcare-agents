"""提示词加载器。

提示词与代码分离（`prompts/` 目录），原因有三：
1. 队友与评审可以直接审阅提示词，不需要读 Python；
2. 提示词版本可与结果一起追溯（`prompt_version` + `prompt_sha256`）；
3. 改提示词不需要动业务代码，降低误改风险。

约定：文件名 `<agent>.<task>.v<版本>.md`，改内容就新增版本文件并同步 Agent 的 `prompt_version`。
缺文件时**直接报错**（而不是静默回退到旧文本），否则部署漏拷 `prompts/` 会变成难查的行为漂移。
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_VERSION = "v1"


class PromptNotFound(FileNotFoundError):
    """提示词文件缺失：属于部署/配置错误，必须显式失败。"""


def prompt_path(name: str, version: str = DEFAULT_VERSION, root: Path | None = None) -> Path:
    base = Path(root) if root else PROMPT_ROOT
    return base / f"{name}.{version}.md"


@lru_cache(maxsize=32)
def _read(path_str: str, mtime_ns: int) -> str:
    return Path(path_str).read_text(encoding="utf-8").strip()


def load_prompt(name: str, version: str = DEFAULT_VERSION, root: Path | None = None) -> str:
    path = prompt_path(name, version, root)
    if not path.is_file():
        raise PromptNotFound(
            f"缺少提示词文件：{path}。请检查 prompts/ 是否随代码一起部署，"
            f"或把 Agent 的 prompt_version 调整到存在的版本。"
        )
    return _read(str(path), path.stat().st_mtime_ns)


def prompt_meta(name: str, version: str = DEFAULT_VERSION, root: Path | None = None) -> dict[str, Any]:
    path = prompt_path(name, version, root)
    text = load_prompt(name, version, root)
    return {
        "name": name,
        "version": version,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "chars": len(text),
        "path": str(path),
    }


def available(root: Path | None = None) -> list[dict[str, Any]]:
    base = Path(root) if root else PROMPT_ROOT
    items = []
    for path in sorted(base.glob("*.v*.md")):
        stem = path.name[: -len(".md")]
        name, _, version = stem.rpartition(".v")
        items.append({"name": name, "version": f"v{version}", "path": str(path)})
    return items
