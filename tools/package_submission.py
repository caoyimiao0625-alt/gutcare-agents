"""打包提交件：把仓库打成可直接交付的 zip。

做法：
- 以 `git ls-files --cached --others --exclude-standard` 为准，包含当前工作区新增文件；
- 包内统一放在 `gutcare-agents/` 目录下，解压即用；
- 附带 `PACKAGE_MANIFEST.json`（提交号、时间、每个文件的字节数与 sha256），
  便于评审核对材料完整性、也便于团队之间确认“发的是哪一版”。

用法：
    python tools/package_submission.py                       # → dist/GutCare_agents_<日期>.zip
    python tools/package_submission.py --out dist/x.zip
    python tools/package_submission.py --include-replays     # 带上 replays/（默认已包含被跟踪的录制）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PREFIXES = ("data/", ".git/", "__pycache__/")
EXCLUDE_SUFFIXES = (".pyc", ".pyo")
EXCLUDE_NAMES = {".env"}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tracked_files() -> list[Path]:
    try:
        names = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("未检测到 git，改用目录遍历（会排除 data/ 与 __pycache__）")
        names = [
            str(p.relative_to(ROOT)).replace("\\", "/")
            for p in sorted(ROOT.rglob("*"))
            if p.is_file()
        ]
    files = []
    for name in sorted(set(names)):
        if not name:
            continue
        if any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        if any(name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
            continue
        if Path(name).name in EXCLUDE_NAMES:
            continue
        path = ROOT / name
        if path.is_file():
            files.append(path)
    return files


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--name", default="gutcare-agents")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d")
    out = Path(args.out) if args.out else ROOT / "dist" / f"GutCare_agents_{stamp}.zip"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    files = tracked_files()
    if not files:
        print("没有可打包的文件", file=sys.stderr)
        return 1

    try:
        commit = git("rev-parse", "HEAD")
        commit_short = git("rev-parse", "--short", "HEAD")
        status = git("status", "--porcelain")
    except Exception:
        commit = commit_short = "unknown"
        status = ""

    manifest = {
        "package": out.name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "git_commit_short": commit_short,
        "working_tree_clean": not status,
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "entrypoint": f"{args.name}/README.md",
        "self_check": [
            f"cd {args.name}",
            "python -m pip install -r requirements.txt",
            "python -m unittest discover -s tests",
            "python -m gutcare_agents serve",
            "python tools/smoke_test.py",
        ],
        "files": [
            {
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            }
            for p in files
        ],
        "excluded": ["data/（运行期任务库/临时输出）", "__pycache__/", ".env（本地密钥）"],
    }

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            arcname = f"{args.name}/{path.relative_to(ROOT).as_posix()}"
            archive.write(path, arcname)
        archive.writestr(
            f"{args.name}/PACKAGE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2),
        )

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"已生成：{out}  ({size_mb:.2f} MB)")
    print(f"提交：{commit_short}  文件：{len(files)}  工作区{'干净' if not status else '有未提交改动'}")
    if status:
        print("提示：工作区有未提交改动，打包内容包含当前工作区文件，请核对清单")
    print("包内清单：PACKAGE_MANIFEST.json")
    print("解压后自检：python -m unittest discover -s tests  然后  python -m gutcare_agents serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
