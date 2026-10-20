"""PDF input preprocessing: embedded text first, bounded page rendering otherwise."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import AgentError, ErrorCode


@dataclass
class PdfInput:
    page_count: int
    processed_pages: int
    text: str
    needs_visual: bool


def inspect_pdf(path: Path, max_pages: int = 5, max_bytes: int = 12 * 1024 * 1024) -> PdfInput:
    if not path.is_file():
        raise AgentError(ErrorCode.FILE_NOT_FOUND, "PDF 文件不存在")
    if path.stat().st_size > max_bytes:
        raise AgentError(ErrorCode.FILE_TOO_LARGE, "PDF 文件超过处理大小限制")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise AgentError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "PDF 文本提取依赖 pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise AgentError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "加密 PDF 无法解析")
        count = len(reader.pages)
        if count == 0:
            raise AgentError(ErrorCode.INPUT_DATA_INSUFFICIENT, "PDF 没有页面")
        selected = min(count, max_pages)
        pages = [(reader.pages[i].extract_text() or "").strip() for i in range(selected)]
    except AgentError:
        raise
    except Exception as exc:
        raise AgentError(ErrorCode.IMAGE_UNREADABLE, "PDF 损坏或无法读取") from exc
    text = "\n\n".join(f"[第 {i} 页]\n{page}" for i, page in enumerate(pages, 1) if page)
    return PdfInput(count, selected, text[:12000], any(len(page) < 20 for page in pages))


def render_pdf(path: Path, output_dir: Path, *, max_pages: int = 5, max_side: int = 1800, timeout_s: float = 30) -> list[Path]:
    binary = shutil.which("pdftoppm")
    if not binary:
        raise AgentError(ErrorCode.UNSUPPORTED_MEDIA_TYPE, "扫描版 PDF 需要 Poppler 的 pdftoppm")
    prefix = output_dir / "page"
    try:
        subprocess.run(
            [binary, "-f", "1", "-l", str(max_pages), "-scale-to", str(max_side), "-png", str(path), str(prefix)],
            check=True, capture_output=True, timeout=timeout_s,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        raise AgentError(ErrorCode.IMAGE_UNREADABLE, "PDF 页面渲染失败") from exc
    pages = sorted(output_dir.glob("page-*.png"))
    if not pages:
        raise AgentError(ErrorCode.IMAGE_UNREADABLE, "PDF 未渲染出页面")
    return pages
