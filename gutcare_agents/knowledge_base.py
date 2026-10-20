"""轻量知识检索（零依赖）。

为什么不用向量库：比赛周期内第一版需要“可复现、可解释、可离线”。
这里用中文字符二元组 + TF-IDF 余弦相似度，召回结果直接对应到知识条目 id，
便于在建议里做引用校验；若后续接入 embedding，只需替换 `_score` 的实现，
引用与降级逻辑不变（检索不到就明确说“无审核依据”，不允许模型自行补知识）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)
ASCII_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9\-]+")
CJK = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class KnowledgeDoc:
    doc_id: str
    title: str
    text: str
    source: str = ""
    review_status: str = "unverified_demo"
    tags: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{' '.join(self.tags)}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "review_status": self.review_status,
            "tags": self.tags,
        }


def tokenize(text: str) -> list[str]:
    """中文用字符二元组，英文用单词；不需要分词器，召回足够稳定。"""
    tokens: list[str] = []
    chars = [c for c in text if CJK.match(c)]
    tokens.extend("".join(chars[i : i + 2]) for i in range(len(chars) - 1))
    tokens.extend(w.lower() for w in ASCII_WORD.findall(text))
    return tokens


def parse_doc(path: Path) -> KnowledgeDoc:
    raw = path.read_text(encoding="utf-8")
    match = FRONT_MATTER.match(raw)
    meta: dict[str, str] = {}
    body = raw
    if match:
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        body = match.group(2)
    return KnowledgeDoc(
        doc_id=meta.get("id") or path.stem,
        title=meta.get("title") or path.stem,
        text=body.strip(),
        source=meta.get("source", ""),
        review_status=meta.get("review_status", "unverified_demo"),
        tags=[t.strip() for t in meta.get("tags", "").replace("，", ",").split(",") if t.strip()],
    )


class KnowledgeBase:
    def __init__(self, docs: Iterable[KnowledgeDoc]) -> None:
        self.docs = list(docs)
        self._build_index()

    @classmethod
    def load(cls, directory: Path | str) -> "KnowledgeBase":
        directory = Path(directory)
        docs = [parse_doc(p) for p in sorted(directory.glob("*.md"))]
        return cls(docs)

    def _build_index(self) -> None:
        self._tf: list[dict[str, float]] = []
        df: dict[str, int] = {}
        for doc in self.docs:
            tokens = tokenize(doc.full_text)
            counts: dict[str, float] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0.0) + 1.0
            total = sum(counts.values()) or 1.0
            self._tf.append({k: v / total for k, v in counts.items()})
            for token in counts:
                df[token] = df.get(token, 0) + 1
        n = max(1, len(self.docs))
        self._idf = {token: math.log((1 + n) / (1 + freq)) + 1.0 for token, freq in df.items()}
        self._norms = [
            math.sqrt(sum((value * self._idf.get(token, 1.0)) ** 2 for token, value in tf.items())) or 1.0
            for tf in self._tf
        ]

    def search(self, query: str, top_k: int = 3, min_score: float = 0.06) -> list[tuple[KnowledgeDoc, float]]:
        if not self.docs or not query:
            return []
        counts: dict[str, float] = {}
        for token in tokenize(query):
            counts[token] = counts.get(token, 0.0) + 1.0
        total = sum(counts.values()) or 1.0
        q = {k: v / total for k, v in counts.items()}
        qnorm = math.sqrt(sum((value * self._idf.get(token, 1.0)) ** 2 for token, value in q.items())) or 1.0
        scored: list[tuple[KnowledgeDoc, float]] = []
        for idx, doc in enumerate(self.docs):
            tf = self._tf[idx]
            score = 0.0
            for token, value in q.items():
                if token in tf:
                    idf = self._idf.get(token, 1.0)
                    score += value * idf * tf[token] * idf
            score = score / (qnorm * self._norms[idx])
            if score >= min_score:
                scored.append((doc, round(score, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    @property
    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]

    def get(self, doc_id: str) -> KnowledgeDoc | None:
        return next((d for d in self.docs if d.doc_id == doc_id), None)

    def describe(self) -> dict[str, Any]:
        return {
            "documents": len(self.docs),
            "review_status": sorted({d.review_status for d in self.docs}),
            "ids": self.doc_ids,
        }
