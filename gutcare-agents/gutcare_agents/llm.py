"""模型适配层与三种运行模式。

- 统一走 OpenAI 兼容的 /chat/completions（DeepSeek 兼容该协议），
  更换供应商只需改 GCA_BASE_URL / GCA_*_MODEL。
- `replay` 模式把真实响应按“请求指纹”缓存到 replays/ 并可直接提交到仓库：
  断网也能演示、评委机能复现、决赛不会因第三方故障翻车。
- 不在此层做业务判断：结构化校验在 validation.py，降级策略在 orchestrator.py。
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import AgentError, ErrorCode


@dataclass
class LLMResponse:
    text: str
    model: str
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    mode: str = "live"
    raw: dict[str, Any] = field(default_factory=dict)

    def usage(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "mode": self.mode,
        }


def extract_json(text: str) -> Any:
    """从模型输出中稳健地取出 JSON。

    模型偶尔会包 ```json 代码块或加一句说明；这里只做“取第一个完整 JSON 值”，
    不做语义修补——修补会掩盖真实的结构错误。
    """
    if text is None:
        raise AgentError(ErrorCode.MODEL_INVALID_OUTPUT, "模型返回为空")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (stripped.find("{"), stripped.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise AgentError(ErrorCode.MODEL_INVALID_OUTPUT, "模型输出不含 JSON", detail={"head": stripped[:200]})
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : idx + 1])
                except json.JSONDecodeError as exc:
                    raise AgentError(
                        ErrorCode.MODEL_INVALID_OUTPUT,
                        f"JSON 解析失败: {exc}",
                        detail={"head": stripped[start : start + 200]},
                    ) from exc
    raise AgentError(ErrorCode.MODEL_INVALID_OUTPUT, "JSON 不完整", detail={"head": stripped[start : start + 200]})


def image_data_uri(path: Path, max_side: int, max_bytes: int) -> tuple[str, str]:
    """把本地图片转成 data URI；过大时等比缩放到 max_side。

    返回 (data_uri, sha256)，后者用于 replay 指纹，保证同一张图命中同一缓存。
    """
    path = Path(path)
    if not path.is_file():
        raise AgentError(ErrorCode.FILE_NOT_FOUND, f"图片不存在: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if len(raw) > max_bytes or True:
        # 统一走一次 PIL 规整：修正 EXIF 方向、限制长边、控制体积（模型侧更稳定）
        try:
            from PIL import Image

            import io

            with Image.open(path) as img:
                img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=88, optimize=True)
                raw = buf.getvalue()
                mime = "image/jpeg"
        except Exception:  # 图片损坏时保留原始字节，让上层给出明确失败
            pass
    import base64

    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", digest


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mode = settings.mode
        self.settings.ensure_dirs()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def complete(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        model: str | None = None,
        images: list[Path] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse:
        model = model or self.settings.text_model
        temperature = self.settings.temperature if temperature is None else temperature
        max_tokens = max_tokens or self.settings.max_tokens
        images = images or []

        content: list[dict[str, Any]] | str
        image_digests: list[str] = []
        if images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": user}]
            for path in images:
                uri, digest = image_data_uri(path, self.settings.max_image_side, self.settings.max_image_bytes)
                image_digests.append(digest)
                parts.append({"type": "image_url", "image_url": {"url": uri}})
            content = parts
        else:
            content = user

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        key = self._cache_key(model, messages, temperature, max_tokens, image_digests, agent)

        if self.mode == "replay":
            cached = self._read_replay(key)
            if cached is not None:
                return LLMResponse(
                    text=cached["text"],
                    model=cached.get("model", model),
                    latency_ms=0,
                    prompt_tokens=cached.get("prompt_tokens", 0),
                    completion_tokens=cached.get("completion_tokens", 0),
                    cached=True,
                    mode="replay",
                )
            if not self.settings.replay_fallback:
                raise AgentError(
                    ErrorCode.MODEL_UNAVAILABLE,
                    f"replay 模式缺少录制结果（agent={agent}, key={key[:12]}）；请用 live 模式先录制，"
                    "或设置 GCA_REPLAY_FALLBACK=1 允许缺省时退回 mock（会在结果里标注）",
                )
            # 明确标注的兜底：演示现场缺录制时不让整场挂掉，但绝不把它当成真实模型结果
            from .mock import mock_text

            payload = mock_text(agent, user, images, context)
            return LLMResponse(
                text=payload + "\n<!-- replay_mock_fallback -->",
                model="mock(fallback)",
                mode="replay_mock_fallback",
            )

        if self.mode == "mock":
            from .mock import mock_text

            return LLMResponse(text=mock_text(agent, user, images, context), model="mock", mode="mock")

        if not self.settings.api_key:
            raise AgentError(
                ErrorCode.MODEL_UNAVAILABLE,
                "未配置 GCA_API_KEY，无法使用 live 模式；可改用 GCA_MODE=mock 或 replay",
            )

        started = time.monotonic()
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        body = self._post_with_retry(payload)
        latency = int((time.monotonic() - started) * 1000)
        choice = (body.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        usage = body.get("usage") or {}
        response = LLMResponse(
            text=text,
            model=body.get("model", model),
            latency_ms=latency,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            mode="live",
        )
        if not text:
            raise AgentError(ErrorCode.MODEL_INVALID_OUTPUT, "模型返回空内容", agent=agent)
        self._write_replay(
            key,
            {
                "agent": agent,
                "model": response.model,
                "text": text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        return response

    def complete_json(
        self,
        *,
        agent: str,
        system: str,
        user: str,
        images: list[Path] | None = None,
        model: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[Any, LLMResponse]:
        response = self.complete(
            agent=agent, system=system, user=user, images=images, model=model, context=context
        )
        return extract_json(response.text), response

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _post_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self._post(payload)
            except AgentError as exc:
                last = exc
                if not exc.retryable or attempt >= self.settings.max_retries:
                    raise
                # 退避更宽容：部分平台的免费/低价模型有严格的 RPM/TPM 限制
                time.sleep(min(2 ** attempt, 8) * 1.0)
        raise last or AgentError(ErrorCode.MODEL_UNAVAILABLE, "模型调用失败")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.settings.base_url}/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "ignore")[:500]
            except Exception:
                pass
            if exc.code in (401, 403):
                raise AgentError(ErrorCode.MODEL_UNAVAILABLE, f"鉴权失败({exc.code})，请检查 GCA_API_KEY", detail={"body": detail})
            if exc.code == 429:
                raise AgentError(ErrorCode.MODEL_RATE_LIMITED, "触发限流", detail={"body": detail})
            if exc.code >= 500:
                raise AgentError(ErrorCode.MODEL_UNAVAILABLE, f"服务端错误({exc.code})", detail={"body": detail})
            if exc.code == 400 and "image" in detail.lower():
                raise AgentError(
                    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                    "当前模型/接口拒绝了图像输入；请确认 GCA_VISION_MODEL 支持视觉，或改走文本输入降级路径",
                    retryable=False,
                    detail={"body": detail},
                )
            raise AgentError(ErrorCode.MODEL_INVALID_OUTPUT, f"请求被拒绝({exc.code})", retryable=False, detail={"body": detail})
        except urllib.error.URLError as exc:
            raise AgentError(ErrorCode.MODEL_UNAVAILABLE, f"网络不可达: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AgentError(ErrorCode.MODEL_TIMEOUT, "模型调用超时") from exc

    # ------------------------------------------------------------------
    # Replay 缓存
    # ------------------------------------------------------------------
    def _cache_key(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        image_digests: list[str],
        agent: str,
    ) -> str:
        # 图像内容用摘要代替，避免把 base64 写进指纹
        normalized = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                content = [c for c in content if c.get("type") != "image_url"]
            normalized.append({"role": m["role"], "content": content})
        material = {
            "agent": agent,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": normalized,
            "images": image_digests,
        }
        return hashlib.sha256(
            json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _replay_path(self, key: str) -> Path:
        return self.settings.replay_dir / key[:2] / f"{key}.json"

    def _read_replay(self, key: str) -> dict[str, Any] | None:
        path = self._replay_path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_replay(self, key: str, payload: dict[str, Any]) -> None:
        path = self._replay_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
