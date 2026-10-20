"""运行时配置。

全部通过环境变量注入，密钥不写进代码与文档。
三种运行模式：
  live   —— 真实调用模型 API
  mock   —— 不联网，返回确定性的“模拟结构化输出”（用于联调与断网演示）
  replay —— 读取真实调用录制下来的响应缓存（用于稳定演示与可复现评测）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass
class Settings:
    mode: str = "mock"                     # live | mock | replay
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    text_model: str = "deepseek-chat"
    vision_model: str = "deepseek-chat"    # 上线前用一张真实报告图冒烟确认
    embedding_model: str = ""              # 留空表示只用本地检索
    timeout_s: float = 60.0
    max_retries: int = 2
    temperature: float = 0.2
    max_tokens: int = 1600

    fixture_root: Path = field(default_factory=lambda: WORKSPACE_ROOT / "testdata_v4")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    replay_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "replays")
    knowledge_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge")
    prompts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    rules_path: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge" / "safety_rules.json")

    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "gutcare_agents.sqlite3")
    http_host: str = "127.0.0.1"
    http_port: int = 8100
    internal_token: str = ""               # 为空表示不校验（仅本地开发）
    backend_base_url: str = ""              # F08 业务后端内部 API
    backend_service_token: str = ""         # 只用于后端内部 API，不写入日志
    log_level: str = "INFO"
    max_image_bytes: int = 12 * 1024 * 1024
    max_image_side: int = 2048
    #: replay 模式下若缺少录制结果：False=直接报错（默认，避免把 mock 当成真实结果）
    replay_fallback: bool = False

    #: 建议生成的数据充足性门槛（低于门槛必须降级，不允许硬编结论）
    min_days_coverage: int = 3
    min_records_for_trend: int = 5

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.replay_dir, self.knowledge_dir):
            path.mkdir(parents=True, exist_ok=True)
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(
                f"缺少提示词目录：{self.prompts_dir}（部署时需与代码一起拷贝 prompts/）"
            )

    @property
    def live_ready(self) -> bool:
        return bool(self.api_key)

    def describe(self) -> dict[str, object]:
        """对外暴露的配置摘要（绝不包含密钥）。"""
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "text_model": self.text_model,
            "vision_model": self.vision_model,
            "text_model_configured": bool(self.text_model),
            "api_key_present": bool(self.api_key),
            "backend_configured": bool(self.backend_base_url and self.backend_service_token),
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "fixture_root": str(self.fixture_root),
            "rules_path": str(self.rules_path),
            "prompts_dir": str(self.prompts_dir),
            "replay_fallback": self.replay_fallback,
        }


def load_settings(**overrides: object) -> Settings:
    mode = _env("GCA_MODE", "mock").lower()
    if mode not in {"live", "mock", "replay"}:
        mode = "mock"
    settings = Settings(
        mode=mode,
        base_url=_env("GCA_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        api_key=_env("GCA_API_KEY"),
        text_model=_env("GCA_TEXT_MODEL", "deepseek-chat"),
        vision_model=_env("GCA_VISION_MODEL", _env("GCA_TEXT_MODEL", "deepseek-chat")),
        embedding_model=_env("GCA_EMBEDDING_MODEL"),
        timeout_s=_env_float("GCA_TIMEOUT_S", 60.0),
        max_retries=_env_int("GCA_MAX_RETRIES", 2),
        temperature=_env_float("GCA_TEMPERATURE", 0.2),
        max_tokens=_env_int("GCA_MAX_TOKENS", 1600),
        http_host=_env("GCA_HOST", "127.0.0.1"),
        http_port=_env_int("GCA_PORT", 8100),
        internal_token=_env("GCA_INTERNAL_TOKEN"),
        backend_base_url=_env("GCA_BACKEND_BASE_URL").rstrip("/"),
        backend_service_token=_env("GCA_BACKEND_SERVICE_TOKEN"),
        log_level=_env("GCA_LOG_LEVEL", "INFO").upper(),
        min_days_coverage=_env_int("GCA_MIN_DAYS", 3),
        min_records_for_trend=_env_int("GCA_MIN_RECORDS", 5),
        replay_fallback=_env("GCA_REPLAY_FALLBACK", "0") in ("1", "true", "True", "yes"),
    )
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise AttributeError(f"未知配置项: {key}")
        setattr(settings, key, value)
    # 测试包路径：文档里承诺可用 GCA_FIXTURE_ROOT 指定（命令行/CI/容器都用它）
    fixture_env = _env("GCA_FIXTURE_ROOT")
    if fixture_env and "fixture_root" not in overrides:
        settings.fixture_root = Path(fixture_env)
    # 任务库路径可单独指定：录制/评测时用干净库，避免幂等复用到旧结果
    db_env = _env("GCA_DB")
    if db_env and "db_path" not in overrides:
        settings.db_path = Path(db_env)
    settings.ensure_dirs()
    return settings
