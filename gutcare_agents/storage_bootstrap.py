"""存储单例工厂（CLI 与 HTTP 服务共用，避免重复建连接/重复建表）。"""
from __future__ import annotations

from functools import lru_cache

from .config import Settings
from .store import Store


@lru_cache(maxsize=4)
def _store_for(path: str) -> Store:
    return Store(path)


def get_store(settings: Settings) -> Store:
    return _store_for(str(settings.db_path))
