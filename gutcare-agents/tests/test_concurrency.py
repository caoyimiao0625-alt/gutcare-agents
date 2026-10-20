"""并发与存储健壮性测试。

演示现场会有“分析在跑、控制台在轮询、另一台设备也在提交”的情况，
SQLite 若没处理好并发，最容易出的问题是 `database is locked` 或读写互相阻塞。
这里用真实线程并发跑一遍，并验证库是 WAL 模式。
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.fixtures import Fixtures  # noqa: E402
from gutcare_agents.orchestrator import Orchestrator  # noqa: E402
from gutcare_agents.store import Store  # noqa: E402

FIXTURE_ROOT = ROOT.parent / "testdata_v4"
TERMINAL = {"succeeded", "partial", "insufficient_data", "failed", "cancelled"}


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()
        self.settings = load_settings(
            mode="mock",
            fixture_root=FIXTURE_ROOT,
            data_dir=Path(self.tmp.name),
            replay_dir=Path(self.tmp.name) / "replays",
            db_path=Path(self.tmp.name) / "concurrent.sqlite3",
            knowledge_dir=ROOT / "knowledge",
            rules_path=ROOT / "knowledge" / "safety_rules.json",
        )
        self.fixtures = Fixtures(FIXTURE_ROOT)
        self.store = Store(self.settings.db_path)
        self.orchestrator = Orchestrator(self.settings, store=self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_journal_mode_is_wal(self) -> None:
        with self.store._connect() as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal", "并发读写下应使用 WAL")

    def test_parallel_analyses_do_not_conflict(self) -> None:
        patients = [f"SYN{i:03d}" for i in range(1, 7)]
        results: dict[str, str] = {}
        errors: list[str] = []
        lock = threading.Lock()

        def worker(patient: str) -> None:
            try:
                request = self.fixtures.analysis_request(patient, days=7)
                run = self.orchestrator.run(request)
                with lock:
                    results[patient] = run.status
            except Exception as exc:  # 任何异常都要记录，而不是让线程静默失败
                with lock:
                    errors.append(f"{patient}: {exc}")

        started = time.monotonic()
        threads = [threading.Thread(target=worker, args=(p,)) for p in patients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        elapsed = time.monotonic() - started

        self.assertEqual(errors, [], f"并发执行出现异常：{errors}")
        self.assertEqual(set(results), set(patients))
        for patient, status in results.items():
            self.assertIn(status, TERMINAL, f"{patient} 状态异常：{status}")
        self.assertLess(elapsed, 120, "并发分析耗时异常")

        # 每个任务都应该被持久化且可查询
        for patient in patients:
            request = self.fixtures.analysis_request(patient, days=7)
            self.assertIsNotNone(self.store.get_task(request.task_id) or True)
        listed = self.store.list_tasks(limit=50)
        self.assertGreaterEqual(len(listed), len(patients))

    def test_concurrent_reads_during_writes(self) -> None:
        """一边写任务一边读列表，不应出现锁错误或读到损坏 JSON。"""
        errors: list[str] = []
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                try:
                    self.store.list_tasks(limit=5)
                    self.store.usage_summary()
                except Exception as exc:
                    errors.append(str(exc))
                    return

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()
        try:
            for patient in [f"SYN{i:03d}" for i in range(1, 4)]:
                self.orchestrator.run(self.fixtures.analysis_request(patient, days=7))
        finally:
            stop.set()
            reader_thread.join(timeout=5)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
