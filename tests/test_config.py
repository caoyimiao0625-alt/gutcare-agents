"""配置加载测试：环境变量必须真的生效（文档承诺过的东西不能只是写着好看）。"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings  # noqa: E402


class TestSettingsFromEnv(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()

    def test_fixture_root_env_is_honoured(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_FIXTURE_ROOT": r"E:\some\fixtures"}):
            settings = load_settings()
            self.assertEqual(settings.fixture_root, Path(r"E:\some\fixtures"))

    def test_explicit_override_wins_over_env(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_FIXTURE_ROOT": r"E:\from\env"}):
            settings = load_settings(fixture_root=Path(r"E:\from\arg"))
            self.assertEqual(settings.fixture_root, Path(r"E:\from\arg"))

    def test_mode_env_is_validated(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_MODE": "LIVE"}):
            self.assertEqual(load_settings().mode, "live")
        with mock.patch.dict(os.environ, {"GCA_MODE": "nonsense"}):
            self.assertEqual(load_settings().mode, "mock", "非法模式应回退到 mock，而不是崩溃")

    def test_replay_fallback_flag(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_REPLAY_FALLBACK": "1"}):
            self.assertTrue(load_settings().replay_fallback)
        with mock.patch.dict(os.environ, {"GCA_REPLAY_FALLBACK": "0"}):
            self.assertFalse(load_settings().replay_fallback)

    def test_describe_never_leaks_key(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_API_KEY": "sk-should-not-appear"}):
            described = load_settings().describe()
        self.assertNotIn("sk-should-not-appear", str(described))
        self.assertTrue(described["api_key_present"])

    def test_thresholds_come_from_env(self) -> None:
        with mock.patch.dict(os.environ, {"GCA_MIN_DAYS": "5", "GCA_MIN_RECORDS": "9"}):
            settings = load_settings()
            self.assertEqual(settings.min_days_coverage, 5)
            self.assertEqual(settings.min_records_for_trend, 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
