"""交付工具测试：打包脚本与部署自检脚本。

这两条命令是**评审/队友会真的去执行的入口**，所以必须本身也有测试：
- 打包：清单齐全、不含 data/、含 prompts/knowledge、带提交号与 sha256；
- 自检：对真实起的服务跑一遍，全部 PASS 且退出码为 0。
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from gutcare_agents.api import make_server  # noqa: E402
from gutcare_agents.config import load_settings  # noqa: E402

FIXTURE_ROOT = ROOT.parent / "testdata_v4"


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


class TestPackageSubmission(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_package_manifest_and_contents(self) -> None:
        import package_submission

        out = Path(self.tmp.name) / "pkg.zip"
        with mock.patch.object(sys, "argv", ["package_submission.py", "--out", str(out)]):
            self.assertEqual(package_submission.main(), 0)
        self.assertTrue(out.is_file())

        with zipfile.ZipFile(out) as archive:
            names = archive.namelist()
            manifest = json.loads(archive.read("gutcare-agents/PACKAGE_MANIFEST.json"))

        self.assertGreater(manifest["file_count"], 50)
        self.assertTrue(manifest["git_commit"])
        self.assertEqual(len(manifest["files"]), manifest["file_count"])
        for item in manifest["files"][:5]:
            self.assertEqual(len(item["sha256"]), 64)

        # 必须包含运行所需目录，且不含运行期数据
        for required in ("prompts/medical.report.v1.md", "knowledge/safety_rules.json", "console/index.html"):
            self.assertTrue(
                any(name == f"gutcare-agents/{required}" for name in names),
                f"包内缺少 {required}",
            )
        self.assertFalse(any("/data/" in name for name in names), "不应打包 data/")
        self.assertFalse(any(name.endswith(".pyc") for name in names), "不应打包 __pycache__")
        self.assertFalse(any(name.endswith("/.env") for name in names), "不应打包 .env")


class TestDocsConsistency(unittest.TestCase):
    def test_document_paths_exist(self) -> None:
        import check_docs

        checked, missing = check_docs.check()
        self.assertGreater(checked, 50, "应扫描到足够多的路径引用")
        detail = "\n".join(f"{path}:{lineno} {token}" for path, lineno, token in missing)
        self.assertFalse(missing, f"文档引用了不存在的路径：\n{detail}")


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestDemoRun(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_demo_runs_and_writes_artifacts(self) -> None:
        import demo_run

        out = Path(self.tmp.name) / "demo"
        with mock.patch.object(sys, "argv", ["demo_run.py", "--mode", "mock", "--out", str(out)]):
            self.assertEqual(demo_run.main(), 0)

        self.assertEqual(len(demo_run.STEPS), 6, "演示序列应包含 6 个步骤")
        expected = [
            "01_report_extract.json",
            "02_meal.json",
            "03_safety.json",
            "04_analysis.json",
            "05_reuse.json",
            "06_degraded.json",
            "transcript.md",
        ]
        for name in expected:
            self.assertTrue((out / name).is_file(), f"缺少产物 {name}")

        transcript = (out / "transcript.md").read_text(encoding="utf-8")
        self.assertIn("演示运行记录", transcript)
        self.assertIn("总耗时", transcript)

        # 演示里最关键的两个可信度点必须能在产物中找到
        degraded = json.loads((out / "06_degraded.json").read_text(encoding="utf-8"))
        self.assertEqual(degraded["unconfirmed"]["report"]["advice"], [])
        analysis = json.loads((out / "04_analysis.json").read_text(encoding="utf-8"))
        for item in analysis["report"]["advice"]:
            self.assertTrue(item.get("evidence_obs_ids") or item.get("knowledge_ids"))


@unittest.skipUnless((FIXTURE_ROOT / "inputs/test_catalog.json").is_file(), "缺少测试包")
class TestSmokeTestScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = _TmpDir()
        cls.settings = load_settings(
            mode="mock",
            fixture_root=FIXTURE_ROOT,
            data_dir=Path(cls.tmp.name),
            replay_dir=Path(cls.tmp.name) / "replays",
            db_path=Path(cls.tmp.name) / "smoke.sqlite3",
            knowledge_dir=ROOT / "knowledge",
            rules_path=ROOT / "knowledge" / "safety_rules.json",
            internal_token="smoke-token",
            http_host="127.0.0.1",
            http_port=0,
        )
        cls.server = make_server(cls.settings)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def test_smoke_passes_against_live_server(self) -> None:
        import smoke_test

        argv = [
            "smoke_test.py",
            "--base-url",
            f"http://127.0.0.1:{self.port}",
            "--token",
            "smoke-token",
        ]
        with mock.patch.object(sys, "argv", argv):
            code = smoke_test.main()
        self.assertEqual(code, 0)
        self.assertFalse([r for r in smoke_test.RESULTS if r[0] == "FAIL"])
        passed = [r for r in smoke_test.RESULTS if r[0] == "PASS"]
        self.assertGreaterEqual(len(passed), 10)

    def test_smoke_reports_failure_when_service_down(self) -> None:
        import smoke_test

        smoke_test.RESULTS.clear()
        argv = ["smoke_test.py", "--base-url", "http://127.0.0.1:59999"]
        with mock.patch.object(sys, "argv", argv):
            code = smoke_test.main()
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
