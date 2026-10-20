"""模型适配层测试：JSON 解析、图像处理与三种模式的行为边界。

这些是“模型不可控”的地方，也是最容易在演示现场出问题的地方：
模型多写一句解释、包一层代码块、返回被截断的 JSON、图片太大——都必须有确定行为。
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.config import load_settings  # noqa: E402
from gutcare_agents.errors import AgentError  # noqa: E402
from gutcare_agents.llm import LLMClient, extract_json, image_data_uri  # noqa: E402


class _TmpDir:
    def __init__(self) -> None:
        base = ROOT / "data" / "testtmp"
        base.mkdir(parents=True, exist_ok=True)
        self.name = str(base / f"t{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


def make_settings(tmp: str, **overrides):
    base = dict(
        mode="mock",
        data_dir=Path(tmp),
        replay_dir=Path(tmp) / "replays",
        db_path=Path(tmp) / "t.sqlite3",
        knowledge_dir=ROOT / "knowledge",
        rules_path=ROOT / "knowledge" / "safety_rules.json",
    )
    base.update(overrides)
    return load_settings(**base)


class TestJsonParsing(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_code_fence(self) -> None:
        text = '```json\n{"recognizable": true, "foods": []}\n```'
        self.assertEqual(extract_json(text)["recognizable"], True)

    def test_prefixed_explanation(self) -> None:
        text = '好的，结果如下：\n{"status": "ok"}\n以上。'
        self.assertEqual(extract_json(text)["status"], "ok")

    def test_nested_braces_in_strings(self) -> None:
        text = '{"note": "这里有个 } 符号", "value": 2}'
        self.assertEqual(extract_json(text)["value"], 2)

    def test_truncated_json_raises(self) -> None:
        with self.assertRaises(AgentError) as ctx:
            extract_json('{"a": {"b": 1}')
        self.assertEqual(ctx.exception.code, "model_invalid_output")

    def test_empty_response_raises(self) -> None:
        with self.assertRaises(AgentError):
            extract_json("   ")

    def test_no_json_raises(self) -> None:
        with self.assertRaises(AgentError):
            extract_json("抱歉，我无法处理这个请求。")


class TestImageHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_image(self, size: tuple[int, int], name: str = "big.png") -> Path:
        from PIL import Image

        path = Path(self.tmp.name) / name
        Image.new("RGB", size, (200, 120, 90)).save(path, format="PNG")
        return path

    def test_large_image_is_downscaled(self) -> None:
        path = self._make_image((4000, 3000))
        uri, digest = image_data_uri(path, max_side=1024, max_bytes=20 * 1024 * 1024)
        self.assertTrue(uri.startswith("data:image/"))
        self.assertEqual(len(digest), 64)
        import base64

        raw = base64.b64decode(uri.split(",", 1)[1])
        from PIL import Image
        import io

        with Image.open(io.BytesIO(raw)) as img:
            self.assertLessEqual(max(img.size), 1024)

    def test_small_image_still_works(self) -> None:
        path = self._make_image((320, 240), "small.png")
        uri, digest = image_data_uri(path, max_side=2048, max_bytes=20 * 1024 * 1024)
        self.assertIn("base64,", uri)
        self.assertEqual(len(digest), 64)

    def test_missing_image_raises(self) -> None:
        with self.assertRaises(AgentError) as ctx:
            image_data_uri(Path(self.tmp.name) / "nope.png", 1024, 1024)
        self.assertEqual(ctx.exception.code, "file_not_found")

    def test_digest_is_stable_across_normalization(self) -> None:
        path = self._make_image((640, 480), "stable.png")
        _, first = image_data_uri(path, max_side=2048, max_bytes=20 * 1024 * 1024)
        _, second = image_data_uri(path, max_side=2048, max_bytes=20 * 1024 * 1024)
        self.assertEqual(first, second, "同一文件的缓存键必须稳定")


class TestModes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _TmpDir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_mock_mode_never_calls_network(self) -> None:
        client = LLMClient(make_settings(self.tmp.name, mode="mock"))
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("不应联网")):
            response = client.complete(agent="trend", system="s", user="u", context={})
        self.assertEqual(response.mode, "mock")
        self.assertEqual(response.model, "mock")

    def test_live_without_key_fails_fast(self) -> None:
        client = LLMClient(make_settings(self.tmp.name, mode="live", api_key=""))
        with self.assertRaises(AgentError) as ctx:
            client.complete(agent="trend", system="s", user="u")
        self.assertEqual(ctx.exception.code, "model_unavailable")
        self.assertIn("GCA_API_KEY", ctx.exception.message)

    def test_replay_missing_file_reports_clearly(self) -> None:
        client = LLMClient(make_settings(self.tmp.name, mode="replay", replay_fallback=False))
        with self.assertRaises(AgentError) as ctx:
            client.complete(agent="trend", system="s", user="u")
        self.assertEqual(ctx.exception.code, "model_unavailable")
        self.assertIn("replay", ctx.exception.message)

    def test_replay_fallback_is_labelled(self) -> None:
        client = LLMClient(make_settings(self.tmp.name, mode="replay", replay_fallback=True))
        response = client.complete(agent="trend", system="s", user="u", context={"facts": {}, "record_ids": []})
        self.assertEqual(response.mode, "replay_mock_fallback")
        payload = json.loads(response.text.split("\n<!--")[0])
        self.assertTrue(payload.get("_mock"))

    def test_replay_hit_returns_recorded_text(self) -> None:
        settings = make_settings(self.tmp.name, mode="replay")
        client = LLMClient(settings)
        key = client._cache_key(
            settings.text_model, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], 0.2, 1600, [], "trend"
        )
        client._write_replay(key, {"text": '{"ok": true}', "model": "recorded-model"})
        response = client.complete(agent="trend", system="s", user="u", temperature=0.2, max_tokens=1600)
        self.assertEqual(response.mode, "replay")
        self.assertTrue(response.cached)
        self.assertEqual(response.model, "recorded-model")
        self.assertEqual(extract_json(response.text)["ok"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
