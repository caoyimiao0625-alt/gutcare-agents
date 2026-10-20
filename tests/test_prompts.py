"""提示词加载与版本一致性测试。

为什么单独测：提示词已从代码移到 `prompts/`，一旦文件没随代码部署、或改了内容忘记升版本，
系统会静默用错提示词——这类问题在演示现场最难查。这里把它变成启动即失败。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gutcare_agents.agents import AGENT_CLASSES  # noqa: E402
from gutcare_agents.prompts import (  # noqa: E402
    PROMPT_ROOT,
    PromptNotFound,
    available,
    load_prompt,
    prompt_meta,
    prompt_path,
)


class TestPromptLoading(unittest.TestCase):
    def test_every_prompt_file_present(self) -> None:
        expected = {
            "medical.report",
            "medical.report_text",
            "medical.ocr",
            "medical.advice_doc",
            "diet.meal",
            "stool.image",
            "trend.narrative",
            "advice.generate",
            "conversation.assistant",
        }
        found = {item["name"] for item in available()}
        self.assertTrue(expected <= found, f"缺少提示词：{expected - found}")

    def test_agent_declared_prompt_exists(self) -> None:
        for name, cls in AGENT_CLASSES.items():
            key = getattr(cls, "prompt_key", "")
            version = getattr(cls, "prompt_version", "v1")
            if not key:
                continue
            with self.subTest(agent=name, prompt=key):
                text = load_prompt(key, version)
                self.assertGreater(len(text), 200, "提示词过短，可能误删")
                self.assertIn("JSON", text, "提示词应要求结构化输出")

    def test_medical_three_prompts_load(self) -> None:
        texts = {
            name: load_prompt(name, "v1")
            for name in ("medical.report", "medical.ocr", "medical.advice_doc")
        }
        self.assertIn("[遮挡]", texts["medical.report"])
        self.assertIn("[不清]", texts["medical.ocr"])
        self.assertIn("模拟", texts["medical.advice_doc"])
        # 三套提示词必须彼此不同，避免“同一段话改个名字”
        self.assertEqual(len(set(texts.values())), 3)

    def test_prompt_meta_is_stable_and_short(self) -> None:
        meta = prompt_meta("trend.narrative", "v1")
        self.assertEqual(meta["version"], "v1")
        self.assertEqual(len(meta["sha256"]), 16)
        again = prompt_meta("trend.narrative", "v1")
        self.assertEqual(meta["sha256"], again["sha256"])

    def test_missing_prompt_fails_loudly(self) -> None:
        with self.assertRaises(PromptNotFound):
            load_prompt("medical.report", "v99")

    def test_prompt_path_layout(self) -> None:
        self.assertTrue(str(prompt_path("diet.meal", "v1")).endswith("diet.meal.v1.md"))
        self.assertEqual(PROMPT_ROOT.name, "prompts")

    def test_prompts_do_not_contain_placeholders_left_unfilled(self) -> None:
        for item in available():
            text = Path(item["path"]).read_text(encoding="utf-8")
            self.assertNotIn("{unknown", text)
            self.assertNotIn("TODO", text)
            self.assertNotIn("FIXME", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
