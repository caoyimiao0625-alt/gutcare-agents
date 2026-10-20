# 提示词目录

本目录存放各 Agent 的系统提示词，**与代码分离**，便于队友审阅、评审追溯和版本管理。

## 命名约定

```
<agent>.<task>.v<主版本>.md     例如：medical.report.v1.md
```

- 修改提示词时**新建版本文件**（例如把 v1 复制为 v2 后再改），不要原地改历史版本；
- 同时把对应 Agent 类里的 `prompt_version` 改成新版本号；
- `AgentResult` 会带 `prompt_version` 与 `prompt_sha256`，同一任务可定位到当时用的提示词；
- 测试 `tests/test_prompts.py` 会校验“每个 Agent 声明的版本文件真实存在”，防止改名后静默回退。

## 当前清单

| 文件 | 使用方 | 职责 |
|---|---|---|
| `medical.report.v1.md` | Medical（报告抽取） | 只抽取可见字段，保留原文、单位与遮挡标记，不把疑似写成确诊 |
| `medical.report_text.v1.md` | Medical（文本抽取，视觉不可用时的降级） | 从 OCR/转写文本抽取，契约与报告抽取一致，不补造文本里没有的字段 |
| `stool.image.v1.md` | Stool（排便图片） | 先评估图像质量，再给出 Bristol 形态候选；禁止图像疾病推断 |
| `medical.ocr.v1.md` | Medical（文字转写） | 逐行转写，遮挡写 `[遮挡]`，不还原被遮信息 |
| `medical.advice_doc.v1.md` | Medical（模拟医嘱文书） | 抽取医嘱条目并保留“仅测试”标记，不推断诊断/剂量 |
| `diet.meal.v1.md` | Diet | 只描述可见食物，禁止热量与营养估算 |
| `trend.narrative.v1.md` | Trend | 只解释程序算好的事实表，不得引入新数字或因果表述 |
| `advice.generate.v1.md` | Advice（历史版本） | 初版：引用要求与禁令 |
| `advice.generate.v2.md` | Advice（当前） | 增加“输出前自检”：禁用词逐字段检查、数字只允许事实表原样出现、不确定性字段同样受约束 |
| `conversation.assistant.v1.md` | Conversation | 只依据已有结果回答，引用编号必须可解析 |

## 注意

- 提示词里的“会被下游校验删除”不是修辞：`validation.py` 会真的删除无依据条目、
  违规表述与无法落地的数字，所以不要把提示词写成“尽量”“建议不要”，要写清后果。
- 任何提示词改动都要重跑 `tests/` 与 `eval/run_eval.py`；涉及模型行为的改动需要重新录制 `replays/`。
