# GutCare 多智能体系统（智能体开发方交付物）

> 2026“仓颉杯”智能编程大赛 · 综合系统赛道 · 智能体侧实现
> 版本 0.2.0 · 2026-10-17

本仓库只负责**推理侧**：多智能体编排、多模态理解、跨来源时序分析、医学知识检索、
建议生成与输出校验。账户、数据归属、患者确认、业务任务状态、通知与确定性风险
规则的**业务执行**由业务后端负责，双方通过 HTTP 契约交互（见 `docs/openapi.yaml`）。

**后端同伴联调请先看 [`docs/后端联调交付说明.md`](docs/后端联调交付说明.md)。**
本仓库是 2026-10-20 的干净交付快照；不含服务令牌、数据库、模型录制数据和独立测试包。
F08 可先按 `mock` 模式验证协议；F09 独立回调和 F03/F04 图片候选回调尚未接入。

当前 F08 冻结任务接入、状态/证据映射和联调步骤见 `docs/智能体详细开发文档.md`。
公开论文与现有实现的逐项对照、答辩表述及未完成的评测见 `docs/科研依据与实现映射.md`。
最新业务后端源码的 F08 路由级联调可运行 `python tools/verify_latest_backend_f08.py --backend-checkout <后端仓库目录>`；该检查使用后端真实 FastAPI 路由和模型，但以内存任务代替 PostgreSQL，具体结果见详细开发文档。
F09 建议的严格格式转换位于 `gutcare_agents/f09_adapter.py`；当前只生成并校验载荷，尚未向业务后端提交独立 F09 回调。

设计依据：`设计方案_App版.md`、`详细功能设计.md`、
`APP及业务后端与智能体系统开发边界(1).md`、`GutCare智能体独立任务与协作机制说明.md`。

---

## 1. 快速开始

需要 Python 3.12+。PDF 文本提取需安装 `requirements.txt` 中的 pypdf；扫描版 PDF 还需安装 Poppler 的 `pdftoppm`（Docker 镜像已包含）。默认 `mock` 模式：不联网、不编造识图结果。

```powershell
cd gutcare-agents
python -m pip install -r requirements.txt
$env:GCA_FIXTURE_ROOT = "<本机测试包目录>"   # 可选；F08 最小联调不需要

python -m gutcare_agents cases                       # 列出测试包样例
python -m gutcare_agents run --case TREND001         # 单 Agent：趋势分析
python -m gutcare_agents analyze --patient SYN001    # 多 Agent：综合分析
python -m gutcare_agents analyze --patient SYN001 --source sqlite   # 同上，但记录来自 SQLite（只读，模拟后端数据库）
python -m gutcare_agents safety --case SYM001        # 独立安全规则
python -m gutcare_agents serve                       # HTTP 服务 + 演示控制台 http://127.0.0.1:8100/
python -m unittest discover -s tests -v               # 回归测试
python eval/run_eval.py --mode mock                  # 评测报告
python tools/make_samples.py                         # 重新生成联调样例
python tools/smoke_test.py                           # 部署自检（服务起来后跑，一条命令出 PASS/FAIL）
python tools/package_submission.py                   # 打包提交件（dist/*.zip，附 PACKAGE_MANIFEST.json）
python tools/demo_run.py                             # 一键跑完 6 步演示序列，落盘每步 JSON + transcript.md
python tools/check_docs.py                           # 校验文档里引用的仓库路径都存在
python tools/live_smoke.py                           # 拿到 Key 后第一件事：验证接口与视觉，并录制 replays/
python eval/ablation.py --mode live --cases 5        # 消融实验（需要真实模型）
```

接入真实模型（不把密钥写进代码 / 仓库）：

```powershell
$env:GCA_MODE = "live"
$env:GCA_API_KEY = "sk-..."          # 只放在运行环境
$env:GCA_BASE_URL = "https://api.deepseek.com"
$env:GCA_VISION_MODEL = "deepseek-chat"   # 上线前用一张真实报告图冒烟确认视觉输入可用
```

### 一键启动（Windows，推荐给队友 / 评审）

```powershell
.\run.ps1                        # mock 模式 + 自动打开控制台 http://127.0.0.1:8100/
.\run.ps1 -Mode replay           # 断网演示（读 replays/ 录制结果）
.\run.ps1 -Mode live -Port 8200  # 真实模型（会提示输入 Key，不落盘）
```

`run.ps1` 会自动寻找 Python、推断测试包路径、启动服务并打开控制台；
另有 `Makefile`（`make test / eval / samples / serve / docker`）供非 Windows 环境使用。

### 三种运行模式

| 模式 | 用途 | 行为 |
|---|---|---|
| `live` | 真实调用 | 调 `/chat/completions`（OpenAI 兼容），自动把响应录制到 `replays/` |
| `mock` | 联调 / 断网演示 | 不联网；回填确定性内容，识图类明确返回“未执行真实识别”**不伪造能力** |
| `replay` | 演示 / 复现 / 评审 | 读 `replays/` 里录制好的真实响应，断网也能完整演示，结果稳定可复现 |

> `replay` 是应对“评审期间第三方服务故障”的关键设计：先 live 跑一遍关键病例，
> 将经审查、脱敏的 `replays/` 通过单独渠道交付，现场演示与评测才可使用录制结果。
> 若某个病例缺少录制，默认**直接报错**；设置 `GCA_REPLAY_FALLBACK=1` 可在缺录制时退回 mock，
> 并标注 `mode=replay_mock_fallback`（仅演示兜底，不会被当成真实模型结果）。

---

## 2. 目录结构

```
gutcare_agents/
  contract.py       统一契约：AgentResult / Observation / TaskRequest / TaskRun
  errors.py         稳定错误码（联调契约的一部分）
  config.py         运行配置（三模式、模型、路径、阈值）
  store.py          SQLite：任务、Agent 结果、缓存、用量、校验发现、风险提示
  llm.py            OpenAI 兼容模型适配 + live/mock/replay + 重试与 JSON 稳健解析
  mock.py           mock 输出（明确标注，不伪造识图结果）
  stats.py          确定性统计（计数/分布/缺失/时间对齐/充足性判定）
  rules.py          独立安全规则引擎（纯程序、同步、毫秒级）
  validation.py     三层输出校验（结构→证据→安全/数值落地）
  knowledge_base.py 中文字符二元组 + TF-IDF 检索（零依赖，可换 embedding）
  fixtures.py       适配层：JSON / 只读 SQLite 两条数据路径（正式联调换成后端授权请求映射）
  orchestrator.py   阶段化编排：intake → domain → trend → advice → validate
  agents/           6 个业务 Agent，各自独立提示词、工具、失败语义
  api.py            HTTP 接口（RFC 9457 风格错误，内部 Bearer 鉴权）
  f08_adapter.py    后端冻结输入、状态和证据回调适配；SQLite 派发/回调日志与重启恢复
  cli.py            命令行入口
knowledge/          知识条目（6 条，全部标记 unverified_demo）+ 安全规则 JSON
prompts/            7 份系统提示词（按 <agent>.<task>.v<版本>.md 命名，与代码分离）
console/index.html  演示控制台（单页，无外部 CDN，离线可用）
docs/               联调说明、OpenAPI、系统设计章节、第三方使用说明、测试报告、答辩问答、演示脚本与分镜、提交检查表、联调样例
docs/samples/       真实代码路径生成的请求/响应样例（成功/降级/不足/错误）
eval/run_eval.py    评测脚本（确定性指标）
eval/ablation.py    消融实验（单模型 vs 多智能体）
tests/              139 个回归用例（契约/编排/F08 接入与故障恢复/图像与 PDF 边界/提示词一致性）
tools/make_samples.py 重新生成联调样例，避免文档漂移
tools/smoke_test.py    部署自检脚本（健康/鉴权/规则/分析/降级逐项 PASS/FAIL）
tools/package_submission.py 打包提交件（git ls-files 为准，附清单与 sha256）
tools/demo_run.py      一键演示序列（6 步，输出 data/demo/<时间戳>/ 与 transcript.md）
tools/check_docs.py    文档路径一致性检查（文档提到的仓库文件必须存在）
tools/live_smoke.py    真实模型冒烟与录制（接口/鉴权/视觉/关键病例 → replays/，退出码区分故障类型）
docs/ci.workflow.example.yml  CI 模板：编译检查 + 回归测试 + 起服务自检 + 打包；待 GitHub workflow 权限开通后安装到仓库根目录的 .github/workflows/ci.yml
replays/            live 调用录制结果（replay 模式与离线演示使用）
Dockerfile          Python 3.12 镜像，内含 pypdf 与 Poppler
run.ps1 / Makefile  一键启动与常用命令
```

---

## 3. Agent 职责（为什么不是“换了名字的同一段提示词”）

| Agent | 独立任务 | 输入 | 关键工具 | 失败语义 |
|---|---|---|---|---|
| Orchestrator | 组织全流程 | 请求 + 授权数据 | DAG 执行、超时、重试、幂等、复用 | 部分成功显式传播 |
| Medical | 报告字段抽取（图片、PDF、文本）/ 可见文字转写 / 医嘱文书抽取 | 报告图片、PDF 或 OCR 文本 | PDF 内嵌文字提取；扫描页渲染后交视觉模型；保留原文件证据 | 失败保留原文件，转手动录入 |
| Diet | 单餐识菜 | 食物图片或文字 | 视觉模型；禁止热量/营养估算 | 无法识别仍允许手动保存 |
| Stool | 排便记录校验；图片质量评估与 Bristol 候选分类 | 结构化记录或单张排便图片 | 手动记录走纯程序；图片走视觉模型，候选需患者确认 | 图片失败转手动选择，不影响记录校验 |
| Trend | 趋势与时间关联 | 确认记录 + 时间窗 | **确定性统计** + 模型叙述 | 模型失败仍返回统计事实 |
| Advice | 个性化建议 | 档案 + Trend 观察 + 检索知识 | RAG + 证据引用 | 数据不足则不给建议 |
| Conversation | 解释已有结果、整理问题 | 已有结果 + 知识 | 只读工具；药物/诊断类请求**不调模型**直接安全回应 | 服务不可用给兜底文案 |
| Validator | 输出复核 | 所有 Agent 结果 | 结构与证据校验（不是大模型 Agent） | 删除不合规内容 |

**关键分工证据**：Trend 的主要工作在 `stats.py`（程序），模型只叙述；Stool 的记录校验不调模型、图片候选才调视觉模型；
Conversation 有独立的“决策类问题拦截”；Medical 有三种输出契约。

---

## 4. 编排与数据契约

阶段：`intake`（输入校验 + 安全前置）→ `domain`（Medical/Diet/Stool 并行，失败不阻塞）
→ `trend`（确定性统计 + 模型叙述）→ `advice`（检索 + 证据）→ `validate`（汇总校验）。
领域阶段每个 Agent 只接收自己的文件与记录；本次识别结果传给 Trend 时保留“待患者确认”标记，
可供核对但不进入已确认统计，也不会作为 Advice 的个性化建议依据。
阶段之间检查取消标记（`POST /v1/tasks/{id}/cancel`）：取消只影响后续阶段，已完成结果保留。

`AgentResult` 统一字段（对齐《协作机制说明》§8）：

```
task_id, agent, agent_version, prompt_version, model_id, status
patient_id, input_record_ids, input_versions, input_snapshot_hash
observations[{obs_id, kind, statement, source_record_ids, source_refs, confidence, value}]
uncertainties[{code, message, affects}]
needs_confirmation, confirmation_fields
data{}, findings[{code, severity, message, target, action}], usage{}, error{}
```

`kind` 必须区分 `fact / patient_report / derived_stat / model_inference / knowledge`，
避免把模型推测写成患者事实。

**输入口径**（来自测试包 `docs/字段与验收.md`）：只有 `status=confirmed` 参与分析；
`pending_confirmation / deleted / superseded` 排除；`occurred_at` 为发生时间、
`created_at` 为录入时间；缺失一律不解释为“没发生”。

**提示词版本化**：7 份系统提示词在 `prompts/`（`<agent>.<task>.v<版本>.md`），结果带
`prompt_version` 与 `prompt_sha256`，可定位到当时的具体文本；缺文件启动即报错，不静默回退。
改提示词请新增版本文件并同步 Agent 的 `prompt_version`，`tests/test_prompts.py` 会校验一致性。

---

## 5. 三层输出校验（现场可演示“被删掉”）

1. **结构**：必填字段、状态枚举、`needs_confirmation` 必须给出字段清单；
2. **证据**：每条观察/建议的引用必须能解析回本次输入记录、知识条目或图像区域，
   解析不了就**丢掉**该条并记录一条 `finding`（不是“提醒模型注意”）；
3. **安全**：禁用表述（加量/停药、确诊、因果、绝对化、精确热量、凭图诊断、延误就医）
   + **数值落地**：叙述中带单位的数字必须出现在事实表里。安全否定句
   （“不要自行停药”“不能据此认定因果”）不会被误杀。

---

## 6. 两条安全路径

| 路径 | 触发 | 实现 | 与模型的关系 |
|---|---|---|---|
| 即时危险征象 | 提交症状/排便记录 | `POST /v1/safety/check`、`rules.py` | **完全不调用模型**，`< 50ms`，模型故障仍工作 |
| 生成结果校验 | 每次分析返回前 | `validation.py` | 检查识别、分析、建议与助手回答 |

规则文件 `knowledge/safety_rules.json` 标记 `review_status: unverified_demo`，
阈值与文案均为测试示例，真实上线前必须由具备资质的专业人员逐条审核。

---

## 7. 评测结果（`eval/report.json`）

| 项目 | 结果 | 说明 |
|---|---|---|
| Trend 记录筛选与计数 | **90/90 完全一致**（id 精确率 1.0，召回 1.0） | 与测试包 `business_answers.json` 逐条比对 |
| 安全规则阳性 | **9/9 召回**（患者主动报告便血） | 测试包中唯一可结构化判定的阳性类别 |
| 安全规则假阳性 | **0/1017** | 正常排便/症状记录不误报 |
| 结构合规 | 通过 | 所有观察都有可解析来源 |
| 真实模型 · 视觉（报告/食物） | 报告 13 项字段、食物 3 项可见食材 | 硅基流动 `Qwen/Qwen3-VL-30B-A3B-Instruct`，2026-10-17 实测 |
| 真实模型 · 综合分析 | 5/5 succeeded（各 3 条建议，0 条被删） | `replay` 复现同一结果 |
| 真实模型 · 医嘱抽取（10 份） | 逐字条目召回 **1.0**（10/10），测试标记识别 **1.0** | 与 `advice_answers.json` 比对 |
| 断网演示（严格 replay） | 自检 12 PASS / 0 FAIL | 无 mock 兜底，全部来自录制 |
| **消融（单模型 vs 多智能体，8 例）** | 不支持结论率 **29.1% → 0%**；违规表述率 **4.2% → 0%**；耗时 22.1s → 47.7s；token 6.3k → 8.4k | `eval/ablation.py --mode live --cases 8`，8/8 完成 |
| 医嘱抽取 | 需 live/replay，mock 跳过 | 无模型时如实报告“未测” |

**明确未测**：报告字段抽取准确性、食物识别准确性、OCR 字符错误率、排便图像分类、
临床有效性——测试包没有对应金标准，本仓库不作声称。

消融实验（`eval/ablation.py`）的对齐条件：同一基础模型、同一批病例与时间窗、
**两边都提供相同的确定性事实表与相同的检索知识**，唯一差别是架构（单次调用 vs 多智能体 + 校验）。
mock 模式下脚本写 `skipped`，不产出假数字。

---

## 8. 联调样例与异常演练

`docs/samples/` 由 `tools/make_samples.py` 跑**真实代码路径**生成（不是手写文档，避免漂移）：

| 样例 | 覆盖点 |
|---|---|
| `report_extract` / `advice_doc_extract` / `meal_recognize` / `stool_image_recognize` / `stool_record` / `trend_analyze` | 单任务请求与响应 |
| `safety_check.{normal,blood,black_tarry}` | 规则不触发 / L2 / L3 |
| `analysis_full` / `analysis_degraded` / `analysis_insufficient` | 完整、部分成功、数据不足 |
| `error_{patient_scope_violation,model_unavailable,agent_timeout}` | problem+json 错误形态 |

异常场景可在运行时注入，便于现场演示“模型挂了会怎样”：

```powershell
python -m gutcare_agents analyze --patient SYN002 --fault advice
# HTTP: {"options": {"simulate_faults": ["advice"], "simulate_delay_ms": 800}}
```

注入故障的任务不进入幂等复用；上游失败以 `upstream_trend_failed` 显式传给下游，
不会被伪装成“数据不足”。

---

## 9. 部署

```powershell
docker compose up --build      # 默认 replay 模式，断网可演示；端口 8100
```

镜像基于 `python:3.12-slim`，已安装 PDF 处理依赖；`data/` 为运行期任务库，
`../testdata_v4` 只读挂载给演示接口使用。

---

## 10. 已知限制（写给自己，也写给评审）

1. 排便图片分类流程已经接通，但测试包没有真实粪便图像和标注集；Bristol 准确率尚未验证，输出只能作为待确认候选；
2. 视觉输入格式需在拿到 Key 后**冒烟确认**；若模型拒绝图像，`Medical/Diet/Stool` 会返回
   `unsupported_media_type` 并降级到手动录入路径，不会静默伪造结果；
3. 知识库仅 6 条示例条目，覆盖不足以支撑真实医学建议，全部标记 `unverified_demo`；
4. 安全规则阈值未经临床审核；
5. SQLite 任务库用于开发期状态追踪，正式环境由业务后端承担任务权威状态。

---

## 11. 与业务后端的联调清单

| 方向 | 内容 |
|---|---|
| 后端 → 智能体 | `task_id`、内部 `patient_id`、分析类型、时间范围、授权记录（含 `record_id`/`version`/`status`）、档案快照、文件访问方式、敏感数据授权范围、幂等键 |
| 智能体 → 后端 | `status`（`ok/partial/failed/insufficient_data/...`）、`agent_name`、错误码、模型/流程版本、`input_record_ids/versions`、观察与证据、不确定性、`needs_confirmation` |
| 责任边界 | 智能体内部如何拆分与调度 Agent 由本仓库决定；业务任务状态、确认落库、通知、规则执行工程由后端负责 |

详见 `docs/openapi.yaml`。

---

## 12. 交付材料索引（对应综合系统赛道提交要求）

| 赛题要求 | 本仓库对应内容 |
|---|---|
| 项目说明（背景/功能/创新点） | `README.md` §1–§7 |
| 系统设计文档（总体架构/模块/接口/数据流） | `docs/系统设计_智能体章节.md`（可直接并入团队系统设计文档） |
| 接口设计 | `docs/openapi.yaml` + `docs/联调接入说明.md` |
| 可运行程序与部署文档 | 启动命令、`requirements.txt`、`Dockerfile`、`docker-compose.yml`、`.env.example` |
| 演示视频素材 | 演示控制台（阶段时间线/证据链/被拦截条目）+ `--fault` 异常演练 + `replay` 离线模式 |
| 第三方模型/API/数据集使用说明 | `docs/第三方模型与数据使用说明.md` |
| 测试与验证材料 | `tests/`、`eval/report.json`、`eval/ablation.json`、`docs/samples/` |
| 版本控制历史 | 本仓库 git 提交历史（从工程骨架开始提交，不做补造） |

按赛题 6 项材料逐条对照的清单见 `docs/提交材料清单.md`。

