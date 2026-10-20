# ADR-0019：F14 心情记录与受控智能体输入

- 状态：Accepted（2026-10-16）
- 范围：患者心情记录、授权、本人访问与删除、F08 冻结快照投影、历史时间线和 F13 导出/清理；不实现心理健康诊断、危机判断或心理治疗。

## 背景

产品新增心情记录，希望在患者知情授权下供综合分析使用。该信息属于高敏健康信息，并可能反映心理状态；现有 F08 快照禁止自由备注，且 Agent 只能读取逐任务冻结快照。现有认证契约和项目规则均冻结邮箱+密码，而新 UI 流程出现手机号或邮箱入口，需确定单一认证契约。

## 决策

1. **认证保持邮箱+密码。** P0 不采集手机号，不实现手机号/SMS 注册或登录。APP 注册、登录、找回密码界面只接受邮箱；关联手机号 UI 在进入联调前移除。新增心情功能不构成修改认证边界的理由。
2. **新增业务记录 `health.mood_record`。** 患者手动提交 `occurred_at`、主观等级 `mood_level`（1–5）、最多 5 个唯一受控 `emotion_tags`，以及可选 `note`（≤500 字）。稳定标签码为 `calm/content/joyful/sad/anxious/irritable/overwhelmed/tired/other`。写 DTO 严格拒绝未知字段；等级非临床量表，标签不等于诊断。备注只为患者本人回顾保存，不供规则、趋势或 Agent 使用。
3. **分别取得心情处理与智能共享单独同意。** 新增 `mood_data_processing` 与 `mood_ai_sharing` 两个 consent type，各自绑定服务端配置的 policy version 和摘要哈希。创建/修改记录要求 `health_data_processing` 与 `mood_data_processing` 有效；读取和删除本人既有记录作为数据主体访问/删除权利保留。F08 快照仅在 `health_data_processing`、`mood_data_processing`、`mood_ai_sharing`、`third_party_ai` 全部有效时纳入心情数据。缺少任一心情专项授权时，只排除心情记录，不阻断其他健康数据分析。
4. **新增 APP API。** `GET/POST /api/v1/mood-records` 与 `GET/PATCH/DELETE /api/v1/mood-records/{record_id}`。创建须幂等键；PATCH/DELETE 使用 `record_version`；列表游标稳定、按时间过滤；越权资源统一 404，冲突返回 RFC 9457 409。心情加入 `/history/timeline` 的 `mood` 类别和 F13 导出类别 `mood`。删除立即软删除并排除详情、时间线与新快照，单记录物理清理期限为 7 天。
5. **Agent 仅能读取冻结的结构化投影。** `mood_record` snapshot payload 精确限于 `occurred_at`、`mood_level`、`emotion_tags`；记录 ID、版本和受控 locator 由外层 InputRecord 提供。禁止 `note`、附件、推断标签、账号标识或原始心理文本。创建任务时与授权撤回/账户冻结共用患者 consent fence；撤回先提交则阻止新的含心情快照并取消已有非终态含心情任务。Agent 不持业务库凭据，不按患者自行查询。结果只能描述被引用记录可支持的时间关联和不确定性，不能断言情绪导致症状、诊断心理疾病、生成心理治疗或危机判断。
6. **最小化日志与副作用。** 心情备注不得进入日志、崩溃附件、错误详情、通知、推送、埋点、指标、F10 规则事实、F11 趋势或建议回调原始输入。F11 时间线仅返回受控类别、发生时间与详情 target。读取本人详情是唯一常规 API 回传备注的路径；F13 用户明确选择 `mood` 导出时可包含本人记录和备注，沿用既有加密、同源 gateway 与一次性短时下载。
7. **通过 Alembic 和既有服务实现。** 在当前 revision `0023_f02_conflict_group_identity` 之后新增迁移，扩展 `ck_patient_consent_type`、创建 `health.mood_record` 和所需幂等/索引/约束，启用并强制 RLS、患者复合归属约束及最小角色授权；同步 OpenAPI、Agent schema、数据字典、隐私、测试和追踪。继续使用 FastAPI 模块化单体，不新增 mood 微服务或数据库；禁止 ORM 自动建/改 schema。

## 后果

- 两种新增授权必须有独立的正式告知文案、版本、服务端 SHA-256 摘要配置，并由 APP 明示目的、字段、智能体处理者和撤回效果。部署缺少任一配置时，不能授权或纳入快照。
- 患者无心情授权时仍可使用原健康记录和分析功能；已授权记录的本人读取/删除不被撤回操作锁死。
- 新增枚举/API/schema 为向后兼容的增量契约；发布 APP 前应先部署兼容后端，再展示新功能。
- 迁移覆盖真实 PostgreSQL 18 的 RLS、并发幂等、权限撤回竞态、删除、导出和恢复重放；独立验收前追踪状态为 `specified`/`implemented`，不得标记为 `verified`。

## 验收映射

- T105：心情 DTO、CRUD、版本、幂等、RLS 与跨患者隔离。
- T106：单独授权、撤回、任务取消和并发栅栏。
- T107：Agent snapshot 结构白名单、备注排除和证据验证。
- T108：敏感备注日志/通知/规则/指标排除。
- T109：混合时间线、签名游标、类别过滤和本地访问。
- T110：7 天删除、F13 导出、注销清理和备份恢复重放。

## G1 契约扩展（2026-10-17，已采纳）

1. **结构化心情趋势。** F11 允许患者查看本人未删除记录的 `mood_level` 确定性聚合，不要求 Agent 共享授权，不把数据发送给 Agent；只按患者 IANA 时区 day/week bucket 返回 `has_data`、`record_count` 与 1–5 档 `mood_level_counts`。`note` 与 `emotion_tags` 不参与趋势；该聚合不是临床量表、诊断或因果结论。
   `/trends` 不因 mood 授权缺失而整体失败：仅 `mood` 子结构返回 `MoodTrendUnavailable`，其他趋势仍返回。`MoodTrendAvailable` 必含 `availability=available` 与 buckets，空数组表示授权有效但范围内无数据；`MoodTrendUnavailable` 必含 `availability=consent_required`、`reason_code=CONSENT_REQUIRED`、去重的 `missing_consents`（限 health_data_processing/mood_data_processing），且不得包含 buckets。两种对象均封闭并由 `availability` 区分。
2. **预览与冻结快照计数。** `GET /analysis-tasks/preview` 返回 `mood_record_count_in_range`、`mood_record_count_eligible`、`mood_record_count_excluded_by_consent` 与受控 `mood_exclusion_reasons`。预览不保证未来快照；任务创建在 consent fence 内重筛授权和记录。缺少基础 F08 `health_data_processing` 或 `third_party_ai` 时整体拒绝预览；只缺 mood 专项授权时仅排除 mood。`AnalysisTask.input_summary.included_mood_record_count` 与任务快照同事务持久化，表示实际纳入数，历史和重试不可按当前记录重算；旧任务按不可变快照回填。
3. **标签数据库双层约束。** `emotion_tags` 继续采用 JSONB；DTO 与数据库共同强制数组、0–5 项、仅允许固定受控字符串码且唯一。Alembic 创建无表依赖的 `IMMUTABLE`/`STRICT` 校验函数和 CHECK，并锁定安全 search_path、所有权与最小执行权限；受控码变化必须与 DTO/OpenAPI/迁移同步。
4. **契约校验细节。** OpenAPI mood trend bucket、AnalysisTask summary、预览计数和 AnalysisResult/AdviceEvidence `source_type=mood_record` 为冻结契约；`MoodDetail` 使用独立完整的 `additionalProperties: false` schema，避免 `allOf` 与封闭基 schema 的冲突。F07 `patchActivityRecord` 明确保留 422 字段验证错误响应。

G1 对应验收：A51–A52、T111–T115。新增契约已定义不代表业务代码/数据库迁移完成；实现仍按 Alembic 与独立验收门禁执行。

## H2 契约澄清（2026-10-17，已采纳）

1. F08 分析范围不使用隐含默认边界：`periodic_summary` 必须同时提供 `requested_from` 与 `requested_to`；`comprehensive_health_analysis` 可同时省略两端（不限定时间）或同时提供两端。任一类型只提供单端均返回 RFC 9457 422。预览响应回显规范化范围，POST 与预览共用相同校验及半开区间 `[from,to)` 语义。
2. 失败任务重试沿用患者 shared consent fence，并检查旧不可变快照。若旧快照含 `mood_record`，重试前必须确认四项快照授权均仍有效；缺少心情专项授权时拒绝并且不创建任务或幂等账本。无心情快照在基础分析授权仍有效时可重试。
3. 0025 downgrade 只从分析任务幂等响应缓存移除新增 `input_summary` 字段，保留响应中的其他字段及账本行；不会删除任务或快照数据。
