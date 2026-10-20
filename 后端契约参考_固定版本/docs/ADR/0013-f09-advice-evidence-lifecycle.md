# ADR-0013：F09 建议与证据业务闭环

- 状态：已采纳
- 日期：2026-10-16
- 范围：建议结构化回调、患者建议读取/归档/反馈、受控证据投影；不包含 Agent 生成逻辑、F12 通知或知识库访问。

## 决策

1. F08 `analysis_task.result_payload` 仍是分析结果，不直接成为建议。独立 internal Agent advice callback 仅可关联本患者 `succeeded` 或 `partial` 的 F08 任务，使用专用 analysis worker Bearer 身份；按 task 与 `callback_id` 记录规范化请求摘要，同摘要幂等重放、异摘要 409。建议及证据在同一事务持久化，task/result 归属和状态在事务中复核。
2. Advice schema 版本化、未知字段拒绝，必须含 `category`、`priority`、`title`、`body`、`uncertainty`、`applicability`、`safety_boundary`、`version`。类别、优先级为固定枚举；适用性和安全边界为有界结构化声明；正文长度受限。安全约束通过 schema、明确安全声明和来源要求执行，不以关键词探测代替规则。拒绝诊断、因果结论、处方及剂量调整类 advice type；须提供患者快照证据或版本化知识引用作为依据。
3. 患者 evidence 必须逐项精确匹配对应 task snapshot 的 `source_type/source_id/source_version/locator`；locator 不得扩张。Knowledge evidence 只接受 `knowledge_ref/version/title/publisher` 和可选安全 HTTPS citation URL，不访问 `gutcare_knowledge`，不接受正文、chunk 原文或摘录。证据接口仅返回受控 DTO。
4. F09 由 `analysis.health_advice`、`analysis.advice_evidence`、callback ledger、当前反馈和反馈历史组成。advice 与 patient 复合外键；患者可列出、读取、标记已读、归档及提交 `helpful/not_helpful/not_applicable` 反馈。反馈更新有乐观版本、幂等键及不可变变更历史；跨患者统一 404。
5. lifecycle 仅为 `new/read/archived`；`basis_updated` 是独立布尔值，并记录最近更新时间和原因。来源版本更新、删除或医疗事实被替代时同步标记；读时也投影检测变更以覆盖遗漏事件。read/archive 不清除此标记。来源删除立即清理 locator、excerpt 和访问能力；建议历史正文保留。知识引用只保留版本化引用元数据。
6. F10 risk alert 独立且展示优先。建议不能关闭、降低、延后或取代风险提示；如当前存在需紧急处理的风险，API DTO 显示 risk priority context，常规建议不改变风险状态。
7. 新表 ENABLE/FORCE RLS、患者复合 FK；仅 `gutcare_app` 可访问患者 advice，`gutcare_analysis_worker` 对 advice callback 按最小范围写入，Agent 无数据库角色。feedback/write 由患者作用域及行锁串行化；用户授权撤回/冻结与 advice callback 使用同一 advisory consent fence，撤回先提交则拒绝写入。账号注销在 freeze 事务中追加撤回任何仍授予的 health/AI consent，保证独立 advice callback 可通过授权栅栏识别冻结，而不扩大 worker 对 IAM 账户数据的读取权限。

## 验收与后果

新增 Alembic 迁移、OpenAPI 与 `agent-v1` 契约、数据字典/数据库/隐私/测试/追踪同步。测试编号从首个未占用的 T72 起。真实 PostgreSQL 18 + MinIO 集成及恢复覆盖 callback 幂等与异摘要、任务状态及迟到回调、患者快照证据边界、知识引用严格校验、状态/反馈并发、来源更新/删除脱敏及 `basis_updated`、跨患者隔离、授权撤回/冻结竞争、worker 最小权限和恢复。追踪状态为 `implemented`，独立验收后才可改为 `verified`。

知识库数据库保持独立且由智能体团队维护；业务服务只保存引用，不直连 `gutcare_knowledge`。
