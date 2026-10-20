# ADR-0012：F08 冻结快照分析任务闭环

- 状态：已采纳
- 日期：2026-10-15
- 范围：F08 业务任务创建、冻结输入、受限 Agent 执行、结果校验与任务查询；不包含模型推理、F09 建议实体/API、通知和 outbox。

## 决策

1. 患者通过 `POST /api/v1/analysis-tasks` 提交 `analysis_type`、可选时间范围及 `Idempotency-Key`。当前须同时授予 `health_data_processing`、`third_party_ai`，账户和档案可用，且范围内至少有一条已确认、未删除的数据。创建事务持患者授权栅栏，校验数据、创建任务及不可变 `task_input_snapshot`，保存可复现快照摘要。只有任务级输入含明确允许的健康字段、来源 ID/版本与定位；不含报告正文、对象 URL、自由文本备注或完整 prompt。
2. 同一患者、路由与幂等键同请求摘要重放返回最初完整响应；不同摘要返回 409。患者可列出/获取自己的任务、取消尚未完成任务、显式重试 `failed` 任务。取消仅允许 pending/processing；重试只允许 failed，并创建同快照的新任务尝试，不重写历史尝试。任务状态为 `pending → processing → succeeded | partial | needs_input | failed | cancelled`；已取消或其他终态拒绝迟到回调。
3. 业务 Worker 是唯一原子领取者，通过 `FOR UPDATE SKIP LOCKED` 写入有期限租约；过期 processing 可重新领取。Worker/Agent 只能通过带最小权限 Bearer 的内部 HTTP 按 task ID 获取冻结快照及提交严格结构化结果/失败。Agent 不可访问业务数据库，不可按患者查询或自由读取历史。
4. 服务端验证结果 schema、允许字段、长度/类型与证据引用。每个证据必须匹配该任务快照中的患者、来源 ID、版本和 locator；越界结果整体拒绝。只持久化规范化结构化结果和证据，不持久化 prompt、模型原始输出或报告正文。结果属于 F08 analysis result，不构成建议或诊断/处方。
5. 患者授权撤回、账号冻结与任务创建使用同一患者 advisory consent fence。撤回/冻结先提交的，阻止新任务与新快照；已有非终态任务在同事务转 cancelled，迟到回调拒绝。已冻结快照作为任务审计事实保留至数据生命周期清理，但患者不因此获得读取已删除来源原文的能力。来源删除不改变快照字节；快照读取在每次访问时校验授权和来源状态，来源已删时不返回原文或链接，只允许处理快照内最小结构化事实。
6. 任务、不可变快照、结果、幂等收据均使用患者 RLS + FORCE RLS、患者复合外键和最小角色授权。Agent 无数据库角色；worker 不可修改患者授权或业务资料。PostgreSQL 18 是并发、RLS 与迁移验收权威。任务终态直接持久化；outbox 留待 F12。

## 验收与后果

新增 Alembic 迁移和患者/内部 API。测试编号从未占用的 T66 起；真实 PostgreSQL 18、MinIO 与备份恢复门禁覆盖幂等、跨患者、快照不可变、来源删除、claim/lease、retry/cancel/迟到回调、越界 evidence 与 Agent 不可用时记录 CRUD。追踪状态在独立验收前为 `implemented`。
