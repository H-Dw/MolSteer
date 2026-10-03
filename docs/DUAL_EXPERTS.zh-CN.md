# MolThinker 双专家

当前直接执行契约见[剩余需求与标量注入](RESIDUAL_NATIVE_SCALAR.zh-CN.md)。下文涉及预算、提议拒绝、分段控制或共同下降的内容为历史适配器记录，不适用于当前 MolThinker → FLOWR.ROOT 路径。

新默认配置使用 `dual_expert`。MolReader 仍只诊断风险；MolThinker 内部依次运行生物专家和数学专家，各自有独立的 ReAct 对话及可调度的 Researcher。MolExecutor 验证并执行声明式产物，MolMonitor 保留观测、调力和要求修订的职责。

## 生物判断与数学设计

生物专家从完整诊断出发，把所有 finding 放入优化、约束、监测或延期方向。`BiologyPlan` 保存机制分类、排序理由、证据、修复判据、化学状态、反证和不确定性。`preservation_conditions` 必须引用显式约束方向 ID，以便交接到可检查的约束表达式。排序没有固定加权分数；不能以可求导性替代生物重要性。

数学专家逐方向检索原始知识，输出 `MathematicalDesign`。它保存可接受集合、尺度来源、原公式与具体来源位置、迁移假设、推导摘要、导数路径、边际敏感性及备选方案。`function_lineage` 的 source_id、locator 和 original_formula 必须匹配真实检索结果。没有命中也要记录实际检索；不足以支持执行时提交 `design_only`。

数学专家可以调用 `request_biology_revision`，最多回议两轮。生物目标或必要保持条件只能由生物专家改写。可执行设计必须先通过 `test_mathematical_design` 的实际数值检查，再提交同一内容。MolExecutor 随后独立复核。必需方向延期、必要约束未实现或回议预算耗尽，都会停止当前设计的执行路径。

## 配置和泛化接口

`thinker.experts` 的 `biology`、`mathematics`、`researcher` 分别引用 `models` 配置。省略某个角色时继承 `agents.molthinker.model`。例如：

```json
{"thinker":{"architecture":"dual_expert","experts":{"biology":"chemistry_model","mathematics":"reasoning_model","researcher":"retrieval_model"},"max_discussions":2,"max_research_requests":4,"research_max_steps":6,"research_max_searches":3,"research_result_limit":5,"external_research":true}}
```

三个模型名称需在 `models` 中定义，继续使用既有 `models → providers` 端点及凭据引用。代码注入模型时，键为 `molthinker.biology`、`molthinker.mathematics`、`molthinker.researcher`。Researcher 按需初始化。没有 `thinker` 配置的旧文件仍使用 `single`；显式 `offline` 仍运行历史确定性领域算法，不生成虚构专家调用。

`AgentRuntime` 新增 `model_dynamics`、`fetch_fn`、`research_providers` 参数。宿主也可通过 inference_adapter 的 `describe_dynamics()` 提供 `ModelDynamicsContext`。模型类型、时间、预测参数化、坐标变换、可编辑原子、注入约定和导数可用性均由宿主声明；默认 unknown/unavailable。版本 2 直接执行适配器还需声明 `capabilities={"reward_versions":["2.0.0"],"live_preflight":true}`，并负责实际预检查及正确执行该版本控制契约。需要重规划时必须返回新的 StatePacket，防止旧诊断驱动新状态。

`ResearchProvider` 使用 `search(query, limit)` 和 `fetch(source_id)`。内置本地 Markdown、Europe PMC 和 arXiv；`search_fn`/`fetch_fn` 接入宿主通用 Web 客户端，`research_providers` 可注入其他已配置提供器。默认先查本地，之后按缺口使用外部来源。每轮规划最多四次研究请求，每次六轮工具循环、三次搜索。相同状态、语料版本和问题复用证据快照。

搜索记录不能直接冒充已读取证据。Researcher 必须 fetch 后才能提交相应 claim；全文不可用时保留摘要限制、错误类型和缺口。结果为 `ResearchEvidencePacket`，不自动激活奖励。源记录按稳定身份关联，不同段落保留独立 observation ID。知识目录按 Markdown 章节和表格检索，返回原式、变量、梯度对象、前提、哈希与行位置；历史表格解析器保持原有语义。

## 表达和控制契约

RewardSpec `2.0.0` 将观测量、局部表达、跨方向策略和约束分开。观测量包括距离、受体距离、弧度角、锚点偏移、方向点积、周期二面角、有符号三重积和 MMFF 应变。原子、受体配对和视图绑定诊断证据。MMFF 要求验证化学图、质子化和适用性，只提供固定图的一阶包络导数。

表达式是有界 JSON 树：`observable`、带单位和来源的 `constant`，以及 add/subtract/multiply/divide、relu/abs/sqrt/power、sin/cos/periodic_difference、minimum/maximum、sum/mean。最多 128 节点、12 层；检查单位、定义域、几何退化和非有限值。度数常量转换为弧度。每个方向最终必须是无量纲非负缺陷量；约束量为零才满足条件。普通字符串不作为代码执行。

跨方向只接受有理由支持的单目标、maximum、lp_norm（1<p≤8）或 `common_descent`。前者是标量势，后者逐方向经真实 Jacobian 求导，再进行掩码空间的最小范数求解。共同下降系数来自求解，不是固定奖励权重。硬约束通过候选检查执行；一般非线性约束 QP、任意代码、自动新增原子和新类别控制器不在该版本表达能力内。

FLOWR 编译保留表达和控制策略。每步重新计算实时梯度，检查真实注入及裁剪后方向，并在同时间候选上检查各目标与约束。预测图或相关当前图变化使旧设计失效，需要重新读取和绑定；不静默替换参考。探测恢复坐标、条件缓存及随机流。无可行共同下降、求解未收敛或导数不可用时不产生额外引导。MolMonitor 路径同样使用该控制器，并保存冲突记录。

## 验证和运行

```bash
.venv/Scripts/python.exe -X utf8 -m pytest tests -q
.venv/Scripts/python.exe -X utf8 scripts/run_dual_expert_smoke.py
```

冒烟脚本使用现有 t=0.50 案例，输出真实调用状态、专家产物、来源记录和推导文件。凭据缺失会写 `skipped`，不会换成脚本化模型。两位专家是否实际调用 Researcher 单独报告，未调用不等于已覆盖该路径。

离线测试覆盖工具调度、来源绑定、预算、异常、数学函数及导数、共同下降、非恒等 Jacobian、混合视图、裁剪和随机流恢复。坐标副本测试不认证真实 FLOWR 控制或终态收益；GPU 续生成和未执行的实时检查明确记为 `not_run`。检查点保存交接产物及工具观察，仍是审计快照，不是可自动恢复的 LLM 对话。
