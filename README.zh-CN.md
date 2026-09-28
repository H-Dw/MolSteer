# MolSteer

MolSteer 分为 MolReader、MolThinker、MolExecutor 和 MolMonitor。新增的 [LangChain/LangGraph Agent 系统](docs/AGENT_SYSTEM.zh-CN.md)默认通过 API 调用 LLM，四个 Agent 可独立选择服务与模型，统一配置见 [configs/README.md](configs/README.md)。新 Agent 默认启用[基于证据的创造性奖励 Skill](skills/molthinker-reward-creativity/SKILL.zh-CN.md)；确需约束式多目标求解时可参考[冲突感知控制 Skill](skills/molthinker-conflict-aware-control/SKILL.zh-CN.md)。原有确定性 creativity/selection 命令仍可用于复现，但 `think` 现在要求显式选择模式；FLOWR.ROOT 实时梯度引导、恢复和通用 flow/diffusion 回调接口保留。

四个 Agent 默认经 OpenRouter 调用 `z-ai/glm-5.3`；API 凭据由宿主环境变量 `OPENROUTER_API_KEY` 注入。

- MolReader：43 个独立指标、多视图 StatePacket、只报告风险的中英文 DiagnosticReport。
- MolThinker：知识检索、适用性检查、奖励选择或组合，以及可执行 RewardProgram。
- MolExecutor：可微奖励、明确的梯度注入、模型适配、自动启动脚本和完整运行检查点。
- MolMonitor：匹配原生参考、局部时序异常识别、动态力度搜索、独立质量检查，以及向 MolThinker 发送可恢复的函数修订请求。

当前实际接入并验证的模型是 FLOWR.ROOT，通用 diffusion 接口已通过数值测试，其他预训练模型仍需对应适配。奖励组合器支持已实现的基元，不能自动实现任意知识库公式。

新增[完整阶段检查点与相同奖励的起始时间对照](docs/EXACT_RESTART.zh-CN.md)，直接保存原生轨迹的自条件缓存和 RNG，并验证精确续跑。

[MolMonitor 架构与文献依据](docs/MOLMONITOR.zh-CN.md)说明动态控制、异常定位、反馈摘要及函数修订后的恢复约定。

[基于无引导对照的奖励与主动图搜索](docs/OUTCOME_GUIDANCE.zh-CN.md)说明自然修复归因、连续 MMFF 梯度、方向相互作用、埋藏极性代价、实际分类张量试探，以及根据独立评分和梯度冲突修订目标。文档保留了动态修订未改善终态的测试反例。

[增广拉格朗日执行与对偶状态恢复](docs/AUGMENTED_LAGRANGIAN.zh-CN.md)说明局部约束注册、持久乘子更新、不可补偿守门条件、checkpoint lineage，以及 t=0.75 精确恢复验证。

请参阅[Linux 安装与运行](docs/LINUX.zh-CN.md)、[执行架构与接入说明](docs/EXECUTOR.zh-CN.md)、[历史环境记录](docs/ENVIRONMENT.zh-CN.md)、[MolReader 属性评估](docs/READER_ATTRIBUTES.zh-CN.md)、[英文说明](README.md)。本次真实续生成结果见 experiments/guidance；此前的离线副本试算保留在 examples。

Skills 提供[creativity](skills/molthinker-reward-creativity/SKILL.zh-CN.md)和[selection](skills/molthinker-reward-selection/SKILL.zh-CN.md)两种模式。技能正文不包含实现命令或发布信息。

[可选 Researcher 与离散概率引导](docs/RESEARCHER_AND_DISCRETE_GUIDANCE.zh-CN.md)说明 MolThinker 内部的证据调研流程、真实类别导数边界、通用引导基元，以及接入验证与完整优化之间的区别。

[Researcher 完整流程](docs/RESEARCHER_PIPELINE.zh-CN.md)进一步接通结构化调研存储、可选运行模式、动态影响与类别力度、完整续推与恢复测试，以及离线 HTML 报告。
