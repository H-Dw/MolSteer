# LangChain / LangGraph Agent 重构

## 架构与兼容边界

新 Agent 层位于 `src/molsteer/agents`，旧版领域算法与已有实验文件保留。LangChain 负责各 Agent 的模型与工具接口，LangGraph 负责状态流转和有界反馈循环。API 是新层默认运行方式；离线模式必须显式选择，不会在 API 失败时偷偷退回规则算法。

`configs/agents.json` 采用 `agents → models → providers` 引用结构。四个 Agent 默认通过 OpenRouter 使用 `z-ai/glm-5.3`，端点为 `https://openrouter.ai/api/v1`；可分别改用其他模型、服务端点和凭据引用。运行真实 API 前，需要由宿主环境注入 `OPENROUTER_API_KEY`。安装依赖使用项目虚拟环境，不改变系统 Python。

默认模型启用 OpenRouter reasoning 参数。`ChatOpenRouter` 在同一 Agent 的临时多轮工具调用中传递 `reasoning_details`；审计 trace 与 checkpoint 不保存供应商的私有 reasoning 内容。

旧 `think` / `demo` 命令保留其历史语义；新 Agent 调度不是把历史离线实验重新标记为 LLM 实验。

## 密钥边界

`configs/secrets.local.json` 仅供宿主凭据加载器使用，初始为占位符；环境变量优先。配置文件保存凭据引用，不保存明文凭据。模型上下文、领域 StatePacket、工具参数和审计文件不应包含凭据。

已为本地密钥配置 Git、Docker 与 Agent 忽略规则，并为 Claude 文件工具设置显式读取/编辑拒绝规则。**忽略规则不是操作系统隔离或绝对防上传机制**：拥有同一用户权限的其他程序仍可能读取文件；人工压缩整个文件夹也不会自动遵守 `.gitignore`。不要上传该文件，也不要上传运行目录或含密钥的环境快照。生产环境建议由独立运行账户和 secret manager 注入凭据，使用网络出口白名单；外部计算应运行在隔离工作进程或容器，不能继承模型 API 密钥。

这里的“不可读取”指 LLM/Agent 工具不可读取，而不是让需要认证的宿主运行时也无法读取。未设置阻止所有进程读取的文件 ACL，因为那将使 API 调用本身失效。

## Agent 职责

### MolReader

各特征获取路径通过独立工具暴露。事实与风险诊断保留来源、表示空间、原子映射、单位和不可用状态。工具不能由 LLM 任意指定本机路径以读取文件。已保存的数据读取与重新执行外部评估应区分，不把工具名称当成测量已运行的证明。

### MolThinker

采用 ReAct（用户所述 RecAct 的思考—执行循环）：先规划，再按需调用 StatePack、知识检索和计算工具，消费 observation 后修订方案。Web 搜索和外部计算由宿主注册的适配器提供；未接入时明确返回 unavailable，不捏造检索结果。

默认指引为 `skills/molthinker-reward-creativity/SKILL.md`，先挖掘最小且足够的核心目标，再按证据、可接受集合和实时导数路径选择函数形状与组合策略。`skills/molthinker-conflict-aware-control/SKILL.md` 用于需要约束式多目标求解的情形。API 奖励提交现在记录目标分组、未采用候选的去向及声明式目标树；当前可执行树算子为 `maximum`、`mean` 与 `lp_norm`。如果合适的函数形状或约束超出后端能力，Agent 可提交 `design_deferred` 设计记录，停止编译。数值检查只是坐标副本验证，不能替代 FLOWR 完整续推。

`optimization.conflict_weights` 求解单纯形上的最小范数多目标梯度组合。输入必须已拉回同一可编辑变量、坐标系和度量；函数对固定坐标投影，并显式使用正尺度归一化。输出无量纲权重、余弦冲突、方向导数、对偶间隙与驻点状态。它不是完整约束求解器，也不设置梯度注入强度。零梯度/相反梯度产生的驻点不能当成成功修复。

### MolExecutor

执行采用“构造可审查产物 → 验证 → 测试 → 有界修复 → 放行”的顺序，不直接执行 LLM 输出的任意 Python。现有有限差分、固定原子保护、有限性和约束检查继续作为硬门禁。正式 inference 需要宿主提供真实生成器适配器与明确的可编辑范围、预算、状态和 RNG 恢复约定。离线坐标试验不能证明真实生成器收益。

### MolMonitor

随机生成过程应以局部窗口和阶段匹配的参考衡量，而不是对一次峰值立即重规划。监控使用稳健的 median/MAD 基线、预热、连续异常、恢复连续数和冷却期。异常先交给 Executor 调整注入强度；有限次数尝试后仍异常才触发 Thinker 函数修订。非有限数和硬约束违规应停止不安全提议，不因 LLM 的解释而放行。

推荐真实生成器上监测：有效控制/原生更新范数比、梯度方向反转、局部结构违规率、归一化目标变化、独立质量指标及其可用性。不同时间方向/噪声阶段不要共享一个未经校准的全局阈值。观测不足不等于健康，降低局部奖励也不等于提高终态质量。

## 可复现的决策记录，而非私有思维链

保存可审计的结构化决策：初始任务计划、简短依据、引用的 evidence IDs、工具参数/结果或哈希、奖励和执行产物、失败测试、修复次数、监控事件、配置/Skill/输入的版本与哈希。不要保存供应商私有 reasoning/thinking blocks、密钥或完整认证异常。

应区分三个层次：

1. **决策审计**：能解释最终选择依赖了什么公开证据。
2. **流程恢复**：当前 `.checkpoint.json` 是带哈希校验的审计产物快照，包含输入、计划、RewardSpec、测试结果和配置；明确标记 `automatic_resume_supported: false`。本次未启用 LangGraph 持久化中断续跑。外部生成器恢复仍使用原项目适配器的 checkpoint/RNG 协议。
3. **科学复现**：另外保存模型/适配器版本、生成器权重、硬件与数值设置、自条件状态、匹配 RNG、预算与独立评价。API 返回值通常不能保证逐 token 重现。

仅恢复 Agent JSON 并不等价于恢复生成器，也不能把保存一份“思维文字”当成实验可复现。

## 运行入口

在项目根目录安装：

```bash
python -m pip install -e ".[test]"
```

注入 OpenRouter 凭据后运行新 Agent 入口（会产生 API 费用）：

```bash
python -X utf8 -m molsteer agents --packet examples/5i0b_A__5vef_M77/ligand_002/t_0.50/StatePacket.json --report examples/5i0b_A__5vef_M77/ligand_002/t_0.50/DiagnosticReport.json
```

离线验收使用显式的 `--offline`，不调用 LLM，也不声称是真实 API 运行。CLI 不提供通过任意模块路径执行生成器的选项；正式 inference 由宿主 Python 集成注册的适配器完成。

## 当前实现范围与验收

- Reader tools 读取绑定 StatePacket 的 geometry、chemistry、uncertainty 三条证据路径，并调用原领域诊断器。没有把全部 43 项指标的原始文件采集与外部评分逐项重新接线。
- Thinker 的 API 工具循环可提交计划、检索与选择/重参数化已支持的奖励基元。Skill 的完整 `ConflictAwareControlDecision` 是设计契约；运行时保存的是较小的 `ControlPlan`，不声称自动完成任意公式推导、全部约束 QP 或离散分支求解。`conflict_weights` 是可调用的数值库函数，尚非自动应用到所有奖励的控制器。
- Executor 的自动修复范围是声明式程序检查和有界测试参数，不能自主编写并部署任意新 Python 后端。新公式需要受审查的后端实现。Web 和外部计算需注入宿主 callable；默认不联网执行这些工具。
- 新工作流正式推理接入协议为 `adapter(packet=..., reward_spec=..., execution_result=validation, request=...)`，返回 `{'done': bool, 'metrics': {...}}`；`request` 含 segment/strength/run_id/monitor_event。适配器必须实际消费 strength，并负责实时梯度、硬约束和生成器状态；不能把旧 FLOWR CLI 配置直接当成该 callable。
- 当前监控对每个指标分别维护基线，不混合原始量纲，异常不会污染正常基线。它尚未自动按 diffusion 时间阶段分桶；生产阈值仍需按真实轨迹校准。
- 验收：完整测试 **144 passed**，包含四个 API Agent 的模拟 tool-call 测试、OpenRouter reasoning 消息传递、真实领域数值测试、监控先 retune 后 replan 的图级测试与审计快照完整性测试。另有一个原有 PyTorch Tensor 转标量警告。没有运行真实付费 API、在线检索或 GPU 生成器推理。

## 验证方法

Windows 中文系统的旧测试使用默认编码读取 UTF-8 样本，建议显式启用 UTF-8：

```bash
.venv/Scripts/python.exe -X utf8 -m pytest tests -q
```

测试不得要求真实 API key，不应访问在线搜索。真实 API 和 GPU inference 的通过状态必须单独报告；占位密钥环境下不宣称完成了这些验证。
