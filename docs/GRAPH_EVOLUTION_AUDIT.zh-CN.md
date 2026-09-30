# MolSteer 中间态图变化审计与优化方案

日期：2026-09-30。审计基线：本地 MolSteer `778cde4`，含当前工作区与已下载的 SCNet 实验记录。

本次交付是代码审计、行为验证和实施方案；没有修改采样器、奖励实现、已保存的实验配置或历史结果。

后续实施说明（同日）：下文记录的是修改前的审计基线。随后已按用户要求删除 ExpertReward 全图一致性检查，并加入可配置的化学图复核通路。当前实现、参数含义与验证范围见 [GRAPH_REVIEW.zh-CN.md](GRAPH_REVIEW.zh-CN.md)；初始图偏好参数仍保留，尚未默认删除。

## 结论

`t=0.50` 的预测图是临时假设。原子类型、形式电荷、键阶、连接关系均应允许在后续采样中变化。原子槽位保持对应关系，不意味着槽位的元素或化学角色必须保持不变。

当前实现没有统一的全局锁图。原生 FLOWR 积分器继续更新原子、电荷和键类别，但不同奖励路径对图变化的处理不一致：expert 路径存在初始图一致性门禁；历史 GraphGuard 实验显式禁止接受偏离初始图的引导；creativity 存在初始类别概率的软偏好；selection 和 AgentMixedReward 则可能在换图后继续使用旧几何参照。不能只删除图检查而保留失效的化学解释。

## 逐路径检查

| 路径 | 代码位置（仓库相对路径） | 当前行为 | 判断 |
| --- | --- | --- | --- |
| 原生采样 | `flowr_root/flowr/models/integrator.py:211`、`:223`、`:250`；`src/molsteer/molexecutor/flowr.py:135` | 每步分别更新 atomics、charges、bonds | 允许元素、电荷、键阶与连接关系变化 |
| 常规引导循环 | `src/molsteer/molexecutor/engine.py:93`、`:145` | 先执行原生更新，再评估额外坐标提议；提议失败时保留原生结果 | 拒绝引导不等于回滚原生图变化 |
| ExpertReward | `src/molsteer/molexecutor/expert_reward.py:22` | 对 atom_ids、元素、形式电荷和完整键阶矩阵计算摘要，与源 StatePacket 一致才允许使用表达式；prediction 和被使用的 state 视图均检查 | 确有全图一致性门禁；即使远离目标区域的一个类别变化，也可能使整个 expert 引导不可用 |
| 历史 GraphGuard | `outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_graphguard_20260929/run_graphguard.py:10` | 当前预测图或候选图不等于 t=0.50 参考图即停用奖励或拒绝引导提议 | 显式初始图门禁；仅为该历史诊断实验的包装器，未接入默认运行入口 |
| creativity | `src/molsteer/molexecutor/program.py:65`、`:131`、`:176`；`src/molsteer/molthinker/creativity.py:18` | 按当前图重新查询 MMFF 键长、键角；同时以初始概率 p0 计算类别偏离代价，默认 lambda_graph=0.1 | 允许换图，但在接受阶段偏向初始假设 |
| selection | `src/molsteer/molexecutor/program.py:149` | 继续使用 term 中固定的 atom_ids、lower、upper、scale | 不锁图，但旧键/角消失后可能仍受到原罚项牵引 |
| AgentMixedReward | `src/molsteer/molexecutor/program.py:214`；`src/molsteer/molexecutor/agent_bridge.py:19`、`:77` | 按 state/prediction 视图使用固定奖励项；编译字段未保留 graph_dependent、hypothesis_atom_ids、conditions 等适用性信息 | 与 expert 的全图门禁相反，这里存在适用性检查不足；graph_policy 的文字不能实现自动重绑定 |
| local_first | `src/molsteer/molexecutor/local_first_reward.py:139`、`:229` | 稳定性 gate 控制几何阶段激活；换图候选另受几何、应变阈值约束 | persistence_gate 不是锁图；但实验脚本设置 graph_change_max_geometry_loss=1e-5、graph_change_strain_allowance_kcal_mol=0，可能过早排除可在后续修复的新图 |
| augmented_lagrangian | `src/molsteer/molexecutor/augmented_lagrangian_reward.py:361` | graph_change_policy 默认为 allow_if_feasible，也支持 reject | 存在可选硬禁改图开关；比较的是候选与同一步原生基线，不是 t=0.50。审计的 configs/scripts/experiments 中未发现启用 reject 的配置 |
| affinity_structure / outcome_aware | `src/molsteer/molthinker/affinity.py:7`、`:35`；`src/molsteer/molthinker/outcomes.py:66` | 默认 retain_initial=False，lambda_graph=0，当前图重新绑定几何；控制契约明确允许 element、formal_charge、bond_order、connectivity | 已有可复用的开放图处理方向，但依赖相应实时评价器和匹配的原生对照 |
| 离线坐标副本 | `src/molsteer/molexecutor/offline.py:18`、`:87` | 固定快照与图签名用于验证已绑定的奖励 | 是离线数值试验的适用性限制，不应照搬为整个实时采样的化学身份约束 |
| 梯度预检 | `src/molsteer/molmonitor/live_gradient.py:32` | 有限差分正负扰动需位于同一离散分支，才把比较视为坐标导数检验 | 同一步的微分检查，不是跨时间锁图；但边界附近可能导致整次预检失败 |

`FlowrRootAdapter` 还显式拒绝未支持的 inpainting 模式，见 `flowr.py:84`。当前适配器的原子数/槽位数固定；这与允许槽位的元素、电荷和连接关系改变是两件事。

## 主要优化方案

### 1. 把图变化处理统一为奖励重绑定

建议运行策略使用 `allow_and_rebind`：允许原生类别演化，图变化触发适用性更新和必要的局部诊断。`t=0.50` 的图与概率保留为审计快照，不作为持续采样的化学身份要求。

对每个奖励方向显式记录：作用视图、涉及槽位、所依赖的元素/电荷/键及其局部环境、参考参数来源、失效条件、重绑定方法、当前绑定版本。局部环境不能只检查直接相连的几个原子；芳香性、共轭和 MMFF 类型可能受更远的变化影响。

- **可重算的几何项**：对每一张当前候选图重新确定实际存在的键、角、邻域与 MMFF 类型，重算参照；新键加入，已不存在的键/角记为不适用。不能把参数缺失记为零损失。
- **与化学身份无关的项**：在坐标系、槽位映射和自身前提仍成立时继续使用。受体排斥也需要随当前元素更新半径；并非所有坐标项都与类别无关。
- **依赖特定化学角色的项**：例如供受体、带电中心、特定官能团的表达式，先重新诊断、推导和验证。任意 expert 表达式中的常量无法可靠地自动替换成新化学意义。

旧项不适用、新参照缺失、目标满足要分别记录。对有 required 目标或跨项组合的表达式，不能直接删去一项后仍宣称同一个目标获得改善；应产生新绑定版本，或把该组评价记为不可用。其余确实独立的目标可以继续，原生采样仍正常推进。

### 2. 移除默认初始图偏好

新生成的常规 program 默认使用 `lambda_graph=0`，不因偏离 t=0.50 的类别概率单独惩罚新图。原有非零配置保留为可复现的消融实验，不修改历史产物。

若需要控制类别提议幅度，可单独研究相对“同一步原生预测分布”的信任域，明确它只限制额外干预幅度，并不规定化学身份不变。不能通过滚动更新 p0 再分别给不同候选使用不同参照，造成评分不可比。

### 3. 让候选可行性与初始身份解耦

保留有限值、张量尺寸、槽位映射、键矩阵对称性、无自键、控制预算等基本约束。sanitization、连通性、严重碰撞等评价针对当前候选图，不要求与初始图相同。

“所有原子连通”约束的是连通分量数，不规定具体哪些原子相连；连接关系可以改变。当前化学可行性检查还会停用无法解码的中间预测上的引导，这仍是探索限制，需要在后续方案中与终态要求分别设计。

local_first 对换图候选的近零几何损失阈值需要校准。中期可考虑时间相关的容忍度或短程续推后的评价，后期再收紧，不能把一张刚出现的新图必须立即几何完美作为未经检验的默认前提。具体阈值应通过配对实验选择。

### 4. 保证跨图比较可解释

每一步在相同时间、自条件缓存与匹配随机条件下比较原生基线和引导候选；两者分别按自身图建立几何参照，使用相同的目标定义、尺度与聚合规则。

避免通过减少被评价的键/角数量“改善”损失。明确覆盖范围与归一化方式；局部目标消失只能说明原缺陷的定义改变，不能自动证明终态质量提升。MMFF 总能量不应直接作为不同分子间的优劣排序；同图参考下的应变及跨图指标也需要明确适用性与校准范围。

### 5. 区分允许改图与主动引导改图

常规 run_suffix 对当前坐标求导，再把位移注入坐标；原子、电荷、键继续由原生积分器采样。坐标扰动可能间接改变类别预测，但这不是对离散类别的直接优化。

如需主动选择更优图，可接入现有 categorical proposal、graph_search 或 discrete_gradient 路径。候选必须重新前向计算和评价，不能沿用修改类别之前的亲和力 head，也不能把 argmax 视为具有真实坐标导数。现有类别搜索受固定槽位数、词表、候选预算和变换族限制。

## t=0.50 示例应如何处理

对于历史诊断中的 N(1)=C(14)–C(10)，`[1,14,10]` 的 119.788°、1–14 的 1.290 Å 只属于当时的图假设。

如果后续预测变为 N(10)=N(1)=N(14)，中心改为槽位 1，10–14 不再成键：

1. 停止把 `[1,14,10]` 当作旧结构的键角修复目标。
2. 基于新图评估 `[10,1,14]`、新的键阶、元素、电荷与几何参照。
3. 按新图重新检查价态、连通性、碰撞与相关化学警示。
4. 用完整续推后的独立指标判断质量，不用旧角度窗口宣称“已修复”。

这允许找到不同化学图，但不预先判断该例中的新图更优。

## 已完成的行为验证

### 源码相关回归

使用仓库现有测试，涵盖 executor、affinity guidance、Agent bridge、expert 图绑定及拒绝提议后的原生路径/RNG 保持、增广拉格朗日控制器：**26 passed**。

首次运行有 3 项因 Windows 默认 GBK 解码 UTF-8 fixture 失败；用 Python `-X utf8` 重跑后全部通过。没有为本次审计修改测试或运行时代码。

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_executor.py tests/test_affinity_guidance.py tests/test_agent_flowr_bridge.py tests/test_expert_system.py::test_compiled_expert_reward_binds_live_graph_and_formula tests/test_expert_system.py::test_engine_rejected_expert_probes_preserve_native_path_and_rng tests/test_augmented_lagrangian.py -q
```

其中 expert 现有测试明确断言图变化时抛出异常；测试通过证明当前门禁确实存在，不代表满足开放图设计要求。

### 合成分子的独立检查

用三原子 CCC 作为初始图，保持坐标与槽位数不变，受体放在无严重碰撞的位置，分别构造以下有效图并调用 `MolecularReward.feasible`、`chemistry`、`graph_cost`：

| 变化 | 候选 | 实际变化的类别头 | 可行性拒绝原因 | 当前图几何参照数 | 初始图代价 |
| --- | --- | --- | --- | ---: | ---: |
| 元素 | COC | atomics | 无 | 3 | 2.046742 |
| 形式电荷 | C[CH-]C | charges | 无 | 3 | 2.046742 |
| 连接关系（闭环） | C1CC1 | bonds | 无 | 6 | 2.046742 |
| 键阶 | C=CC | bonds | 无 | 3 | 2.046742 |

四例 MMFF 几何参照均无缺失。这直接验证常规可行性检查允许上述变化，也验证初始概率偏好会对每一种变化施加正代价。这里 p0 为构造的 one-hot，数值仅用于定位行为，不代表真实模型概率或采样成功率；未执行奖励接受比较或真实 GPU 续推。

另读取历史 GraphGuard execution_summary，确认该分支 50 步中只有 1 步取得梯度、0 次接受引导。该结果来自已有产物，并非本次重新运行。

## 实施顺序与验收条件

1. 先统一可执行的图变化策略和逐奖励项的绑定契约，保留历史 frozen-reference 实验的可复现性。
2. 让新建 program 默认不偏好初始图；为直接几何项实现当前图参照重建，同时修复 Agent 编译时适用性信息丢失。
3. 用逐方向的适用性检查、重绑定和重新验证替换 expert 的全图相等门禁；任意化学表达式不得未经验证自动改写。
4. 校准 local_first 的换图阈值；改善梯度预检在分支边界处的处理，边界样本不能误报导数通过，也不应被解释为必须保留初始图。
5. 对元素、电荷、键阶、新增/删除边、远端无关图变化、新图 MMFF 缺失、原目标消失和终态未改善等分别建立回归案例；使用相同完整检查点和匹配随机状态进行真实续推对照。

验收必须同时证明：允许图变化；不会沿用失效的旧参照；可行性失败不会回滚原生采样；奖励改善没有来自评价项丢失；终态质量用当前图上的独立指标报告。仅提高引导接受次数不足以证明优化有效。
