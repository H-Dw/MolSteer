# MolMonitor：动态力度控制与奖励修订路由

MolMonitor 已接入 MolSteer 的推理循环。其快路径向 MolExecutor 返回本步接受的梯度力度；慢路径生成 RewardRevisionRequest 和 MolThinkerRevisionContext，并保存可恢复的完整运行状态。没有可调用的推理 Agent 时停在修订检查点，不能把“生成了请求”表述为“Agent 已完成修订”。

## 1. 文献依据与迁移边界

| 文献 | 可借鉴机制 | 在本模块中的处理 |
|---|---|---|
| [ASCED，CVPR 2025](https://arxiv.org/abs/2503.16218) | 时间上的局部异常变化、稳健阈值 | 将观察对象改为分子几何特征；变化快只作为警报，与几何恶化和化学检查联合判断。 |
| [APG，ICLR 2025](https://arxiv.org/abs/2410.02416) | 引导方向分解、范数限制 | 保留有效注入量控制；不直接把图像 CFG 的投影方向用于分子奖励梯度。 |
| [Understanding and Improving Training-free Loss-based Diffusion Guidance，NeurIPS 2024](https://arxiv.org/abs/2403.12404) | 错配梯度与自适应步长 | 原文的噪声范数/梯度平方范数尺度依赖扩散参数化；FLOWR 不直接套用，而以实际位移和候选检查控制。 |
| [Variational Control for Guidance in Diffusion Models，ICML 2025](https://proceedings.mlr.press/v267/pandey25a.html) | 以终点代价优化轨迹控制 | 适合进一步加入短程展望，减少贪心一步改善与终态收益的错配；当前未实现 DTM。 |
| [Dynamic Classifier-Free Diffusion Guidance via Online Feedback，ICLR 2026](https://openreview.net/forum?id=z9YC9bvfUL) | 候选力度的在线比较 | 借鉴有限候选搜索；当前直接检查分子指标，没有训练或声称拥有论文中的潜空间评价器。 |
| [Training-free Multi-objective Diffusion Model for 3D Molecule Generation，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c8ff6807d1f362bb22b4f0be7b66e9ca-Abstract-Conference.html) | 多性质目标可能冲突 | 持续目标冲突反馈给 MolThinker，不无限增大 η。论文的量子性质实验不是蛋白结合亲和力验证。 |

ASCED 原文的式 4 分析加权 score 的时间差。所核对的公开代码提交为 `7501e05ec6f45f075ccfabaff6b687e04af18d2d`：[detection.py](https://github.com/YuCao16/ASCED/blob/7501e05ec6f45f075ccfabaff6b687e04af18d2d/src/detection.py) 对相邻输入差做平滑，以 median + 3×1.4826×MAD 检测，再筛选空间区域；[diffusion_util.py](https://github.com/YuCao16/ASCED/blob/7501e05ec6f45f075ccfabaff6b687e04af18d2d/src/diffusion_util.py) 的该输入序列来自预测的 pred_xstart。MolMonitor 是受这些机制启发的分子适配，并非 ASCED 原算法复现，也不复制其图像尺寸或噪声注入规则。

## 2. 什么情况下视为异常

同时维护两个比较对象：

1. 同一原始检查点、相同随机状态的无引导参考轨迹，用于判断这一阶段通常变化多快。
2. 当前已被引导的分支上，本步不加额外梯度的 native-next 提案，用于隔离“这一次注入”带来的变化。

它们不能混为一谈：历史上已经施加的引导会改变当前状态，native-next 不等于完整无引导参考分支。

观察预测终点的原子对距离，以及按当前图 MMFF 参照归一化的键长、键角偏差。键/角特征携带原子身份、电荷、相关键阶和参照标识；化学图改变后，只有定义相同的局部量才比较时间变化。新的有效化学图不自动等于 broken。噪声状态 X_t 的暂时化学无效也不直接判为终态损坏。

对归一化特征 f 定义变化率：

\[
v_j(t)=\frac{f_j(t+\Delta t)-f_j(t)}{\Delta t}.
\]

参考限值使用当前原生区间变化、最近四个原生区间的稳健离散程度和物理尺度下限：

\[
L_j(t)=\max\{v_{floor},\;3|v^0_j(t)|,\;\operatorname{median}(|v^0_j|)+4\cdot1.4826\operatorname{MAD}(|v^0_j|)\}.
\]

默认 v_floor=2 个归一化单位/生成进度；它是待校准的控制参数。单条原生轨迹不是统计总体，这一限值没有假阳性率或置信区间保证。当前实现使用变化率和持续性，未把只有两个粗斜率得到的“加速度”伪装成可靠异常分类器。

本步候选被拒绝的条件包括：奖励已有的化学/碰撞/口袋硬检查失败；新增或恶化的严重几何偏差；高变化率同时伴随同图几何恶化；奖励没有同时间改善；周期检查发现同图应变明显恶化。只有变化率高、但几何改善的候选不会因此被拒绝。没有参数、未收敛和未测量均记录为覆盖不足。

## 3. 为什么要细化 checkpoint

t=0.50、0.75、1.00 只能给出两个区间平均变化率。它们可能掩盖短暂尖峰、正负变化抵消和中途改图，不能可靠标定每步的变化率上限。

`build_monitor_reference.py` 从真实 t=0.50 runtime 复演同一原生后缀，记录每 0.01 的预测终点特征，共 51 帧，同时验证 final 状态、自条件缓存、时间与随机数逐张量未变。仍保留三点粗参考摘要供对照。动态引导每 0.05 保存完整恢复状态；发生修订事件时立即保存。仅有稀疏参考时，模块不自动升级力度，明确请求补齐参考。

FLOWR 的进度方向是 0→1；final 的额外 head 在模型时间 1−1e−4 评价，而几何参考的生成进度标记为 1。这两个时间字段明确分开。

## 4. 尽可能大的可接受力度

本模块搜索的是“声明范围内，实际测试过且通过约束的最大有效注入”，不声称求得全局最大安全 η：

\[
\delta_t(\eta)=\operatorname{BudgetClip}[\eta\,\operatorname{Inject}(\nabla_{X_t}R,\Delta t)],\qquad
\eta_t^*=\arg\max_{\eta\in H_t,\;C_t(\eta)}\|\delta_t(\eta)\|_2.
\]

H_t 包含上一次力度附近的扩张/收缩候选、下限及预算饱和上界，每步最多九个提案；重复的裁剪后位移不重复评价。若有效位移相同，选择更小的名义系数。所有候选共享同一 native-next 状态、自条件缓存和时间，不额外执行采样或消耗 sampler RNG。

默认 η 起点 30、范围 0.01–10000，扩张因子 4。单步和累计位移上限由实验明确给定，不能自动突破。图会离散跳变，约束通过性不保证对 η 单调，因此这里采用有限探测，不声称二分法能找到全局边界。总奖励改善仍不足以保证终态改善，需要独立检查。

默认每五步、从 t=0.75 开始计算 MMFF94s 弛豫代理。该量只作非可微检查；不会改变提交坐标。同图可与原生参考直接对照；新图则与自身弛豫后的局部极小值比较，按重原子数归一化。新图自身应变超过默认 1 kcal/mol/重原子，或该检查不可用时，请求独立质量复核。这是明确声明的筛查参数，不是跨分子通用的化学有效性定律，也不是跨图比较 MMFF 总能量。缺参和未收敛不能当作通过。

## 5. 动态路由

| 观察 | 路由 |
|---|---|
| 存在通过检查的有效提案 | MolExecutor：提交候选，并记录 η、有效 η、实际位移和预算消耗 |
| 无提案通过，尚未形成持续事件 | MolExecutor：本步沿原生更新，降低下次试探中心 |
| 早期奖励不适用或参考不足 | MolExecutor：不施加新增梯度，保留原因 |
| 位移预算耗尽 | MolExecutor：停止新增引导；不把预算问题错发为函数缺陷 |
| 多步无可接受力度、后期评价持续不可用、affinity/独立质量持续冲突，或新图质量检查持续异常 | MolThinker：发出函数修订请求并保存运行状态 |
| 达到修订次数上限 | 停止新增引导并报告限制，避免路由循环 |

默认持续窗口为三步、事件冷却五步、修订上限两次。暂停发生在一次原生更新及已接受提案完成后；检查点保存其完整一致状态，而不是只保存坐标。

应变每五步更新一次，其警报在下一次检查前保持有效。三步持续性不代表三次独立力场计算。修订 Agent 也可明确返回 `stop_guidance`：保留当前函数和全部预算记录，仅运行原生剩余步骤以评估后果。这种停止不代表轨迹已被修复。

## 6. 给 MolThinker 的摘要内容

- **绑定与目标**：StatePacket、原始诊断对应身份、当前 RewardProgram、检查点/受体/模型、坐标单位和时间约定。
- **异常证据**：特征位置、原子身份和电荷、坐标、当前值/参照/容差、原生变化率、观察变化率及阈值来源。
- **反事实比较**：当前分支本步无注入结果、每个候选及最终提交值；分清完整原生参考与 native-next。
- **力度干预结果**：已尝试 η、有效位移、裁剪后有效 η、奖励收益、每个拒绝原因。先回答“减小力度是否能解决”。
- **目标与独立质量**：四个 affinity head 分列，几何/碰撞、可比较的应变及适用性，不把总奖励掩盖的恶化省略。
- **时间证据**：最近事件窗口、持续性、图变化、累计预算和修订历史。
- **解释边界**：观察与“奖励漏项/尺度失衡/方向冲突”等假设分开，缺失梯度分解或不确定性不能捏造。
- **修订契约**：允许改哪些奖励项，哪些硬约束/归一化/映射必须保留，以及用什么共同指标复核。

`prepare_revision_context` 依据事件从已提供知识库检索适用机制，形成推理任务。上下文只保留有正检索分数的相关函数与原公式/来源，并以摘要及摘要哈希绑定完整奖励，避免把既有检索表和所有历史局部项重复塞入提示。它不把文档内容当作可执行指令，也不在没有推理 Agent 的情况下假称已进行创造性推导。

## 7. 模块接口与恢复

- `features.py`：构造带单位、局部映射和适用性的观测。
- `reference.py`：原生参考、粗/密采样覆盖与稳健变化率阈值。
- `controller.py`：力度候选、选择、持续性、冷却和路由状态。
- `runtime.py`：接入适配器，执行候选检查、提交及完整状态保存。
- `feedback.py`：生成紧凑 RewardRevisionRequest。
- `graph_review/changes.py`、`graph_review/session.py`：默认关闭的图变化识别与 Reader → Thinker → Executor 复核。
- `molthinker/feedback.py`：接收/检索上下文，校验修订响应。

通用 Executor 在配置包含启用的 `monitor` 时进入该循环；`monitor.enabled=false` 明确关闭。图复核是其子模块，须另行显式设置 `monitor.graph_review.enabled=true`，默认关闭，并服从父开关。Agent 分段监控通过 `agents.molmonitor.enabled` 控制，其图复核开关 `monitoring.graph_review_enabled` 也默认关闭；详见[父子开关与图复核流程](MOLMONITOR_GRAPH_REVIEW.zh-CN.md)。该循环需要适配器实现 live prediction、endpoint、native step、注入尺度、序列化/恢复接口。控制器本身不依赖 FLOWR 的网络结构，但当前集成验证只覆盖 FLOWR.ROOT。

修订响应包含 request_id、parent_program_id、新 program、理由与验证计划。用 Executor 的 `--revision-response` 和 `--resume` 加载挂起状态。身份、引用、硬阈值、监控策略及预算不允许静默变化；累计路径、自条件及 RNG 保留。新奖励须通过当前状态的 live gradient preflight 后续推理。

环境沿用已有 PyTorch/RDKit/NumPy；没有新增 package。验证输出同时保留所有成功、拒绝和修订事件，独立 final Reader 评价仍是必需步骤。局部候选检查不能提供无 broken 的全轨迹数学保证。

## 8. 调用示例

在原有 Executor 配置中增加以下字段；reference 必须绑定本次原始完整 runtime。单步/累计预算继续使用已有 budget 配置。

```json
{
  "monitor": {
    "enabled": true,
    "graph_review": {"enabled": false},
    "reference": "/data1/dhuang/flowr_root/output/molsteer_monitor_20260923/reference.json",
    "knowledge_path": "/data1/dhuang/flowr_root/MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md",
    "policy": {"initial_eta": 30.0, "max_eta": 10000.0}
  }
}
```

```bash
cd /data1/dhuang/flowr_root
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.molexecutor.runner \
  --config /path/execution.json --output /path/fresh-output
PYTHONPATH=MolSteer/src .venv/bin/python -m molsteer.molexecutor.runner \
  --config /path/execution.json --resume /path/resume_revision.pt \
  --revision-response /path/response.json --output /path/fresh-revision-output
```
