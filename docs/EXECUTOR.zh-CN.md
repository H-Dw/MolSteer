# 实时引导与模型接入

MolThinker 默认采用 `molthinker-reward-creativity`。显式指定 `--mode selection` 可使用保留的参考函数选择流程。未带后缀的 Skill 作为兼容入口，默认路由至 creativity。两套说明均提供英文与中文；实现细节、运行命令和环境信息保留在技术文档中。

Skills 为 Agent 提供推导规范。当前确定性组合器支持有限的、经过实现与核验的奖励基元，并非能够自动实现任意知识库公式的符号发现系统。新可观测量需补充可微实现及适用性检查。

## 四模块职责

MolReader 提供状态、诊断和证据；MolThinker 明确目标、约束与 RewardProgram；MolExecutor 计算穿过实时模型的梯度，并通过模型适配器注入原生推理；MolMonitor 检查数值、约束和生成质量。

通用入口为 `interfaces.py` 中的 `guided_step`。接入其他 flow/diffusion 模型时，提供可微终点预测、原生更新函数、模型对应的梯度注入系数、可编辑掩码、累计预算和提案检查函数。返回后续坐标、更新后的预算和记录。被拒绝的引导返回原生更新结果。

正向 flow 的时间步与反向 diffusion 的协方差系数不能通用替换。模型适配器必须声明自己的符号和尺度。模型权重冻结，每步状态与自条件缓存断开历史计算图。当前实际验证的预训练模型是 FLOWR.ROOT；diffusion 的接口约定已有小型数值测试，尚未在另一套预训练 diffusion 模型上验证。

FLOWR 专属适配器在 `flowr.py`，在线循环在 `engine.py`；奖励基元与组合在 `program.py`；配置启动和脚本输出在 `runner.py`。实时有限差分检查及独立质量评估属于 MolMonitor。

## 注入方式

\[
X_{t+\Delta t}^{\rm proposal}=\operatorname{NativeStep}(X_t,\hat X_1,t,\Delta t)
+\Delta t\eta\nabla_{X_t}R(\hat X_1(X_t,t),G).
\]

原积分器保留 `(predicted-current)/(1-t)`、坐标噪声与离散类别采样。额外梯度不再除以一次 `1-t`。梯度通过当前真实模型前向传播回传到当前状态坐标。仅为选中的 batch 分子添加位移，其余分子保留原生更新。

提案检查在同一个后续时间、相同自条件缓存下比较原生与引导终点。无效或断连的图、新增或加重的严重碰撞、超出终点位移限制、奖励退化会触发拒绝，最多尝试四个逐次减半的提案。无法评估完整目标或参照时暂停该步额外引导。这些约束保护新增引导，不代表原生生成过程或最终结果必然有效。

本实验明确允许目标分子的全部原子坐标移动。通用接口支持固定掩码；当前 FLOWR 历史恢复流程会拒绝其独立的 inpainting 模式，以免错误承诺固定原子保持。

## 奖励与实验设置

selection 按给定旧公式计算平方区间罚，使用两个初始预测视图的固定距离窗口，权重为 1、距离尺度为 1 Å，不叠加原始噪声视图的代价。固定窗口保留了旧模板的字面含义；化学图变化后其物理适用性存在限制，记录中保留这一变化。

creativity 实现给定的 log-mean-exp 平衡、辅助加和与图偏好。共同目标为局部几何、碰撞、质心保持、位移。几何项合并证据区域内的键长与键角残差，使用联合 pseudo-Huber 代价，随当前有效化学图重新查询参照。[RDKit 参数接口](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html)提供参照，但查询成功不等于物理合理性已验证。

图偏好固定使用原始上传阶段预测的类别分数，不能随步骤重新设定。离散图固定时，`C_G` 对坐标的梯度为零；它用于化学图改变时的候选接受检查，没有添加未声明的直通梯度。形式电荷不代替部分电荷。

共同目标集合不含残余应变和特定接触模式：目前缺少扣除局部项后的能量分解及明确有利接触。只采用质心保持项。亲和力 head 和无目标范围的描述符只监测，不自动设为优化目标。

参数：`tau=0.1`、`rho=0.05`、`lambda_graph=0.1`，目标权重均为 1；键长容差及尺度为当前参照的 10%，角度为 30°；碰撞及质心尺度 1 Å、位移尺度 0.5 Å；蛋白/分子内碰撞半径系数 0.75/0.70，严重重叠阈值 0.4 Å；终点相对初始预测的最大原子位移 1 Å。梯度强度为 1，单步额外位移上限 0.02 Å，累计额外路径上限 0.5 Å。均为未校准演示参数，没有按结果调优。累计注入路径并不是分子沿原生轨迹的总位移。

## 恢复与精度边界

旧 `state.pt` 缺失自条件缓存和随机数状态，重放不能完全复现。因此采用显式恢复策略：原始状态的每个张量精确恢复，三个分支共用重建的缓存和随机状态。重建上下文下的新 head 预测可能与历史 head 不同，甚至发生 argmax 图变化。报告必须说明此限制，不能称为历史轨迹的精确续跑。

新增检查点保存状态、先验、自条件、时间、步号、CPU/CUDA/Python/NumPy 随机状态、来源信息、奖励身份及已用引导预算。采用张量和基本容器，可用 `weights_only=True` 加载。恢复时仍需原受体、参考配体、预处理流程和模型权重；不允许静默更换目标上下文。

探索运行中，启用 TF32 的小扰动导数检查失败。交付适配器默认使用最高 float32 矩阵精度并关闭 TF32；启动前记录多种步长的有限差分、类别图是否稳定、梯度方向及相对误差，若不通过声明的筛查标准则停止执行。成功反向传播本身并不能证明有限扰动与梯度一致。[PyTorch 自动求导说明](https://docs.pytorch.org/docs/stable/autograd.html)。

## Agent 自动启动

```bash
cd /data1/dhuang/flowr_root
.venv/bin/python -m molsteer think --mode creativity \
  --packet /path/StatePacket.json --report /path/DiagnosticReport.json \
  --knowledge /path/knowledge.md --output /path/creativity.json
.venv/bin/python -m molsteer.molexecutor.runner \
  --config MolSteer/experiments/guidance/execution.json --emit /path/new-job
.venv/bin/python /path/new-job/run_guidance.py
```

请使用新的输出目录。中途恢复可将 `resume_checkpoint` 指向保存的运行检查点，并选择相同 arm，保持奖励不变。历史重放流程的 `start_step` 默认是 50。质量评估入口为 `python -m molsteer.molmonitor.evaluate_generation --root RUN --config CONFIG`。

部署只更新独立 MolSteer 包并新增输出目录，不覆盖原模型源文件及原始分子快照。
