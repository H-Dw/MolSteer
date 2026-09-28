# 完整阶段检查点与相同奖励的起始时间对照

阶段入口为 `integrations/flowr_root/stage_runner.py`。每个阶段保留原有五个分子文件，新增 `runtime.pt` 与 `runtime.json`。runtime 保存完整 batch，包含实际自条件缓存、采样先验、时间网格、Python/NumPy/PyTorch CPU/所有 CUDA 随机数状态、口袋输入与编码、坐标原点、精度、实际 CUDA 设备及模型/入口哈希。各 ligand 的 runtime 文件是同一 batch 文件的硬链接。

保存完整 batch 是为了保持离散类别抽样的随机数消耗顺序。final 检查点中的原生时间保持 1.0，最终报告 head 单独使用正常的末端时间校正。

原始 `state.pt` 没有保存自条件与 RNG。旧 MolExecutor 的 `resume_*.pt` 已保存这两个字段，但其初始上下文来自重放。新增 runtime 则直接记录本次原生轨迹中的真实现场，不能混称。

## 复现流程

完整命令见[英文说明](EXACT_RESTART.md)。远端原阶段入口已先备份为 `stage_runner.before_runtime_20260923.py`，历史 state/head/SDF 未覆盖。新生成必须使用新输出目录。

阶段入口现在显式设置 CUDA 设备。原设备解析函数返回未编号的 `cuda`，仅设置 `mp_index` 不能保证消耗相应 GPU 的 RNG。MolExecutor 保存并检查实际设备编号，设备不一致时拒绝静默恢复；跨设备迁移需要显式映射随机数发生器。

`prepare_exact_experiment.py` 只在 t=0.50 运行 MolReader 与 MolThinker。两个起点共用同一个 `RewardProgram.json`，并把 `reward_reference_stage` 固定为 t=0.50，以保证参考坐标 X0 和固定图偏好 p0 相同。`saved_stage` 和 `resume_checkpoint` 分别选择 t=0.25 或 0.50。通用执行入口也支持 `reward_reference_stage`，未设置时沿用 `saved_stage`。

这是利用 t=0.50 诊断信息进行的离线回溯控制实验，不代表没有未来信息的在线决策。没有为 t=0.25 重新设计奖励。

每个起点执行 η=0、1、10、100、300；所有分支恢复真实检查点中的随机数状态，没有重新设种子。另做两次无引导消融：仅在起点清空一次自条件缓存，后续由原生采样恢复正常更新。该消融用于测量敏感性，不是恢复策略。

两个起点均使用每步 0.02 Å、每原子累计注入路径 0.5 Å 的限制，提前开始不增加总路径预算。初始 head 必须逐位一致；无引导续跑必须精确复现同次原生生成的最终状态、structure/affinity 输出、自条件和 RNG，之后才比较引导。

早期预测断连或化学图无效时，按照同一奖励的适用性条件暂停引导。有限差分预检在未干预原生轨迹的第一个可评估点进行，每个实际步骤仍独立检查适用性；每个正式分支均重新恢复原检查点。

## 新分子的评估

不冻结化学图。按原槽位记录元素、电荷和键变化，同时匹配重原子化学角色，避免把氮原子槽位互换误认为重原子骨架变化。比较分子式、总电荷、MW、logP、TPSA、QED、HBD/HBA、可旋转键、SA 代理、局部几何、应变代理、筛选警示和四个 affinity 预测头。保留改变身份的候选，但不将其应变值解释为原分子的构象改善百分比。

本次数据分别位于 `output/crossdocked_100target_stage_test_exact_20260923` 和 `output/molsteer_exact_restart_20260923`。设备不匹配的初稿及被用户修正替代的 t=0.25 奖励推导已移入 `*_device_draft_20260923`，不进入正式比较。`validation/` 保留有限差分、精确恢复及文件字段核查证据。未安装额外依赖。
