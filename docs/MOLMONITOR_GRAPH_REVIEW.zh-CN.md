# MolMonitor 与可选图复核的调用约定

代码检查日期：2026-09-30。图复核实现位于 `src/molsteer/molmonitor/graph_review/`：`changes.py` 负责图变化证据，`session.py` 负责实时复核和新奖励安装。原 `graph_change.py` 保留兼容导入。

## 运行监控与奖励控制

Agent 工作流按 `MolReader → MolThinker → MolExecutor → MolMonitor` 执行。MolReader 读取 StatePacket 并形成诊断；MolThinker 设计奖励；MolExecutor 编译并做数值检查。MolMonitor 使用各指标的滚动 median/MAD 识别持续异常，先降低 `strength`，达到调节次数上限后请求 MolThinker 修订奖励。监控 Agent 只能确认宿主决定，不能绕过硬停止。关闭 `agents.molmonitor.enabled` 后，工作流直接从执行器结束或继续执行下一个分段，不创建监控模型，也不调用监控节点。

FLOWR 逐步监控是另一条运行路径。显式启用 `monitor` 时，执行器进入 `run_monitored_suffix`，对齐同起点的原生参考，测量局部几何变化，搜索通过检查的有效注入力度；持续无可接受提议、奖励不适用或独立质量冲突会生成奖励修订请求，保存恢复检查点。此请求本身不表示 MolThinker 已完成修订。

关闭 `monitor` 后执行普通 `run_suffix`，没有动态力度搜索、参考轨迹监控或图复核。奖励梯度仍经过实时模型回传到当前坐标，原生采样每步执行一次，再对额外引导提议检查奖励、化学约束和位移预算。拒绝提议时保留本步原生更新。默认位移预算为每原子每步 0.02 Å、累计 0.5 Å；实验可以显式配置。

## 图复核默认关闭

FLOWR 配置将图复核放在监控配置下面，须同时显式打开父开关和子开关：

```json
{
  "monitor": {
    "enabled": true,
    "reference": "/absolute/path/to/matched_reference.json",
    "graph_review": {
      "enabled": true,
      "agent_config": "/absolute/path/to/agents.json",
      "max_reviews": 4
    }
  }
}
```

省略 `graph_review` 或省略其 `enabled` 均表示关闭；父 `monitor.enabled=false` 时子开关无效。旧的顶层 `graph_review` 配置报迁移错误，避免关闭监控时意外调用付费 API。Agent 分段复核由 `monitoring.graph_review_enabled` 控制，默认 `false`，并服从 `agents.molmonitor.enabled`。

开启后，只监测奖励依赖的图视图。原子元素、形式电荷或键类别变化触发 `MolMonitor → MolReader → MolThinker → MolExecutor`：读取当前图，重新识别化学角色和奖励支持，编译新奖励并验证实时梯度。纯坐标变化不会触发化学复核；明确标记为与图无关的项也不因无关类别变化触发。复核保留原始槽位、采样状态、自条件、随机数状态和累计位移预算。失败或预算耗尽时暂停额外梯度，引导外的原生采样继续。

## 5i0b 五档测试约定

`scripts/test_5i0b_api_weights.py` 读取 `test` 内已复制的单分子轨迹及配体、口袋输入。在真实 t=0.50 检查点生成多视图 StatePacket，调用真实 MolReader、MolThinker 和 MolExecutor API；API 模式不会回退到离线奖励。数值验证成功后编译一次奖励，所有权重分支从同一检查点恢复，逐步运行至 t=1.00。

测试关闭 Agent 和生成器两层 MolMonitor，并关闭图复核。仅给 molecule_000 的可编辑原子注入引导；为保留原生分类采样的 RNG 布局，仍保留原始批次上下文。

测试将用户的 t=0.50 检查、单分子权重机制实验和关闭监控的要求作为 `UserTaskContext` 传入真实 API。Biology 须保留诊断的 scope/views，不能用含噪状态的图有效性证据推断终态预测无效。工具提交失败时只返回可信的契约规则、字段路径和固定提示；不回显被拒绝的公式、输入值或任意异常文本。来源公式仍要求与实际检索片段逐字对应，测试不降低这项验证。

五档含义为全局 `R_w=wR`，其中 `w∈{1,10,100,300,500}`。普通标量奖励直接缩放梯度；会归一化方向的专家控制器在选出控制方向后再乘以 w。奖励内部各目标的相对参数、执行力度及位移预算不随档次改变。较大权重可能因预算裁剪得到相同实际位移，须同时读取梯度范数、接受步数、有效位移和最终分子质量。

运行前后以文件哈希、大小及修改时间核对 data 与复制源；输出仅写入 test。保存原生精确重放核验、每档实时有限差分检查、逐步引导 trace 和运行摘要。只有实际完成的分支可以记为已测试；本地回归测试不能替代远程 GPU 与真实 API 测试。

`scripts/analyze_5i0b_api_weights.py` 对已完成的五档实验和原生对照重新运行终态 MolReader，保留化学有效性、连通性、碰撞、当前图几何、应变代理、模型亲和力及全部失败原因。真实 API 请求日志只记录 Agent 名称、请求标识、耗时和 token 计数；不保存凭据、提示正文或模型私有推理。
