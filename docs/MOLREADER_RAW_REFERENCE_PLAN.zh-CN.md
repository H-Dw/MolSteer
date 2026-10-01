# 原生推理多节点参考：设计与实现

分析对象仍是传入的当前生成状态。用户通过 `reader.raw_reference` 选择同一次原生推理中已经保存的后续节点，提供对照材料；不启动生成器，也不从时间数值推断哪个节点是 final。

## 数据与证据

使用 `RawInferenceManifest` 指向原始 StatePacket。清单声明 raw 路径、时间方向、锚点、节点角色与来源。当前输入必须与锚点内容一致；节点必须属于同一 target/ligand。清单声明属于同一次原生路径，内容哈希只能验证文件和锚点绑定，不能独立证明采样历史。路径读取由主机配置控制，Agent 不获得任意文件访问权。

新产物 `RawTrajectoryComparison` 与当前 DiagnosticReport 分开，保留每个节点的时间、表示、化学身份、测量方法、阈值、来源哈希和引用。state、prediction、decoded SDF 不混用。局部支持区域的元素、形式电荷、邻接关系变化时，记录重新定型；不能把同一槽位的新化学对象称为旧缺陷已经修复。

## 比较与决策

1. 对当前有证据的风险沿已选择节点追踪：持续、首次未再触发原筛查、复发、重新定型、测量缺口。缺失测量和缺失局部覆盖不算修复。给出修复发生的采样区间，不声称知道区间内的精确时刻。
2. 对局部接触、表面埋藏、构象几何和化学身份等变化进行多因素对照；同时读取配置声明的亲和力、SA、稳定性等全局趋势。局部变化与全局指标改善同步，只形成候选关联，不构成区域贡献分解或因果证明。
3. 输出可复核的三类决策材料：持续缺陷的额外修复、原生路径已出现修复的提前干预评估、后续区域特征的增强评估。候选不自动获得优先级、权重或执行许可。
4. 生物专家整合当前证据和这些材料，比较独立缺陷覆盖、自然修复能力、提前修复的耦合风险、化学变化及可控通道，选择最小充分目标集。数学专家按所选目标检索知识库，确定目标集合、局部响应、候选函数；原生 final 坐标和图不自动成为目标约束。

## 接入方式

参考功能默认关闭。参考时间列表、是否加入显式 final、比较表示、因子和全局指标方向由 config 指定；优化起点来自当前输入包。Reader 和两位专家获得简要对比概览及分页读取工具，可记录公开解释和决策草稿。辅助记录和工具调用均不成为新的提交 gate。当前奖励的数值证据仍绑定当前包；跨时间材料使用独立引用命名空间。

离线验证包括动态节点选择、相反时间方向、持续与晚修复、化学身份改变、测量缺口、区域与全局指标关联、配置关闭、来源绑定和实际 Agent 工具接入。本轮不运行真实推理或 API。

## 配置和文件格式

`configs/agents.json` 已增加默认关闭的 `reader.raw_reference`。启动前设置：

- `enabled`：是否读取保存的 raw 对照。
- `manifest_path`：仓库内清单路径。清单里的 packet 路径也相对仓库根目录；不修改输入文件。
- `reference_times`：要读取的后续时间数值列表，来自本次实验的变量。列表不需要包含优化起点；起点就是当前传入的 StatePacket。空列表配合 `include_final=true` 表示只加显式 final。
- `include_final`：是否加入清单中明确标为 final 的节点。final 不由最大数值或某个固定时间常数判断。
- `views`：选择 state、prediction、sdf 中的表示；每条轨迹只比较同一种表示，不以 SDF 代替 noisy state。
- `factors`：选择八类现有生物物理面板中的一部分，空列表读取全部。`analyses` 选择持续缺陷、后续修复和区域关联候选。
- `outcome_metrics`：明确指标、数值字段路径、表示、改善方向和可选 `min_delta`。已提供 pKd 代理、SA 代理、MMFF strain 代理的配置，均不是奖励权重。需要其他亲和力头或独立评分时，修改该列表。
- `cross_view_associations`：默认 false，只将同一表示的局部变化与全局变化组成关联。若实验显式需要跨表示线索，可开启；此时记录 `cross_view_coevolution`，生物/数学专家仍需考虑表示差异及真实控制映射。
- `group_regional_candidates`：默认 true，同一表示、同一原子支持集合的多项关联合并为一个展示条目，包含全部来源 ID。共享原子只用于资料组织，不证明共享机制，也不意味着只需一个干预。全部逐项变化仍可在 `regional_changes` 展开。

以下示意以变量生成实际 JSON。`start_packet_path`、`saved_raw_nodes`、`selected_times` 和 `raw_run_id` 由实验提供，未写入系统或 Skills。保存节点本身是已经测量并校验的 enriched StatePacket，清单不触发新的测量或生成。

```python
manifest = {
    "kind": "RawInferenceManifest",
    "schema_version": "1.0",
    "mode": "raw",
    "trajectory_id": raw_run_id,
    "time_direction": raw_time_direction,  # increasing 或 decreasing
    "anchor_packet_path": start_packet_path,
    "nodes": [
        {"node_id": item.node_id,
         "time": item.packet["identity"]["stage_t"],
         "role": item.role,  # intermediate 或 final，由原实验声明
         "packet_path": item.packet_path}
        for item in saved_raw_nodes
    ],
    "provenance": raw_run_provenance,
}
config["reader"]["raw_reference"].update(
    enabled=True,
    manifest_path=saved_manifest_path,
    reference_times=selected_times,
    include_final=use_final_reference,
)
```

数据流：当前包 → 当前 DiagnosticReport；配置清单 + 当前锚点 → RawTrajectoryComparison → Reader 公开比较解释 → 生物专家整合与目标选择 → 数学专家知识检索、目标集合、局部响应、候选构造。两位专家都能分页读取原始比较事实和具体来源。

`record_raw_reference_analysis` 保存 Reader 的可选公开解释。生物专家可在 `record_biology_decision(stage="raw_reference_review", ...)` 中记录 temporal 候选的取舍、局部可控性、提前干预的风险、关联的其他解释，并引用 `comparison_id` 和 `rr_` 来源；记录自动传给数学专家。完整比较、解释和工作草稿保存到 Agent checkpoint，可独立审计。

参考文件缺失、错误的 raw 声明、主体不匹配或锚点过期时，比较产物明确标为 unavailable/partial，当前诊断流程仍可继续。既有奖励绑定检查保持有效：`rr_` 引用不能冒充当前 `ev_` / `me_` 数值证据。不增加强制参考读取或草稿提交 gate。

## 实现验证记录

2026-10-02 完成以下离线检查，未调用真实 API、生成器或外部评分程序：

- MolSteer `tests/` 完整回归：374 项通过。随后完成表示隔离和区域资料合并的最后修改，针对 raw 参考、前向决策和 AgentRuntime 的 51 项补测全部通过。新增回归验证 API 形式的真实工具循环和两位专家的初始输入，使用注入的确定性测试模型，不代表真实 GLM 推理性能测试。
- 动态配置覆盖任意起点/参考时间、不同时间方向、显式 final 选择、表示和因子选择；风险轨迹覆盖缺口、复发、化学重定型、局部参考改变、平面原子顺序变化。后续证据不能替换当前数值证据，锚点过期不继续沿用。
- 只读试读仓库内现有 5i0b 示例的两个保存中间态，比较结果写入 `outputs/raw_reference_saved_example/`。完整底层变化保留 10,849 条；改进后的同表示关联和资料合并得到 126 个区域评估条目，均未自动选为优化目标。原始两个 StatePacket 的 SHA256 在读入前后相同。
- 此示例没有提供 raw final，因此不声称发现了持续到 final 的缺陷或确定晚期修复。109 条风险轨迹中，25 条涉及化学/图变化，72 条起点条件未确定，12 条仍受缺失测量或定义变化影响；这些情况没有被记为已经修复。
- 测试范围为 MolSteer 自身的 `tests/`。全仓收集还包含 FLOWR 附带测试，当前 CPU 环境缺少它们的 `biotite` 依赖；未执行这些生成器测试。

本轮验证的是证据读取、表示/身份区分、配置和 Agent 数据流。提前修复或区域增强是否实际改善生成质量，需要后续受控生成对照，不能由上述软件验证替代。
