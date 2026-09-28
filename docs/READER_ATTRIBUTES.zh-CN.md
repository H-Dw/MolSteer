# MolReader属性补充与职责划分

需要补充。现有43项指标能够识别主要结构问题，但奖励选择还需要知道某个量是否可评估、梯度能到哪里、哪些原子可动、目标是否存在，以及反馈由谁产生。本次采用独立特征提取模块，既保留原始测量，又增加有来源的可用性事实。

## 已实现的补充

| 输入文档中的属性 | 当前处理 | 负责提供的模块 |
|---|---|---|
| t、total_steps、schedule | 读取阶段和manifest；根据已核验runner识别flow及0→1方向；计算名义剩余步数 | MolReader/state_capabilities |
| discrete_channel | 扩展为hard_sample / probabilities / decoded_discrete_graph，保留每个视图的含义 | MolReader/state_capabilities |
| has_x0_hat | 规范为has_endpoint_estimate，并保留本生成器的X_hat_1语义 | MolReader/state_capabilities |
| has_jacobian | 快照上下文中has_live_jacobian=false；不等于系统永远不能提供梯度 | MolReader读取；未来MolExecutor运行适配器声明 |
| can_unfold_suffix | 未提供运行适配器，保持未知 | 未来MolExecutor |
| movable_dofs、masks | 输入没有可编辑声明，保持未知；演示掩码单独存在于ExecutionMonitor | 设计约束与MolExecutor |
| atom_count_fixed | 当前活跃原子数可测；采样过程是否允许增删原子仍未知 | MolReader读取计数；运行适配器声明不变量 |
| n_particles、weight_variance、ESS | 保存候选数与活跃种群分开；运行种群与权重未知 | MolExecutor/MolMonitor |
| budget_oracle_per_step | 未提供预算，保持未知 | 运行配置 |
| graph_valid、valence_ok、arom_ok | 按state/prediction/sdf分开；通过消毒仅说明当前声明图自洽 | MolReader/chemical_readiness |
| 连通性 | 独立component_count，不与价态混为一个布尔值 | 原connectivity指标与补充事实 |
| protonation_ok | 未经验证，保持未知；RDKit隐式氢仅为假设 | 分子准备流程提供证据 |
| charge_ok | 拆分formal_charge_complete与partial_charges_available | MolReader；后者需要电荷分配流程 |
| bond_len_viol、angle_viol | 保留拓扑界表的计数；另列MMFF局部键长、角度越界数 | 原指标与chemical_readiness |
| clash_count、max_penetration | 分别汇总蛋白碰撞和分子内碰撞；蛋白穿透深度保留评估分母 | 原指标与chemical_readiness |
| plane_dev_max | 汇总已评估的芳环、SP2双键平面；显式说明不涵盖所有酰胺/共轭体系 | 原planarity指标与补充事实 |
| chirality_ok | 当前构象的手性信息可测；没有目标R/S时，正确性保持未知 | 原stereochemistry；目标约束提供参照 |
| sasa_dev | 原burial_sasa仍保留；没有基团目标面积，偏差未知 | 设计目标与未来可微后端 |
| shape_coverage、rmsd_to_ref | 原shape描述符保留；缺少指定且已映射的目标参照，不虚构覆盖度或RMSD | 设计参照 |
| desc_2d | 原MW、logP、QED、SA、TPSA、氢键计数、旋转键、形式电荷等继续保留 | 原独立指标 |
| pocket_ready | 拆分坐标可用、坐标对齐、力场准备，避免全有/全无的误判 | MolReader读取及分子准备流程 |
| priors、spec、off_target_pockets | 明确缺失；文档示例不能当作该分子的真实输入 | 设计目标与后续适配器 |
| delta_E、delta_norm | 诊断快照中未知；本次副本试算后由Monitor实际计算 | MolMonitor |
| rebound、pass_rate、validity_rate、atom_stability_rate、diversity | 需要受控采样或种群历史，本次未知 | MolMonitor |

新增坐标快照、图摘要、类别熵和方法来源，使RewardSpec能独立定位计算对象，并在输入变化时拒绝沿用旧奖励。原子/键的局部top-three概率及阶段变化已有独立chemistry_context，不再复制成新风险计数。

## 输入文档需要修正的解释

1. **时间方向不能统一写成“越大越噪声”。** 此处FLOWR为0噪声、1终点。采样总步数、当前连续时间、进度与剩余步数应分开表达；最终head还有近终点时间微调。
2. **离散通道应包含概率。** 当前保存的head输出是概率分布。把它当logits再次softmax会改变原证据。
3. **缺少Jacobian不自动指向SPSA或重采样。** SPSA需要可重复调用的评价器、可扰动变量和预算；重采样需要实时种群及权重。没有这些能力时只应保留离线推导。
4. **graph_valid=false不意味着任何几何量均无定义。** 已知元素、有效世界坐标下的半径碰撞仍可计算；依赖完整化学图、氢及力场参数的量必须单独判断。
5. **pocket_ready不能覆盖所有功能。** 蛋白原子坐标足以支持特定重原子碰撞筛查，不能支持完整静电、力场接触或严格氢键分析。
6. **未知值不等于false或0。** 没有手性参照、没有可评估非键原子对、没有活跃种群均不能报告“已通过”。示例事实表不属于当前分子的测量。

## 对5i0b / ligand_002 / t=0.50的影响

预测图通过RDKit检查，但这不确认其化学身份稳定；原噪声态仍存在PAD和局部价态矛盾。拓扑界表的键长越界数为0，MMFF局部键长越界数为1，两者参考不同，必须并存。预测终点的蛋白碰撞为0，有6320个被评估原子对；这与未完成受体质子化/电荷准备并不矛盾。

局部1–14单键概率约0.341、无键约0.293、双键约0.213。当前奖励以冻结这一图假设为条件；不能将隐式氢或单键身份当作已经验证的化学事实。全分子局部弛豫能量下降111.704 kcal/mol只作为背景，不将其分配给该三原子区域。

## 补充方式

当前已落地的三类特征模块为`state_capabilities.py`、`chemical_readiness.py`、`objective_context.py`，由`enrichment.py`汇总。可计算的从已验证输入派生；无法从快照获知的显式标记为未知，并记录原因。

未来接入时，应由生成器适配器提供可编辑掩码、Jacobian/后缀能力、实际粒子权重及预算；由设计规格提供目标区间、参照形状和脱靶列表；由分子准备流程提供质子化、部分电荷及受体类型；由Monitor在真实控制后记录反弹及批次统计。当前没有为缺少目标的性质额外安装昂贵评价器，也没有伪造这些属性。
