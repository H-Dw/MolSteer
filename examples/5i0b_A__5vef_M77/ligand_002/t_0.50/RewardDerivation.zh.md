# MolThinker — 奖励函数推导

5i0b_A__5vef_M77 / ligand_002 / t_0.50

StatePacket: `sp_7c2bbc227fdd4e662ab161c2` · RewardSpec: `rw_4098beec3d554ffe2a657933`

检索知识库中的21项函数；在当前化学图假设下构建局部坐标奖励。实际生成器执行仍受阻，本次仅在坐标副本上验证数值行为。

## 推导出的奖励项

R = −Σ Eᵢ; Eᵢ = ½ wᵢ {max[(ℓᵢ−vᵢ)/sᵢ, 0]² + max[(vᵢ−uᵢ)/sᵢ, 0]²}.

单侧碰撞罚省略上界项。不同视图分别求和，原状态与预测终点不相加。权重均为1，距离尺度1 Å、角度尺度10°，是未校准的演示设置。

| View | Atoms | Observable | Bounds | Scale | Knowledge | Evidence |
|---|---|---|---|---|---|---|
| prediction | [1, 10] | 1–3端点距离（键长/键角耦合代理） | 1.786254–3.077091 angstrom | 1.0 | G01, line 24 | ev_b426fc1a42649d46af8f |
| prediction | [1, 14] | MMFF参照键长窗口 | 1.305900–1.596100 angstrom | 1.0 | G01, line 24 | ev_7f9b0b23003da23a9390 |
| state | [3] | 配体—蛋白最小距离 | ≥2.475000 angstrom | 1.0 | G02, line 25 | ev_dcbe941e2040b9b8d244 |
| state | [1] | 配体—蛋白最小距离 | ≥2.475000 angstrom | 1.0 | G02, line 25 | ev_66a39522a0aba6109d18 |

## 化学假设与证据限制

- [1, 10]: order 0 p=0.439962; order 1 p=0.408524; order 2 p=0.139508.
- [1, 14]: order 1 p=0.341107; order 0 p=0.292530; order 2 p=0.212557.
- [10, 14]: order 1 p=0.590613; order 0 p=0.285790; order 2 p=0.073467.

端点1–3距离是键角与键长共同决定的代理量。其角度MMFF提示作为同一局部问题的支持，不再添加重复罚项。MMFF键长窗口依赖当前元素、形式电荷、隐式氢和键级；类别变化后必须重新推导。全分子应变不作为该位置的局部能量。

## 检索与适用性决策

| Function | Role | Decision | Reason |
|---|---|---|---|
| G01 · 通用 flat-bottom 双侧／单侧区间罚 | coordinate_energy | conditional_offline | 局部测量及边界可用；仅在冻结化学假设及明确演示掩码后离线验证 |
| P02 · 跨界面 van der Waals 势 | coordinate_energy | deferred | 有蛋白坐标，但没有经过核验的受体力场类型 |
| P01 · MMFF94 分子内能 | coordinate_energy | deferred | 全分子应变可作证据，但化学图稳定性、质子化及参数适用性未确认 |
| G02 · 成对最小距离位阻罚（含排除球） | coordinate_energy | conditional_offline | 局部测量及边界可用；仅在冻结化学假设及明确演示掩码后离线验证 |
| G03 · 锚点／保持二次势 | coordinate_energy | deferred | 缺少锚点坐标及可移动区域声明 |
| G04 · 方向一致罚（方向性药效团） | coordinate_energy | deferred | 缺少目标方向及已准备的特征类型 |
| G05 · 立体化学与共轭平面几何势 | coordinate_energy | deferred | 没有具有有效参照的手性/平面异常 |
| G06 · 形状／体积占据势 | coordinate_energy | deferred | 缺少指定参照形状及对齐 |
| G07 · 局部环境相似核势 | coordinate_energy | deferred | 缺少可微环境描述符与参照库 |
| P03 · 跨界面静电势 | coordinate_energy | deferred | 形式电荷不能替代部分电荷及介电模型 |
| P04 · xTB 原子力范数稳定性损失 | coordinate_energy | deferred | 缺少xTB评价器、化学验证及调用预算 |
| P05 · Vina / torchvina 总分（含五个分项与柔性校正） | coordinate_energy_or_population_score | deferred | 缺少Vina准备、评分后端及实时后缀导数 |
| P06 · 可微 SASA 目标势（候选） | coordinate_energy | deferred | 缺少基团面积目标与可微SASA后端 |
| P07 · ESP 表面相似势（候选） | coordinate_energy | deferred | 缺少对齐的ESP参照及部分电荷 |
| S01 · 目标值型高斯核 reward 权重 | population_weight | deferred | 缺少目标值、容差及实时种群 |
| S02 · softmax / tilted 重要性采样权重 | population_weight | deferred | 亲和力数值不能单独确定目标方向、温度及实时种群 |
| S03 · 目标/脱靶选择性 reward | population_score | deferred | 缺少同一候选的配对脱靶评分 |
| S04 · SPSA 零阶梯度估计 | gradient_estimator | deferred | 属于估计方法；缺少可重调用评价器、可扰动自由度及预算 |
| S05 · Pareto 非支配排序 + 结构净空（SAES） | selection_strategy | deferred | 缺少多目标声明、候选种群及结构距离 |
| S06 · 批次完整性分数（噪声深度校准） | hyperparameter_strategy | deferred | 单个已保存状态不能提供编辑深度校准批次 |
| S07 · 三群结构罚与化学有效性罚 | population_strategy | deferred | 缺少目标片段与三种群搜索适配器 |

未覆盖的诊断风险（例如价态冲突、PAD、断连）保留为诊断证据；本实现不会伪造可微的类别修复奖励。SPSA是估计方法；没有可重复调用的评价器时，不因缺梯度而自动启用。

## 执行边界

当前只有离线坐标副本的梯度。保存张量不提供生成器雅可比；活跃原子掩码不是可编辑掩码。三份已保存分子也不代表运行中的粒子种群。若接入生成器，需要明确的可编辑自由度、预算及端点梯度映射。

## 离线数值验证

- Penalty: 0.03282699 → 5.6170305e-13; reward: -0.03282699 → -5.6170305e-13.
- Finite differences: True; maximum error 4.458e-10.
- Demo movable atoms: [1, 10, 14]; maximum displacement 0.124730 Å.
- Fixed atoms unchanged: True; original snapshot unchanged: True.
- New bond-window violations: 0; new protein clashes: 0.

| Atoms | Before | After | Unit |
|---|---|---|---|
| [1, 10] | 1.537599 | 1.786253 | angstrom |
| [1, 14] | 1.244057 | 1.335957 | angstrom |

本次仅验证公式及局部能量下降，没有重新运行完整分子生成，不证明final质量、结合能力或化学稳定性得到改善。原始生成文件未修改。
