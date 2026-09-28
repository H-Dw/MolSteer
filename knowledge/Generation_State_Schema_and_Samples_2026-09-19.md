# 生成状态事实表：属性定义、样例与下游使用说明

> 用途：给「负责挑选控制函数的 LLM」一张结构化状态卡。数值由脚本/外部工具给出，LLM 只做路由与解释。
> 说明：取值缺失的**可选字段**一律省略（等价于 JSON 的 `null`），TOML 与 JSON 两种格式严格等价。

## 1. 属性定义表

| 属性名 | 类型 | 取值范围/约束 | 含义说明 |
|---|---|---|---|
| `t` | int 或 float | $t\in\mathbb{Z}$，$0\le t\le\text{total\_steps}$；连续时间参数化时 $t\in[0,1]$ | 当前去噪或流匹配的噪声时间步。值越大表示信息被破坏越严重，决定哪些评价函数的输入语义仍然成立。 |
| `total_steps` | int | $\ge 1$ | 采样总步数。与 `t` 共同给出「还剩多少步可修」的时间预算。 |
| `schedule` | enum(string) | $\in\{\text{ddpm},\text{sde},\text{flow}\}$ | 生成器的时间参数化与更新类型。决定梯度注入形式（反向均值、score 或速度场）与系数，三者不可共用。 |
| `movable_dofs` | array[enum(string)] | $\subseteq\{\text{coords},\text{rigid},\text{torsion},\text{latent},\text{hidden}\}$，非空 | 当前状态允许被改动的自由度集合。决定梯度最终落在原子坐标、刚体位姿、扭转、潜变量还是隐藏态。 |
| `discrete_channel` | enum(string) | $\in\{\text{hard},\text{logits}\}$ | 元素、键型、电荷等离散通道是硬类别还是连续松弛量。取 `hard` 时离散身份只能靠重采样，不能由坐标梯度替代。 |
| `has_x0_hat` | bool | $\{0,1\}$ | 当前步能否得到近干净估计 $\hat x_{0\|t}$。取 $0$ 时只能在带噪态上评价，或改用黑箱评分与重采样。 |
| `has_jacobian` | bool | $\{0,1\}$ | 能否对去噪器/解码器求雅可比并把梯度传回当前状态。取 $0$ 时需 SPSA 零阶估计或粒子重采样。 |
| `can_unfold_suffix` | bool | $\{0,1\}$，**可选** | 能否保存中间变量并展开完整去噪后缀做终态回传。缺失视为 $0$。 |
| `masks` | object(string→enum) | 值为 $\in\{\text{fixed},\text{editable},\text{scaffold},\text{arm}\}$，也接受链 ID；**可选** | 固定片段、编辑区、arm/scaffold 与链的归属标签。决定哪些原子允许移动，以及区域类函数是否可用。 |
| `atom_count_fixed` | bool | $\{0,1\}$ | 原子数是否固定。取 $0$ 时涉及原子数变化的操作不能写成坐标势，需要出生/删除或离散编辑。 |
| `n_particles` | int | $\ge 1$ | 同时维护的粒子或轨迹数。取 $1$ 时所有种群级方法（重采样、Pareto、结构净空）不可用。 |
| `weight_variance` | float | $\ge 0$，**可选**（随粒子法） | 粒子权重方差。骤增表示权重退化，应改用控制漂移或多样性选择。 |
| `budget_oracle_per_step` | int | $\ge 0$ | 每步允许调用外部评价器（Vina、xTB、MMFF 等）的次数。取 $0$ 时只能用不依赖 oracle 的几何项或内部条件。 |
| `graph_valid` | bool | $\{0,1\}$ | 键图能否被化学工具解析且原子、键、电荷自洽。取 $0$ 时一切物理化学类函数无定义。 |
| `valence_ok` | bool | $\{0,1\}$ | 价态与连通性是否通过硬检查。取 $0$ 时先做离散重采样或拒绝。 |
| `arom_ok` | bool | $\{0,1\}$ | 芳香性与键型标注是否通过硬检查。取 $0$ 时平面性、键角等精细几何势的界表不可信。 |
| `protonation_ok` | bool | $\{0,1\}$ | 质子化状态是否已确定。取 $0$ 时静电势、ESP 与氢键相关项的梯度方向可能本身错误。 |
| `charge_ok` | bool | $\{0,1\}$ | 形式电荷与部分电荷是否已分配。取 $0$ 时不得启用 Coulomb、ESP 等电荷依赖函数。 |
| `bond_len_viol` | int | $\ge 0$ | 越界的共价键长计数。$>0$ 时应调用键长区间罚，而不是笼统的「结构优化」。 |
| `angle_viol` | int | $\ge 0$ | 越界的键角或 1–3 距离计数。$>0$ 时用 1–3 距离罚或完整力场键角项修复。 |
| `clash_count` | int | $\ge 0$ | 低于最低分离距离的原子对**数量**（内部与界面合计或分列）。$>0$ 时启用位阻/碰撞势。 |
| `max_penetration` | float | $\ge 0$，单位 Å | 最深的空间穿透**深度**。与 `clash_count` 互补：计数给范围，深度给强度与优先级。 |
| `chirality_ok` | bool | $\{0,1\}$ | 手性中心有向体积符号是否与指定 R/S 一致。取 $0$ 时优先拒绝或局部重采样，不建议靠坐标下降翻转。 |
| `plane_dev_max` | float | $\ge 0$，单位 Å | 需共面原子集（芳环、酰胺、双键）的最大离面距离。偏大时用平面偏差势或 improper dihedral 约束。 |
| `sasa_dev` | float | $x\in\mathbb{R}$，单位 Å²，**可选** | 指定基团的可及表面积与目标 $A^*$ 之差，正值表示过度暴露。仅在需要埋藏/暴露目标时启用 SASA 势。 |
| `shape_coverage` | float | $\in[0,1]$，**可选** | 目标点云或目标体积被当前原子云覆盖的比例。偏低时用形状占据势，并优先对称 Chamfer 或最优传输。 |
| `rmsd_to_ref` | float | $\ge 0$，单位 Å，**可选** | 刚体对齐后与参考结构/模板的 RMSD。偏大时启用模板参考势或固定片段保持势。 |
| `desc_2d` | object，**可选** | $\text{QED}\in[0,1]$、$\text{SA}$、$\text{logP}$、$\text{TPSA}$、$\text{MW}>0$、$\text{HBD},\text{HBA},\text{rings},\text{rot\_bonds}\in\mathbb{Z}_{\ge0}$、$\text{formal\_charge}\in\mathbb{Z}$ | 二维图级描述符集合。用于判断 QED、logP、TPSA 等是否偏离窗口，决定是否启用全局性质代理或条件引导。 |
| `pocket_ready` | bool | $\{0,1\}$ | 口袋原子、残基、电荷与表面是否已准备齐全。取 $0$ 时所有口袋依赖函数（Vina、跨界面 vdW/静电、软表面）不可用。 |
| `pocket_frame_aligned` | bool | $\{0,1\}$ | 配体与口袋是否处于同一参考系。取 $0$ 时不得计算任何跨界面距离类函数。 |
| `priors` | object，**可选** | 键可含药效团类型与方向、目标残基、参考 shape/ESP、保留片段、排除体积；可为空对象 | 设计先验。决定锚点二次势、方向一致罚、排除球、形状与 ESP 势是否可启用以及匹配是否唯一。 |
| `spec` | object | $\text{mode}\in\{\text{target},\text{interval}\}$，$\text{direction}\in\{+1,-1\}$，`hard_thresholds` 为字符串数组 | 目标规格：是逼近单点目标值还是落入区间、越大越好还是越小越好、哪些指标是硬门槛。决定用平方误差还是区间罚。 |
| `off_target_pockets` | array[string]，**可选** | 可为空数组 | 脱靶口袋标识。非空才能计算选择性 reward；为空时不得声称做了选择性控制。 |
| `delta_E` | float，**可选** | $x\in\mathbb{R}$ | 一步引导前后能量/损失之差 $E_{after}-E_{before}$。用于判断该函数是否真的在下降，正值时须降低强度或停用。 |
| `delta_norm` | float，**可选** | $\ge 0$，单位 Å | 本步坐标修正量 $\lVert\delta_t\rVert$。超过每原子位移上限时须裁剪，否则样本被推离生成流形。 |
| `rebound` | bool，**可选** | $\{0,1\}$ | 修正后若干步是否回弹。取 $1$ 说明该梯度未被采样器吸收，应改为噪声重参数化或粒子选择。 |
| `pass_rate` | float，**可选** | $\in[0,1]$ | 硬检查（价态、键长、碰撞、手性等）通过率。持续偏低说明应回到离散修复而非继续加梯度。 |
| `validity_rate` | float，**可选** | $\in[0,1]$ | 当前批次有效分子比例。与 `atom_stability_rate` 一起用于校准编辑/去噪强度。 |
| `atom_stability_rate` | float，**可选** | $\in[0,1]$ | 当前批次原子稳定比例。与 `validity_rate` 的乘积用于选择最浅的有效噪声深度。 |
| `diversity` | float，**可选** | $\in[0,1]$ | 候选集合的结构多样性。持续偏低时改用 Pareto 排序与结构净空，而非继续加大同一梯度。 |
| `ESS` | float，**可选** | $\in[1,\text{n\_particles}]$ | 有效粒子数。接近 $1$ 表示权重坍缩，应改用控制漂移、重采样校正或增加粒子数。 |

## 2. 事实表样例：TOML

```toml
[[state]]
t = 350
total_steps = 500
schedule = "ddpm"
movable_dofs = ["coords"]
discrete_channel = "logits"
has_x0_hat = true
has_jacobian = true
can_unfold_suffix = false
masks = { ligand = "editable", pocket = "fixed", frag_1 = "fixed" }
atom_count_fixed = true
n_particles = 8
weight_variance = 0.42
budget_oracle_per_step = 1
graph_valid = true
valence_ok = true
arom_ok = true
protonation_ok = true
charge_ok = true
bond_len_viol = 2
angle_viol = 3
clash_count = 4
max_penetration = 0.85
chirality_ok = true
plane_dev_max = 0.21
sasa_dev = 12.5
shape_coverage = 0.66
rmsd_to_ref = 1.8
desc_2d = { qed = 0.52, sa = 3.4, logp = 2.1, tpsa = 78.0, mw = 412.5, hbd = 2, hba = 6, rings = 3, rot_bonds = 6, formal_charge = 0 }
pocket_ready = true
pocket_frame_aligned = true
priors = { pharmacophore_types = ["donor", "hydrophobic"], keep_fragment = "frag_1", exclusion_volumes = [] }
spec = { mode = "interval", direction = -1, hard_thresholds = ["no_clash", "valence_ok"] }
off_target_pockets = []
delta_E = -1.35
delta_norm = 0.34
rebound = false
pass_rate = 0.71
validity_rate = 0.88
atom_stability_rate = 0.93
diversity = 0.47
ESS = 5.2

[[state]]
t = 480
total_steps = 500
schedule = "ddpm"
movable_dofs = ["coords"]
discrete_channel = "hard"
has_x0_hat = false
has_jacobian = false
masks = { ligand = "editable", pocket = "fixed" }
atom_count_fixed = false
n_particles = 1
budget_oracle_per_step = 0
graph_valid = false
valence_ok = false
arom_ok = false
protonation_ok = false
charge_ok = false
bond_len_viol = 0
angle_viol = 0
clash_count = 1
max_penetration = 1.9
chirality_ok = false
plane_dev_max = 2.4
pocket_ready = true
pocket_frame_aligned = false
spec = { mode = "interval", direction = -1, hard_thresholds = [] }

[[state]]
t = 0
total_steps = 500
schedule = "flow"
movable_dofs = ["rigid", "torsion"]
discrete_channel = "hard"
has_x0_hat = true
has_jacobian = false
can_unfold_suffix = true
masks = { chain_A = "fixed", chain_B = "fixed" }
atom_count_fixed = true
n_particles = 1
weight_variance = 0.0
budget_oracle_per_step = 8
graph_valid = true
valence_ok = true
arom_ok = true
protonation_ok = true
charge_ok = true
bond_len_viol = 0
angle_viol = 0
clash_count = 0
max_penetration = 0.0
chirality_ok = true
plane_dev_max = 0.0
sasa_dev = 0.0
shape_coverage = 1.0
rmsd_to_ref = 0.0
desc_2d = { qed = 1.0, sa = 1.0, logp = 0.0, tpsa = 0.0, mw = 18.02, hbd = 0, hba = 0, rings = 0, rot_bonds = 0, formal_charge = 0 }
pocket_ready = false
pocket_frame_aligned = false
priors = {}
spec = { mode = "target", direction = 1, hard_thresholds = [] }
off_target_pockets = ["ABL1", "EGFR"]
delta_E = 0.0
delta_norm = 0.0
rebound = false
pass_rate = 1.0
validity_rate = 1.0
atom_stability_rate = 1.0
diversity = 0.0
ESS = 1.0
```

## 3. 事实表样例：JSON（与上等价）

```json
{
  "state": [
    {
      "t": 350,
      "total_steps": 500,
      "schedule": "ddpm",
      "movable_dofs": ["coords"],
      "discrete_channel": "logits",
      "has_x0_hat": true,
      "has_jacobian": true,
      "can_unfold_suffix": false,
      "masks": { "ligand": "editable", "pocket": "fixed", "frag_1": "fixed" },
      "atom_count_fixed": true,
      "n_particles": 8,
      "weight_variance": 0.42,
      "budget_oracle_per_step": 1,
      "graph_valid": true,
      "valence_ok": true,
      "arom_ok": true,
      "protonation_ok": true,
      "charge_ok": true,
      "bond_len_viol": 2,
      "angle_viol": 3,
      "clash_count": 4,
      "max_penetration": 0.85,
      "chirality_ok": true,
      "plane_dev_max": 0.21,
      "sasa_dev": 12.5,
      "shape_coverage": 0.66,
      "rmsd_to_ref": 1.8,
      "desc_2d": { "qed": 0.52, "sa": 3.4, "logp": 2.1, "tpsa": 78.0, "mw": 412.5, "hbd": 2, "hba": 6, "rings": 3, "rot_bonds": 6, "formal_charge": 0 },
      "pocket_ready": true,
      "pocket_frame_aligned": true,
      "priors": { "pharmacophore_types": ["donor", "hydrophobic"], "keep_fragment": "frag_1", "exclusion_volumes": [] },
      "spec": { "mode": "interval", "direction": -1, "hard_thresholds": ["no_clash", "valence_ok"] },
      "off_target_pockets": [],
      "delta_E": -1.35,
      "delta_norm": 0.34,
      "rebound": false,
      "pass_rate": 0.71,
      "validity_rate": 0.88,
      "atom_stability_rate": 0.93,
      "diversity": 0.47,
      "ESS": 5.2
    },
    {
      "t": 480,
      "total_steps": 500,
      "schedule": "ddpm",
      "movable_dofs": ["coords"],
      "discrete_channel": "hard",
      "has_x0_hat": false,
      "has_jacobian": false,
      "masks": { "ligand": "editable", "pocket": "fixed" },
      "atom_count_fixed": false,
      "n_particles": 1,
      "budget_oracle_per_step": 0,
      "graph_valid": false,
      "valence_ok": false,
      "arom_ok": false,
      "protonation_ok": false,
      "charge_ok": false,
      "bond_len_viol": 0,
      "angle_viol": 0,
      "clash_count": 1,
      "max_penetration": 1.9,
      "chirality_ok": false,
      "plane_dev_max": 2.4,
      "pocket_ready": true,
      "pocket_frame_aligned": false,
      "spec": { "mode": "interval", "direction": -1, "hard_thresholds": [] }
    },
    {
      "t": 0,
      "total_steps": 500,
      "schedule": "flow",
      "movable_dofs": ["rigid", "torsion"],
      "discrete_channel": "hard",
      "has_x0_hat": true,
      "has_jacobian": false,
      "can_unfold_suffix": true,
      "masks": { "chain_A": "fixed", "chain_B": "fixed" },
      "atom_count_fixed": true,
      "n_particles": 1,
      "weight_variance": 0.0,
      "budget_oracle_per_step": 8,
      "graph_valid": true,
      "valence_ok": true,
      "arom_ok": true,
      "protonation_ok": true,
      "charge_ok": true,
      "bond_len_viol": 0,
      "angle_viol": 0,
      "clash_count": 0,
      "max_penetration": 0.0,
      "chirality_ok": true,
      "plane_dev_max": 0.0,
      "sasa_dev": 0.0,
      "shape_coverage": 1.0,
      "rmsd_to_ref": 0.0,
      "desc_2d": { "qed": 1.0, "sa": 1.0, "logp": 0.0, "tpsa": 0.0, "mw": 18.02, "hbd": 0, "hba": 0, "rings": 0, "rot_bonds": 0, "formal_charge": 0 },
      "pocket_ready": false,
      "pocket_frame_aligned": false,
      "priors": {},
      "spec": { "mode": "target", "direction": 1, "hard_thresholds": [] },
      "off_target_pockets": ["ABL1", "EGFR"],
      "delta_E": 0.0,
      "delta_norm": 0.0,
      "rebound": false,
      "pass_rate": 1.0,
      "validity_rate": 1.0,
      "atom_stability_rate": 1.0,
      "diversity": 0.0,
      "ESS": 1.0
    }
  ]
}
```

样例说明：第 2 条省略了全部可选字段（`can_unfold_suffix`、`weight_variance`、`sasa_dev`、`shape_coverage`、`rmsd_to_ref`、`desc_2d`、`priors`、`off_target_pockets` 与全部反馈项），表示早期高噪声、图尚未成立、无 oracle 预算的状态。第 3 条取边界值：$t=0$、`n_particles=1`、`ESS=1.0`、`weight_variance=0`、`diversity=0`、各类违例计数为 $0$，且 `movable_dofs` 不含 `coords`（枚举临界：只能改刚体与扭转，逐原子梯度的函数一律不可用）。

## 4. 下游「选择函数的 LLM」短报告

高 $t$ 或图不成立禁用物理化学类；自由度定梯度落点。易混：$t$ 与 `has_x0_hat`、`clash_count` 与 `max_penetration`。缺字段即退回黑箱重采样。规则：①图不成立→离散重采样；②无雅可比→SPSA；③硬违例 $>0$→先区间与碰撞势。
