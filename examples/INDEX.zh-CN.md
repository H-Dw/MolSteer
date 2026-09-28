# MolSteer 结果索引

四份DiagnosticReport均由增强StatePacket生成，保留原有516条指标观察及189条来源证据。每份报告继续按局部问题合并重复来源，并将噪声态与覆盖限制分开。

| 分子 | 阶段 | 当前预测终点的主要风险 | 报告 |
|---|---|---|---|
| 5i0b / ligand_002 | t=0.50 | `[1,14,10]`角度67.669°，1–14键长1.2441 Å；相应类别有明显歧义 | [中文](5i0b_A__5vef_M77/ligand_002/t_0.50/DiagnosticReport.zh.md) · [English](5i0b_A__5vef_M77/ligand_002/t_0.50/DiagnosticReport.en.md) |
| 5i0b / ligand_002 | t=0.75 | `[10,1,14]`角度159.243°、1–14距离1.0794 Å；元素、电荷及角中心已改变 | [中文](5i0b_A__5vef_M77/ligand_002/t_0.75/DiagnosticReport.zh.md) · [English](5i0b_A__5vef_M77/ligand_002/t_0.75/DiagnosticReport.en.md) |
| 2pqw / ligand_001 | t=0.50 | C3–O9单键1.2186 Å，相对MMFF参照偏短 | [中文](2pqw_A__2rhy_MLZ/ligand_001/t_0.50/DiagnosticReport.zh.md) · [English](2pqw_A__2rhy_MLZ/ligand_001/t_0.50/DiagnosticReport.en.md) |
| 2pqw / ligand_001 | t=0.75 | 重叠亚胺筛选规则合并为一处警示；不等同于已证实化学无效 | [中文](2pqw_A__2rhy_MLZ/ligand_001/t_0.75/DiagnosticReport.zh.md) · [English](2pqw_A__2rhy_MLZ/ligand_001/t_0.75/DiagnosticReport.en.md) |

## 5i0b / ligand_002 / t=0.50 奖励推导

- [中文推导](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardDerivation.zh.md) · [English derivation](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardDerivation.en.md)
- [RewardSpec](5i0b_A__5vef_M77/ligand_002/t_0.50/RewardSpec.json) · [21项知识检索记录](5i0b_A__5vef_M77/ligand_002/t_0.50/RetrievalTrace.json)
- [增强StatePacket](5i0b_A__5vef_M77/ligand_002/t_0.50/StatePacket.json) · [ExecutionMonitor](5i0b_A__5vef_M77/ligand_002/t_0.50/ExecutionMonitor.json)

预测终点奖励由两个区间罚组成：1–3端点距离`d(1,10)`采用1.786254–3.077091 Å，1–14键采用MMFF参照±10%的1.3059–1.5961 Å。二者均为条件性几何奖励；不额外将同一角度的MMFF提示再计为一个罚项。

另为原噪声态两处配体—蛋白碰撞生成单侧距离罚，保存在独立state组，未与预测终点求和，也未纳入本次坐标下降试算。

本次只允许坐标副本中的`[1,10,14]`移动。局部总罚从0.03282699降至5.617×10⁻¹³；有限差分最大误差4.458×10⁻¹⁰；最大原子位移0.124730 Å；未增加所检查的键长窗口异常及蛋白碰撞。由于1–14单键概率约0.341，这不确认真实键型、化学稳定性或终态质量。

## 验证记录

[真实数据完整性检查](verification.json)保留证据计数、五类负向校验及来源检查。[单元与回归测试记录](../validation/test_summary.json)记录26项原测试和10项新测试；[环境记录](../validation/environment_changes.json)记录新增依赖。
