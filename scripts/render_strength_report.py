"""Render the paired strength experiment without choosing only favorable runs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);main=root/'molsteer_strength_sweep_20260923'
    analysis=json.loads((main/'analysis.json').read_text())
    schedule=json.loads((main/'schedule_comparison.json').read_text())
    records=analysis['records'];lookup={(r['seed'],r['strength']):r for r in records}
    rows=['| η | 原配对状态 S0 | 配对状态 S1 | 配对状态 S2 |','|---:|---:|---:|---:|']
    for weight in [0,.3,1,3,10,30,100,300]:
        values=[f"{lookup[s,weight]['strain']:.3f}"+('*' if not lookup[s,weight]['same_final_graph'] else '') for s in range(3)]
        rows.append(f"| {weight:g} | "+' | '.join(values)+' |')
    phase=[r for r in analysis['phases'] if r['seed']==0 and r['strength']==1]
    dynamics=['| 时间段 | 引导/原生单步位移范数比 | 平均方向余弦 | 逆向步数/可评估步数 |','|---|---:|---:|---:|']
    for x in phase:
        dynamics.append(f"| {x['lo']:.2f}–{x['hi']:.2f} | {x['raw_guidance_to_native_ratio']*100:.2f}% | {x['cosine_drift_gradient']:.3f} | {x['opposing_steps']}/{x['count']} |")
    tailrows=['| 配对状态 | 无引导 | 全程 η=30 | η=30，仅引导至 t=0.9 |','|---|---:|---:|---:|']
    for x in schedule['records']:
        tailrows.append(f"| S{x['seed']} | {x['unguided_strain']:.3f} | {x['constant_30_strain']:.3f} | {x['tailoff_strain']:.3f} |")
    zh=r'''# 新奖励的引导强度、原生推理冲突及后续生成优化

原引导强度偏小确实是改善有限的原因之一，但不是唯一原因。实测还发现：当前参数化下梯度带来的实际位移较小，后期原生更新会削弱局部位移，且当前奖励在几何达标后主要优化“保持接近中间态”的目标，与整体 MMFF 应变下降并不一致。

本次完成 24 次权重对照和 3 次时段消融，共 27 次真实 FLOWR.ROOT 续生成；均从同一个 t=0.50 分子状态继续到 1.00。

## 实验设置

测试外部引导强度 η=0、0.3、1、3、10、30、100、300。这里的 η 乘在注入生成过程的梯度前；奖励内部 w、τ、ρ、λ_G 均保持不变。所有运行使用相同的最高 float32 精度，单步额外位移上限为 0.02 Å、每原子累计注入路径上限为 0.5 Å，原有化学与碰撞等接受检查保持不变。

S0 保留此前运行检查点的随机状态；S1、S2 在同样的分子状态和自条件缓存上，仅将后续随机状态设为 2026092301、2026092302。每个状态内部各权重使用共同随机数。三组都是同一个分子的后续随机性测试，不是三个独立分子样本。

历史 state.pt 缺少自条件缓存与 RNG 的限制仍然保留：原始状态张量精确恢复，隐藏上下文沿用此前重建的共享运行检查点，不宣称精确复现历史隐藏轨迹。本次 η=1 的 S0 最终坐标与类别张量，与上一轮交付逐位相同；新增监测未改变该基准结果。

## 权重扫描结果

下表为 final 的 MMFF 局部松弛应变代理，单位 kcal/mol，较低通常较好，但仅作为同一化学图的构象比较指标。

'''+ '\n'.join(rows)+r'''

*S0、η=300 改变了最终 canonical SMILES。其 10.771 不能当作原分子获得约 45% 的构象改善，不参与同图配对平均。

- 对原配对状态 S0，η=100 在本次扫描的同图结果中最低：19.635→18.737，约下降 4.57%；相对原 η=1 的 19.585 下降约 4.33%。
- 在保持三组最终图可比的权重中，η=30 的配对相对变化平均约改善 1.04%，但 S2 恶化约 3.84%。η=100 的三组平均变化反而恶化约 0.37%。
- η=300 只在 S0 改图，其余两组仍是原 canonical SMILES；S2 的应变代理升至 20.907。强度过大并不稳定地产生更好结果。
- 24 个 final 均通过已返回的 PoseBusters 检查，未检出蛋白/分子内碰撞或局部 MMFF 阈值异常。筛选警示仍在：S0、η=300 为 2 个原始命中，其他结果为 4 个命中。这些阈值通过不等于完全无风险。

S0、η≤100 的最终 canonical SMILES 为：

`[N-]=[N+]=NC1OC(n2cnc3c(N)ncnc32)C(O)C1O`

S0、η=300 的最终 canonical SMILES 为：

`N=NCC1OC(n2cnc3c(N)ncnc32)C(O)C1O`

后者保留含 N=N 的筛选警示。模型预测 pKi 从无引导的约 5.997 升至 6.487，而 pKd 从约 5.419 降至 5.302；不同预测终点并不一致，不能据此宣称结合能力改善。

## 是否与原始 inference 存在冲突

本次记录了原生位移与梯度的方向余弦。对 S0、η=1：

'''+ '\n'.join(dynamics)+r'''

46 个可计算梯度的步骤中，35 步梯度与原生漂移方向的内积为负；多数对抗较弱，不能解释为完全反向，也不表示实现中的梯度符号写错。实时有限差分通过，候选还需通过同一后续时间的奖励不退化检查。该夹角衡量固定时刻的局部坐标方向，不能替代时间与化学图同时变化后的总奖励变化。

对当前线性 FLOWR 更新，在同一随机噪声和自条件缓存下，一次额外位移 δ 经下一原生坐标更新后的局部变化满足：

\[
\delta_{\rm next}=(1-a)\delta+a\Delta\hat X_1,\qquad a=\frac{\Delta t}{1-t}.
\]

若终点预测对 δ 的响应很弱，原生步骤会削弱这次调整。S0、η=1 在 t=0.99 处的一步局部保留比例约为 0.263；这是固定共同上下文的一步估计，不是完整轨迹的因果归因。整条轨迹相对无引导的最大原子槽位坐标差曾达 0.0223 Å，final 仅约 0.00319 Å。

## 为什么增大权重仍会失败

**当前参数化下引导位移小。** 在 S0 的 t=0.60–0.75，当前状态梯度范数与终点坐标梯度范数之比平均约 0.035。前者对模型缩放坐标求导，后者对 Å 坐标求导，该比值包含坐标尺度影响，不能视作尺度无关的预测器梯度衰减率。更直接的证据是 η=1 的引导位移仅为原生位移的很小一部分；这解释了其作用偏弱，但不意味着可以无限提高强度。

**位移上限使名义权重失真。** S0 的 η=1、10、30、100 最大累计注入路径分别为约 0.050、0.383、0.500、0.500 Å；η=30 和 100 分别有 37、45 个步骤发生裁剪。之后提高 η 会改变各原子预算消耗与轨迹分配，不会等比例提高有效位移。

**奖励与质量指标有偏差。** S0、η=1 在 t≥0.9 的几何项和碰撞项都为零，剩余可微目标是质心与位移保持；图偏好在固定离散图内没有坐标梯度。原奖励没有直接优化全局 MMFF 应变，而且保持参照来自 t=0.50 的中间预测，并非已验证的低能构象。

S2 是明确反例：η 从 0 提高至 100，最终奖励从 −0.44269 提高到 −0.40563，但应变代理从 19.114 升至 20.686。奖励确实变好，独立质量指标却变差。因此问题不是简单地“优化没有生效”，而是当前奖励改善方向不总与该质量指标一致。

**离散图和后续采样会放大差异。** 只注入坐标梯度，也可能通过后续预测概率和类别采样间接改变化学图。S0、η=300 出现了这种情况；本实现没有直接对离散类别进行连续求导。

## 最后 0.1 时段关闭引导的配对消融

附加方案：η(t)=30，0.50≤t<0.90；η(t)=0，0.90≤t≤1.00。其奖励和控制上限均不变。

'''+ '\n'.join(tailrows)+r'''

三组配对在 t=0.9 之前的坐标与类别轨迹完全一致。关闭后期引导后，三组都比全程 η=30 的应变代理更低，分别降低 0.139、0.206、0.343 kcal/mol；与此同时，其最终奖励都略差。这直接支持：最后阶段的奖励目标与整体应变改善存在偏差。

该分段方案相对无引导的配对相对改善平均约 2.24%，高于全程 η=30 的约 1.04%；但 S2 仍恶化约 2.04%。三组最终 canonical SMILES 均保持不变，PoseBusters 返回检查通过、碰撞未检出，原筛选警示仍保留。不能据此宣称已有稳定通用的最佳引导策略。

## 当前结论与后续设计依据

对这一次原始随机状态，增大强度或关闭后期引导，均可取得比原 η=1 更低的应变代理。跨配对状态看，单纯增大权重不足以保证优化。默认强度未被自动调高。

后续值得验证的调整是按几何缺口决定引导强度、在几何达标后减弱中间态位置保持，以及在图与参数适用性得到确认时补充不重复计数的应变目标。这些属于后续奖励设计方向，未混入本次固定奖励的权重对照。当前数据不支持直接将 η=100 或 300 设为通用默认值。

## 交付与复现

![引导强度、预算与方向冲突](molsteer_strength_sweep_20260923/strength_analysis.png)

- [逐运行比较 CSV](molsteer_strength_sweep_20260923/comparison.csv)
- [完整分析与分阶段动力学](molsteer_strength_sweep_20260923/analysis.json)
- [时段消融结果](molsteer_strength_sweep_20260923/schedule_comparison.json)
- [原配对状态、η=100 的 final 诊断](molsteer_strength_sweep_20260923/seed_0/eta_100/evaluation/creativity/final/DiagnosticReport.zh.md)
- [原配对状态、η=300 的 final 诊断](molsteer_strength_sweep_20260923/seed_0/eta_300/evaluation/creativity/final/DiagnosticReport.zh.md)
- [第三组、η=100 的 final 诊断](molsteer_strength_sweep_20260923/seed_2/eta_100/evaluation/creativity/final/DiagnosticReport.zh.md)
- [分段引导的 final 诊断](molsteer_strength_tailoff_20260923/seed_0/eta_30/evaluation/creativity/final/DiagnosticReport.zh.md)

工程新增：逐步方向、梯度传递、裁剪、提案拒绝、局部保留比例和张量轨迹监测；权重扫描、评估、分析及报告脚本；可选 guidance_interval。没有改变奖励内部配方。20 项执行/推导测试通过，新增动力学测试覆盖方向冲突与原生步骤对注入位移的消除。27 次运行均保留配置、随机状态来源、奖励摘要、轨迹和后续检查点；54 个阶段结果完成独立质量评估及中英文诊断。

远端输出位于 `/data1/dhuang/flowr_root/output/molsteer_strength_sweep_20260923` 与 `molsteer_strength_tailoff_20260923`。未新增依赖，使用现有 matplotlib 输出图表，原生成快照与模型权重保持不变。
'''
    (root/'Analysis.zh-CN.md').write_text(zh,encoding='utf-8')
    en=r'''# Guidance strength, native inference and terminal molecular quality

Weak guidance is one cause of the small initial improvement, but not the only cause. This experiment ran 24 paired strength continuations plus 3 late-guidance ablations for the same saved molecule from t=0.50 to 1.00.

External eta was varied across 0, 0.3, 1, 3, 10, 30, 100 and 300. The reward composition, highest float32 precision, 0.02 Å per-step cap and 0.5 Å per-atom injected-path cap remained fixed. S0 preserves the earlier shared runtime RNG; S1/S2 change only suffix randomness to 2026092301/2026092302. All arms within each suffix use common randomness. These are three paired continuations of one molecule, not independent molecular samples.

Historical SC/RNG reconstruction remains a limitation. Saved state tensors are restored exactly, but no exact historical hidden-state continuation is claimed. The instrumented eta=1 S0 result exactly reproduced all earlier final coordinate and categorical tensors.

## Final MMFF relaxation proxy (kcal/mol)

'''+ '\n'.join(rows).replace('原配对状态 S0','Original paired suffix S0').replace('配对状态 S1','Suffix S1').replace('配对状态 S2','Suffix S2')+r'''

*S0 at eta=300 changes canonical SMILES and is excluded from same-graph paired averages. Its 10.771 is not evidence of a 45% conformational improvement of the original molecule.

For S0, eta=100 gives the lowest same-graph strain proxy in this grid: 19.635 to 18.737 (approximately 4.57%). Across all three same-graph suffixes, eta=30 gives a mean paired percentage improvement of approximately 1.04%, but worsens S2 by 3.84%. Eta=100 is worse on average. All 24 final endpoints pass the returned PoseBusters checks with no detected protein/intraligand clashes or local MMFF threshold violations. Structural screening warnings remain.

## Measured interaction with the native sampler

At eta=1 in S0, guidance/native step-norm ratios average 3.45%, 1.11%, 0.84% and 0.43% across time windows [.50,.60), [.60,.75), [.75,.90) and [.90,1.00]. Corresponding mean drift/gradient cosines are -0.022, -0.089, -0.132 and -0.205. In 35 of 46 evaluable steps the dot product is negative, usually weakly so. This is local directional opposition, not evidence of an incorrect gradient sign or a guaranteed decrease of a time-varying reward.

For linear FLOWR coordinates, with common noise and fixed self-conditioning, a perturbation obeys:

\[
\delta_{next}=(1-a)\delta+a\Delta\hat X_1,\quad a=\Delta t/(1-t).
\]

Weak endpoint response can attenuate the injected displacement. The one-step local retention estimate at t=.99 is approximately 0.263 for S0/eta=1. This is not long-horizon causal attribution. The largest raw atom-slot coordinate difference from the unguided trajectory reaches 0.0223 Å but is only approximately 0.00319 Å at the final endpoint.

The mean current-state/endpoint gradient-norm ratio is approximately 0.035 over t=.60-.75. The numerator differentiates model-scaled coordinates and the denominator differentiates Å coordinates; this parameterization-dependent ratio is not a scale-invariant predictor attenuation factor. The small guidance/native displacement ratios provide more direct evidence of weak effective injection. Increasing eta eventually encounters clipping: S0 eta=30 and 100 have 37 and 45 clipped steps, respectively, and both reach the 0.5 Å injected-path cap.

## Objective mismatch

For S0/eta=1, geometry and clash costs are zero at t>=.9. Remaining coordinate gradients preserve centroid and proximity to the t=.5 endpoint. That intermediate endpoint is not a validated low-energy reference. The discrete graph preference has no coordinate derivative within a fixed branch, and the reward does not directly optimize global MMFF strain.

S2 demonstrates the distinction: raising eta from 0 to 100 improves the final reward from -0.44269 to -0.40563 while worsening strain from 19.114 to 20.686 kcal/mol. The optimizer can improve its declared score while worsening an independent quality measure.

Eta=300 changes the S0 graph to `N=NCC1OC(n2cnc3c(N)ncnc32)C(O)C1O`, retaining N=N screening alerts. The other branches retain `[N-]=[N+]=NC1OC(n2cnc3c(N)ncnc32)C(O)C1O`. S0's predicted pKi increases while predicted pKd decreases; these heads do not establish improved binding. Coordinate guidance can indirectly affect subsequent categorical predictions/sampling; it does not directly differentiate discrete identities.

## Paired late-off ablation

Keep eta=30 for .50<=t<.90, then use eta=0 for the final .10:

'''+ '\n'.join(tailrows).replace('配对状态','Suffix').replace('无引导','Unguided').replace('全程 η=30','Constant eta=30').replace('η=30，仅引导至 t=0.9','eta=30 until t=.9')+r'''

Coordinates and categorical trajectories match exactly before t=.9. Late-off improves the strain proxy over constant eta=30 in all three paired suffixes by 0.139, 0.206 and 0.343 kcal/mol, while making the final reward slightly worse. This supports a concrete late-stage mismatch between the reward and the global strain proxy.

The mean paired percentage improvement relative to unguided is approximately 2.24%, compared with 1.04% for constant eta=30. S2 still worsens by approximately 2.04%; the experiment does not establish a robust universally best schedule. The default strength was not automatically increased. Future hypotheses include deficit-dependent strength, tapering intermediate-pose preservation, and a validated nonduplicative strain objective; those changes were not introduced into this fixed-reward comparison.

![Strength analysis](molsteer_strength_sweep_20260923/strength_analysis.png)

Data: [comparison CSV](molsteer_strength_sweep_20260923/comparison.csv), [full analysis](molsteer_strength_sweep_20260923/analysis.json), [late-off comparison](molsteer_strength_sweep_20260923/schedule_comparison.json). Each run retains molecular tensors, SDFs, traces and runtime checkpoints. All 54 assessed stages include StatePackets and English/Chinese risk-only reports. Twenty reasoning/execution tests pass, including new directional-conflict and displacement-retention checks. No new dependencies were installed; original generation snapshots and model weights remain unchanged.
'''
    (root/'Analysis.en.md').write_text(en,encoding='utf-8')


if __name__=='__main__':main()
