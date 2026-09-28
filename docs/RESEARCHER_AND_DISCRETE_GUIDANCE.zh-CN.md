# 证据调研与离散类别引导

下述初始可行性工作现已扩展为[可选完整流程](RESEARCHER_PIPELINE.zh-CN.md)，包含完整续推和恢复测试。下述单步结果仍按其原有范围解释。

Researcher 是 **MolThinker 内部的可选能力**，MolSteer 仍保留四大模块。默认 creativity 流程会在残余化学问题或外部评分提示坐标优化不足时考虑该方向；检索到文献不等于自动激活奖励。

## 通用信息流

MolReader 提供原子映射后的中间态预测、匹配无引导 clean 化学图、基团连接关系、极性埋藏和相互作用、类别概率与原生轨迹变化。Researcher 确认靶点与参考配体身份，检索原始实验 SAR、匹配分子对、阴性类似物及暴露/选择性代价，产出 HypothesisPacket。MolThinker 选择可检验假说并绑定证据、变量与导数路径；MolExecutor 执行，MolMonitor 将局部和终态反证回传。

`molthinker/researcher.py` 提供检索问题模板和证据契约检查。联网由 agent 的 web 工具承担，当前没有新增独立爬虫或外部 LLM 服务。必须区分“整个化合物有效”和“当前替换有实验支持”。同靶点并不意味着跨骨架可直接迁移。规范见中英文 `molthinker-researcher` Skill。

## FLOWR.ROOT 的真实导数边界

当前 `fm_pocket.py` 在调用生成器前，对当前态的原子、键、电荷类别做 argmax。实测 live pKd 对这三组输入概率的导数均为 `None`。同一次前向的输出类别概率与 affinity 是并列预测，后者对前者的导数也均为 `None`。这说明计算图未提供该导数，不能解释为化学身份没有影响。

优先在原生类别积分器采样之前，对预测概率建立可微目标。新增 `molexecutor/discrete_gradient.py`：

- `motif_score`：以平均对数边缘支持度描述已映射的原子/键/电荷组合，以平滑形式容纳互斥方案。完整连接化学由调用方验证；它不是化学图联合概率或效力预测。
- `guided_probabilities`：求真实概率梯度，中心化后指数更新；通过 KL 回溯、指数改变量裁剪、固定区域掩码、对称键、保留对角线和原有零概率支持限制干预。不修改输入或 RNG。
- `apply_to_prediction`：只替换目标分子的类别预测及相应自条件条目。原 affinity 仍描述干预前那次前向；采样后必须重新评分。

对声明的软目标 J，未裁剪形式为：

\[
g_{\alpha c}=\partial J/\partial p_{\alpha c},\qquad
q_{\alpha c}\propto p_{\alpha c}\exp\{\eta_G[g_{\alpha c}-\sum_d p_{\alpha d}g_{\alpha d}]\}.
\]

实现进一步裁剪指数并回溯，要求局部代理得分上升且类别 KL 不越界。它改变终点类别提议，**不等同于精确 CTMC 条件转移率**。坐标的 Å 预算与类别 KL 预算分开。平均期望价态正常不能代替采样化学图有效性检查。

该基元**尚未自动接入常规 suffix 引擎**。目前验证了软导数与一次真实原生采样步骤，尚未证明完整轨迹的药化图改变能提高性质。全面启用前还需要连接化学检查、轨迹记录、类别控制状态保存，以及零引导重现和续跑核验。

## 奖励候选与边界

有证据后，可组合软药效团匹配、靶点相关作用、埋藏未满足极性代价和较小的基团偏好项。供受体身份依赖键与电荷环境，不能仅由元素判断。硬 SMARTS 与 sanitization 用于评估，不能当作普通梯度来源。

应变作为可解释的取舍项或超出容忍范围的代价，不应自动压制一切增加分子内能量的方向；化学无效和新增严重碰撞另设约束。学习型性质模型需验证软输入及噪声态适用性。期望 embedding 或直通估计会改变模型契约，必须说明近似；本轮未隐式引入这些操作。

CTMC Discrete Guidance/Taylor 近似支持离散转移梯度引导的可行性，但不证明本终点启发式能改善口袋亲和力（[论文](https://arxiv.org/html/2406.01572v3)、[作者实现](https://github.com/hnisonoff/discrete_guidance)）。其他离散 classifier-based/free 方法有不同预测器和训练要求（[论文](https://arxiv.org/html/2412.10193v2)、[实现](https://github.com/kuleshov-group/discrete-diffusion-guidance)）。PharmacoBridge 通过训练的桥模型从空间药效团生成分子，不能直接当作不改生成器的奖励（[论文](https://arxiv.org/html/2412.19812v2)）。[mmpdb](https://github.com/rdkit/mmpdb) 可组织相同结构环境下的匹配分子对证据。

## 复现

`scripts/research_discrete_feasibility.py` 依赖此前核验的 release 和全新输出目录，先核验真实类别接口，再从相同完整 t=0.50 检查点完成两个坐标策略消融。分子特定的探针仅存在于实验脚本，通用基元和 Skill 不含特定原子编号。

远端结果目录为 `output/molsteer_researcher_discrete_20260924`。DiscreteGradientAudit、runs、provenance、comparison 和统一评估材料分别记录接入可行性与分子质量。本轮未安装新包；见[环境说明](ENVIRONMENT.zh-CN.md)。
