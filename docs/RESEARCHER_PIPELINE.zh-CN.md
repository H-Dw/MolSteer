# 可选 Researcher：证据存储、动态影响与完整测试

Researcher 位于 MolThinker 内部，默认 **off**。`shadow` 绑定证据但不改变采样；`active` 必须选择可执行假说。MolSteer 仍保持四大模块。

## 结构化存储与调用

`molthinker/research/` 分离了检索提供器、日志与不可变发布、证据影响预算。`researcher.py` 保留化学假说检查；`research_rewards.py` 将证据绑定到奖励，并执行影响权重修订。

每次调研保存 `request.json`、`searches.jsonl`、`sources.json`、`raw/*.json`、`hypotheses.json`、`ResearchPacket.json`。来源按 DOI 或数据库身份去重。检索时间、查询式、失败、耗时、返回记录及响应哈希全部保留。发布时核对引用 ID/URL，并绑定 StatePacket、诊断、无引导结果和运行检查点。发布后不可原地修改，内容哈希用于后续核验。检索文本仅作为证据，不作为执行指令。

提供器实际调用 Europe PMC 获取元数据和摘要；agent 负责结构化综合、适用性、反证、化学映射与导数契约。本模块没有声称自动调用外部 LLM，也不把摘要检索当成穷尽全文综述。[接口说明](https://europepmc.org/RestfulWebService)

通用命令分为 `molsteer.cli research-search --request ... --output ...` 和 `research-publish --store ... --hypotheses ... --analysis ...`，使用现有 `.venv/bin/python` 及 `PYTHONPATH=MolSteer/src`。请求包含 subject、bindings、capabilities、queries 和可选 page_size。agent 阅读检索结果后，使用返回的 source_ids 编写假说，再发布证据包；不是检索完成即自动改分子。完整输入示例见本次准备实验脚本。

原 `think` 入口新增 `--research-packet`、`--research-mode off|shadow|active`、可重复的 `--research-hypothesis`、`--research-allow-exploratory` 和 `--research-fixed-influence`。省略时不增加检索或引导。当前 active 只支持已测试的 outcome-aware 执行契约；来源错配和不支持的后端明确报错。

## 动态影响与执行

当前实验影响上限为：来源质量 × 靶点相关性 × (0.25 + 0.75 × 具体替换支持) × 背景匹配 × (1 − 反证代价)；初始权重取上限的一半。五个因素都需显式给出 [0,1] 有限值；延期假说为零。这是未校准的控制预算，不是成功概率。来源数量不作为乘数。

MolMonitor 使用相同状态、自条件、RNG 下的成对观测；未知值不算改善。反复出现评分/应变回退时降低研究影响及类别力度；只有反复观察到实际图收益才允许有界提高。连续没有实际作用时降低研究影响，不自动增加力度。MolThinker 按证据上限处理权重请求，MolExecutor 使用对应力度。规则和每次前后值均可追踪；没有虚构的 LLM 调用。

`research_guidance.py` 在原生类别采样前更新原子/键/电荷概率及配套自条件，随后对两个真实原生步骤的新预测进行检查。候选无效或无法比较时恢复原生分支。分别记录概率变化、当前态采样变化、预测终点图变化；保存单步 KL、累计“各步最大槽位 KL 之和”、控制历史、权重、力度，以及 MMFF 参考、自条件和 RNG。坐标力度现允许明确设为零，用于仅类别对照。

类别目标是映射基团的平均对数支持度，而不是亲和力或图联合概率。原模型 argmax 边界未被修改；不声称对硬化学类别求到了普通梯度。坐标方向来自该步实时预测，并在类别步骤后以新预测验证；它不是穿过离散采样的导数。终点时刻的控制器更新只作为记录，对已结束的生成无作用。

## 测试与报告

准备脚本重新生成 t=0.50 的 StatePacket/诊断，检索文献，并声明关闭、固定、动态、仅类别四个实验。执行脚本完整续推 0.50–1.0，核验关闭模式与原结果精确一致、shadow 不改变采样、动态模式从 0.75 精确恢复，以及其他 batch 分子不变。评分脚本使用 Vina、Vinardo、MMFF、ProLIF、PoseBusters，对原姿势与另存优化副本分别评估并核对坐标准备。

HTML 报告内嵌全部 StatePacket、诊断、证据和决策摘要、奖励公式、检索日志、动态轨迹、分数、带编号二维分子图、同坐标系姿势投影、残余风险、验证及备份。展示可审计推导摘要，不输出私有内部思考过程。验证不完整时不能生成“全部通过”的结论。原始检索摘要保存在证据文件，HTML 使用来源元数据和自行撰写的摘要。

测试仅覆盖一个匹配检查点，不等于统计或实验效力证明。当前不覆盖原子数变化、其他模型适配、pH/水/受体 ensemble。报告复用已安装的 markdown-it-py 和 Matplotlib，本轮没有安装新包。
