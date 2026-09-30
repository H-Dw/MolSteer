# SCNet 上 5i0b / ligand_002 / t=0.50 的 MolSteer × FLOWR.ROOT 推理测试

## 测试范围

- 日期：2026-09-29（Asia/Shanghai）。
- 目标：`5i0b_A__5vef_M77 / ligand_002`。只用 `t=0.50` 的状态和诊断设计奖励；从同一个完整运行检查点分别续推无引导和梯度引导分支至 `t=1.00`。
- 远端：SCNet Notebook 容器，`/root/private_data/MolSteer`；FLOWR.ROOT 在 `flowr_root/`。
- 凭据：由远端环境提供 `OPENROUTER_API_KEY`。本记录不保存密钥、SSH 密码或私有模型推理内容。

## 环境与输入确认

| 项目 | 观察 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，24564 MiB；检查时约 24081 MiB 空闲 |
| Python | `/opt/conda/envs/molsteer-flowr/bin/python`，3.12；PyTorch 2.5.1+cu121，RDKit 2026.03.6 |
| MolSteer 远端提交 | `94e09963fe643574afa1dcd935742b9acf8c087a` |
| 本地仓库提交 | `778cde401ed49cfc38f7dc8ddb95eedd0a9a3bbe`；相对远端只增加部署说明文档提交 |
| FLOWR 权重 | 本地 `tmp/flowr_root_v2.2.ckpt`，1,007,216,503 字节；远端整文件 SHA-256 已验证为 `b818f41dc12ffb6bc558bb0ad997055581e07cd9e49dcac1b794ed9993c46e4c` |
| 受体和参考配体 | `crossdocked_100target_stage_test_exact_20260923/inputs/5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb` 和同前缀 `.sdf`；已传至远端 `flowr_root/input_5i0b/`，本地/远端哈希分别一致为 `0e351338cb9eed087a11b7df62804dadb289b03811bf68f3744f5eac103b3b78` 和 `336b3554274af510034c7ce64262b90c767ae93b81cdbd548cc7cfb92addb5f5` |
| 远端初始状态 | 项目源码与环境已就绪，但没有模型权重、输入和 `runtime.pt`；因此不能直接复用本地示例声称完成远端续推 |

## 执行记录

1. SSH 接入后确认环境变量存在、GPU 可见、`molsteer`/`molreader` 可导入，且 `stage_runner.py` 和 `molsteer agents` 入口可运行。
2. 从本地传输 FLOWR.ROOT v2.2 权重及 5i0b 的受体和参考配体。SSH 权重传输中途停滞；用[官方仓库所链接的 Google Drive 文件](https://github.com/jule-c/flowr_root/blob/main/README.md#checkpoints)的 HTTP Range 补齐未传区间。补齐前逐字节比较文件开头的 1 MiB，完成后验证整文件 SHA-256 与本地一致。官方仓库将 v2.2 列为联合生成与亲和力模型。
3. 运行阶段捕获：

   ```bash
   python integrations/flowr_root/stage_runner.py \
     --output-root flowr_root/output/scnet_5i0b_stage_20260929 \
     --input-root flowr_root/input_5i0b \
     --checkpoint flowr_root/checkpoints/flowr_root_v2.2.ckpt \
     --gpu 0 --precision highest --target 5i0b_A__5vef_M77
   ```

   生成 3 条配体轨迹；`ligand_002/t_0.50/runtime.pt` 为 5.0 MiB，SHA-256 `c9a2409df784af2c98d6cd00158f929905a015967613580f0e2588914a2f05d6`。阶段捕获报告 8.65 秒；保存 `t_0.25`、`t_0.50`、`t_0.75` 的完整运行状态。
4. 用 `scripts/prepare_exact_experiment.py` 对新阶段进行 MolReader 识别和 MolThinker 的确定性参考奖励准备，输出到 `flowr_root/output/scnet_5i0b_preparation_20260929/`。生成的 StatePacket ID 为 `sp_213cf2af932e040a852bcef3`，诊断含 1 个预测终点风险区域和 3 组当前噪声状态风险。
5. 首次 API Agent 启动因 SCNet 同一文件系统的两种挂载路径不一致而停止：`/root/private_data/MolSteer` 与配置根 `/public/home/daweihuang/MolSteer` 指向同一 inode，但知识库路径的目录包含检查按字符串路径进行。改用 `/public/home/daweihuang/MolSteer/knowledge/...` 后重新启动；没有修改输入或跳过检查。
6. 默认双专家 API Agent 的 `scnet_5i0b_20260929_api01` 运行失败：Researcher 多次超出搜索预算，随后 `fetch_source` 参数校验失败并耗尽工具修复次数；工作流明确返回 `status=failed`，没有生成已验证奖励，也没有被送入 FLOWR.ROOT。审计文件保存在远端 `outputs/agent_runs/`。
7. 为检查真实梯度链路，用第 4 步绑定当前状态的确定性参考奖励先做配对续推：

   ```bash
   python -m molsteer.molexecutor.runner \
     --config flowr_root/output/scnet_5i0b_preparation_20260929/configurations/t_0.50.json \
     --arm unguided --arm creativity
   ```

   该运行的梯度有限差分预检通过，两个分支都从同一 `runtime.pt` 恢复到第 50 步；结果见下文。
8. 将本次 API 配置单独复制到 `flowr_root/output/scnet_5i0b_preparation_20260929/agents_single.json`，只将 `thinker.architecture` 改为 `single`、`external_research` 改为 `false`，以 `scnet_5i0b_20260929_api02` 重试。结果为 `status=design_deferred`、`errors=[]`：Agent 提出了角度与键长的直接物理残差，但认为当前 CLI 输入没有声明可用的实时导数、编辑掩码和图变化后的参照重绑定规则，没有提交可执行 RewardSpec。因此没有调用 `agent_continuation.py`；该入口要求已验证的 Agent 检查点，不能以设计草稿替代。
9. 根据同一 t=0.50 诊断的原始证据，以现有 `flat_bottom_angle` 和 `flat_bottom_distance` 基元构造两项直接局部奖励，另开 `scnet_5i0b_direct_20260929`，通过同一梯度预检并从同一 `runtime.pt` 配对续推。构建脚本和运行配置随产物保存。
10. 直接奖励引用 t=0.50 的原子角色、角度和键长；第 51 步预测图开始改变，因此另用显式图门禁包装相同奖励，运行 `scnet_5i0b_graphguard_20260929`。门禁仅在当前预测图与 t=0.50 参考图一致时允许评价和接受提议；图不同时令本步引导不可用，由原生采样器继续。这是为审计固定参照而额外加入的诊断策略，不属于 FLOWR.ROOT 或 MolSteer 默认限制，也不应作为允许自由图优化时的生产策略。

## t=0.50 识别

- 预测终点局部区域：N(1)、C(10)、C(14)；中心 14 的 `[1,14,10]` 角度 78.396°，相对当前图假设下的 MMFF 参照 119.788°；1–14 双键长 1.1536 Å，参照 1.2900 Å。图类别仍有歧义，数值只在当前图假设下成立。
- 当前噪声状态：声明成键类别造成明显的价态异常；N(3) 到 LEU447:CD1 为 2.0029 Å，N(1) 到 VAL335:CG2 为 2.4112 Å。噪声图的异常不直接等同于最终产物缺陷。
- 受体类型、质子化、部分电荷和独立亲和力评价仍未充分验证；这些缺口不按零风险处理。

## 确定性参考奖励与续推结果

参考 `RewardProgram.json` 的 ID 为 `rp_f1b2c630cec93effc90ae0e2`，局部原子集合为 `[1,10,14]`。其四类活动目标为 geometry、clash、pocket、movement；使用 `R = -τ log mean exp(wF/τ) - ρΣwF - λ_graph C_G`，本次固定参数 `τ=0.1`、`ρ=0.05`、`λ_graph=0.1`、四个权重均为 1。图偏好只参与离散候选接受，不产生坐标梯度。该程序及参数是仓库确定性参考实现，尚未经本目标校准，不能视为 API Agent 的验证产物。预算为强度 1.0、单步每原子最多 0.02 Å、累计最多 0.5 Å。

| 配对分支 | 续推步数 | 有梯度步数 | 接受引导步数 | 最大累计注入路径 | 最终 pKd head |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无引导 | 50 | 0 | 0 | 0 Å | 5.427943 |
| 参考奖励引导 | 50 | 44 | 40 | 0.032020 Å | 5.428281 |

两条最终 SDF 均可解析且通过 RDKit sanitize，最终坐标帧 RMSD 约为 0.000156 Å。MolReader 对两者各报告 1 个相同类型的结构筛选区域（azido/diazo 等子结构规则）；MMFF 局部弛豫能量下降分别约 19.559 和 19.513 kcal/mol。最终局部图已从 t=0.50 的 N(1)–C(14)–C(10) 变为 N(10)=N(1)=N(14)，其中 10–14 不再成键；因此不能沿用原来的 `[1,14,10]` 角度参照来声称修复。pKd 的 `+0.000338` 仅是同一 FLOWR 模型 head 的微小变化，不是独立亲和力证据。本单例证明梯度确实接入实时续推，但没有证明终态质量或结合性能改善。

## 直接局部奖励与图变化门禁

直接奖励的程序 ID 为 `rp_a32ed7b19e31e5437dac5360`，以诊断证据 `ev_144436af92aa72d14f7f` 和 `ev_9dfe8e30a31eb609b13e` 为来源，在 prediction 视图定义两个区间残差。令 `I(v;l,u,s) = ½·ReLU((l-v)/s)² + ½·ReLU((v-u)/s)²`，则 `R = -I(θ[1,14,10];89.788°,149.788°,30°) - I(d[1,14];1.161 Å,1.419 Å,0.129 Å)`。

角度窗口来自 119.788° ± 30°，键长窗口来自 1.290 Å ± 10%。两项都是无量纲、区间内零罚的坐标函数，角度在非退化几何下可微；噪声态价态和口袋近距警示仅监测，不伪装成坐标可修复目标。控制预算与参考奖励相同。该程序是本次证据驱动的实验构造，未通过 API Agent 审核。

| 运行 | 有梯度步数 | 接受引导步数 | 最大累计注入路径 | 最终 pKd head | 与配对无引导终态的坐标帧 RMSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 直接奖励，无图门禁 | 44 | 42 | 0.084150 Å | 5.425444 | 0.005178 Å |
| 直接奖励，有图门禁 | 1 | 0 | 0 Å | 5.427943 | 0 Å |

两次运行的梯度有限差分预检均通过。第 50 步的当前预测 SMILES 为 `N=CCC1OC(n2cnc3c(N)ncnc32)C(O)C1O`，同一步原生更新后的下一时刻预测为 `N=NCC1OC(n2cnc3c(N)ncnc32)C(O)C1O`；到第 51 步，奖励记录中的预测图已与 t=0.50 不同。无门禁运行有 43 步的有效奖励评价落在不同于 t=0.50 的图上，仍沿用旧原子角色，故其 42 次接受只证明执行路径可工作，**不能解释为修复原始 N(1)=C(14)–C(10) 问题**。42 次被接受的坐标提议中，下一时刻候选相对同一时刻原生基线均未改变预测图；本例最终图变化主要来自原生类别采样，而非本次坐标奖励直接选择了另一张图。最终 SDF 与无引导有相同的 N(10)=N(1)=N(14) 图且通过 RDKit sanitize，MolReader 均报告同类结构筛选警示；FLOWR 自身 pKd head 相对无引导下降约 0.002499。图门禁运行在第 50 步取得梯度，但该步的下一时刻候选已不满足初始图；0 次接受、终态坐标及 SDF 与无引导逐字节一致。这里“参照失效”只表示固定原子角色的局部几何罚项失去原先化学含义，**不表示图变化错误或应该被冻结**。

## 图变化的含义与当前优化边界

- `t=0.50` 的 N(1)=C(14)–C(10) 是当时预测终点的类别 `argmax`，并非需要保留的真实化学结构。FLOWR.ROOT 在后续步继续采样原子类别、形式电荷和键类别；中间态改图是正常的搜索自由度。改变原子或键后，119.788° 和 1.290 Å 这两个数值对新图未必有同样含义。只用旧坐标罚项继续优化，可能改善一个已不存在的键角，同时错过更好的化学图。
- 本次常规 `run_suffix` 对当前坐标求导，经过实时预测终点后把有界位移加到坐标；类别仍由原生积分器采样。坐标扰动理论上可间接改变下一次类别预测，但本例 42 次已接受提议中没有一次在同一步把候选预测图改成不同于原生基线的图。它没有对原子/键类别执行显式的目标驱动梯度更新。
- 参考 creativity 奖励会随**当前**预测图重新生成 MMFF 键角参照，允许换图；但 `lambda_graph=0.1` 的图代价用 t=0.50 冻结的类别概率作排序偏好，且口袋/位移项继续以初始位置为基准。直接奖励和 Agent 的固定 atom ID、固定角度/距离窗口则不会自动更换化学语义。化学有效性、连通性、严重碰撞和终点位移检查只约束新增引导提议；它们不是对原生采样器的禁止改图命令。
- 仓库另有可选类别温度提议、`outcome_aware` 的有限化学假设枚举，以及研究性软类别概率引导。它们在本次三个配对分支均未启用，不能把本次结果当作这些方法的效果。现有枚举在固定原子槽位数内运行，受候选数、改动槽位数、预设变换与模型词表限制；还要求匹配的原生对照、可解码的化学图和评价器。类别概率引导同样只是修改采样提议分布，不能保证最终图改变或最终质量提升。
- 要允许并有效利用原子与键的变化，后续实验需要把目标写成对不同有效图均有意义的终态效用，并在图变时按当前图重算几何或切换化学假设；对离散候选使用配对原生对照和独立评价。固定参照图门禁仅是本次定位奖励失配的对照实验。

## 来源校验

- 阶段捕获脚本 SHA-256：`cd371012e995ccf6f8343e3e66fce430e924bf9a1652840457dea2c2268901e0`。
- 新 StatePacket SHA-256：`cc6266ccd947d8bb323e3d4fbebb27b57756dea7a863ff8509b62ce6e1dd49a2`。
- 新 DiagnosticReport SHA-256：`cf98b5f0a7e824d5952aef56847afd40aae9025403b63ff31575151e8d134f60`。
- 参考 RewardProgram SHA-256：`536f0d22b0671831e969e945e45b494e091f3416b12bea428a9030e80feab6d8`。
- 直接 RewardProgram SHA-256：`a8608f4554b1fc62530e3ff0570c99ec35b567c66271fe9fb8d9085a315310a2`。
- 图门禁运行脚本 SHA-256：`9e6a71602b6d461a1601b19b0eb3286c2c5b8e0d55bd241494411d4b3b167e93`。
- 远端桥接/Agent 相关测试：`11 passed`。

## API 奖励结果

`api01` 为工作流失败，`api02` 为有依据的非执行性设计延期。两者都没有通过 Agent 奖励验证门禁，均未用于 FLOWR 推理。`api02` 特别指出：仅用 1–3 端点距离代理可能通过拉长 14–10 距离来满足窗口，而角度仍不合格；它还缺少图变化后的应用策略。这与上述直接奖励实验中观察到的图漂移一致。

## 产物与复现边界

- [本次 t=0.50 诊断](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_preparation_20260929/reasoning/t_0.50/DiagnosticReport.zh.md)、[StatePacket](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_preparation_20260929/reasoning/t_0.50/StatePacket.json)、[参考奖励](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_preparation_20260929/reasoning/t_0.50/RewardProgram.json)。
- [直接奖励构建脚本](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_direct_20260929/build_program.py)、[直接奖励程序](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_direct_20260929/RewardProgram.direct.json)、[图门禁运行脚本](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_graphguard_20260929/run_graphguard.py)。
- [参考运行摘要](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_preparation_20260929/continuations/t_0.50/experiment_summary.json)、[直接运行摘要](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_direct_20260929/run/experiment_summary.json)、[图门禁运行摘要](../outputs/scnet_5i0b_20260929/flowr_root/output/scnet_5i0b_graphguard_20260929/run/experiment_summary.json)及各自的梯度预检、逐步 trace、终态 SDF。
- [API 双专家审计](../outputs/scnet_5i0b_20260929/outputs/agent_runs/scnet_5i0b_20260929_api01.checkpoint.json)、[API 单 Agent 设计延期审计](../outputs/scnet_5i0b_20260929/outputs/agent_runs/scnet_5i0b_20260929_api02.checkpoint.json)。
- 本地复制的 `reference_artifacts.tar.gz` 和 `followup_artifacts.tar.gz` 与远端 SHA-256 一致，分别为 `1c1989cb113794e99aad9e9d2db4f395dd9f8f24700a8d90f94c72c2adb6f13a`、`6930b15bd1a5f6f5fc3a3ee5b34992e61f57fc7b738cec582efe72e314ca3b99`。模型权重因体积未复制到交付目录，但远端原文件保留并已校验。
- 结论仅针对该单检查点、单采样种子和当前未校准预算；不存在独立实验亲和力验证。进一步实现需要在图变化时重新识别化学假设并重建奖励，或启用并验证以终态效用为目标的离散类别引导，而非继续沿用失效的固定角度参照。
