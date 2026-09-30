# 5i0b / ligand_002：图变化复核与 FLOWR.ROOT 梯度续推实测

日期：2026-09-30（Asia/Shanghai）。本页单独记录 `5i0b_A__5vef_M77 / ligand_002` 的 `t=0.50→1.00` 测试。远端为 `ksai.scnet.cn:10545` 的 SCNet Notebook，仓库 `/root/private_data/MolSteer`。本文不含 SSH 密码、API 密钥或模型私有推理文本。

## 测试定义与来源

使用 2026-09-29 在同一容器捕获的真实 FLOWR.ROOT v2.2 阶段：`flowr_root/output/scnet_5i0b_stage_20260929/5i0b_A__5vef_M77/ligand_002/t_0.50/runtime.pt`。它保存了第 50 步、模型、自条件和随机数状态；SHA-256 为 `c9a2409df784af2c98d6cd00158f929905a015967613580f0e2588914a2f05d6`。两支从此同一状态恢复，均运行到第 100 步；本次没有重新随机生成一个 `t=0.50` 起点。阶段脚本 SHA-256 为 `cd371012e995ccf6f8343e3e66fce430e924bf9a1652840457dea2c2268901e0`。

远端实测为 RTX 4090（24,564 MiB）、Python 3.12.14、PyTorch 2.5.1+cu121；FLOWR v2.2 权重、受体和配体输入均存在。MolSteer 优化代码先提交到 GitHub，再由远端 `git pull --ff-only origin main` 拉取，最终源码提交为 `64e30d4`；远端完整测试为 **207 passed**。保留远端原有未提交数据。开始时曾经通过归档传入源码做预检，正式运行前已恢复原文件并改用 GitHub 拉取后的提交。

## `t=0.50` 重新识别

`scripts/prepare_exact_experiment.py` 从该阶段重新生成 StatePacket `sp_a5662e7e4de6bb9f80da5f10`、DiagnosticReport 和确定性参考程序；Reader 索引 80 条证据，报告 1 个预测终点结构风险区域。重点为原子槽位 1、10、14：当前预测图中 N(1)–C(14)–C(10) 的局部几何与基于这张暂时假设图的 MMFF 参照有偏差。`t=0.50` 图仅是当时的类别预测，不是要求保留的化学结构。

API 单 Agent 使用当前配置和同一新诊断进行了真实设计尝试，结果为 `design_deferred`、`errors=[]`，**没有可执行的已验证 API 奖励**。它指出候选派生器没有提供直接角度 `[1,14,10]` 的软约束项；1–3 端点距离不能替代角度；硬约束与现有软距离原语的语义也不一致。因此 API 草稿没有送入 FLOWR 推理。路径双挂载（`/root/private_data` 与 `/public/home/daweihuang`）导致首次知识文件路径校验失败，改用后者的绝对路径后才执行该 API 设计。

为完成可执行链路测试，另行显式启用 **offline** Agent；其奖励检查点状态为 `validated`，不可称为 API Agent 的产物。初始程序 `rp_fa6350791b8c933e9624123d` 具有 4 项：prediction 视图上原子 1–10 的区间距离 `[1.7705, 3.0508] Å`、1–14 的区间距离 `[1.161, 1.419] Å`，均声明依赖化学图；state 视图上原子 3、1 对各自受体参考点的最小距离下限均为 `2.475 Å`，均声明为纯坐标项。四项权重各 1，`lambda_graph=0`。这是筛选性几何／近距奖励，没有独立亲和力证据，也没有补上 API 所要求的角度原语。

## 代码变化与试运行

提交 `f85d16e` 删除了 expert 的全图一致性检查，加入 MolMonitor 的图变化事件、MolReader 新图读取、MolThinker 奖励复核、MolExecutor 重新编译与实时梯度预检。保留原生槽位、RNG、自条件和累计位移预算；原子、形式电荷、键类别仍由 FLOWR 原生采样。候选若换图不会仅因“与 `t=0.50` 图不同”而拒绝；化学不可解码、严重碰撞或奖励回退仍可拒绝新增引导提议。若原生更新后的图改变，当前步先暂停额外引导，下一步复核后继续；原生采样不回滚。

代码中仍有 `same_graph` 字段，用于判断两次 MMFF 应变数值能否在同一化学图下直接比较；不同图会标记需要质量复核，而不会因这项比较被强制改回初始图。`agent_mixed` 本次程序的 `lambda_graph=0`，没有初始图类别偏好。

第一次配对试运行设置 `max_graph_reviews=4`。因为初始 Agent 程序含两个 **不依赖化学类别**的 state 最小距离项，旧触发逻辑仍把噪声态的每步类别波动送入复核，预算迅速耗尽：引导分支仅 3 步取得梯度、0 步接受。这个试运行只用于定位复核触发过宽的问题。随后提交 `64e30d4` 将 FLOWR 实时复核限定到奖励项声明的 `graph_dependent` 视图，远端拉取后定向测试 **13 passed**。若旧程序未声明依赖性，仍保守监测相关视图。

最终配对运行使用该修正和 `max_graph_reviews=20`。起点奖励及第 51 步重设计奖励均通过实时有限差分预检。第 51 步预测图的 **原子 14 从 C(0) 改为 N(0)**，事件 `gc_e44eace29e388c528d94541f` 触发一次 Reader→Thinker→Executor 复核，得到新 StatePacket `sp_2cc7f9108a5bb3a93bfe61ed` 和已验证程序 `rp_af75395c5258868ce73e8331`。旧图条件下的两个 prediction 键长窗口被撤掉；新程序只保留两个 graph-independent 的 state 最小距离项。之后不再因为与这些纯坐标项无关的类别变化反复调用 Agent。编辑槽位仍受宿主原生槽位约束。

| 分支 | 续推步数 | 有梯度步数 | 接受引导步数 | 已验证图复核 | 最大累计注入路径 | 终态 FLOWR pKd head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无引导原生 FLOWR.ROOT | 50 | 0 | 0 | 0 | 0 Å | 5.42794323 |
| offline Agent + 图变化复核 | 50 | 44 | 41 | 1 | 0.00429059 Å | 5.42802334 |

两支终态 SDF 都通过 RDKit 解析和 sanitize，均为 20 个原子且连接图相同；坐标帧直接 RMSD 为约 `0.000378 Å`。模型内部 pKd head 差为 `+0.0000801`，远小于可据此宣称结合力改善的程度。终态 MolReader 两支均报告 1 个相同类别的 `structural_screening` 区域（槽位 1、10、14）；没有证据表明引导消除了该结构筛选警示。引导确实接入坐标续推，但此次位移很小，单例不证明分子质量或亲和力提高。

## 实际分子图

下图直接由保存的阶段和两条终态 SDF 用 RDKit 绘制，标注原始原子槽位，粉色标出 1、10、14。**两条终态图相同是观测结果，不是测试要求或接受门禁。**`t=0.50` 的 `N=CC…` 在后续原生类别采样后，两条分支均变为叠氮形式 `[N-]=[N+]=N…`；原子 10、14 的元素身份也已变化。

![t=0.50 暂时预测图、无引导终态、梯度引导终态](assets/5i0b_t050_direct_guided_graphs_20260930.png)

```text
t=0.50 暂时预测: N=CC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O
无引导 t=1.00:  [N-]=[N+]=N[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O
引导 t=1.00:    [N-]=[N+]=N[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O
```

这里没有执行离散类别梯度更新。坐标位移理论上可间接影响后续类别预测；本次最终两支的类别恰好收敛为同一图。引导分支采用图变化后重设计的空间距离目标，不能把 41 次接受解释为修复了 `t=0.50` 旧图的键长或键角。

## 复现与产物

以下命令假定远端既有权重和上述完整阶段检查点。输出目录须使用尚不存在的新名称。`agents_offline.json` 由 `configs/agents.json` 改为 `mode=offline`、`thinker.architecture=single`、`thinker.external_research=false`，并为运行设置独立 `trace_dir`；API 配置使用相同结构的 `mode=api`。模型凭据仅来自远端环境。

```bash
cd /root/private_data/MolSteer
git pull --ff-only origin main
python scripts/prepare_exact_experiment.py \
  --input-root flowr_root/output/scnet_5i0b_stage_20260929 \
  --output flowr_root/output/scnet_5i0b_graph_review_20260930/preparation \
  --model-root flowr_root \
  --knowledge knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md \
  --gpu 0
python -m molsteer agents \
  --config flowr_root/output/scnet_5i0b_graph_review_20260930/agents_offline.json \
  --packet flowr_root/output/scnet_5i0b_graph_review_20260930/preparation/reasoning/t_0.50/StatePacket.json \
  --report flowr_root/output/scnet_5i0b_graph_review_20260930/preparation/reasoning/t_0.50/DiagnosticReport.json \
  --knowledge /public/home/daweihuang/MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md \
  --run-id UNIQUE_OFFLINE_RUN_ID
python integrations/flowr_root/agent_continuation.py \
  --agent-checkpoint outputs/agent_runs/scnet_5i0b_graph_review_20260930/UNIQUE_OFFLINE_RUN_ID.checkpoint.json \
  --flowr-config flowr_root/output/scnet_5i0b_graph_review_20260930/preparation/configurations/t_0.50.json \
  --flowr-root flowr_root \
  --output-root flowr_root/output/UNIQUE_GRAPH_REVIEW_RUN \
  --graph-review-agent-config flowr_root/output/scnet_5i0b_graph_review_20260930/agents_offline.json \
  --max-graph-reviews 20
```

远端最终实验目录为 `flowr_root/output/scnet_5i0b_graph_review_20260930/offline_guided_v2/`，含已编译奖励、桥接来源清单、梯度预检、逐步 trace、图复核事件、新旧程序和两条终态 SDF。Agent 审计位于 `outputs/agent_runs/scnet_5i0b_graph_review_20260930/`。关键结果已复制到本地 `outputs/scnet_5i0b_graph_review_20260930/artifacts.tar.gz`；本地和远端压缩包 SHA-256 均为 `5f5c9139ecff2ad7afe44772f05d61b96d9ce3ee7c267493f457f68c1907d879`。模型权重与完整运行状态仍保留远端，没有把凭据或大型权重纳入报告仓库。
