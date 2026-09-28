# 环境安装与复现

已部署位置：`/data1/dhuang/flowr_root/MolSteer`。运行环境：`/data1/dhuang/flowr_root/.venv/bin/python`。原始生成输入仍位于`output/crossdocked_100target_stage_test`；本次派生结果位于`output/molsteer_reports_20260923`。

## 依赖及本次变更

现有Python 3.12.14、PyTorch 2.5.1+cu121、RDKit 2026.03.6、NumPy、SciPy、PoseBusters和ProLIF继续使用。初始环境没有pip模块，最初使用uv安装；下文独立评估阶段已补装pip。未重建虚拟环境或升级Torch/CUDA/RDKit。

实际新增四个验证依赖：jsonschema 4.26.0、jsonschema-specifications 2025.9.1、referencing 0.37.0、rpds-py 2026.6.3。具体锁定见`requirements-extra.lock`；安装前后包清单保存在`validation/environment_before.json`和`validation/environment_after.json`。另外将本项目安装为editable包。

```bash
cd /data1/dhuang/flowr_root/MolSteer
/home/dhuang/.local/bin/uv pip install --python /data1/dhuang/flowr_root/.venv/bin/python -r requirements-extra.lock
/home/dhuang/.local/bin/uv pip install --python /data1/dhuang/flowr_root/.venv/bin/python --no-deps -e .
```

在新环境安装需先按所在机器的CUDA条件准备PyTorch及RDKit/NumPy，随后安装项目依赖。PoseBusters/ProLIF是可选测量后端；缺失时保持覆盖限制。无需为本次推导安装Vina、xTB、MACE、向量数据库或在线模型客户端。

## 复现四份报告与奖励试算

```bash
/data1/dhuang/flowr_root/.venv/bin/python -m molsteer demo \
  --input-root /data1/dhuang/flowr_root/output/crossdocked_100target_stage_test \
  --baseline-root /data1/dhuang/flowr_root/output/molreader_localized_reports_v2_20260922 \
  --output-root /data1/dhuang/flowr_root/output/molsteer_reports_20260923 \
  --knowledge /data1/dhuang/flowr_root/MolSteer/knowledge/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md
```

demo读取并核验此前完整StatePacket，在保留指标观察的基础上重新提取控制属性，生成新证据包身份及中英文诊断；只对指定5i0b / ligand_002 / t=0.50执行奖励推导与副本试算。此前其他阶段资料不会被改写。

单独对增强证据包推导：

```bash
/data1/dhuang/flowr_root/.venv/bin/python -m molsteer think \
  --packet /path/StatePacket.json --report /path/DiagnosticReport.json \
  --knowledge /path/Molecular_Generation_Control_Functions_Representative_Table_2026-09-19.md \
  --output /path/RewardSpec.json
```

原有43个独立指标仍支持`python -m molreader.metrics.<metric>`，整包读取仍支持`python -m molreader pack`。在新生成样例上，先生成基础包，再通过`enrich_packet(packet, contexts)`补充控制属性。当前demo入口专门复现已有四个案例。

## 验证

```bash
cd /data1/dhuang/flowr_root/MolSteer
/data1/dhuang/flowr_root/.venv/bin/python -m unittest discover -s tests/reader_regression -v
/data1/dhuang/flowr_root/.venv/bin/python -m unittest discover -s tests -p test_thinker.py -v
/data1/dhuang/flowr_root/.venv/bin/python scripts/verify_delivery.py \
  --reports /data1/dhuang/flowr_root/output/molsteer_reports_20260923 \
  --baseline /data1/dhuang/flowr_root/output/molreader_localized_reports_v2_20260922
```

测试覆盖原指标行为、表格转义符、完整21项检索、方法/奖励区分、区间内零罚、梯度方向、有限差分、退化几何、实际执行拒绝及真实结果完整性。该历史离线试验与新增的实时续生成实验分开保存。

坐标微分使用PyTorch自动求导，并以双精度中心差分验证；参见[PyTorch自动求导文档](https://docs.pytorch.org/docs/stable/autograd.html)。局部MMFF参照延续原有RDKit参数查询，参数可获得不等于已验证化学适用性；参见[RDKit力场辅助接口](https://rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html)。


## Live MolExecutor / 实时引导

No additional packages were installed for the live executor. Existing PyTorch, RDKit and FLOWR dependencies were retained. Highest float32 matmul precision is required by the delivered numerical preflight; TF32 is disabled. The Python/CUDA environment was not upgraded.

本次实时 MolExecutor 未新增安装包，继续使用现有环境。默认最高 float32 矩阵精度并关闭 TF32；未升级 Python、Torch、CUDA 或 RDKit。模型源文件与原始快照保持不变。

See [execution architecture](EXECUTOR.md) / [执行说明](EXECUTOR.zh-CN.md). `molsteer think` defaults to creativity; pass `--mode selection` for reference selection. The `demo` command intentionally retains the historical offline example.

```bash
cd /data1/dhuang/flowr_root/MolSteer
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m unittest discover -s tests/reader_regression -v
../.venv/bin/python -m molsteer.molexecutor.runner --config experiments/guidance/execution.json --emit /path/new-job
../.venv/bin/python /path/new-job/run_guidance.py
```

Use a fresh output directory in the JSON configuration. The original historical state requires explicit reconstructed-context restoration; newly saved runtime checkpoints preserve self-conditioning, RNG and guidance budget.

## 原生运行现场捕获

本次完整阶段检查点与起始时间实验未新增依赖，使用现有 PyTorch、RDKit、PoseBusters 和 matplotlib。恢复时使用 runtime 记录的同一实际 CUDA 设备与最高 float32 精度。参见[完整检查点说明](EXACT_RESTART.zh-CN.md)；初始 head 与无引导 final 的精确复现通过后，再解释引导差异。
## 动态 MolMonitor

MolMonitor 使用现有 PyTorch、NumPy 和 RDKit，未安装或升级额外包。独立评估继续使用已安装的 PoseBusters 和 matplotlib。参考轨迹记录、控制器配置与函数修订后的续跑方法见 [MolMonitor 说明](MOLMONITOR.zh-CN.md)。候选力度探测共用同一次原生推进和自条件上下文，不额外抽取采样噪声。

## 独立终态评估的环境增量

新增 Vina 1.2.7、Meeko 0.8.0，通过 ensurepip 补装 pip 25.0.1。保留 NumPy 2.5.3、RDKit 2026.3.6、ProLIF 2.2.1、Gemmi 0.7.5、MDAnalysis 2.10.0 及现有 PyTorch/CUDA。requirements-evaluation.lock 只锁定本次新增的两个包，不是全环境安装清单。

```bash
cd /data1/dhuang/flowr_root
.venv/bin/python -m ensurepip
.venv/bin/python -m pip install --no-deps -r MolSteer/requirements-evaluation.lock
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/prepare_independent_eval.py --model-root /data1/dhuang/flowr_root --output /path/new-evaluation/preparation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/evaluate_guidance_independently.py --model-root /data1/dhuang/flowr_root --output /path/new-evaluation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/analyze_independent_evaluation.py --root /path/new-evaluation
PYTHONPATH=MolSteer/src .venv/bin/python MolSteer/scripts/audit_independent_evaluation.py --root /path/new-evaluation
```

实验脚本复用已完成的精确恢复与 MolMonitor 运行。分析和核验脚本当前要求新评估目录与 molsteer_affinity_validation_20260923 位于同一父目录，复现时请使用该位置下的新目录。molreader 下的收集器可独立复用。分项梯度脚本仅重新读取已有 GPU 检查点，不生成新分子。

结果位于 output/molsteer_independent_evaluation_20260923。完整受体刚体配准至原口袋坐标系，移除晶体配体 67U、保留 SEP；Meeko 准备没有移动受体重原子。小分子固定重原子，仅先松弛 H。Vina/Vinardo 原姿势评分只产生 PDBQT 的 0.001 Å 舍入，局部优化保存为独立副本。CPU=2，Vina 固定 seed=20260923，24 Å 立方盒中心 [13.7376,34.2223,14.51965]，局部优化 max_steps=200。本轮未执行全局 docking、随机种子扫描、pH ensemble 或在线结构上传。fasudil 的 Meeko 宏环表示含胶合虚拟原子，坐标核验会排除这些虚拟点。

十个案例的固定重原子 H 松弛及完整 MMFF 松弛均收敛。40 项测试通过。audit.json 核验源结构哈希、配体准备坐标舍入、受体重原子保留、分项能量守恒、原始诊断观察保留。进一步说明见 [终态证据与推导扩展](OUTCOME_EVIDENCE.zh-CN.md)。

## 基于无引导对照的奖励执行

本次连续 MMFF 梯度、方向/SASA 代理、互变异构体搜索与目标反馈没有安装新包，复用上述 RDKit、PyTorch、Vina、Meeko。实际环境为 Python 3.12.14、PyTorch 2.5.1+cu121、RDKit 2026.3.6、Vina 1.2.7、Meeko 0.8.0。力场/独立评分使用 CPU，模型导数使用 GPU；离散试探增加额外计算预算。详见[执行与测试](OUTCOME_GUIDANCE.zh-CN.md)。

50 项测试与实时梯度有限差分检查通过；无引导/旧奖励精确重现和 t=0.75 重启检查单独保存。这些验证实现忠实性，不保证普遍的分子优化收益。本例动态修订不如固定新奖励，复现时应保留该负面结果。

## Researcher 与类别接口核验

可选 Researcher 证据契约和概率梯度基元不需要安装新包，复用上述环境。扩展后 53 项测试通过。使用全新目录运行 `scripts/research_discrete_feasibility.py`，参数 `--prior output/molsteer_outcome_guidance_20260923/release --output output/your_fresh_researcher_experiment`；启动方式同上，设置 `PYTHONPATH=MolSteer/src` 并使用 `.venv/bin/python`。脚本执行真实单步类别核验与两个完整坐标消融，不自动部署文献诱导的完整离散策略。

统一评估复用相同受体准备，通过 `evaluate_guidance_independently.py --cases-json ...` 明确输入。原姿势与局部优化副本分别保留。文献查询使用 agent 已有 web 工具，没有配置新的远程服务。

## 完整可选 Researcher 与 HTML 报告

完整流程复用现有环境，没有新增安装。自动文献检索使用 Python 标准库 HTTPS 客户端访问公共 Europe PMC REST 接口。HTML 报告使用已安装的 `markdown-it-py` 和 `matplotlib`；新环境若缺少这两个可选报告依赖，再单独安装即可。核心引导不依赖报告渲染。RDKit 生成带编号的分子 SVG；报告内嵌数据和图表，不依赖浏览器 CDN。

详见[流程与调用](RESEARCHER_PIPELINE.zh-CN.md)。准备、续推、评估和报告入口分别为 `prepare_researcher_experiment.py`、`run_researcher_experiment.py`、`evaluate_researcher_experiment.py`、`render_researcher_report.py`，使用全新实验目录。失败后可显式核对奖励身份并跳过已完成分支继续执行，同时保留原失败日志。完整运行核验包括关闭/shadow 不改变采样、t=0.75 研究控制状态恢复、其他 batch 不变。

## 增广拉格朗日 evaluator

增广拉格朗日 evaluator 与对偶状态 checkpoint 协议没有新增依赖，继续使用现有 PyTorch、RDKit、Vina 和 Meeko 环境。详见[执行与恢复约定](AUGMENTED_LAGRANGIAN.zh-CN.md)。
