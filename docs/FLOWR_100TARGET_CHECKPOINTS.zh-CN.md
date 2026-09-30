# 100-target 数据预处理与精确续推

入口为 `scripts/preprocess_flowr_100targets.py`，默认配置为 `configs/flowr_100target.json`。
使用 MolSteer 现有的 `integrations/flowr_root/stage_runner.py` 构造 FLOWR.ROOT v2.2 模型和预处理参数，直接调用其模型、口袋编码器和原生 integrator。旧的两靶点 stage runner 和历史数据不变。

## 输入与运行

从 Pocket2Mol / TargetDiff 共用的 `data/split_by_name.pt` 读取 `test`，要求恰好 100 对，保持原始次序，逐项检查受体/参考配体及 SHA-256。不会重新划分、随机选取或使用训练集。配对文件相对于 `data/crossdocked_pocket10` 解析。默认保留所有受体链，避免把非 A 链靶点错误裁空；口袋采用现有 MolSteer 的 holo、7 Å 截取设置。

在远程服务器运行：

```bash
cd /root/private_data/MolSteer
PY=/root/private_data/.miniforge3/envs/molsteer-flowr-dtk/bin/python

# 不加载模型，核验完整测试集并保存展开后的输入清单。
"$PY" scripts/preprocess_flowr_100targets.py generate --dry-run \
  --report outputs/flowr_100target_preflight.json

# 有 GPU 的实例：先验证第 0 个靶点的三个分子。
"$PY" scripts/preprocess_flowr_100targets.py generate --target-index 0 \
  --output outputs/flowr_100target_smoke

# 全部 100 个靶点，每个三个分子；输出目录必须尚不存在。
"$PY" scripts/preprocess_flowr_100targets.py generate \
  --output outputs/flowr_100target_checkpoints
```

配置中的相对路径均相对于 MolSteer 仓库；可用 `--split`、`--data-root`、`--model-root`、`--checkpoint`、`--stage-runner`、`--output` 覆盖路径。默认 GPU 为逻辑设备 0，可用 `--gpu` 改变。DTK/ROCm 的 GPU 同样通过 PyTorch `cuda` 接口访问。没有 GPU 时默认报错；`--device cpu` 是显式 CPU 选项，CPU 性能和内存需求需单独评估。

默认 100 步、linear 网格、Euler、uniform-sample 类别采样、无引导、无末尾额外 corrector iteration，保留正常的最终 prediction head。`--integration-steps` 必须为正的 10 的倍数，避免把不在网格上的时间点冒充 checkpoint。默认种子为 20260930，各靶点以配对路径计算稳定独立种子，因此单靶点重试不受任务顺序影响。完整 batch 布局和实际 RNG 都记录在文件中。

## 输出

```text
outputs/flowr_100target_checkpoints/
  run.json
  000_2z3h_A_rec_1wn6_bst_lig_tt_docked_3/
    batch_000.pt       # 一份完整批次轨迹 + 去重张量池
    molecule_000.pt    # 小型索引，指向 batch_000.pt 的 slot 0
    molecule_001.pt    # slot 1
    molecule_002.pt    # slot 2
  ... 共 100 个靶点目录
```

完整成功运行产生 **300 个分子索引、2400 个逻辑分子 checkpoint**；若每个靶点为一个 batch，则共享数据只保存为 100 个批次文件。批次拆分时每个真实 batch 各保存一份。通过 `load_bundle` 打开任意分子索引，可以得到以下完整逻辑数据：

| 字段 | 内容 |
| --- | --- |
| `checkpoints` | 键为 `0.30`、`0.40`、`0.50`、`0.60`、`0.70`、`0.80`、`0.90`、`1.00` |
| 每个 checkpoint | 完整 batch 的 `curr`、`cond`、`times`、`step_index`，Python / NumPy / PyTorch CPU / 全部可见 GPU RNG，UTC 时间、累计积分耗时、推理阶段 |
| `shared` | 原始 prior、口袋张量、坐标原点、缓存口袋编码、精确时间网格、batch 大小、模型/积分器参数 |
| `molecule_index` / `batch_index` | 靶点内分子编号 / 其在原始采样 batch 中的位置 |
| `target` / `inputs` | 测试集条目、路径、哈希、原始 PDB/SDF 字节 |
| `generator_args` / `seed` | 完整生成参数和该靶点的种子 |
| `provenance` | 权重和 FLOWR/预处理源码哈希、Python/PyTorch/依赖版本、设备、数值计算设置及允许记录的环境变量 |
| `final` | 最终结构和亲和力张量、世界坐标预测、最终 RNG、head 耗时 |
| `molecule` | 解码得到的 SDF 文本、化学有效性/解码失败信息 |
| `verification` / `status` | 一致性验证结果、运行状态和错误信息 |

**v2 分子文件只保存批次文件名、批次身份、molecule_index 和 batch_index。** 完整 batch 的 prior、口袋、8 个 checkpoint、RNG 和最终输出只存一份。各分子的 SDF、sanitized 状态等专属字段分别保留在批次文件的 members 表中，不能因为大部分张量相同而丢弃。

批次内进一步按 dtype、shape、stride 和原始字节做 SHA-256 去重：跨时间点相同的 mask、RNG、静态张量，以及归一化预测和世界坐标预测中相同的非坐标张量，均引用同一份张量数据。保留精度、符号零和 NaN 的原始字节，不采用浮点降精度或有损压缩。恢复仍使用完整原始 batch，从而保留类别随机采样顺序。

共享文件以原子替换提交，所有分子索引保持稳定；数据校验和与成员信息在同一批次文件内提交，避免更新到一半导致旧索引指向失效哈希。读取时核验批次身份、成员映射、元数据和全部张量校验和。Linux 默认以内存映射加载历史数据，恢复时只克隆需要的 live state。返回的历史张量可能共享存储，应视为只读。移动数据时必须携带整个靶点目录。

checkpoint 位于对应积分步完成后、下一次 forward 前；不会为了捕获中间结构额外调用模型。`1.00` 保存 t=1 的原生积分状态，恢复后执行标准 `-1e-4` 时间偏移的最终 prediction head。最终分子使用这个 head 的输出，而不是把 t=1 的当前状态直接当成解码结果。checkpoint 中时间张量保留实际浮点值，不按十进制标签重建。

三个分子指三条原始采样轨迹。解码失败或化学无效仍保留张量与错误记录，不重采样替换；不承诺三个有效且互异的分子。`run.json` 分别汇报 generated、decoded、sanitized 数量。

## 一致性验证与恢复

默认每个 batch 完成后执行两类验证：

1. 同一 prior 和初始 RNG 调用上游 `model._generate`，对比全部最终预测张量和 RNG，检查捕获代码保持原生行为。
2. 重新读取已落盘的分子文件，从八个 checkpoint 分别恢复，比对剩余各保存点的全部状态、自条件、时间和 RNG，以及最终结构、亲和力、世界坐标和 RNG。张量比较使用 `torch.equal`，不使用误差容忍。

校验增加约 3.8 倍积分步开销，加上原始生成总计约 4.8 倍；实际时间受推理和文件保存影响。`--no-verify` 可跳过，但结果会明确标为 `completed_unverified`。各 checkpoint 使用临时文件、刷新和原子替换保存；异常会记录为 failed/interrupted，已保存的最近状态仍可恢复。批次遇到异常不会悄悄记为完成；其他靶点可继续，整个任务以非零退出码报告失败。

使用同一推理环境恢复：

```bash
"$PY" scripts/preprocess_flowr_100targets.py resume \
  --file outputs/flowr_100target_checkpoints/000_2z3h_A_rec_1wn6_bst_lig_tt_docked_3/molecule_001.pt \
  --t 0.30 \
  --output outputs/resumed_molecule_001_t030.pt
```

恢复过程加载相同权重后，直接恢复保存的口袋、batch、自条件和 RNG，无需原始数据目录，无需重新生成 t=0 到保存点的前缀。原始文件保留；恢复结果写入新文件。原始文件有最终输出时，恢复自动对比；中断文件缺少最终输出时会标记 `resumed_without_original_final`，不冒充已与原始最终结果核验。t=1.00 也支持恢复及最终 head 比对。

`--model-root`、`--checkpoint`、`--stage-runner` 允许重新定位文件路径，内容哈希必须一致。源码、权重、模型参数、数值设置、依赖版本或设备布局不同会拒绝严格恢复。GPU 和 CPU 的结果不能互相声称逐位相同。默认开启严格确定性算法，若当前后端没有所需的确定性实现，会明确失败。必须在实际部署环境完成真实模型验证，才能对该环境的轨迹作精确续推结论；本地随机测试模型通过不等于实际 GPU 模型通过。

读取示例：

```python
from molsteer.preprocessing.checkpoints import load_bundle
bundle = load_bundle("molecule_001.pt")  # 自动解析 v2 索引；同时兼容 v1 完整文件
checkpoint = bundle["checkpoints"]["0.30"]
j = bundle["batch_index"]
coords = checkpoint["curr"]["coords"][j]  # 当前归一化坐标；分析时可选取该分子
mask = checkpoint["curr"]["mask"][j].bool()
active_coords = coords[mask]
# 续推时必须保留完整 bundle，不能用上面的单分子切片代替它。
```

当前格式由本入口的 `resume` 读取；不是旧 MolExecutor 的单时间点 `runtime.pt` 格式，不能直接作为旧执行器的 `--resume` 参数。数据包不含模型权重；续推仍需要同一模型权重和匹配的软件/硬件环境。

## 指定时刻的信息读取脚本

`scripts/read_flowr_checkpoint.py` 只需 CPU、PyTorch 和 NumPy，不加载 FLOWR 模型、模型权重，也不恢复或消耗保存的 RNG。v2 分子入口会自动找到同目录的共享批次，并核验元数据、所有张量哈希及分子索引；旧 v1 完整文件也可读取。

远程实例使用 `/opt/conda/envs/molsteer-flowr/bin/python`。在 `/root/private_data/MolSteer` 中执行：

```bash
PY=/opt/conda/envs/molsteer-flowr/bin/python

# target 68（第 69 个目标）、分子 1（第二个分子），读取精确的 t=0.50。
"$PY" scripts/read_flowr_checkpoint.py \
  --root outputs/flowr_100target_checkpoints \
  --target-index 68 --molecule-index 1 --t 0.5 \
  --json outputs/target068_molecule001_t050_info.json \
  --export-selected outputs/target068_molecule001_t050_analysis.pt

# 也可直接指定分子入口；不提供 --json 时，信息打印到标准输出。
"$PY" scripts/read_flowr_checkpoint.py \
  --file outputs/flowr_100target_checkpoints/068_5d7n_D_rec_4jt9_1ns_lig_tt_min_0/molecule_001.pt \
  --t 1.0
```

所有序号从 **0** 开始；仅接受 `0.3, 0.4, …, 1.0` 中实际存在的保存时刻，不将 `0.35` 取整为附近的 checkpoint。输出路径必须是新文件，脚本拒绝覆盖已有数据。

JSON 包括 target、molecule_index、batch_index、原始 batch 大小、种子、保存状态、积分步数、捕获时间、推理耗时、RNG 摘要哈希、CPU/GPU RNG 张量信息、完整推理环境/依赖版本、源码及模型哈希、校验记录，以及所选分子的张量形状、类型和哈希。RNG 完整数值仍在 `read_at()` 返回的 `state["rng"]` 中；JSON 摘要不会打印数千个随机数状态整数。

`--export-selected` 输出普通 `torch.load(..., weights_only=True)` 可读的字典，包括当前分子的 `curr`、`cond`、`times`、世界坐标 `coords_world`、有效原子索引和有效原子坐标。保留原始 dtype 和 padding，张量已复制，后续修改不会影响共享存储。这个文件标记 `resume_supported=False`，仅供分析，不能代替原批次续推。键 `curr`、`cond` 中的坐标是模型归一化坐标；`coords_world` 按保存的 `coord_scale` 与 pocket COM 换算，并将 padding 位置置零。

也可在 Python 中导入脚本函数：

```python
import sys
sys.path.insert(0, "/root/private_data/MolSteer/scripts")
from read_flowr_checkpoint import read_at, selected_state

bundle, state = read_at(
    "/root/private_data/MolSteer/outputs/flowr_100target_checkpoints/"
    "068_5d7n_D_rec_4jt9_1ns_lig_tt_min_0/molecule_001.pt", 0.5
)
analysis = selected_state(bundle, state)
coords = analysis["active_coords_world"]
rng = state["rng"]            # 原始完整批次 RNG；不要按单分子切片。
environment = bundle["provenance"]["environment"]
# bundle/state 的共享映射张量按只读使用；分析修改应使用 selected_state 返回的副本。
```

`t=1.00` 的 checkpoint 保存的是最终预测 head 运行**之前**的状态；最终分子预测位于 `bundle["final"]["world_prediction"]`。`capture.inference_seconds_this_segment` 是从本次 rollout 起点累计到该保存时刻的积分耗时，不是相邻 0.1 区间的单独耗时，也不包含文件写入和校验耗时。读取报告中的 `recorded_verification` 是原推理保存的验证记录；读取脚本本身仅做存储完整性检查，不运行新的推理。

## 仅补齐未完成目标

在 Linux 推理实例执行：

```bash
"$PY" scripts/complete_flowr_100targets.py \
  --root outputs/flowr_100target_checkpoints --audit-only

"$PY" scripts/complete_flowr_100targets.py \
  --root outputs/flowr_100target_checkpoints --expected-pending 32
```

脚本读取原 `run.json` 中的配置和 100-target 清单，核验所有标为完成的目标，记录其全部 `.pt` 文件哈希；仅为失败/缺失目标重新生成三个原始样本，沿用每个目标由全局种子和 split pair 得到的确定性种子。加载一次模型，逐目标执行原生生成和八个时刻的严格回放验证。若某个“已完成”目标损坏，立即报错，不将它偷偷纳入重生成。

补跑先写入同级 `flowr_100target_checkpoints_completion_<UTC>/generated/`。只有目标完整且通过验证，才将新目录移入正式输出；原来的残缺目录移动至同次补跑目录的 `previous_incomplete/` 保留。原始 `run.json`、已完成文件哈希清单及本次推理环境保存在同目录的 `original_run.json`、`preserved_targets.json`、`completion.json`。结束时再次核验原有完整文件哈希未改变。正式 `run.json` 合并更新目标状态，但原始 provenance 保留；各分子 bundle 的 provenance 才是其实际推理环境，混合实例运行不能用一份环境记录替代。

`--expected-pending` 用于防止选错输出目录；后续重试时应填实际剩余数量或省略。脚本对活动状态拒绝启动，并使用进程锁防止两个补跑入口同时写入。意外终止后，应先核对进程与同次补跑目录，保留记录后再恢复状态，不能直接忽略活动状态。

## 旧文件审计与迁移

```bash
# 只读审计：逐字段核验共享部分，记录各文件 SHA-256 和可节省的张量字节数。
python -m molsteer.preprocessing.migrate \
  --source outputs/flowr_100target_checkpoints --audit --report outputs/storage_audit.json

# 输出到新目录，保留旧文件；需要足够的额外磁盘空间。
python -m molsteer.preprocessing.migrate \
  --source outputs/flowr_100target_checkpoints \
  --output outputs/flowr_100target_compact --report outputs/storage_migration.json

# 空间紧张时可显式选择原地迁移：先写共享文件并重读验证，再原子替换分子文件为索引。
python -m molsteer.preprocessing.migrate \
  --source outputs/flowr_100target_checkpoints --in-place --report outputs/storage_migration.json
```

迁移要求生成进程已停止。只接受分子齐全、8 个保存点和最终结果均完整的目标；失败或部分写入的目标会拒绝并记录，返回非零退出码。共享字段任何差异都会拒绝合并；molecule_index、batch_index 和专属 molecule 信息单独核验并全部保留。完整目标中的多个 batch 按各自身份分别迁移。

原地迁移不是直接删除两个“副本”：先验证所有原文件，写入并重新读取共享文件，逐字节核验所有张量及所有元数据，再依次原子替换分子入口。每个靶点的 `storage_migration.json` 保留原始文件哈希、大小、映射、验证结果和节省字节数。中断后 v1/v2 混合目录仍可读取，重试同一迁移可完成转换。迁移保留原 provenance，不改写历史推理已验证的事实；`replay_status=not_run` 明确表示本次存储迁移没有执行 GPU 续推。

已有 v1 文件的恢复仅放行已审计的两份旧存储/入口代码哈希，FLOWR 源码、采样循环 `trajectory.py`、stage runner、模型权重和环境仍须匹配。不会通用地忽略源码差异。新旧格式的回放测试从每个时间点比较后续状态、自条件、预测和 RNG。真实 GPU 再验证继续使用上文的 `resume` 命令。

若需要还原独立的旧格式文件，可在有足够空间时执行：

```python
from molsteer.preprocessing.checkpoints import load_bundle, atomic_save
atomic_save(load_bundle("molecule_001.pt"), "molecule_001_standalone.pt")
```

## 初版交付验证（生成前记录，2026-09-30）

本地新增测试及相关执行器回归共 31 项通过；远程新增测试 16 项通过。远程完整 `test` 清单与 200 个输入文件已校验，预检报告在 `outputs/flowr_100target_preflight.json`。这些随机测试模型测试覆盖落盘恢复和上游采样循环，但没有运行真实 FLOWR 权重的全量推理。

当前连接的 SCNet 实例返回 `torch.cuda.is_available() == False`、设备数量 0，因此未启动 300 分子生成，未取得真实模型的 GPU 续推一致性结论。换到有加速卡的实例后，先运行上述单靶点验证，再运行全量命令。验证范围见 `validation/flowr_preprocessing_20260930.json`。

## v2 实际迁移验证（2026-09-30）

后续真实生成运行完成了 68 个目标，其余目标遇到磁盘空间不足。本次已把这 68 个完整目标原地转换为 68 个共享批次文件和 204 个分子索引。两个仍含部分分子文件的不完整目录（068、071）未改动，未重跑其余失败目标。

| 检查点文件指标 | 实测结果 |
| --- | ---: |
| 迁移前 | 1,457,811,936 bytes（1390.28 MiB） |
| 迁移后 | 489,822,208 bytes（467.13 MiB） |
| 减少 | 967,989,728 bytes（923.15 MiB，66.4%） |
| 其中原批次重复张量 | 947,734,976 bytes（903.83 MiB） |
| 批次内部额外张量去重 | 6,079,200 bytes（5.80 MiB） |

文件指标不含迁移审计 JSON 与源码备份。每个分子入口均在迁移后重新加载，核验完整状态、RNG、专属结果和索引与原文件一致。原始源码归档保存在远程 `validation/flowr_checkpoint_v1_sources_20260930.zip`；每个目标和汇总报告保留原 `.pt` 文件 SHA-256。

本地 44 项测试通过，远程随机测试模型的 3 × 8 条续推通过。当前实例没有 GPU，未重新执行真实 FLOWR 模型的 GPU 后缀；原始生成已有的验证记录保留在数据包中。本次严格区分“实际数据无损重建”和“新执行的 GPU 回放”。详细结果见 `validation/flowr_storage_v2_20260930.json`，远程完整报告为 `validation/flowr_storage_migration_20260930.json`。
