# Linux 上的 MolSteer × FLOWR.ROOT 实测

本页记录 `ml-apus.bio.sustech.edu.cn` 上的 `5i0b_A__5vef_M77/ligand_002` 实验。顶层仓库是 `/data1/dhuang/MolSteer`；生成器及其模型、虚拟环境和输出位于 `/data1/dhuang/MolSteer/flowr_root`。根目录 `.gitignore` 的 `/flowr_root/` 规则排除整个生成器目录。旧路径 `/data1/dhuang/flowr_root` 保留为指向新位置的符号链接，因为早期 `runtime.pt` 和 StatePacket 记录了这个绝对路径。

## 环境激活

FLOWR.ROOT 原环境是 `flowr_root/.venv`，含 Python 3.12、PyTorch 2.5.1+cu121 与 RDKit 2026.03.6。MolSteer 使用 `flowr_root/.venv-molsteer` 叠加环境：它复用原环境的科学计算依赖，并从顶层 Git 仓库以 editable 方式导入 MolSteer。搬迁后，两个环境的激活脚本、命令入口和 `.pth` 路径已更新。

```bash
cd /data1/dhuang/MolSteer
source integrations/flowr_root/activate.sh
python -m pytest tests -q
```

脚本默认激活 `.venv-molsteer`，同时设置 `FLOWR_ROOT` 与 `PYTHONPATH`；仅运行原生成器时可使用 `MOLSTEER_FLOWR_ENV=base source integrations/flowr_root/activate.sh`。实测 146 项测试通过。`OPENROUTER_API_KEY` 当前由远端交互式 shell 提供；SSH 非交互会话、调度器和服务进程需在各自环境中注入，不能依赖交互式配置自动传递。

GitHub 私有仓库在远端仍未配置认证。已认证本机将最新提交通过 Git bundle 传入，并快进顶层仓库；未来直接 `git pull` 需要配置仓库凭据，或继续使用 bundle。不要把令牌写入仓库或作业配置。

## 阶段捕获与迁移后的兼容性

`integrations/flowr_root/stage_runner.py` 会自动查找嵌套的 FLOWR 包，也可读取显式 `FLOWR_ROOT`；调用 FLOWR 参数解析器后恢复调用者的 `sys.argv`。`--target` 可只生成指定靶标。以下命令已在新目录完成 100 步生成，并写出 `t_0.25`、`t_0.50`、`t_0.75` 与最终阶段的完整运行状态：

```bash
cd /data1/dhuang/MolSteer
source integrations/flowr_root/activate.sh
python integrations/flowr_root/stage_runner.py \
  --output-root flowr_root/output/molsteer_stage_runner_moved_20260928 \
  --input-root flowr_root/output/crossdocked_100target_stage_test_exact_20260923/inputs \
  --checkpoint flowr_root/checkpoints/flowr_root_v2.2.ckpt \
  --gpu 1 --precision highest --target 5i0b_A__5vef_M77
```

新捕获的 `ligand_002/t_0.50/runtime.pt` 已用 MolExecutor 在 GPU 1 恢复到第 50 步。改变 `stage_runner.py` 会改变其源哈希；旧精确捕获必须继续使用自身输出目录保存的旧版 `stage_runner.py`，不能换成新的集成脚本。适配器在目录搬迁后以实际文件身份比较模型、受体、配体和阶段路径，同时核验源哈希、模型内容哈希、积分步、GPU 与数值精度。

用户指定的 `crossdocked_100target_stage_test/5i0b_A__5vef_M77/ligand_002/t_0.50` 原始阶段没有 `runtime.pt`，适合读取证据，不能冒充精确续推。下面的 Agent 结果来自 `crossdocked_100target_stage_test_exact_20260923` 的 StatePacket；其来源哈希与精确 `runtime.pt` 匹配。新捕获与旧精确捕获也不可交叉组合。

## GLM-5.3 Agent 奖励的真实续推

OpenRouter 的四 Agent 工作流对精确捕获生成了 `glm53_5i0b_fixed_20260928.checkpoint.json`：状态 `validated`，4 个奖励项覆盖 `state` 和 `prediction` 两个视图，2 组离线数值测试通过。审计文件已复制到顶层 `outputs/agent_runs/`。`integrations/flowr_root/agent_continuation.py` 检查 Agent 审计摘要、奖励验证、阶段文件哈希、完整运行状态、受体/模型来源、GPU 与预算，然后将两种视图分别映射到实时 `X_t` 和可微的 `X̂₁`。只允许经过验证的区间距离、角度和最小距离项；编辑掩码限于奖励项引用的原子。

可在新的输出目录重跑同一实验：

```bash
cd /data1/dhuang/MolSteer
source integrations/flowr_root/activate.sh
python integrations/flowr_root/agent_continuation.py \
  --agent-checkpoint outputs/agent_runs/glm53_5i0b_fixed_20260928.checkpoint.json \
  --flowr-config flowr_root/output/molsteer_openrouter_5i0b_20260928/exact_preparation/configurations/t_0.50.json \
  --flowr-root flowr_root \
  --output-root flowr_root/output/NEW_AGENT_CONTINUATION
```

输出目录必须尚不存在。该入口会先生成 `RewardProgram.agent.json`、执行配置及来源清单，再用同一完整检查点和 RNG 运行 `unguided`、`agent` 两条分支；梯度有限差分预检是硬门禁。历史实测产物在 `flowr_root/output/molsteer_agent_glm53_5i0b_20260928/`。此次从第 50 步运行到第 100 步，Agent 分支 44 步具有可用梯度，37 步接受引导，最大累计注入路径约 0.00257 Å；无引导分支接受数为 0。

两条分支的最终 SDF 均可解析，分子连接图一致，MolReader 在该表示下未报告蛋白或分子内碰撞。最终坐标帧 RMSD 约 0.000112 Å；原生 pKd head 分别为 5.427947（Agent）和 5.427954（无引导），SDF MMFF 应变代理分别约为 19.567 和 19.546 kcal/mol。本单例证明 Agent 奖励已经接入实时续推，但不构成质量提升证据。后续应在多个靶标和随机种子上配对运行，并用独立评价指标判断收益。
