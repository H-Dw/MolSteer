# Linux 上与 FLOWR.ROOT 配合：5i0b_A__5vef_M77 实测

本页记录在 `ml-apus.bio.sustech.edu.cn`、`/data1/dhuang/flowr_root` 上验证过的接入方式。MolSteer 仓库位于同级 `MolSteer-github/`；原有 `MolSteer/` 源码目录和 FLOWR.ROOT 仓库保持独立。

## 环境与代码

远端 FLOWR.ROOT 使用 Python 3.12、PyTorch 2.5.1+cu121 和 RDKit 2026.03.6。以 `git -C MolSteer-github rev-parse HEAD` 核对具体代码版本。私有 GitHub 仓库尚未在远端配置认证，因此首次同步使用从已认证本机传输的 Git bundle；以后的直接 `git pull` 仍需配置远端 GitHub 凭据，或继续传输新 bundle 并快进本地分支。不要在文档或配置中放访问令牌。

为避免改动 FLOWR.ROOT 的 `.venv`，使用 `.venv-molsteer` 叠加环境。其 `flowr_base.pth` 指向 FLOWR.ROOT 原虚拟环境的 `site-packages`，使 MolSteer 使用原有的模型、CUDA 和 RDKit 依赖；MolSteer 的 LangChain/OpenRouter 依赖安装在叠加环境中。这个方案已经通过测试，但升级任一环境后应重新检查导入来源和依赖版本。

```bash
cd /data1/dhuang/flowr_root
.venv-molsteer/bin/python -m pytest MolSteer-github/tests -q
```

实测结果：144 项通过。当前远端的 `OPENROUTER_API_KEY` 由交互式 shell 提供；批处理、调度器和 SSH 非交互命令必须在自身进程环境中注入该变量。不要假设登录用户配置会自动传给作业。

## 样例输入与恢复边界

指定的原始样例为：

```text
output/crossdocked_100target_stage_test/5i0b_A__5vef_M77/ligand_002/t_0.50/
```

其 `state.pt`、模型预测和 `ligand.sdf` 可供 MolReader 读取，但没有完整的 `runtime.pt`。它不能直接做严格的历史续推。用该目录生成了 `output/molsteer_openrouter_5i0b_20260928/original_StatePacket.json`，覆盖 129 条观测。

同一靶标另有带完整运行状态的精确捕获：

```text
output/crossdocked_100target_stage_test_exact_20260923/5i0b_A__5vef_M77/ligand_002/t_0.50/runtime.pt
```

精确捕获与原始目录的 `state.pt`、`world_prediction.pt` 哈希不同。续推时，`saved_stage`、`reward_reference_stage`、`runtime.pt`、受体/参考配体、模型 checkpoint 和记录的 `stage_runner.py` 必须来自同一捕获；不能把原始样例的证据或基线混进精确捕获。该检查点要求 GPU 1、`float32_matmul_precision=highest`、原记录的 stage runner 哈希和第 50 步状态。FLOWR 适配器会检查源代码、模型路径、受体/配体身份、积分网格、数值精度、GPU 索引和恢复状态。

已生成可审查的配置及奖励程序：

```text
output/molsteer_openrouter_5i0b_20260928/exact_preparation/configurations/t_0.50.json
output/molsteer_openrouter_5i0b_20260928/exact_preparation/reasoning/t_0.50/RewardProgram.json
```

可用相同配置在新的输出目录分别跑引导与无引导对照：

```bash
cd /data1/dhuang/flowr_root
CFG=output/molsteer_openrouter_5i0b_20260928/exact_preparation/configurations/t_0.50.json
.venv-molsteer/bin/python -m molsteer.molexecutor.runner \
  --config "$CFG" \
  --output output/molsteer_openrouter_5i0b_20260928/flowr_creativity \
  --arm creativity
.venv-molsteer/bin/python -m molsteer.molexecutor.runner \
  --config "$CFG" \
  --output output/molsteer_openrouter_5i0b_20260928/flowr_unguided \
  --arm unguided
```

输出目录必须为空。引导运行通过梯度有限差分预检，恢复至第 50 步，续推 50 步，其中 40 步接受引导提议；只编辑批次索引 2。最终产物位于各自的 `.../<arm>/5i0b_A__5vef_M77/ligand_002/final/`。同 RNG 无引导对照也完成。两份最终 SDF 可解析、连接图相同，MolReader 未发现蛋白碰撞；引导与无引导的原生 pKd head 分别约为 5.42825 和 5.42795，坐标帧 RMSD 约 0.00156 Å。单例和如此小的差异不足以证明生成质量改善；应扩大靶标和随机种子，并用独立指标评价。

## GLM-5.3 Agent 与生成器的接口

`configs/agents.json` 默认让四个 Agent 通过 OpenRouter 调用 `z-ai/glm-5.3`，启用 reasoning。两轮真实 API 调用已验证模型和 `reasoning_details` 的续传。Agent 的 `RewardSpec` 经声明式验证和离线数值测试后，正式生成仍要求宿主提供 `AgentRuntime(inference_adapter=..., approve_inference=True)`。当前仓库的 FLOWR `runner.py` 接收的是另一种 `RewardProgram` JSON；它并不自动消费 Agent 的 `RewardSpec`。上面的 GPU 续推使用了 `prepare_exact_experiment.py` 生成的确定性奖励程序，不能表述为 GLM 直接控制的续推。

修正 OpenRouter 超时单位后，对精确捕获的 StatePacket/DiagnosticReport 运行了完整的真实 API Agent 工作流。`glm53_5i0b_fixed_20260928` 的状态为 `validated`，四个 Agent 合计留下 21 条审计事件；RewardSpec 有 4 项、2 个视图组，2 组离线数值测试通过。其 `execution_result.mode` 为 `validation_only`，没有调用 FLOWR 生成器。审计产物保存在 `MolSteer-github/outputs/agent_runs/glm53_5i0b_fixed_20260928.{trace,checkpoint}.json`。配置里的 `timeout` 单位是秒；`ChatOpenRouter` 接收毫秒，模型工厂会转换单位。

接通两层需要一个受审查的适配器，至少完成以下工作：

1. 从同一 `StatePacket` 和同一精确捕获编译受支持的 `RewardSpec` 条目到 FLOWR `RewardProgram`；拒绝无法表达的视图、奖励项或图操作，保留证据 ID、单位和坐标映射。
2. 验证图签名、坐标哈希、模型和 stage runner 哈希、受体/配体、GPU、精度、积分步与 RNG；不得跨捕获组合参考预测。
3. 将 `request.strength`、编辑掩码、每步/累计位移预算映射到实时梯度控制，保存完整状态，并返回真正测量的 segment 指标供 MolMonitor 判断。
4. 对 GLM 产出的奖励先运行静态验证、数值梯度预检和小段续推，再与同 RNG 原生对照及独立质量指标比较。禁止把离线坐标试验标记为生成器收益。

在这个桥接适配器通过验证前，API Agent 与 FLOWR 续推应分别运行并分别标记来源。
