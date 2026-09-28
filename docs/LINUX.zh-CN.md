# Linux 安装与运行

在 Linux 上使用 Python 3.10 或更新版本，从仓库根目录创建独立环境。若要运行 CUDA 生成，请先按该机器的驱动和 CUDA 环境安装适配的 PyTorch，再安装 MolSteer。

```bash
git clone https://github.com/H-Dw/MolSteer.git
cd MolSteer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest tests -q
python -m molsteer --help
```

请使用可编辑安装：Agent 的配置、知识库与 Skill 文件仍位于源码目录。`outputs/`、`.venv/` 和本地密钥文件受 Git 忽略。API Agent 需要在本机设置服务凭据与模型 ID；测试不调用模型 API。

## FLOWR.ROOT 生成

实时生成还需要单独安装 FLOWR.ROOT、模型权重及对应数据。下面的路径均由使用者按 Linux 主机实际位置设置；`--input-root` 必须包含该脚本使用的两组受体和配体输入。请在已安装 FLOWR.ROOT 的 Python 环境中运行，并选用未存在的输出目录。

```bash
export FLOWR_ROOT=/path/to/flowr_root
export MOLSTEER_ROOT=/path/to/MolSteer
python "$MOLSTEER_ROOT/integrations/flowr_root/stage_runner.py" \
  --input-root "$FLOWR_ROOT/output/crossdocked_100target_stage_test/inputs" \
  --checkpoint "$FLOWR_ROOT/checkpoints/flowr_root_v2.2.ckpt" \
  --output-root "$FLOWR_ROOT/output/new_exact_stages" \
  --gpu 0 --precision highest
```

随后可按[完整阶段检查点流程](EXACT_RESTART.md)准备续推。`experiments/guidance/*.json` 是原实验记录，包含原主机的绝对路径；在新主机运行时需复制配置，逐项替换模型、输入、检查点、奖励程序和输出路径。历史运行检查点带有模型与生成入口哈希，不能将其当作新主机生成的检查点直接续推。

独立评估需要额外安装依赖：`python -m pip install -e '.[evaluation]'`。`prepare_independent_eval.py` 会在当前 Python 环境或 `PATH` 中查找 `mk_prepare_receptor.py`，也可通过 `--receptor-preparer` 指定。研究审计脚本 `finalize_researcher_audit.py` 通过 `--tests-log` 接收测试日志路径。
