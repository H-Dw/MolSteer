# MolSteer 与 FLOWR.ROOT 环境配置

本配置针对 Linux x86_64、Python 3.12 和 CUDA 12.1 的服务器部署。顶层仓库位于 `/data1/dhuang/MolSteer`，生成器源码位于被 Git 忽略的 `flowr_root/`。环境变量 `OPENROUTER_API_KEY` 由运行进程注入，配置文件和镜像均不包含密钥。

## 远端已安装环境与兼容边界

2026-09-29 检查的主机为 Ubuntu 20.04、RTX 3090、驱动 530.41.03。现有 `flowr_root/.venv` 和 `flowr_root/.venv-molsteer` 均为独立的 Python 3.12 虚拟环境，而不是 Conda 环境。两者使用 PyTorch `2.5.1+cu121`、NumPy `2.5.3`、RDKit `2026.3.6`；generator 还使用 TensorDict `0.5.0`、PyG `2.8.0.post1` 和 Lightning `2.6.5`。Conda 当前没有专用的 MolSteer/FLOWR 环境。

generator 的上游 `flowr_root/pyproject.toml` 与 `uv.lock` 针对 CUDA 13 / GB10 声明 `torch==2.11.*` 和 `tensordict>=0.12,<0.13`，与本机已运行的 CUDA 12.1 组合不一致。因此本配置固定**实测部署**版本，并通过 `PYTHONPATH` 引入生成器源码；不要在这台机器上直接对上游锁文件执行 `uv sync`。这也是 `pip check` 对原有 `flowr-root` editable 安装报告两项版本冲突的原因；该报告不能单独证明现有生成流程失效。

`flowr_root/.venv` 原先缺少 LangChain Agent 包。现已安装 `langchain-core 1.6.5`、LangChain `1.4.2`、LangGraph `1.2.12`、OpenRouter 适配包 `0.2.9` 及相关依赖，并将 MolSteer editable 安装路径改为顶层仓库。修复后 `flowr`、`molsteer` 和 `langchain_core` 可从同一解释器导入，CUDA 仍可用。默认 `activate.sh` 继续选用已安装 Agent 依赖的 `.venv-molsteer`；需要显式使用 generator 原环境时设置 `MOLSTEER_FLOWR_ENV=base`。

该主机的联合进程存在原生扩展导入顺序问题：先导入完整 Agent 模块、再导入 `flowr.gen.generate_from_pdb` 会发生段错误；反向顺序正常，两个既有虚拟环境都可复现。`agent_continuation.py` 因此在加载 Agent 执行模块之前预载并核验指定的 FLOWR 源码。自行编写同进程入口时也应先导入生成器；独立 Agent API 作业不受此顺序约束。

## Conda 文件

| 文件 | 用途 |
| --- | --- |
| [`environment.agent.yml`](../environment.agent.yml) | MolSteer Agent、奖励验证与独立评估；无需生成器源码即可创建 |
| [`environment.yml`](../environment.yml) | MolSteer + FLOWR 生成与续推；包含 generator 的直接运行依赖 |

两个文件固定主依赖版本，由 pip 安装其传递依赖；它们是可重建的环境规格，不是逐个 wheel 哈希锁。创建环境前切换到仓库根目录，因为文件最后以 editable 方式安装 `.`。如已在其他环境中运行作业，请勿原地替换；新建环境后分别验证。

```bash
cd /data1/dhuang/MolSteer
/data1/dhuang/miniconda3/bin/conda env create -f environment.agent.yml
/data1/dhuang/miniconda3/bin/conda env create -f environment.yml

# 在 Bash 中启用 Conda 后选择一个环境
eval "$(/data1/dhuang/miniconda3/bin/conda shell.bash hook)"
conda activate molsteer-flowr
export FLOWR_ROOT="$PWD/flowr_root"
export PYTHONPATH="$FLOWR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import molsteer, flowr, langchain_core, torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pytest tests -q
```

若只运行 Agent，请激活 `molsteer-agent`。如需从纯 Agent 环境执行 FLOWR 续推，应改用 `molsteer-flowr`；两种环境都从宿主进程读取 `OPENROUTER_API_KEY`，不从仓库文件读取。

## Docker 镜像

[`Dockerfile`](../Dockerfile) 提供 `agent` 与 `flowr` 两个构建目标。它以 CUDA 12.1 base 镜像和固定版本 Miniforge 为基础，PyTorch cu121 wheel 安装用户态 CUDA 库，并分别创建上述 Conda 环境。`flowr` 目标要求构建目录内有 `flowr_root/flowr/`；该目录由用户单独管理，不进入 MolSteer Git 仓库。`.dockerignore` 只发送必要的生成器源码与 vendored PoseBusters，跳过虚拟环境、输出、数据和模型权重。

```bash
cd /data1/dhuang/MolSteer
docker build --target agent -t molsteer:agent-cu121 .
docker build --target flowr -t molsteer:flowr-cu121 .

# 离线检查；不会调用 OpenRouter
docker run --rm molsteer:agent-cu121 python -c 'import langchain_core, molsteer; print(langchain_core.__version__)'

# 生成器例子：模型与输出在运行时挂载；密钥从宿主环境传入
docker run --rm --gpus all \
  -e OPENROUTER_API_KEY \
  -v "$PWD/flowr_root/checkpoints:/opt/MolSteer/flowr_root/checkpoints:ro" \
  -v "$PWD/flowr_root/output:/opt/MolSteer/flowr_root/output" \
  molsteer:flowr-cu121 \
  python integrations/flowr_root/stage_runner.py \
    --input-root flowr_root/output/crossdocked_100target_stage_test/inputs \
    --checkpoint flowr_root/checkpoints/flowr_root_v2.2.ckpt \
    --output-root flowr_root/output/new_exact_stages \
    --gpu 0 --precision highest --target 5i0b_A__5vef_M77
```

容器中的新阶段可继续使用 `integrations/flowr_root/agent_continuation.py` 续推。旧的精确检查点包含原宿主机的绝对路径、模型哈希、stage runner 哈希和 GPU 信息；迁移到容器后须按[精确恢复说明](FLOWR_ROOT_LINUX.zh-CN.md)重建或核验这些来源，不能仅因路径已挂载就假定可直接恢复。
