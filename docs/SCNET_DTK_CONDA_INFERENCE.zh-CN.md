# SCNet Notebook：MolSteer 与 FLOWR.ROOT 推理 Conda 环境

记录日期：2026-09-29。目标镜像为 `jupyterlab-pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.11-devel`，本配置只覆盖模型推理、MolSteer Agent 与必要的结构处理，不安装训练数据或训练工具链。登录方式是 `ssh -p 10544 root@ksai.scnet.cn`；密码不要写入脚本或文档。

## 已部署位置与兼容边界

| 项目 | 实测值 |
| --- | --- |
| 加速卡 | `K100_AI`，64 GB；`torch.cuda.is_available()` 为 `True` |
| 镜像 PyTorch | `2.9.0+das.opt1.dtk2604`，`torch.version.hip=6.3.26093` |
| Miniforge | `/root/private_data/.miniforge3` |
| Conda 环境 | `/root/private_data/.miniforge3/envs/molsteer-flowr-dtk`，Python 3.11.16 |
| MolSteer 源码 | `/root/private_data/MolSteer` |
| FLOWR.ROOT 源码 | `/root/private_data/MolSteer/flowr_root` |
| 模型权重 | `flowr_root/checkpoints/flowr_root_v2.2.ckpt` |
| 验证时源码提交 | MolSteer `94e09963fe643574afa1dcd935742b9acf8c087a`；FLOWR.ROOT `739a33c2aab74d074dd24605b9a2e34dc809ae9c` |

该镜像的 PyTorch 是 DTK 定制构建，不能用 PyPI、官方 CUDA 或官方 ROCm 轮子替换。FLOWR.ROOT 上游当前 `pyproject.toml` / `uv.lock` 面向 Python 3.12、PyTorch 2.11 和 CUDA 13；在这个镜像上直接执行 `uv sync --extra gpu` 会破坏 GPU 兼容性。本部署让 Python 3.11 Conda 环境复用镜像的 DTK PyTorch，其余包安装在 Conda 环境内。FLOWR.ROOT 源码在 Python 3.11 下通过了编译检查和一次真实推理，但这不是上游锁文件声明的标准组合。

平台的 `/root/private_data` 是持久个人目录，容量 5 GB；系统盘随镜像保存有 15 GB 的保存限制。环境与权重均放在个人目录，安装时的 Conda 包缓存放在 `/root/.conda/pkgs`，避免包缓存挤占个人目录。Notebook 关闭前仍需按平台要求保存开发环境，个人目录中的文件不依赖该操作。

## 重建步骤

以下命令在该 Notebook 容器中运行。当前机器已经有两份源码；若重建新容器，先把 MolSteer 仓库和 [FLOWR.ROOT 仓库](https://github.com/jule-c/flowr_root)放到上表路径，确保 `flowr_root/flowr/` 和仓库内修改过的 `posebusters/` 均在。

### 1. 安装 Miniforge 并创建 Conda 环境

```bash
curl -fL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /root/Miniforge3-Linux-x86_64.sh
bash /root/Miniforge3-Linux-x86_64.sh -b -p /root/private_data/.miniforge3
export CONDA_PKGS_DIRS=/root/.conda/pkgs
/root/private_data/.miniforge3/bin/mamba create -y \
  -p /root/private_data/.miniforge3/envs/molsteer-flowr-dtk \
  python=3.11 pip setuptools wheel
```

### 2. 隔离安装推理依赖

首次执行 `pip install` 时，**不要先添加镜像的 `site-packages` 到 Conda 环境**。否则 pip 会把镜像基础包当作本环境的可卸载依赖。本次部署中首次尝试触及了基础包，已中断并恢复；隔离安装后再次核对了基础 PyTorch 与 GPU。

先给 pip 提供仅用于解析依赖的 DTK PyTorch 版本元数据；真正的 PyTorch 模块仍由镜像提供。完整包清单和约束分别在 [`scnet-dtk-inference.requirements.txt`](../configs/scnet-dtk-inference.requirements.txt) 与 [`scnet-dtk-inference.constraints.txt`](../configs/scnet-dtk-inference.constraints.txt)。

```bash
export MOLSTEER_ROOT=/root/private_data/MolSteer
export MOLSTEER_ENV_PREFIX=/root/private_data/.miniforge3/envs/molsteer-flowr-dtk
export ENV_SITE="$MOLSTEER_ENV_PREFIX/lib/python3.11/site-packages"
mkdir -p "$ENV_SITE/torch-2.9.0+das.opt1.dtk2604.dist-info"
cat > "$ENV_SITE/torch-2.9.0+das.opt1.dtk2604.dist-info/METADATA" <<'EOF'
Metadata-Version: 2.1
Name: torch
Version: 2.9.0+das.opt1.dtk2604
Summary: SCNet image DTK PyTorch, imported from /usr/local/lib/python3.11/site-packages
EOF

"$MOLSTEER_ENV_PREFIX/bin/python" -m pip install --no-cache-dir \
  -c "$MOLSTEER_ROOT/configs/scnet-dtk-inference.constraints.txt" \
  -r "$MOLSTEER_ROOT/configs/scnet-dtk-inference.requirements.txt"
"$MOLSTEER_ENV_PREFIX/bin/python" -m pip install --no-deps -e "$MOLSTEER_ROOT"

printf '%s\n' \
  /usr/local/lib/python3.11/site-packages \
  "$MOLSTEER_ROOT/flowr_root" \
  > "$ENV_SITE/scnet_dtk_runtime.pth"
```

`.pth` 文件让同一个解释器同时找到镜像 DTK PyTorch 和 FLOWR.ROOT 源码。FLOWR.ROOT 的上游包元数据要求 Python 3.12 / PyTorch 2.11，所以本配置通过源码路径加载它，不执行 `pip install -e flowr_root`。以后更新环境包时，先暂时移走 `scnet_dtk_runtime.pth`，完成安装后再放回，避免 pip 修改镜像基础包。

### 3. 激活并在 JupyterLab 中使用

```bash
source /root/private_data/.miniforge3/etc/profile.d/conda.sh
conda activate /root/private_data/.miniforge3/envs/molsteer-flowr-dtk
python -m pip check
python -c 'import torch, flowr.gen.generate_from_pdb, molsteer; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

命令行使用 `conda activate` 即可。JupyterLab 中已注册 `MolSteer + FLOWR (DTK)` 内核，指向上述 Conda 解释器。容器重建后如内核消失，在当前环境中重新运行 `python -m ipykernel install --user --name molsteer-flowr-dtk --display-name "MolSteer + FLOWR (DTK)"`。

### 4. 权重与一次最小推理

作者公开的 `flowr_root_v2.2.ckpt` 可从 [Zenodo 记录](https://zenodo.org/records/20069589)下载。文件大小应为 `1007216503` 字节，MD5 应为 `d227b736471c6f6fd90488951571a6fa`。当前远端文件已通过校验。若在新容器中缺失：

```bash
mkdir -p "$MOLSTEER_ROOT/flowr_root/checkpoints"
curl -fL 'https://zenodo.org/records/20069589/files/flowr_root_v2.2.ckpt?download=1' \
  -o "$MOLSTEER_ROOT/flowr_root/checkpoints/flowr_root_v2.2.ckpt"
md5sum "$MOLSTEER_ROOT/flowr_root/checkpoints/flowr_root_v2.2.ckpt"
```

使用仓库自带 BACE 受体和配体，只生成 1 个分子：

```bash
cd "$MOLSTEER_ROOT/flowr_root"
python -m flowr.gen.generate_from_pdb \
  --pdb_file examples/bace_protein.pdb \
  --ligand_file examples/bace_ligands.sdf \
  --arch pocket --pocket_type holo --cut_pocket --pocket_cutoff 7 \
  --gpus 1 --num_workers 0 --batch_cost 20 \
  --ckpt_path checkpoints/flowr_root_v2.2.ckpt \
  --save_dir output/scnet_dtk_smoke_20260929 \
  --max_sample_iter 0 --coord_noise_scale 0.1 \
  --sample_n_molecules_per_target 1 --integration_steps 10
```

重复运行时把 `--save_dir` 改成新的目录。10 步仅用于快速验证；日志提示该步数下类别噪声保护会压制噪声，正式生成应采用上游脚本默认的 100 步，并根据任务设置过滤和采样数。

## 本次验证结果

- Conda 内的 `torch` 模块路径为 `/usr/local/lib/python3.11/site-packages/torch`；版本 `2.9.0+das.opt1.dtk2604`，GPU 张量运算成功。
- `flowr.gen.generate_from_pdb`、`molsteer`、LangChain、RDKit、TensorDict、PyG、Lightning、Biotite、OpenMM 与仓库内 PoseBusters 都能导入。
- `python -m pip check` 输出 `No broken requirements found.`
- BACE 示例推理日志显示 `Using device: cuda`、`Run time=8.0s for 1 molecules` 和 `Sampling finished.`；`samples_bace_protein.sdf` 包含 1 个可由 RDKit 读取、带 3D 构象的 27 原子分子。
- MolSteer Agent 配置、资产、运行时与 FLOWR 桥接的 39 项针对性测试全部通过（`39 passed in 75.84s`）。

本次验证证明这套环境能运行给定检查点的基本 FLOWR 推理；它不证明所有条件生成模式、MolSteer 的在线引导或长期运行都已在 DTK 上验证。Agent 真正调用模型 API 时，还需由运行进程设置相应的 API 密钥，不能把密钥写进 Conda 文件或项目仓库。

## 依据

- [SCNet：该 Notebook 镜像的 DTK PyTorch 二进制兼容说明](https://www.scnet.cn/help/docs/mainsite/ai/practice/application/ultralytics/index.html)
- [FLOWR.ROOT 官方安装及推理说明](https://github.com/jule-c/flowr_root/blob/main/README.md)
- [FLOWR.ROOT 作者公开权重与校验值](https://zenodo.org/records/20069589)
