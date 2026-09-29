# 独立构建 MolSteer + FLOWR.ROOT 镜像

[`Dockerfile.standalone`](../Dockerfile.standalone) 不使用 `COPY`，构建上下文可以为空。它从固定版本的 NVIDIA CUDA 12.1 镜像开始，安装 Miniconda、PyTorch cu121、MolSteer Agent 与 FLOWR.ROOT 所需的 Python 依赖、OpenSSH Server 和 `sudo`。源码由 Git 获取，`MOLSTEER_REF` 与 `FLOWR_REF` 默认固定到已核对的提交；可在构建时用 `--build-arg` 显式更换。

截至 2026-09-29，`jule-c/flowr_root` 可匿名克隆，但 `H-Dw/MolSteer` 对匿名 Git 请求返回 401。构建私有 MolSteer 时需要有该仓库读取权限的 GitHub token。BuildKit 只在拉取源码的步骤挂载 token，不把它写入镜像层、构建参数或 Git remote URL。若以后将 MolSteer 仓库公开，省略 `--secret` 即可。

在 Linux Bash 中，从任意保存了 Dockerfile 的目录构建，无需将 MolSteer 或 FLOWR 源码放入构建目录：

```bash
read -rsp 'GitHub token: ' GITHUB_TOKEN; echo
export GITHUB_TOKEN
docker build --network=host \
  --secret id=github_token,env=GITHUB_TOKEN \
  -t molsteer:standalone-cu121 - < Dockerfile.standalone
unset GITHUB_TOKEN
```

容器默认以前台 `sshd` 保持运行，只允许 `molsteer` 用户用公钥登录；该用户有 `sudo` 权限。首次启动时产生容器自己的 SSH host key。下面的例子只向宿主机回环地址映射 SSH 端口，并从宿主机读取授权公钥、已有模型权重和目标数据。模型权重、生成输入、输出以及 `OPENROUTER_API_KEY` 均不在镜像内。宿主机配置好 NVIDIA Container Toolkit 后，在 `docker run` 中增加 `--gpus all` 即可启用 GPU。

```bash
docker run -d --name molsteer \
  -p 127.0.0.1:2222:22 \
  --env OPENROUTER_API_KEY \
  --mount type=bind,src="$HOME/.ssh/authorized_keys",dst=/run/secrets/ssh_authorized_keys,readonly \
  --mount type=bind,src=/data1/dhuang/MolSteer/flowr_root/checkpoints,dst=/opt/MolSteer/flowr_root/checkpoints,readonly \
  --mount type=bind,src=/data1/dhuang/MolSteer/flowr_root/output,dst=/opt/MolSteer/flowr_root/output \
  molsteer:standalone-cu121

ssh -p 2222 molsteer@127.0.0.1
docker exec molsteer python -c 'import flowr.gen.generate_from_pdb; import molsteer, langchain_core, torch; print(torch.__version__, torch.cuda.is_available())'
```

在容器内，`/opt/conda/envs/molsteer-flowr` 是包含 Agent 和生成器直接依赖的统一环境；`MOLSTEER_ROOT=/opt/MolSteer`，`FLOWR_ROOT=/opt/MolSteer/flowr_root`。SSH 会话也设置这些路径。FLOWR 上游当前的 `pyproject.toml`/`uv.lock` 绑定 PyTorch 2.11 和 CUDA 13，因此镜像按现有服务器已验证的 CUDA 12.1 依赖清单创建环境，并直接从固定的 FLOWR 源码导入，不安装上游包元数据。

下载权重可使用 [FLOWR.ROOT 项目列出的检查点](https://github.com/jule-c/flowr_root#checkpoints)，尤其是 `flowr_root_v2.2.ckpt`。实验输入与运行结果也应挂载在 `flowr_root/output` 下。旧的精确续推检查点带有原主机的绝对路径、源码和权重哈希，跨入容器后必须按[恢复说明](FLOWR_ROOT_LINUX.zh-CN.md)逐项核验或重新生成阶段快照。

Docker 宿主机还需安装并配置 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 才能运行 `--gpus all`。目前 `ml-apus.bio.sustech.edu.cn` 的 Docker 尚未注册 NVIDIA 运行时；镜像构建与 CPU 导入检查不受此影响，GPU 容器生成必须待宿主机运行时配置完成。

2026-09-29 已在该服务器从空构建上下文成功构建 `molsteer:standalone-cu121`（约 10.3 GB）。构建中的 `pip check` 与 FLOWR/Agent 导入检查通过；容器内 `tests/test_expert_system.py` 和 `tests/test_agent_flowr_bridge.py` 共 31 项通过。公钥 SSH 登录后可直接找到统一环境的 Python 与 Conda，`sudo -n` 可用。GPU 容器生成尚未运行。
