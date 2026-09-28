# 增广拉格朗日执行与对偶状态恢复

MolExecutor 现在支持 `evaluator: augmented_lagrangian`。对于非负违反量 (h_j(X,G)\ge 0)，执行器最大化：

\[
R(X,G)=U(X,G)-\sum_j\left[\lambda_j h_j(X,G)+\frac{\kappa_j}{2}h_j(X,G)^2\right].
\]

每个约束独立声明尺度、正的修复余量、初始乘子、二次罚系数、乘子上限、满足阈值和持续满足规则。发生违反时更新 `lambda_j <- min(lambda_max, lambda_j + kappa_j h_j)`；只有连续满足达到设定帧数后，乘子才允许衰减。化学有效性、连通性、新增严重碰撞、受体固定和注入预算仍作为不可补偿的提案约束，不能被任务效用抵消。

约束计算器采用注册表。当前提供局部图条件 MMFF 几何、相对同时间无引导目标的局部 MMFF 松弛应变，以及缺陷支持区域之外的位移约束。新的 flow/diffusion 适配可以注册新的标量违反量，无需修改对偶控制器。可编辑掩码可独立选择缺陷一键邻域或全部配体原子。

`guidance_state.augmented_lagrangian` 保存控制器 schema、RewardProgram 身份、完整约束定义哈希、每个乘子、连续满足计数、更新次数、最后更新 step 和有界历史。外层 runtime checkpoint 同时保存当前生成状态、自条件缓存、RNG、累计路径预算和模型上下文。恢复时若程序、约束定义或约束 ID 改变会直接拒绝；同一步重复更新保持幂等。

FLOWR 后缀执行器在相同下一时刻和相同自条件上下文中比较原生分支与引导提案，每步只依据最终选中端点更新一次对偶变量，每五个积分步保存一次 checkpoint，并记录分量梯度范数、违反量、乘子变化、裁剪和独立 oracle 守门结果。若无引导参考的局部标量缺失，只允许在最大间隔以内的两个有效相邻帧之间线性插值，日志会明确记录插值方式与边界。

5i0b / ligand_002 的验证从真实 t=0.50 runtime 以 eta 100 运行。实时有限差分检查通过，仓库 73 项测试通过；从 t=0.75 恢复后对 2143 个 checkpoint 叶节点逐项比较，连续运行与恢复运行的最终 state、head 预测、world prediction 和 SDF 均逐字节一致。该结果验证当前 FLOWR 适配的执行与恢复一致性，不代表这些约束尺度已经适用于其他分子。

本模块没有新增安装包。运行时需要新输出目录，并在 execution JSON 中提供声明该 evaluator 的 RewardProgram、无引导对照轨迹、参考文件、约束和引导预算。
