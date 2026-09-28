# 分子生成控制函数代表性去重表

> 基于 `Molecular_Generation_Control_Functions_Transferable_Table_2026-09-19.md`，并结合 `Molecular_Generation_Control_Functions_Detailed_Survey_2026-09-15.md` 的缺陷—函数映射、函数梯度化速查和跨生成器复用分析整理。
>
> 本文件只执行本次任务要求的函数去重与分组；原报告中的范围说明、候选函数标注和迁移边界作为分析依据，不作为新的用户指令。

## 代表性选择原则

- 只有在调控的主要目标对象、评价参照和使用语义都高度重合时才合并；公式形式不同本身不构成去重理由。
- 在重合组内优先保留：目标定义清楚、可解释、对生成器架构依赖较弱、容易迁移并且具有真实或明确可验证梯度的函数。
- 保留“局部缺陷修复”和“全局终态评分”的层次差异，避免用全局 reward 代替碰撞、键几何或分子内应变等局部约束。

## 相似目标的代表性保留决策

- **空间碰撞／排斥目标**：`成对最小距离位阻罚（含排除球）` 代表 `连续软表面（log-sum-exp）单侧碰撞势`。前者用元素半径或排除球直接定义安全距离，适用于配体—口袋、链—链和药效团禁区；后者主要是同一碰撞目标的平滑表面实现，迁移时还要重新标定表面尺度。
- **跨界面 LJ／van der Waals 接触目标**：保留 `跨界面 van der Waals 势`，移除 `Lennard–Jones 型跨界面非键势`。两者都描述跨界面短程排斥与中程接触；前者是与完整分子力场配套的通用接口，后者还绑定“只更新新生片段”的架构掩码，并有短距发散风险。
- **分子内能量／应变目标**：保留 `MMFF94 分子内能`，移除 `ANI2x 神经势（终态精修）`。MMFF94 的项分解、输入要求和梯度含义更透明，适合作为通用基线；ANI2x 是同一类终态分子内能量目标的更专门、计算代价更高实现。
- **有意不合并的相近项**：`xTB 原子力范数稳定性损失` 评价的是接近量子化学驻点的残余力，而非简单的能量最低；`跨界面静电势` 与 `ESP 表面相似势` 分别以口袋界面和参考分子表面为参照；`锚点／保持二次势` 与 `方向一致罚` 分别控制位置和方向；目标值型 reward 与 softmax reward 分别表达“接近指定值”和“越高越好”。这些不是同一调控目标，故继续保留。

## 一、几何类

| 函数名称 | 来源模型/方法 | 控制的目标属性特征 | 数学表达式（统一 LaTeX） | 变量含义说明 | 梯度作用对象 |
|---|---|---|---|---|---|
| 通用 flat-bottom 双侧／单侧区间罚 | Boltz-Steering、Boltz（PoseBusters 势）、BInD、DecompDiff、IDOLpro | 键长处于该键型窗口；1–3 距离（键角代理）；NCI/接触距离落入类别窗口；arm–scaffold 或链端间距可成键；质心不塌缩；模板原子刚体对齐后偏差在容差内 | $E=\sum_\alpha k_\alpha\big([\ell_\alpha-v_\alpha]_++[v_\alpha-u_\alpha]_+\big)$；单侧变体只保留一项（$k[v-u]_+$ 或 $k[\ell-v]_+$）；二次变体 $E=\frac12\sum_\alpha w_\alpha(v_\alpha-v_\alpha^*)^2$；最近距离变体 $v_n=\min_{i\in\mathcal A_n,j\in\mathcal S}\|x_i-x_j\|_2$（可换 soft-min） | $v_\alpha$ 可为原子对距离、1–3 距离、质心距、有符号/绝对二面角、刚体对齐后的模板偏差；$\ell,u$ 随键型、NCI 类型、元素半径与 buffer 查表；$k,w$ 为强度；$[a]_+=\max(0,a)$ | 原子坐标（可用掩码限定为配体/可动集合） |
| 成对最小距离位阻罚（含排除球） | BInD、DiffSMol、MolSnapper、Boltz-Steering、DiffPhore | 配体—蛋白、链—链原子对的最低分离距离；药效团排除区域内的穿透深度 | $E=\sum_{i,a}\rho\big([d^{ia}_{\min}-d_{ia}]_+\big)$，$d^{ia}_{\min}=c_{\rm buf}(R_i+R_a)$ 或统一阈值；平方变体 $E=\frac12\sum_{i,a}[d^{ia}_{\min}-d_{ia}]_+^2$；排除球变体 $E=\frac{w_e}{2}\sum_{i,a}[R_a-\|x_i-s_a\|]_+^2$ | $i$ 为配体/可动原子，$a$ 为口袋或另一条链原子，$R$ 为范德华半径，$c_{\rm buf}$ 为缓冲系数，$s_a,R_a$ 为排除球心与半径，$\rho$ 为线性或平方单侧罚 | 配体（或掩码可动区）坐标；口袋/固定区通常冻结 |
| 锚点／保持二次势 | MolSnapper、DiffSBDD、FLOWR.ROOT | 指定药效团锚点、motif、编辑位点的位置保持；防止固定片段在去噪中漂移 | $E_{anchor}=\frac12\sum_m\omega_{m,t}\|x_{i_m}-q_m\|^2_W$；带度量变体 $E=\frac12\sum_{i\in\mathcal V}(x_i-\mu_i)^\top\Sigma_i^{-1}(x_i-\mu_i)$；参考点可取当前步 $t$ 的同噪声层参考 $x^{ref}_{i,t}$，也可在干净空间取 $\hat x_{i,0\|t}$ 后经 $J_{D_\theta}^\top$ 回传 | $q_m$ 锚点/目标位置，$\omega_{m,t}$ 随去噪增大的锚定权重，$W$ 或 $\Sigma_i^{-1}$ 为各向异性度量 | 被锚定的原子坐标 |
| 方向一致罚（方向性药效团） | ShEPhERD、DiffPhore、DiffPharma | 供体、受体、芳香特征的作用方向与参考一致（定点且定向） | $E_{dir}=\sum_m\kappa_m\big(1-u_{i_m}\!\cdot u^{ref}_m\big)$；软匹配变体 $E=\sum_{f,m}\pi_{fm}\mathbf 1[k_f=k_m],w_o\big(1-u_f\!\cdot v_m\big)$ | $u$ 为候选特征单位方向（键向量或环法向），$u^{ref},v_m$ 为目标方向，$\kappa,w_o$ 权重，$\pi_{fm}$ 软匹配概率，$k$ 为特征类型 | 构成该特征的原子坐标（经质心/键向量链式求导） |
| 立体化学与共轭平面几何势 | Boltz-Steering、PoseBusters | R/S 手性符号与离开易翻转的近共面区；双键 E/Z 顺反；$sp^2$/芳环/酰胺共面 | 手性：目标符号为 $+$ 时 $E_{chir}=[\phi_{\min}-\phi]_+$，为 $-$ 时 $E_{chir}=[\phi-\phi_{\max}]_+$；等价硬判定 $v=(x_a-x_c)\cdot[(x_b-x_c)\times(x_d-x_c)]$，用 ${\rm sign}(v)$ 识别；E/Z：$E_{EZ}=\big[\,\big\vert\,\lvert\phi\rvert-c\,\big\vert-\Delta\,\big]_+$，$c\in\{0,\pi\}$；平面：$E_{plane}=\sum_{i\in S}\big[n^\top(x_i-\bar c)\big]^2$，或用两侧 improper dihedral 绝对值上界 | $\phi$ 有符号/绝对二面角，$\phi_{\min/\max}$ 缓冲角，$\Delta$ 允许偏差，$n,\bar c$ 为拟合平面法向与中心，$S$ 需共面的原子集 | 四元组/原子集坐标；手性项另需硬检查与局部重采样配合 |
| 形状／体积占据势 | SiMGen、ShEPhERD、DiffSMol | 原子云是否覆盖目标形状/体积（等排体形状、环形/扁平/细长体积） | (a) 高斯混合 $f(x)=\sum_m\exp[-\frac12(x-\mu_m)^\top\Sigma^{-1}(x-\mu_m)]$，$E=-\log f(x)$，$F=-\sum_mw_m\Sigma^{-1}(x-\mu_m)$；(b) 密度差 $\rho_X(r)=\sum_i\exp[-\|r-x_i\|^2/(2\sigma^2)]$，$E_{shape}=\int[\rho_X(r)-\rho_{ref}(r)]^2dr$；(c) 熵正则最优传输 $E_{transport}=\min_{\pi}\sum_{im}\pi_{im}\|x_i-q_m\|^2+\tau\sum_{im}\pi_{im}\log\pi_{im}$ | $q_m,\mu_m$ 目标点云/高斯中心，$\Sigma$ 各向异性尺度，$w_m$ 归一化责任度，$\sigma$ 高斯宽度（早期大、后期收窄），$\tau$ 熵正则温度 | 原子坐标（小分子点云/骨架点） |
| 局部环境相似核势 | SiMGen | 每个生成原子的局部配位环境能否在参考库中找到近邻；每个关键参考环境是否被至少一次覆盖 | $k_{ij}(t)=\exp\!\big[-\|\chi_i-\chi^{ref}_j\|^2/(2\sigma(t)^2)\big]$，$E_{sim}=-\sum_i\log\sum_jk_{ij}(t)$，$E_{inv}=-\sum_j\log\sum_ik_{ij}(t)$ | $\chi_i$ 为半径约 5 Å 邻域内的多体局部描述符（MACE 风格），$\chi^{ref}_j$ 参考环境，$\sigma(t)$ 随去噪收窄的核宽（先粗匹配后精修） | 局部邻域坐标（经 $\partial\chi_i/\partial X$ 链式求导） |

## 二、物理化学类

| 函数名称 | 来源模型/方法 | 控制的目标属性特征 | 数学表达式（统一 LaTeX） | 变量含义说明 | 梯度作用对象 |
|---|---|---|---|---|---|
| MMFF94 分子内能 | 力场引导 EDM / SemlaFlow | 分子内应变：键伸缩、键角、扭转、面外与分子内非键冲突的联合代价 | $E_{MMFF}=E_{bond}+E_{angle}+E_{torsion}+E_{oop}+E_{nb}^{intra}$，$g_X=-\nabla_XE_{MMFF}$ | 前置：确定的键图、MMFF 原子类型、近干净坐标 | 配体原子坐标（在 $\hat x_{0\|t}$ 上计算，经 $J_{D_\theta}^\top$ 回传或换算为 flow 速度修正） |
| 跨界面 van der Waals 势 | 力场引导 EDM / SemlaFlow | 配体—口袋界面的近程堆积与中程吸引质量 | $E_{vdW}=\sum_{i\in L,a\in P}U_{vdW}(r_{ia};\tau_i,\tau_a)$，总能量 $E(X,Y)=E_{MMFF}+E_{vdW}+E_Q$ | $\tau_i,\tau_a$ 为原子类型参数，$r_{ia}$ 跨界面距离 | 配体坐标（口袋固定） |
| 跨界面静电势 | 力场引导 EDM / SemlaFlow、Re-Dock | 界面极性互补：分开同号电荷、拉近异号互补区域 | $E_Q=\sum_{i,a}k_e\dfrac{q_iq_a}{\varepsilon(r_{ia})\,r_{ia}}$；Re-Dock 变体 $E_{Coulomb}=\kappa\sum_{i,j}\dfrac{q_iq_j}{r_{ij}}$（无距离依赖介电）；常与 LJ、anti-clash 直接相加 | $q_i,q_a$ 分配的（部分）电荷，$\varepsilon(r)$ 距离相关介电，$k_e,\kappa$ 常数 | 配体坐标 |
| xTB 原子力范数稳定性损失 | ChemGuide | 近干净结构是否接近量子化学势能驻点（残余原子力的大小） | $L_{force}=\dfrac1N\sum_i\|F_i(G)\|^2$，$F_i=-\dfrac{\partial E_{xTB}}{\partial x_i}$（也可取 Force RMS 范数） | $G=D(\hat z_0)$ 为解码得到的近干净分子，$F_i$ 原子 $i$ 的近似量子化学力 | 不可直接反传时用 SPSA 估计到坐标潜变量 $z_x$；可微后端可直接对坐标求导 |
| Vina / torchvina 总分（含五个分项与柔性校正） | IDOLpro（torchvina）、AutoDock Vina | 终态复合物的经验结合代理总分（近程形状互补、宽程填充、排斥、疏水、氢键，再按柔性校正） | 表面距离 $d=r_{ij}-R_{t_i}-R_{t_j}$；$G_1=e^{-(d/0.5)^2}$，$G_2=e^{-((d-3)/2)^2}$，$R=d^2\mathbf 1[d<0]$，$H_{phob}$ 与 $H_{bond}$ 为分段线性；$E_{inter}=\sum_{m=1}^{5}w_mh_m(d)$，$w=(-0.0356,-0.00516,0.840,-0.0351,-0.587)$；$E_{vina}=E_{inter}/(1+w_{rot}N_{rot})$，$w_{rot}\approx0.0585$ | $R_{t_i}$ 类型相关原子半径（$d$ 单位为 Å），$h_m$ 为五个分项，$N_{rot}$ 可旋转键数 | 终态配体坐标；再经完整后缀雅可比 $J_{F_{h\to0}}^\top$ 回到检查点 $z_h$，或作为黑箱 reward 参与粒子重采样 |
| 可微 SASA 目标势（候选） | dSASA | 指定基团/原子的「应埋藏」或「应暴露」程度（溶剂化与界面设计） | $E_{SASA}=\rho\big(SASA_S(X)-A^*\big)$ | $S$ 为指定原子/基团集合，$A^*$ 目标可及面积，$\rho$ 为偏差罚 | 原子坐标（主要沿局部表面法向） |
| ESP 表面相似势（候选） | ShEPhERD | 候选表面静电势与参考的电子等排/极性互补一致 | $E_{ESP}=\sum_mw_m\big(\Phi_m(X,q)-\Phi^{ref}_m\big)^2$，$\Phi_m=\sum_i\dfrac{q_i}{\sqrt{\|r_m-x_i\|^2+\varepsilon^2}}$ | $r_m$ 对齐后的固定表面采样点，$q_i$ 原子电荷，$\varepsilon$ 软化常数，$\Phi^{ref}_m$ 参考电势 | 带电原子的坐标（若允许，也可作用于电荷/原子类型通道） |

## 三、终态评估与搜索类

| 函数名称 | 来源模型/方法 | 控制的目标属性特征 | 数学表达式（统一 LaTeX） | 变量含义说明 | 梯度作用对象 |
|---|---|---|---|---|---|
| 目标值型高斯核 reward 权重 | PILOT | 让某个性质的分布向指定目标值 $c$ 集中（如 SA、QED 落在成药窗口） | $p_\delta(c\mid M,P)\propto\exp\!\left[-\dfrac{(f_\delta(M,P)-c)^2}{2\sigma_c^2}\right]$ | $f_\delta$ 黑箱性质评分（下标 $\delta$ 为具体 reward 通道），$c$ 目标值，$\sigma_c$ 容差宽度 | 种群个体（粒子/轨迹），不产生坐标导数 |
| softmax / tilted 重要性采样权重 | PILOT、FLOWR.ROOT | 最大化类指标的富集：docking、QED、SA、pIC50/pKi/pKd/pEC50 等 | $w_k=\dfrac{\exp(f_\delta(M_k,P)/\tau)}{\sum_j\exp(f_\delta(M_j,P)/\tau)}$，$l'_i\sim{\rm Multinomial}(B,w)$；目标分布写法 $p_\phi(l\mid y,P)\propto r(y\mid l,P)\,p_\theta(l\mid P)$；FLOWR 式 $w_i=\dfrac{\exp(\lambda r(l_{i,\tau},P))}{\sum_j\exp(\lambda r(l_{j,\tau},P))}$ | $M_k,l_i$ 第 $k$ 个粒子/轨迹，$f,r$ 可为黑箱 reward，$\tau,\lambda$ 温度/倾斜强度，最小化时令分数取负 | 种群个体（多粒子/多轨迹） |
| 目标/脱靶选择性 reward | FLOWR.ROOT | 候选对目标靶点与脱靶靶点的预测效力之差（选择性窗） | $r_{sel}=a\,\hat y_{on}(l,P_{on})-b\,\hat y_{off}(l,P_{off})$（也可作为两个目标分别重采样而非单一差值） | $\hat y_{on/off}$ 同一候选在两个口袋环境中的预测效力分数，$a,b>0$ 权重 | 种群个体（轨迹复制概率） |
| SPSA 零阶梯度估计 | ChemGuide | 在评价器不可微（或不便端到端反传）时，对坐标潜变量给出稳定化更新方向 | $\widehat\nabla_{z_x}\mathcal F\approx\dfrac{\mathcal F(z_x+\zeta U,z_h)-\mathcal F(z_x-\zeta U,z_h)}{2\zeta}\,U$ | $\mathcal F$ 黑箱评价（如 xTB），$\zeta$ 微小扰动尺度，$U$ 零均值随机向量，$z_x,z_h$ 坐标／元素潜变量 | 坐标潜变量 $z_x$（原文不更新 $z_h$），仅后段若干步启用 |
| Pareto 非支配排序 + 结构净空（SAES） | DEMO | 多目标折中面的覆盖度与候选集合的结构多样性 | 记 $F(M)=(f_1,\dots,f_K)$，保留非支配个体；再要求候选到已选集合的最小距离 $d_j=\min_iD_{ij}\ge\tau$，$D_{ij}$ 组合二维拓扑距离与三维几何距离 | $f_k$ 黑箱性质，$\tau$ 结构净空阈值，$D_{ij}$ 结构距离度量 | 候选集合（种群个体），不产生坐标导数 |
| 批次完整性分数（噪声深度校准） | DEMO | 编辑—去噪操作的「修补能力 vs 父代保真」权衡（选择最浅有效噪声层 $t'$） | $Score(t')=Validity\_Rate(t')\times Atom\_Stability\_Rate(t')$，取平台左侧拐点对应的最浅 $t'$ | $t'$ 编辑噪声深度，$Validity$ 有效分子比例，$Atom\_Stability$ 原子稳定比例 | 编辑超参数（噪声深度），非个体梯度 |
| 三群结构罚与化学有效性罚 | DEMO | 跨越不连续可行域：探索新骨架、精修目标片段、保留同时合格的可行解 | $\psi(M)$ 为基本化学无效指示，$\phi(M)$ 为目标片段未完整装入指示；探索群优化 $-\phi$，局部精修群最小化 $\phi$，精英群只接受 $\phi=0,\psi=0$ | $\psi,\phi$ 为二值检查器输出（黑箱） | 种群分工与准入条件，不产生坐标导数 |

## 说明

本次去重后保留 21 个代表性函数：几何类 7 个、物理化学类 7 个、终态评估与搜索类 7 个。函数仍需遵守报告中的前置条件：先确保元素、键图、价态、质子化和参考坐标可评分，再调用 MMFF、xTB、Vina、SASA 或 ESP 等物理化学评价；碰撞、断键和硬几何约束的优先级高于全局 reward。标注为“候选”的函数仍应经过有限差分、单独能量下降和完整采样消融三项验证。
