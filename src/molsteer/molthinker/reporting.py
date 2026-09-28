"""Human-readable bilingual reasoning, distinct from the risk-only diagnosis."""
def render_derivation(spec, monitor=None, language='en'):
    zh = language=='zh'
    title = '奖励函数推导' if zh else 'Reward derivation'
    i=spec['identity']
    lines=[f"# MolThinker — {title}", '', f"{i['target_id']} / {i['ligand_id']} / {i['stage']}", '',
        f"StatePacket: `{spec['packet_id']}` · RewardSpec: `{spec['reward_id']}`", '',
        ('检索知识库中的21项函数；在当前化学图假设下构建局部坐标奖励。实际生成器执行仍受阻，本次仅在坐标副本上验证数值行为。' if zh else
         'Retrieved all 21 knowledge functions. Local coordinate rewards are conditional on the current chemical hypothesis. Live generator execution remains blocked; this trial checks numerical behavior on a coordinate copy.'), '',
        ('## 推导出的奖励项' if zh else '## Derived reward terms'), '',
        'R = −Σ Eᵢ; Eᵢ = ½ wᵢ {max[(ℓᵢ−vᵢ)/sᵢ, 0]² + max[(vᵢ−uᵢ)/sᵢ, 0]²}.', '',
        ('单侧碰撞罚省略上界项。不同视图分别求和，原状态与预测终点不相加。权重均为1，距离尺度1 Å、角度尺度10°，是未校准的演示设置。' if zh else
         'The upper term is omitted for minimum-distance penalties. Views are evaluated separately. Weights are 1; distance scale is 1 Å and angle scale 10°. These are uncalibrated demonstration settings.'), '',
        '| View | Atoms | Observable | Bounds | Scale | Knowledge | Evidence |', '|---|---|---|---|---|---|---|']
    for t in spec['terms']:
        bounds=f"{t['lower']:.6f}–{t['upper']:.6f}" if t['upper'] is not None else f"≥{t['lower']:.6f}"
        description={'1-3 endpoint distance; coupled bond-length/angle proxy':'1–3端点距离（键长/键角耦合代理）','MMFF reference bond window':'MMFF参照键长窗口','ligand-protein minimum distance':'配体—蛋白最小距离'}.get(t['observable'],t['observable']) if zh else t['observable']
        lines.append(f"| {t['view']} | {t['atom_ids']} | {description} | {bounds} {t['unit']} | {t['scale']} | {t['function_id']}, line {t['knowledge_source']['line']} | {t['reference_evidence_id']} |")
    lines += ['', ('## 化学假设与证据限制' if zh else '## Chemical hypotheses and evidence limits'), '']
    already=set()
    for t in spec['terms']:
        if t['finding_id'] in already or t['view']=='state':continue
        already.add(t['finding_id'])
        for b in t['local_chemical_context']['bonds']:
            p=b.get('top3',[])
            if p:
                lines.append(f"- {b['atom_ids']}: " + '; '.join(f"order {r['category']} p={r['probability']:.6f}" for r in p)+'.')
    lines += ['', ('端点1–3距离是键角与键长共同决定的代理量。其角度MMFF提示作为同一局部问题的支持，不再添加重复罚项。MMFF键长窗口依赖当前元素、形式电荷、隐式氢和键级；类别变化后必须重新推导。全分子应变不作为该位置的局部能量。' if zh else
        'The 1–3 distance couples bond lengths and angle. Its MMFF angle flag is supporting evidence, not an additional penalty. MMFF bond windows depend on current elements, formal charges, implicit hydrogens and bond orders; category changes require rederivation. Global strain is not local energy attribution.'), '',
        ('## 检索与适用性决策' if zh else '## Retrieval and applicability decisions'), '',
        '| Function | Role | Decision | Reason |', '|---|---|---|---|']
    reasons_zh={
        'G03':'缺少锚点坐标及可移动区域声明','G04':'缺少目标方向及已准备的特征类型',
        'G05':'没有具有有效参照的手性/平面异常','G06':'缺少指定参照形状及对齐','G07':'缺少可微环境描述符与参照库',
        'P01':'全分子应变可作证据，但化学图稳定性、质子化及参数适用性未确认',
        'P02':'有蛋白坐标，但没有经过核验的受体力场类型','P03':'形式电荷不能替代部分电荷及介电模型',
        'P04':'缺少xTB评价器、化学验证及调用预算','P05':'缺少Vina准备、评分后端及实时后缀导数',
        'P06':'缺少基团面积目标与可微SASA后端','P07':'缺少对齐的ESP参照及部分电荷',
        'S01':'缺少目标值、容差及实时种群','S02':'亲和力数值不能单独确定目标方向、温度及实时种群',
        'S03':'缺少同一候选的配对脱靶评分','S04':'属于估计方法；缺少可重调用评价器、可扰动自由度及预算',
        'S05':'缺少多目标声明、候选种群及结构距离','S06':'单个已保存状态不能提供编辑深度校准批次',
        'S07':'缺少目标片段与三种群搜索适配器'}
    for r in spec['retrieval']:
        reason=r['decision_reason']
        if zh:
            reason='局部测量及边界可用；仅在冻结化学假设及明确演示掩码后离线验证' if r['decision']=='conditional_offline' else reasons_zh.get(r['function_id'],'没有满足条件的局部证据')
        lines.append(f"| {r['function_id']} · {r['name'] if zh else r['name_en']} | {r['role']} | {r['decision']} | {reason} |")
    lines += ['', ('未覆盖的诊断风险（例如价态冲突、PAD、断连）保留为诊断证据；本实现不会伪造可微的类别修复奖励。SPSA是估计方法；没有可重复调用的评价器时，不因缺梯度而自动启用。' if zh else
        'Unhandled diagnostic risks (such as valence contradictions, PAD tokens or disconnection) remain evidence. This implementation does not invent differentiable categorical repair rewards. SPSA is an estimator and cannot be enabled without a reevaluable oracle.'), '',
        ('## 执行边界' if zh else '## Execution boundary'), '',
        ('当前只有离线坐标副本的梯度。保存张量不提供生成器雅可比；活跃原子掩码不是可编辑掩码。三份已保存分子也不代表运行中的粒子种群。若接入生成器，需要明确的可编辑自由度、预算及端点梯度映射。' if zh else
         'Only coordinate-copy derivatives are available. Saved tensors do not provide the generator Jacobian; active-slot masks are not editable masks. Three saved molecules do not establish a live particle population. Generator integration requires explicit editable degrees of freedom, budget and endpoint derivative mapping.'), '']
    if monitor:
        lines += [('## 离线数值验证' if zh else '## Offline numerical validation'), '',
            f"- Penalty: {monitor['penalty_before']:.8g} → {monitor['penalty_after']:.8g}; reward: {monitor['reward_before']:.8g} → {monitor['reward_after']:.8g}.",
            f"- Finite differences: {monitor['numerical_gradient']['passed']}; maximum error {monitor['numerical_gradient']['maximum_absolute_error']:.3e}.",
            f"- Demo movable atoms: {monitor['demo_movable_atom_ids']}; maximum displacement {monitor['max_atom_displacement_angstrom']:.6f} Å.",
            f"- Fixed atoms unchanged: {monitor['fixed_atoms_unchanged']}; original snapshot unchanged: {monitor['input_snapshot_unchanged']}.",
            f"- New bond-window violations: {monitor['new_bond_window_violations']}; new protein clashes: {monitor['new_protein_clashes']}.", '',
            '| Atoms | Before | After | Unit |', '|---|---|---|---|']
        for r in monitor['terms']:lines.append(f"| {r['atom_ids']} | {r['before']['value']:.6f} | {r['after']['value']:.6f} | {r['unit']} |")
        lines += ['', ('本次仅验证公式及局部能量下降，没有重新运行完整分子生成，不证明final质量、结合能力或化学稳定性得到改善。原始生成文件未修改。' if zh else
            'This validates the formula and local energy descent only. Full generation was not rerun; improved final quality, binding or chemical stability is not established. Original generation files are unchanged.'), '']
    return '\n'.join(lines)
