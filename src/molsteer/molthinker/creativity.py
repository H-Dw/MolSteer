"""Historical deterministic reward baselines; API Agent creativity uses a separate design path."""
from .planner import derive as derive_selection
from molsteer.common import digest

DEFAULT_MODE='creativity'
DEFAULT_SKILL='molthinker-reward-creativity'
SKILLS={m:'molthinker-reward-'+m for m in ('selection','creativity')}


def build_program(selection, mode='creativity'):
    if mode not in SKILLS:raise ValueError('Unknown reward reasoning mode')
    terms=[t for t in selection['terms'] if t['view']=='prediction']
    if not terms:raise ValueError('No supported endpoint objective; additional observations are required')
    region=sorted({i for t in terms for i in t['hypothesis_atom_ids']})
    p=dict(kind='RewardProgram',mode=mode,skill=SKILLS[mode],packet_id=selection['packet_id'],
        identity=selection['identity'],region_atom_ids=region,terms=terms,
        active_objectives=['geometry','clash','pocket','movement'],weights=[1.,1.,1.,1.],
        tau=.1,rho=.05,lambda_graph=.1,graph_epsilon=1e-8,
        bond_tolerance_fraction=.1,angle_tolerance_degrees=30.,
        clash_scale_angstrom=1.,centroid_scale_angstrom=1.,movement_scale_angstrom=.5,
        intra_vdw_ratio=.7,protein_vdw_ratio=.75,severe_overlap_angstrom=.4,
        max_endpoint_displacement_angstrom=1.,
        graph_policy='rebind_local_bond_and_angle_references' if mode=='creativity' else 'frozen_initial_template_for_literal_ablation',
        prerequisite_policy='Reject guidance if any common objective or mandatory constraint cannot be evaluated',
        graph_preference_gradient='Zero within each discrete branch; influences acceptance, no straight-through surrogate',
        constraints=['sanitized_connected_proposal_graph','no_new_or_worsened_severe_clash','endpoint_displacement_bound','fixed_mask','injected_path_budget'],
        inactive_objectives=[dict(name='residual_strain',reason='No matched reference with local terms subtracted'),
            dict(name='binding_affinity',reason='No supplied endpoint target or calibrated differentiable binding objective'),
            dict(name='contact_pattern',reason='No designated validated favorable contacts; centroid retention only')],
        normalization_origin='Explicit uncalibrated demonstration settings; tolerances express meaningful geometric scales',
        source_reward_id=selection['reward_id'],evidence_ids=sorted({e for t in terms for e in t['evidence_ids']}),
        knowledge_source=selection['knowledge_source'],retrieval=selection['retrieval'])
    p['reward']='-tau*log(mean(exp(w*F/tau)))-rho*sum(w*F)-lambda_graph*C_G' if mode=='creativity' else '-sum(interval_penalties)'
    p['program_id']='rp_'+digest(p)[:24]
    return p


def derive(packet, report, knowledge_path, mode=DEFAULT_MODE):
    selection=derive_selection(packet,report,knowledge_path)
    return build_program(selection,mode)


def render_program(p, language='en'):
    zh=language=='zh'
    lines=[f"# {'奖励组合说明' if zh else 'Reward composition'}",'',f"Mode: {p['mode']}",
        f"Skill: {p['skill']}",f"StatePacket: {p['packet_id']}",'',f"`{p['reward']}`",'',
        '局部区域由诊断证据绑定的原子槽位确定；未将具体分子身份写入通用策略。' if zh else
        'The localized region comes from diagnostic evidence; molecular identity is not embedded in the general policy.',
        '几何、碰撞、口袋位置保持、位移使用固定共同目标集合；缺失目标使候选不可比较，不记作零。' if zh else
        'Geometry, clashes, pocket-position retention and movement form a common objective set. Missing objectives invalidate comparison rather than becoming zero.',
        '几何参照随离散图更新；图偏好使用初始边缘分数，仅参与候选接受，不产生坐标梯度。' if zh else
        'Geometry references follow the discrete graph. Frozen initial marginal scores affect acceptance, with no coordinate derivative.',
        '所有权重与控制预算均为待校准实验参数；不保证改进最终质量。' if zh else
        'Weights and control budgets are uncalibrated experimental choices; improved final quality is not guaranteed.','']
    lines += [f"- {x['name']}: {x['reason']}" for x in p['inactive_objectives']]
    return '\n'.join(lines)+'\n'
