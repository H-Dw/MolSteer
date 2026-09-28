"""Localized cross-source evidence and traceable attribution summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
from rdkit import Chem,RDConfig
from rdkit.Chem import ChemicalFeatures,Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from molsteer.common import write_json,file_hash
from molsteer.molreader.outcome_context import attach
from molsteer.molreader.polar_context import measure as polar_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    context=json.loads((root/'ReaderComparisonContext.json').read_text());by={r['label']:r for r in context['observations']}
    factory=ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))
    rows={}
    for label in ['native_final','adaptive_same_budget','fixed_300','fasudil_reference']:
        r=by[label];m=Chem.MolFromMolFile(r['source_sdf'],removeHs=True)
        rows[label]=polar_context(m,r['interface']['interactions'],r.get('metrics',{}).get('burial_sasa',{}).get('values',{}).get('per_atom'))
    base=by['native_final'];guide=by['adaptive_same_budget']
    components={k:dict(native=base['forcefield']['component_strain'][k],guided=guide['forcefield']['component_strain'][k],
        delta=guide['forcefield']['component_strain'][k]-base['forcefield']['component_strain'][k]) for k in base['forcefield']['component_strain']}
    summary=dict(main_pair=dict(base='native_final',guided='adaptive_same_budget',
        changes=context['comparisons']['adaptive_same_budget'],strain_components=components,polar_context=rows),
        measurements=[dict(label=r['label'],pKd=r.get('model_predictions',{}).get('affinity',{}).get('pkd'),
            strain=r['forcefield']['strain_kcal_mol'],vina=r['docking']['vina']['score_only_kcal_mol'],vina_local=r['docking']['vina']['local_optimized_kcal_mol'],
            vinardo=r['docking']['vinardo']['score_only_kcal_mol'],vinardo_local=r['docking']['vinardo']['local_optimized_kcal_mol'],
            interaction_counts=r['interaction_counts']) for r in context['observations']])
    write_json(root/'analysis.json',summary)
    original=root.parent/'molsteer_affinity_validation_20260923/reasoning/t_0.50/StatePacket.json'
    packet=json.loads(original.read_text());write_json(root/'StatePacket.with_outcomes.json',attach(packet,root/'ReaderComparisonContext.json'))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    selected=[base,by['fixed_300'],guide];names=['Native','Fixed eta300','Adaptive']
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    values=[[r['model_predictions']['affinity']['pkd'] for r in selected],[r['forcefield']['strain_kcal_mol'] for r in selected],[r['docking']['vina']['score_only_kcal_mol'] for r in selected]]
    for ax,v,title in zip(axes,values,['Predicted pKd (higher)','MMFF self-strain kcal/mol (lower)','Vina score-only kcal/mol (lower)']):
        ax.bar(names,v,color=['#8c969f','#569db1','#218b77']);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelrotation=15)
        for i,n in enumerate(v):ax.text(i,n,f'{n:.3f}',ha='center',va='bottom' if n>=0 else 'top',fontsize=9)
        ax.margins(y=.18)
    fig.suptitle('Same molecule and matched runtime: modest pose improvement',fontsize=13)
    fig.tight_layout();fig.savefig(root/'paired_evaluation.png');plt.close(fig)
    m=Chem.MolFromMolFile(guide['source_sdf'],removeHs=True);Chem.rdDepictor.Compute2DCoords(m)
    draw=rdMolDraw2D.MolDraw2DCairo(1100,720);opts=draw.drawOptions();opts.addAtomIndices=True;opts.legendFontSize=22
    values=context['comparisons']['adaptive_same_budget']['atom_displacements']
    colors={v['atom_id']:tuple(plt.colormaps['YlOrRd'](v['displacement_angstrom']/.55)[:3]) for v in values}
    draw.DrawMolecule(m,legend='Original tensor-slot IDs; color = world-frame displacement (0 to 0.55 A)',highlightAtoms=list(colors),highlightAtomColors=colors)
    draw.FinishDrawing();(root/'atom_changes.png').write_bytes(draw.GetDrawingText())
    write_json(root/'source_hashes.json',{str(p):file_hash(p) for p in Path(__file__).resolve().parents[1].rglob('*.py') if '__pycache__' not in str(p)})
    print(json.dumps(dict(polar_context=rows,components=components),indent=2))


if __name__=='__main__':main()
