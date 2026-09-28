"""Standalone HTML reporting; no external scripts, invented analysis or hidden data."""
from pathlib import Path
from copy import deepcopy
import html,json,re,io
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from markdown_it import MarkdownIt
from molsteer.common import write_json


def embedded_json(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026').replace('\u2028','\\u2028').replace('\u2029','\\u2029')


def mol_svg(mol,changed=()):
    mol=Chem.Mol(mol);rdDepictor.Compute2DCoords(mol)
    draw=rdMolDraw2D.MolDraw2DSVG(720,400);opts=draw.drawOptions();opts.addAtomIndices=True;opts.clearBackground=False
    draw.DrawMolecule(mol,highlightAtoms=list(changed));draw.FinishDrawing()
    return draw.GetDrawingText()


def pose_svg(reference,candidate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=reference.GetConformer().GetPositions();y=candidate.GetConformer().GetPositions()
    fig,axes=plt.subplots(1,3,figsize=(10,3));fig.patch.set_alpha(0)
    for ax,(i,j),name in zip(axes,[(0,1),(0,2),(1,2)],['XY','XZ','YZ']):
        for mol,z,c in [(reference,x,'#a7b3bf'),(candidate,y,'#14806b')]:
            for b in mol.GetBonds():
                a,d=b.GetBeginAtomIdx(),b.GetEndAtomIdx();ax.plot(z[[a,d],i],z[[a,d],j],color=c,linewidth=1.5)
            ax.scatter(z[:,i],z[:,j],color=c,s=9)
        ax.set_aspect('equal');ax.set_title(name,fontsize=11);ax.set_xlabel('Angstrom');ax.grid(alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();buf=io.StringIO();fig.savefig(buf,format='svg',transparent=True);plt.close(fig)
    return buf.getvalue()


def build(root):
    root=Path(root);read=lambda p:json.loads((root/p).read_text(encoding='utf-8'))
    packet=read('input/StatePacket.json');diagnosis=read('input/DiagnosticReport.json');research=read('research/ResearchPacket.json')
    evaluation=read('evaluation/ReaderComparisonContext.json');cases=[];native=next(v for v in evaluation['observations'] if v['label']=='native_final')
    native_mol=Chem.MolFromMolFile(native['source_sdf'],removeHs=True)
    for row in evaluation['observations']:
        label=row['label'];comparison=evaluation['comparisons'][label];mol=Chem.MolFromMolFile(row['source_sdf'],removeHs=True)
        changed=[i for i,(a,b) in enumerate(zip(native_mol.GetAtoms(),mol.GetAtoms())) if (a.GetAtomicNum(),a.GetFormalCharge())!=(b.GetAtomicNum(),b.GetFormalCharge())]
        cases.append(dict(label=label,smiles=row['smiles'],scores=dict(pkd=row.get('model_predictions',{}).get('affinity',{}).get('pkd'),
            vina=row['docking']['vina']['score_only_kcal_mol'],vinardo=row['docking']['vinardo']['score_only_kcal_mol'],
            strain=row['forcefield']['strain_kcal_mol'],vina_local=row['docking']['vina']['local_optimized_kcal_mol']),
            descriptors=row['descriptors'],interactions=row['interaction_counts'],comparison=comparison,
            posebusters=row.get('metrics',{}).get('posebusters',{}),polar_context=row.get('polar_context'),
            forcefield=row['forcefield'],svg=mol_svg(mol,changed),pose_svg=pose_svg(native_mol,mol),
            sdf=Path(row['source_sdf']).read_text(),source_sha256=row['source_sha256']))
    runs=read('runs.json');traces={}
    for run in runs:
        label=run['label'];path=root/'continuations'/label/'creativity/outcome_trace.jsonl'
        traces[label]=[json.loads(v) for v in path.read_text().splitlines()]
    tests=(root/'tests.log').read_text();validation={p.stem:json.loads(p.read_text()) for p in (root/'validation').glob('*.json')}
    research_view=deepcopy(research)
    for source in research_view['sources']:source.pop('abstract',None)
    research_view['report_view']='Metadata and authored-analysis projection; original immutable ResearchPacket and raw responses are stored alongside the report.'
    data=dict(packet=packet,diagnosis=diagnosis,research=research_view,derivation=read('DerivationSummary.json'),
        programs={p.stem:json.loads(p.read_text()) for p in (root/'programs').glob('*.json')},
        cases=cases,runs=runs,traces=traces,tests=tests,validation=validation,
        provenance=read('execution_provenance.json'),backup=read('BackupSummary.json'),evaluation_audit=read('evaluation_audit.json'))
    md=MarkdownIt('commonmark',{'html':False}).enable('table')
    diagnosis_html=md.render((root/'input/DiagnosticReport.zh.md').read_text(encoding='utf-8'))
    template=(Path(__file__).parent/'templates/research_report.html').read_text(encoding='utf-8')
    output=template.replace('<!--DIAGNOSIS-->',diagnosis_html).replace('/*REPORT_DATA*/',embedded_json(data))
    (root/'report.html').write_text(output,encoding='utf-8')
    count=re.search(r'Ran (\d+) tests',tests)
    checks=dict(test_count=int(count.group(1)) if count else None,unit_tests_passed='\nOK\n' in tests,
        live_checks_passed=all(v.get('all_exact',False) for v in validation.values()),
        expected_sections=['state','diagnosis','thinker','research','monitor','results','validation'],
        all_sections_present=all('id="'+x+'"' in output for x in ['state','diagnosis','thinker','research','monitor','results','validation']),
        no_external_script='src="http' not in output,case_count=len(cases),observation_count=len(packet['observations']),
        includes_full_state_packet=True,includes_full_diagnostic=True,bytes=len(output.encode()))
    if not checks['unit_tests_passed'] or not checks['live_checks_passed'] or not checks['all_sections_present']:raise ValueError('HTML report cannot claim a passing incomplete test suite')
    write_json(root/'report_audit.json',checks)
    return checks
