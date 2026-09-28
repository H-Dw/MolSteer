from pathlib import Path
import json
import numpy as np
from ..core import result,Unavailable,sha256

def compute(ctx):
    path=ctx.options.get('vina_result')
    if not path:raise Unavailable('No provenance-verified Vina score result provided; affinity heads are not Vina')
    data=json.loads(Path(path).read_text())
    if data.get('stage_identity')!=ctx.identity:raise ValueError('Vina result stage mismatch')
    if data.get('view')!=ctx.view:raise ValueError('Vina result representation mismatch')
    if data.get('ligand_sdf_sha256')!=ctx.sources.get('ligand.sdf',{}).get('sha256'):raise ValueError('Vina pose hash mismatch')
    if ctx.view!='sdf':raise Unavailable('Imported Vina score currently supports verified SDF view only')
    if not ctx.receptor_path or data.get('receptor_sha256')!=sha256(ctx.receptor_path):raise ValueError('Vina receptor hash mismatch')
    if data.get('mode')!='score_only' or data.get('coordinates_preserved') is not True:raise ValueError('Only unchanged-pose score_only output is supported')
    value=float(data['score_kcal_mol'])
    if not np.isfinite(value):raise ValueError('Nonfinite Vina score')
    if not data.get('vina_version') or not data.get('preparation'):raise ValueError('Vina version and preparation provenance required')
    return result({'score_kcal_mol':value,'vina_version':data['vina_version'],'preparation':data['preparation'],'source_sha256':sha256(path)},
                  units={'score':'kcal/mol (empirical Vina score)'},method='Verified external Vina score-only import',
                  notes=['Empirical score, not measured binding affinity. Receptor/ligand PDBQT preparation is an external prerequisite.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('vina_score')
