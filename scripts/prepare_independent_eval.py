"""Prepare a documented full receptor in the saved pocket coordinate frame."""
import argparse
import json
import subprocess
import shutil
import sys
from pathlib import Path
import numpy as np
from molsteer.common import write_json,file_hash


def records(path):
    result={}
    for line in Path(path).read_text().splitlines():
        if line[:6].strip() not in ['ATOM','HETATM']:continue
        if line[76:78].strip()=='H':continue
        key=(line[21],line[22:27].strip(),line[12:16].strip())
        result[key]=(line,np.array([float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]]))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--receptor-preparer',help='Path to mk_prepare_receptor.py (defaults to the current Python environment or PATH)')
    a=p.parse_args()
    environment_preparer = Path(sys.executable).with_name('mk_prepare_receptor.py')
    preparer = a.receptor_preparer or (str(environment_preparer) if environment_preparer.is_file()
        else shutil.which('mk_prepare_receptor.py'))
    if not preparer:
        p.error('mk_prepare_receptor.py was not found; install Meeko in the active environment or pass --receptor-preparer')
    root=Path(a.model_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    inputs=root/'output/crossdocked_100target_stage_test_exact_20260923/inputs'
    full=inputs/'5i0b_A.pdb';pocket=inputs/'5i0b_A_rec_5vef_m77_lig_tt_min_0_pocket10.pdb'
    f=records(full);q=records(pocket);shared=sorted(set(f)&set(q))
    x=np.array([f[k][1] for k in shared]);y=np.array([q[k][1] for k in shared]);xc=x.mean(0);yc=y.mean(0)
    u,s,vt=np.linalg.svd((x-xc).T@(y-yc));d=np.diag([1,1,np.linalg.det(u@vt)]);rot=u@d@vt
    error=np.linalg.norm((x-xc)@rot+yc-y,axis=1)
    written=[];removed=[]
    for key,(line,xyz) in f.items():
        if line[17:20]=='67U':removed.append(dict(key=key,reason='original crystallographic ligand removed from receptor'));continue
        v=(xyz-xc)@rot+yc
        written.append(line[:30]+''.join(f'{value:8.3f}' for value in v)+line[54:])
    receptor=out/'receptor_aligned.pdb';receptor.write_text('\n'.join(written)+'\nEND\n')
    write_json(out/'receptor_alignment.json',dict(full_source=str(full),full_sha256=file_hash(full),pocket_source=str(pocket),pocket_sha256=file_hash(pocket),
        matched_atoms=len(shared),rmsd_angstrom=float(np.sqrt(np.mean(error**2))),max_error_angstrom=float(error.max()),
        rotation=rot.tolist(),source_center=xc.tolist(),target_center=yc.tolist(),removed=removed,
        scope='Rigid alignment only; shared pocket coordinates are not overwritten or minimized'))
    command=[str(preparer),'--read_pdb',str(receptor),'-o',str(out/'receptor'),'-p','-j','--write_pdb',str(out/'receptor_prepared.pdb')]
    run=subprocess.run(command,capture_output=True,text=True)
    (out/'receptor_preparation.log').write_text(run.stdout+'\n'+run.stderr)
    write_json(out/'receptor_preparation_command.json',dict(command=command,returncode=run.returncode))
    print(json.dumps(dict(alignment_rmsd=float(np.sqrt(np.mean(error**2))),max_error=float(error.max()),returncode=run.returncode)))
    if run.returncode:print((run.stdout+'\n'+run.stderr)[-7000:]);raise SystemExit(run.returncode)


if __name__=='__main__':main()
