"""Prepared score-only oracle. No docking optimization or invented derivative."""
from pathlib import Path
import numpy as np
from molsteer.common import file_hash
from .chemistry import decode_endpoint,signature


class VinaOracle:
    def __init__(self,config,observables,vocabulary):
        from .flowr import snapshot_rng,restore_rng
        rng=snapshot_rng()
        try:self._initialize(config,observables,vocabulary)
        finally:restore_rng(rng)

    def _initialize(self,config,observables,vocabulary):
        from vina import Vina
        path=Path(config['receptor_pdbqt'])
        if file_hash(path)!=config['receptor_sha256']:raise ValueError('Prepared receptor changed')
        self.engine=Vina(sf_name='vina',cpu=2,seed=20260923,verbosity=0)
        self.engine.set_receptor(str(path));self.engine.compute_vina_maps(center=config['center'],box_size=config['box_size'])
        self.observables=observables;self.vocabulary=vocabulary;self.cache={};self.calls=0

    def score(self,pred):
        from .flowr import snapshot_rng,restore_rng
        rng=snapshot_rng()
        try:return self._score(pred)
        finally:restore_rng(rng)

    def _score(self,pred):
        from meeko import MoleculePreparation,PDBQTWriterLegacy
        mol=decode_endpoint(pred,self.vocabulary);xyz=pred['coords'].detach().cpu().numpy()
        key=(signature(mol),xyz.tobytes())
        if key in self.cache:return self.cache[key]
        data=self.observables.mmff.evaluate(mol,xyz)
        setups=MoleculePreparation().prepare(data['prepared'])
        if len(setups)!=1:raise ValueError('Ambiguous scoring microstate')
        text,ok,error=PDBQTWriterLegacy.write_string(setups[0])
        if not ok:raise ValueError('Scoring preparation failed: '+error)
        self.engine.set_ligand_from_string(text)
        energy=float(self.engine.score()[0]);self.calls+=1
        if not np.isfinite(energy):raise ValueError('Nonfinite independent score')
        self.cache[key]=dict(vina=energy,strain=float(data['strain']),
            mode='score_only',coordinate_rounding_angstrom=.001,
            interpretation='Independent of the differentiable reward, but an uncalibrated empirical score under one prepared state')
        return self.cache[key]
