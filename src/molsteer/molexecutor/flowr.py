"""FLOWR.ROOT bridge: exact prefix replay, differentiable endpoint, native integrator.

The external stage runner is a trusted, hash-recorded preprocessing/serialization
adapter. Its sampling implementation is neither overwritten nor monkey-patched.
"""
import importlib.util
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import torch


def tree_map(fn, obj):
    if torch.is_tensor(obj): return fn(obj)
    if isinstance(obj,dict) or hasattr(obj,'items'): return {k:tree_map(fn,v) for k,v in obj.items()}
    if isinstance(obj,list): return [tree_map(fn,v) for v in obj]
    if isinstance(obj,tuple): return tuple(tree_map(fn,v) for v in obj)
    return obj


def detached(obj):
    return tree_map(lambda x:x.detach(),obj)


def snapshot_rng():
    ns=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
        python=random.getstate(),numpy=[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]])


def restore_rng(rng):
    torch.set_rng_state(rng['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    random.setstate(rng['python'])
    ns=rng['numpy'];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))


class FlowrRootAdapter:
    def __init__(self, config):
        self.config=config
        root=Path(config['model_root']).resolve()
        sys.path.insert(0,str(root))
        path=Path(config['stage_runner'])
        self.source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        module=importlib.util.spec_from_file_location('molsteer_flowr_stage',path)
        runner=importlib.util.module_from_spec(module);module.loader.exec_module(runner)
        self.runner=runner
        oldargv=sys.argv[:]
        try:
            self.args=runner.build_args(config['receptor'],config['reference_ligand'],config['output'],config['checkpoint'])
        finally:
            sys.argv=oldargv
        self.args.mp_index=config.get('gpu',0)
        self.device=torch.device(f"cuda:{config.get('gpu',0)}")
        torch.cuda.set_device(self.device)
        self.precision=config.get('float32_matmul_precision','highest')
        torch.set_float32_matmul_precision(self.precision)
        torch.backends.cuda.matmul.allow_tf32=self.precision!='highest'
        model,hparams,*vocabs=runner.load_model(self.args)
        self.model=model.to(self.device).eval().requires_grad_(False)
        self.hparams=hparams
        transform,interpolant=runner.util.load_util(self.args,hparams,*vocabs[:4])
        system=runner.util.load_data_from_pdb(self.args,remove_hs=hparams['remove_hs'],
            remove_aromaticity=hparams['remove_aromaticity'],ligand_idx=0,chain_id='A',canonicalize_conformer=False)
        dataset=runner.get_dataset(system,transform,vocabs[0],interpolant,self.args,hparams)
        loader=runner.util.get_dataloader(self.args,dataset,interpolant,iter=0)
        batches=list(loader)
        if len(batches)!=1:
            raise ValueError('This resume recipe requires a single original batch; supply another adapter recipe for multiple batches')
        prior,data,_,_=batches[0]
        self.prior=model.builder.extract_ligand_from_complex(prior)
        for k in ['interactions','fragment_mask','fragment_mode']: self.prior[k]=prior[k]
        self.prior=tree_map(lambda x:x.to(self.device),self.prior)
        self.pocket=model.builder.extract_pocket_from_complex(data)
        self.pocket.update(interactions=data['interactions'],complex=data['complex'])
        self.pocket=tree_map(lambda x:x.to(self.device),self.pocket)
        self.batch=prior['coords'].shape[0]
        self.index=config['ligand_index']
        self.curr=tree_map(lambda x:x.clone(),self.prior)
        self.cond=dict(coords=self.prior['coords'],atomics=torch.zeros_like(self.prior['atomics']),bonds=torch.zeros_like(self.prior['bonds']))
        if model.sc_charges:self.cond['charges']=torch.zeros_like(self.prior['charges'])
        # Reject unsupported inpainting rather than silently breaking fixed atoms.
        if model.inpainting_mode or model.graph_inpainting or model._inpaint_self_condition:
            raise ValueError('FLOWR resume recipe currently requires the unmasked generator; use a mask-aware model adapter for inpainting')
        self.times=[torch.zeros(self.batch,device=self.device) for _ in range(3)]
        self.grid=torch.linspace(0,1,self.args.integration_steps+1)
        with torch.no_grad():
            self.equis,self.invs=model.gen.get_pocket_encoding(self.pocket['coords'],self.pocket['atom_names'],
                pocket_atom_charges=self.pocket['charges'].argmax(-1),pocket_bond_types=self.pocket['bonds'].argmax(-1),
                pocket_res_types=self.pocket['res_names'],pocket_atom_mask=self.pocket['mask'])
        if not self.curr['mask'][self.index].bool().all():
            raise ValueError('Resume recipe requires active unpadded target slots; remap padded targets explicitly')
        self.step_index=0
        self.guidance_state={}

    def predict(self, coordinates=None, state=None, times=None, cond=None):
        state=dict(self.curr if state is None else state)
        if coordinates is not None:state['coords']=coordinates
        out=self.model(state,self.pocket,self.times if times is None else times,training=False,
            cond_batch=(self.cond if cond is None else cond) if self.model.self_condition else None,
            pocket_equis=self.equis,pocket_invs=self.invs)
        return self.model._get_predictions(out)

    def endpoint(self, predicted):
        world=self.runner.world_prediction(self.model,predicted,self.pocket)
        result={k:v[self.index] for k,v in world.items() if torch.is_tensor(v) and v.shape[0]==self.batch}
        # Keep live, same-forward affinity tensors. Serialized scalar scores cannot
        # supply gradients, and the heads need not depend on the output coordinates.
        if 'affinity' in predicted:
            result['affinity']={k:v[self.index].reshape(()) for k,v in predicted['affinity'].items()}
        return result

    def world_state_coordinates(self, coordinates):
        """Map the live X_t batch to the same receptor-world frame as StatePacket."""
        world=self.runner.world_prediction(self.model,
            {'coords':coordinates,'mask':self.curr['mask']},self.pocket)
        return world['coords'][self.index]

    def inject(self, gradient, step_size):
        # dX/dt = native velocity + strength * grad_X R. Positive forward time.
        return step_size*gradient

    def describe_dynamics(self):
        """Host facts for mathematical design; a live derivative still needs preflight."""
        return dict(kind='ModelDynamicsContext',model_type='FLOWR.ROOT flow matching',
                    time_convention='Forward integration on the declared grid, t=0 to t=1',
                    prediction_parameterization='model._get_predictions returns endpoint coordinates and categorical heads',
                    coordinate_mapping='runner.world_prediction; checkpoint coord_scale and receptor translation',
                    editable_atom_ids=self.config.get('editable_atom_ids'),live_derivative='not_run',
                    injection_convention='inject(d,dt)=dt*d; d is a minimizing control direction in model coordinates',
                    source='FlowrRootAdapter with stage-runner SHA256 '+self.source_hash)

    def native_step(self, predicted, cond, step_size):
        with torch.no_grad():
            self.curr=detached(self.model.integrator.step(self.curr,detached(predicted),self.prior,self.times,step_size))
            self.cond=detached(cond)
            self.times=self.model._update_times(self.times,step_size)
        self.step_index+=1

    def replay_to(self, step_index):
        while self.step_index<step_index:
            with torch.no_grad():pred,cond=self.predict()
            dt=self.grid[self.step_index+1]-self.grid[self.step_index]
            self.native_step(pred,cond,dt)
            # Preserve extra stage heads from the original sampling procedure.
            if self.step_index in (25,50,75):
                with torch.no_grad():self.predict()
        saved=torch.load(Path(self.config['saved_stage'])/'state.pt',weights_only=True,map_location='cpu')
        differences={k:float((self.curr[k][self.index:self.index+1].cpu()-v).abs().max())
                     for k,v in saved.items() if torch.is_tensor(v)}
        with torch.no_grad():pred,_=self.predict()
        original=torch.load(Path(self.config['saved_stage'])/'structure_affinity_prediction.pt',weights_only=True,map_location='cpu')
        head={k:float((pred[k][self.index:self.index+1].cpu()-v).abs().max())
              for k,v in original.items() if torch.is_tensor(v)}
        check=dict(state_max_abs=differences,head_max_abs=head,step_index=self.step_index)
        if max(differences.values())>self.config.get('replay_tolerance',1e-5):
            if self.config.get('missing_resume_context_policy')!='replay_context_restore_exact_saved_state':
                raise ValueError(f'Prefix replay mismatch: {check}')
            # Original snapshots lack SC/RNG. Never claim reconstructed context is
            # the historical context. All paired arms share this same explicit one.
            for j in range(self.batch):
                path=Path(self.config['saved_stage']).parents[1]/f'ligand_{j:03d}'/Path(self.config['saved_stage']).name/'state.pt'
                row=torch.load(path,weights_only=True,map_location=self.device)
                for k,v in row.items():
                    if torch.is_tensor(v):self.curr[k][j:j+1]=v
            with torch.no_grad():pred,_=self.predict()
            check['restored_state_max_abs']={k:float((self.curr[k][self.index:self.index+1].cpu()-v).abs().max())
                for k,v in saved.items() if torch.is_tensor(v)}
            check['restored_head_max_abs']={k:float((pred[k][self.index:self.index+1].cpu()-v).abs().max())
                for k,v in original.items() if torch.is_tensor(v)}
            check['resume_fidelity']='Exact saved state; replay-reconstructed self-conditioning and RNG. Not an exact historical continuation.'
        else:
            check['resume_fidelity']='Prefix state matches tolerance; context reconstructed by replay, original SC/RNG unavailable for independent verification'
        return check

    def checkpoint(self):
        return tree_map(lambda x:x.detach().cpu().clone(),dict(curr=self.curr,cond=self.cond,
            prior=self.prior,times=self.times,step_index=self.step_index,rng=snapshot_rng(),source_hash=self.source_hash,
            model_checkpoint=self.config['checkpoint'],config=self.config,guidance_state=self.guidance_state,precision=self.precision,
            cuda_device_index=self.device.index,
            pocket_tensors={k:v for k,v in self.pocket.items() if k!='complex'},
            pocket_com=torch.stack([torch.as_tensor(s.com) for s in self.pocket['complex']]),
            pocket_equis=self.equis,pocket_invs=self.invs,grid=self.grid,
            resume_fidelity=getattr(self,'resume_fidelity','Legacy runtime; inspect initial checkpoint provenance')))

    def restore(self, checkpoint):
        def same_file(first, second):
            if str(first)==str(second):return True
            try:return Path(first).resolve(strict=True)==Path(second).resolve(strict=True)
            except (OSError,ValueError):return False

        if checkpoint['source_hash']!=self.source_hash or not same_file(
                checkpoint['model_checkpoint'],self.config['checkpoint']):
            raise ValueError('Resume source/checkpoint mismatch')
        native=checkpoint.get('format')=='flowr_root_live_runtime'
        keys=['receptor','reference_ligand','target_id'] if native else ['receptor','reference_ligand','target_id','ligand_index','saved_stage']
        for key in keys:
            first,second=checkpoint['config'][key],self.config[key]
            equal=same_file(first,second) if key in ('receptor','reference_ligand','saved_stage') else first==second
            if not equal:raise ValueError('Resume context mismatch: '+key)
        if checkpoint.get('precision') and checkpoint['precision']!=self.precision:
            raise ValueError('Resume numerical precision mismatch')
        if 'cuda_device_index' in checkpoint and checkpoint['cuda_device_index']!=self.device.index:
            raise ValueError('Resume CUDA generator device differs; use the recorded device or an explicit RNG remapping adapter')
        if native:
            if checkpoint['config']['integration_steps']!=self.args.integration_steps:
                raise ValueError('Resume integration grid mismatch')
            if bool(self.model.self_condition)!=checkpoint['self_condition_enabled']:
                raise ValueError('Resume self-conditioning configuration mismatch')
            if not hasattr(self,'model_checkpoint_sha256'):
                self.model_checkpoint_sha256=hashlib.sha256(Path(self.config['checkpoint']).read_bytes()).hexdigest()
            if self.model_checkpoint_sha256!=checkpoint['model_checkpoint_sha256']:
                raise ValueError('Resume model checkpoint content mismatch')
        if 'pocket_tensors' in checkpoint:
            systems=self.pocket['complex']
            com=torch.stack([torch.as_tensor(s.com) for s in systems]).cpu()
            if not torch.equal(com,checkpoint['pocket_com'].cpu()):
                raise ValueError('Reconstructed receptor coordinate frame differs from saved frame')
            self.pocket=tree_map(lambda x:x.to(self.device),checkpoint['pocket_tensors'])
            self.pocket['complex']=systems
            self.equis=checkpoint['pocket_equis'].to(self.device)
            self.invs=checkpoint['pocket_invs'].to(self.device)
            if not torch.equal(self.grid,checkpoint['grid'].cpu()):raise ValueError('Resume time grid mismatch')
        for name in ['curr','cond','prior','times']:
            setattr(self,name,tree_map(lambda x:x.to(self.device),checkpoint[name]))
        if self.curr['coords'].shape[0]!=self.batch:raise ValueError('Resume batch size mismatch')
        self.step_index=checkpoint['step_index']
        self.guidance_state=tree_map(lambda x:x.to(self.device),checkpoint.get('guidance_state',{}))
        self.resume_fidelity=checkpoint.get('resume_fidelity','Legacy runtime; inspect initial checkpoint provenance')
        restore_rng(checkpoint['rng'])

    def save_stage(self, output, name, t):
        times=self.model._update_times(self.times,-1e-4) if name=='final' else self.times
        with torch.no_grad():pred,_=self.predict(times=times)
        self.runner.write_stage(Path(output)/self.config['target_id'],self.index,name,t,
            self.runner.slice_batch(self.curr,self.index,self.batch),self.runner.slice_batch(pred,self.index,self.batch),
            self.model,self.pocket,dict(molsteer_guided=getattr(self,'execution_arm','unguided')!='unguided',
                guidance_arm=getattr(self,'execution_arm','unguided'),step_index=self.step_index))
        return pred
