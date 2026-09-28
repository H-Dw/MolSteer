from ..core import result,Unavailable,need_mol,need_protein,evidence,clean

def compute(ctx):
    if not ctx.options.get('posebusters',False):raise Unavailable('PoseBusters was not enabled; run --posebusters')
    try:
        import posebusters
    except ImportError as exc:
        raise Unavailable('Optional dependency posebusters is missing') from exc
    mol=need_mol(ctx);need_protein(ctx)
    # De novo generation has no reference identity/RMSD constraint.
    bust=posebusters.PoseBusters(config='dock')
    frame=bust.bust(mol_pred=mol,mol_cond=str(ctx.receptor_path),full_report=False)
    if len(frame)!=1:raise RuntimeError('Unexpected PoseBusters record count')
    row={str(k):clean(v) for k,v in frame.iloc[0].items()}
    failures=[k for k,v in row.items() if v is False]
    unknown=[k for k,v in row.items() if v is None]
    return result({'config':'dock','version':posebusters.__version__,'checks':row,'failed_checks':failures,
                   'all_reported_checks_pass':not failures if not unknown else None,'unknown_checks':unknown},
                  evidence=[evidence(message='PoseBusters check failed',check=k,localization='Molecule-level; consult local geometry metrics') for k in failures],
                  status='partial' if unknown else 'ok',method='Official PoseBusters dock configuration, no mol_true supplied',
                  notes=['No redocking RMSD or identity comparison to the chemically different reference ligand. Aggregate pass applies only to returned dock checks.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('posebusters')
