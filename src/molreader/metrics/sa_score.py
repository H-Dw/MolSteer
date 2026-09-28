from ..core import result, need_mol, Unavailable

def compute(ctx):
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError as exc:
        raise Unavailable('RDKit Contrib SA_Score and fragment model are required') from exc
    value = float(sascorer.calculateScore(need_mol(ctx)))
    return result({'sa_score': value}, units={'sa_score': 'heuristic 1 easy to 10 difficult'},
                  method='Ertl-Schuffenhauer SA score using RDKit bundled fragment model',
                  notes=['Synthetic-accessibility proxy, not a retrosynthesis route or proof of feasibility.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('sa_score')
