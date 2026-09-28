import numpy as np
from ..core import result,evidence
from ..io import array

def compute(ctx):
    if ctx.view != 'prediction':
        return result(status='not_applicable', notes=['Shared head affinity observation is recorded once in prediction view.'])
    source=ctx.bundles['prediction'].get('affinity',{})
    values={};flags=[]
    for key in ('pic50','pki','pkd','pec50'):
        a=array(source[key]).reshape(-1) if key in source else np.array([])
        value=float(a[0]) if len(a)==1 and np.isfinite(a[0]) else None
        values[key]=value
        recorded=ctx.metadata.get('affinity',{}).get(key) if isinstance(ctx.metadata.get('affinity'),dict) else None
        if isinstance(recorded,list) and len(recorded)==1:recorded=recorded[0]
        if value is None:
            flags.append(evidence(message=f'{key} is missing, non-scalar or nonfinite',severity='warning'))
        elif recorded is not None and not np.isclose(float(recorded),value,atol=1e-5,rtol=0):
            flags.append(evidence(message=f'{key} tensor/JSON mismatch',severity='high',tensor_value=value,json_value=recorded))
    return result(values,evidence=flags,status='partial' if any(v is None for v in values.values()) else 'ok',
                  units={k:'predicted p-activity (-log10 molar endpoint), uncalibrated' for k in values},
                  method='Preserve four independent head outputs and verify JSON correspondence',
                  notes=['All views share the same head observation conditioned on X_t. Four endpoints are not ensemble replicates; their spread is not uncertainty. No binding free energy or experimental potency inferred.'])

if __name__ == '__main__':
    from ..cli import metric_main
    metric_main('affinity')
