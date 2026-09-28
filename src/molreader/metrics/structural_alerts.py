from rdkit.Chem import FilterCatalog
from ..core import result,need_mol,evidence

def compute(ctx):
    mol=need_mol(ctx);rows=[]
    for name in ("PAINS","BRENK"):
        params=FilterCatalog.FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs,name))
        catalog=FilterCatalog.FilterCatalog(params)
        for entry in catalog.GetMatches(mol):
            sites=set()
            for match in entry.GetFilterMatches(mol):
                sites.add(tuple(sorted({int(pair.target) for pair in match.atomPairs})))
            for indices in sorted(sites):
                semantics={'Oxygen-nitrogen_single_bond':'Non-ring O/N to O/N match; may be N-N and does not require oxygen',
                           'quaternary_nitrogen_3':'A positively charged N with two double bonds; not necessarily four heavy-atom neighbors',
                           'diazo_group':'Broad non-ring N=N match, not a complete functional-group assignment'}.get(entry.GetDescription(),'Substructure screening match, not an established chemical defect')
                rows.append(evidence(ctx.ids(indices),message="Structural screening alert",catalog=name,alert=entry.GetDescription(),
                                     interpretation=semantics))
    return result({"alert_count":len(rows),'matched_rule_count':len({(r['catalog'],r['alert']) for r in rows})},evidence=rows,method="RDKit PAINS and BRENK substructure filters; separate spatial matches",
                  notes=["Alerts are screening hypotheses. They do not establish toxicity, assay interference or synthetic infeasibility."])

if __name__ == "__main__":
    from ..cli import metric_main
    metric_main("structural_alerts")
