"""Every public module supports compute(context) and python -m <module>."""
METRICS = [
    "tensor_integrity", "atom_confidence", "bond_confidence", "charge_confidence", "decode_consistency",
    "atom_inventory", "formal_charge", "connectivity", "valence", "chemistry_context", "bond_lengths", "bond_angles", "mmff_local_geometry",
    "torsions", "ring_planarity", "double_bond_planarity", "intramolecular_clashes", "shape", "stereochemistry",
    "molecular_weight", "logp", "tpsa", "hbd", "hba", "rotatable_bonds", "qed", "fraction_csp3", "sa_score",
    "structural_alerts", "mmff_energy", "mmff_strain", "protein_clashes", "protein_contacts",
    "hydrogen_bond_candidates", "hydrophobic_contacts", "salt_bridge_candidates", "aromatic_contacts",
    "burial_sasa", "affinity", "trajectory_change", "posebusters", "prolif", "vina_score"
]
