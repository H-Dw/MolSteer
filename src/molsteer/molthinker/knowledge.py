"""Auditable retrieval over table rows; document text is data, never executable."""
import math
import re
from collections import Counter
from pathlib import Path
from molsteer.common import file_hash

# Reviewed semantics for the supplied table; unknown function names are rejected.
ROUTES = [
 ('G01', 'flat-bottom', 'local_geometry', 'coordinate_energy', ['local_reference', 'declared_graph']),
 ('G02', '成对最小距离', 'protein_clashes intramolecular_sterics', 'coordinate_energy', ['coordinates', 'radii', 'frame']),
 ('G03', '锚点', 'anchor', 'coordinate_energy', ['anchor_reference', 'movable_mask']),
 ('G04', '方向一致', 'direction pharmacophore', 'coordinate_energy', ['direction_reference', 'feature_typing']),
 ('G05', '立体化学', 'stereochemistry planarity', 'coordinate_energy', ['stereo_or_plane_reference', 'declared_graph']),
 ('G06', '形状／', 'shape volume', 'coordinate_energy', ['shape_reference', 'alignment']),
 ('G07', '局部环境', 'environment', 'coordinate_energy', ['descriptor_model', 'reference_library']),
 ('P01', 'MMFF94 分子', 'conformational_strain local_geometry', 'coordinate_energy', ['stable_graph', 'protonation', 'mmff_applicability']),
 ('P02', '跨界面 van', 'protein_clashes interface', 'coordinate_energy', ['ligand_and_pocket_ff_types']),
 ('P03', '跨界面静电', 'electrostatics', 'coordinate_energy', ['partial_charges', 'dielectric_model']),
 ('P04', 'xTB', 'force stability', 'coordinate_energy', ['xtb_oracle', 'stable_graph', 'protonation', 'budget']),
 ('P05', 'Vina /', 'binding affinity', 'coordinate_energy_or_population_score', ['vina_preparation', 'scoring_backend']),
 ('P06', '可微 SASA', 'sasa burial', 'coordinate_energy', ['group_area_target', 'differentiable_sasa']),
 ('P07', 'ESP 表面', 'electrostatics shape', 'coordinate_energy', ['partial_charges', 'aligned_reference_esp']),
 ('S01', '目标值型', 'property_target', 'population_weight', ['target_value_tolerance', 'live_population']),
 ('S02', 'softmax /', 'affinity property_maximize', 'population_weight', ['score_direction', 'temperature', 'live_population']),
 ('S03', '目标/脱靶', 'selectivity', 'population_score', ['matched_off_target_scores', 'weights']),
 ('S04', 'SPSA', 'blackbox gradient', 'gradient_estimator', ['reevaluable_oracle', 'perturbable_dof', 'budget']),
 ('S05', 'Pareto', 'multiobjective diversity', 'selection_strategy', ['objectives', 'population', 'distance_metric']),
 ('S06', '批次完整性', 'validity depth', 'hyperparameter_strategy', ['calibrated_depth_batches', 'parent_identity']),
 ('S07', '三群', 'graph_connectivity chemical_validity', 'population_strategy', ['fragment_target', 'population', 'checkers']),
]
ENGLISH_NAMES = [
 'Flat-bottom interval penalty','Pairwise minimum-distance penalty','Anchor quadratic potential',
 'Directional agreement penalty','Stereochemical and planar geometry potential','Shape/volume occupancy potential',
 'Local environment similarity kernel','MMFF94 intramolecular energy','Interfacial van der Waals potential',
 'Interfacial electrostatic potential','xTB force-norm loss','Vina / torchvina total score',
 'Differentiable SASA target potential','ESP surface similarity potential','Target-value Gaussian reward',
 'Softmax / tilted importance weights','On-target/off-target selectivity reward','SPSA gradient estimation',
 'Pareto ranking and structural diversity','Depth-calibrated batch integrity score','Three-population validity penalties']


def tokenize(text):
    # English words plus Chinese bigrams avoid requiring a tokenizer service.
    words = re.findall(r'[a-z0-9_]+', text.lower())
    for chunk in re.findall(r'[\u4e00-\u9fff]+', text):
        words.extend(chunk[i:i+2] for i in range(max(1, len(chunk)-1)))
    return words


class KnowledgeBase:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = file_hash(path)
        self.entries = []
        section = ''
        for line_no, line in enumerate(self.path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if line.startswith('## '):
                section = line[3:]
            if not line.startswith('|'):
                continue
            cells = [x.strip() for x in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
            if len(cells) != 6 or cells[0] == '函数名称' or all(set(c) <= {'-', ':'} for c in cells):
                continue
            matches = [r for r in ROUTES if r[1] in cells[0]]
            if len(matches) != 1:
                raise ValueError(f'Unreviewed or ambiguous knowledge row at line {line_no}')
            rid, _, tags, kind, needs = matches[0]
            self.entries.append(dict(function_id=rid, name=cells[0], name_en=ENGLISH_NAMES[ROUTES.index(matches[0])], sources=cells[1], target=cells[2],
                formula=cells[3], variables=cells[4], gradient_target=cells[5], section=section,
                tags=tags.split(), role=kind, prerequisites=needs, source=dict(file=self.path.name,
                sha256=self.sha256, line=line_no), original_row=line))
        if len(self.entries) != 21 or len({e['function_id'] for e in self.entries}) != 21:
            raise ValueError('Expected all 21 distinct reviewed knowledge functions')

    def retrieve(self, categories):
        query = ' '.join(sorted(set(categories)))
        terms = set(tokenize(query))
        docs = [Counter(tokenize(' '.join(e['tags']) + ' ' + e['name'] + ' ' + e['target'])) for e in self.entries]
        avgdl = sum(sum(d.values()) for d in docs) / len(docs)
        rows = []
        for e, d in zip(self.entries, docs):
            score = 0.
            for t in terms:
                df = sum(t in doc for doc in docs)
                idf = math.log(1 + (len(docs)-df+.5)/(df+.5))
                tf = d[t]
                score += idf * tf * 2.2 / (tf + 1.2*(.25+.75*sum(d.values())/avgdl))
            matched = sorted(set(categories) & set(e['tags']))
            # Prefer a directly localized observable over a global energy for the same local defect.
            locality_bonus=3. if e['function_id']=='G01' and 'local_geometry' in categories else 0.
            rows.append(dict(**e, retrieval_score=score + 5*len(matched)+locality_bonus,
                retrieval_components=dict(lexical_bm25=score,category_bonus=5*len(matched),locality_bonus=locality_bonus), matched_categories=matched))
        return sorted(rows, key=lambda r: (-r['retrieval_score'], r['function_id']))
