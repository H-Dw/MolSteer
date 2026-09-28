"""Strict extensions to the existing StatePacket contract and reward provenance."""
import json
from pathlib import Path
import jsonschema
import molreader
from molreader.packet import validate_packet
from molsteer.common import digest

FACT = {'type':'object', 'additionalProperties':False,
        'required':['value','status','source','reason'],
        'properties':{'value':{},'status':{'enum':['observed','derived','declared','unavailable']},'source':{},'reason':{}},
        'if':{'properties':{'status':{'const':'unavailable'}}}, 'then':{'properties':{'value':{'type':'null'}}}}


def state_schema():
    schema=json.loads((Path(molreader.__file__).parent/'schemas/state_packet.schema.json').read_text())
    schema['title']='MolSteer enriched StatePacket'
    schema['required'] += ['parent_packet_id','steering']
    schema['properties']['parent_packet_id']={'type':'string','pattern':'^sp_[0-9a-f]{24}$'}
    schema['properties']['steering']={'type':'object','additionalProperties':False,
        'required':['generation','chemical_readiness','objectives','coordinate_snapshots','receptor_atoms','graph_signatures','categorical_uncertainty','scope'],
        'properties':{k:{'type':v} for k,v in [('generation','object'),('chemical_readiness','object'),('objectives','object'),
            ('coordinate_snapshots','object'),('receptor_atoms','array'),('graph_signatures','object'),('categorical_uncertainty','object'),('scope','string')]}}
    schema['properties']['steering']['properties']['outcome_context']=FACT
    return schema


def validate_enriched(packet):
    validate_packet(packet)
    jsonschema.validate(packet,state_schema())
    expected='sp_'+digest(dict(parent=packet['parent_packet_id'],steering=packet['steering']))[:24]
    if packet['packet_id']!=expected:
        raise ValueError('Enrichment content digest mismatch')
    def facts(obj):
        if isinstance(obj,dict):
            if 'status' in obj and 'value' in obj:
                jsonschema.validate(obj,FACT)
            else:
                for value in obj.values():facts(value)
    for key in ['generation','chemical_readiness','objectives']:
        facts(packet['steering'][key])
    for view,snap in packet['steering']['coordinate_snapshots'].items():
        if snap['atom_ids']!=packet['representations'][view]['original_atom_ids']:
            raise ValueError('Coordinate/representation slot mismatch')
        if len(snap['coords_angstrom'])!=len(snap['atom_ids']) or any(len(x)!=3 for x in snap['coords_angstrom']):
            raise ValueError('Invalid coordinates')
        if digest(snap['coords_angstrom'])!=snap['coordinate_hash']:
            raise ValueError('Coordinate content digest mismatch')
    return True


def validate_reward_schema(spec):
    jsonschema.validate(spec,{'type':'object','required':['kind','schema_version','reward_id','packet_id','identity','terms',
        'reward_groups','retrieval','knowledge_source','graph_signatures','coordinate_hashes','runtime_execution'],
        'properties':{'kind':{'const':'RewardSpec'},'schema_version':{'const':'1.0.0'},
            'reward_id':{'type':'string','pattern':'^rw_[0-9a-f]{24}$'},'terms':{'type':'array','items':{'type':'object',
                'required':['term_id','view','family','atom_ids','hypothesis_atom_ids','lower','upper','scale','weight','function_id','reference_evidence_id'],
                'properties':{'view':{'enum':['state','prediction','sdf']},'family':{'enum':['flat_bottom_distance','flat_bottom_angle','minimum_distance']},
                    'scale':{'type':'number','exclusiveMinimum':0},'weight':{'type':'number','minimum':0}}}}}})
    body={k:v for k,v in spec.items() if k!='reward_id'}
    if spec['reward_id']!='rw_'+digest(body)[:24]:
        raise ValueError('Reward content digest mismatch')
