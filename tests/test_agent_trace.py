import json
import pytest
from molsteer.agents.trace import append_trace, save_checkpoint, load_checkpoint


def test_checkpoint_contains_artifacts_and_checked_tool_observations(tmp_path):
    state={'run_id':'trace_test','packet':{'packet_id':'packet'},'plan':{'steps':['inspect']},'config_sha256':'c','skill_sha256':'s'}
    append_trace(state,node='thinker',kind='tool',summary='Evidence retrieved',input_value={'query':'geometry'},output={'rows':[1,2],'api_key':'hidden','thinking':'private'})
    path=save_checkpoint(state,tmp_path)
    result=load_checkpoint(path)
    assert result['artifacts']['plan']['steps']==['inspect']
    assert result['trace'][0]['input']['query']=='geometry'
    assert result['trace'][0]['output']['rows']==[1,2]
    assert result['automatic_resume_supported'] is False
    assert 'hidden' not in path.read_text(encoding='utf-8')
    result['artifacts']['plan']['steps']=['tampered']
    path.write_text(json.dumps(result),encoding='utf-8')
    with pytest.raises(ValueError,match='digest'):
        load_checkpoint(path)


def test_nonfinite_observation_is_recorded_without_invalid_json(tmp_path):
    state={'run_id':'nonfinite'}
    append_trace(state,node='executor',kind='observation',summary='Invalid measurement',output={'metric':float('nan')})
    assert load_checkpoint(save_checkpoint(state,tmp_path))['trace'][0]['output']['metric']=={'unavailable':'nonfinite'}
