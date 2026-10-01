import pytest
import json
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from molsteer.agents.loop import run_tools


@pytest.mark.parametrize('failure',['length','invalid'])
def test_no_partial_calls_execute_before_a_complete_reply(failure):
    invoked=[]
    @tool
    def submit(value:int) -> dict:
        """Submit a complete integer result."""
        invoked.append(value);return {'status':'accepted'}
    class Model:
        def __init__(self):self.count=0
        def bind_tools(self,tools):return self
        def invoke(self,messages):
            self.count+=1
            if self.count==1:
                return AIMessage(content='PRIVATE_REPLY_NEVER_PERSISTED',
                    tool_calls=[{'name':'submit','args':{'value':1},'id':'first','type':'tool_call'}],
                    invalid_tool_calls=([{'name':'submit','args':'{broken','id':'bad','error':'PRIVATE_ERROR'}] if failure=='invalid' else []),
                    response_metadata={'finish_reason':'length' if failure=='length' else 'tool_calls'})
            return AIMessage(content='',tool_calls=[{'name':'submit','args':{'value':2},'id':'second','type':'tool_call'}])
    state={}
    run_tools(Model(),[submit],instructions='Submit a complete result.',context={},state=state,
              node='test',max_steps=3,max_repairs=1,completed=lambda:bool(invoked))
    assert invoked==[2]
    assert 'PRIVATE' not in str(state)


@pytest.mark.parametrize('persistent',[False,True])
def test_response_decoder_retries_once_before_any_tool_runs(persistent):
    executed=[]
    @tool
    def submit(value:int) -> dict:
        """Submit an integer after a complete response."""
        executed.append(value);return {'status':'accepted'}
    class Model:
        calls=0
        def bind_tools(self,tools):return self
        def invoke(self,messages):
            self.calls+=1
            if persistent or self.calls==1:
                raise json.JSONDecodeError('PRIVATE_ERROR','PRIVATE_BODY',0)
            return AIMessage(content='',tool_calls=[{'name':'submit','args':{'value':53},'id':'good','type':'tool_call'}])
    state={};model=Model()
    if persistent:
        with pytest.raises(RuntimeError,match='after one retry'):
            run_tools(model,[submit],instructions='Submit.',context={},state=state,
                node='test',max_steps=3,completed=lambda:bool(executed))
        assert not executed
    else:
        run_tools(model,[submit],instructions='Submit.',context={},state=state,
            node='test',max_steps=3,completed=lambda:bool(executed))
        assert executed==[53]
    assert model.calls==2 and 'PRIVATE' not in str(state)
