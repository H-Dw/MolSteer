"""Complete provider streams, strict raw argument parsing and private continuity."""
import asyncio
import json
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from molsteer.agents.openrouter_transport import OpenRouterChat
from molsteer.agents.trace import _redact


def install_stream(monkeypatch,chunks):
    def fake(self,*args,**kwargs):
        for chunk in chunks:
            yield self._convert_chunk_to_generation_chunk(chunk,AIMessageChunk,{})
    async def afake(self,*args,**kwargs):
        for item in fake(self,*args,**kwargs):yield item
    monkeypatch.setattr(ChatOpenAI,'_stream',fake)
    monkeypatch.setattr(ChatOpenAI,'_astream',afake)
    return OpenRouterChat(model='z-ai/glm-5.3',api_key='test-only',
        base_url='https://example.invalid/v1',streaming=True,use_responses_api=False)


def chunk(delta,finish=None):
    return {'model':'z-ai/glm-5.3','choices':[{'index':0,'delta':delta,'finish_reason':finish}]}


def test_stream_preserves_private_delta_order_and_complete_tool_arguments(monkeypatch):
    first={'type':'reasoning.text','text':'synthetic private A','index':0,'id':'same','format':'unknown'}
    second={**first,'text':'synthetic private B'}
    model=install_stream(monkeypatch,[chunk({'role':'assistant','reasoning_details':[first]}),
        chunk({'reasoning_details':[second],'tool_calls':[{'index':0,'id':'call_1','function':{'name':'inspect','arguments':'{"count":'}}]}),
        chunk({'tool_calls':[{'index':0,'function':{'arguments':'3}'}}]},'tool_calls'),
        {'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15,
            'completion_tokens_details':{'reasoning_tokens':2}}}])
    message=model.invoke('test')
    assert message.tool_calls[0]['args']=={'count':3}
    assert message.response_metadata['finish_reason']=='tool_calls'
    assert message.usage_metadata['output_token_details']['reasoning']==2
    payload=model._get_request_payload([message])['messages'][0]
    assert payload['reasoning_details']==[first,second]
    assert 'synthetic private' not in json.dumps(_redact(message.additional_kwargs))


@pytest.mark.parametrize('async_mode',[False,True])
def test_stream_cannot_execute_repaired_partial_json(monkeypatch,async_mode):
    model=install_stream(monkeypatch,[chunk({'role':'assistant','tool_calls':[{
        'index':0,'id':'call_1','function':{'name':'inspect','arguments':'{"count":3'}}]},'tool_calls')])
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(model.ainvoke('test')) if async_mode else model.invoke('test')


@pytest.mark.parametrize('async_mode',[False,True])
def test_stream_requires_a_terminal_completion_marker(monkeypatch,async_mode):
    model=install_stream(monkeypatch,[chunk({'role':'assistant','tool_calls':[{
        'index':0,'id':'call_1','function':{'name':'inspect','arguments':'{"count":3}'}}]})])
    with pytest.raises(RuntimeError,match='completion marker'):
        asyncio.run(model.ainvoke('test')) if async_mode else model.invoke('test')


def test_incomplete_stream_receipt_contains_only_counts():
    from molsteer.agents.openrouter_transport import IncompleteStreamError
    error=IncompleteStreamError({0:'synthetic private tool argument'})
    assert error.stream_receipt=={'tool_argument_streams':1,'argument_character_count':31}
    assert 'private' not in json.dumps(error.stream_receipt) and 'private' not in str(error)


@pytest.mark.parametrize('async_mode',[False,True])
@pytest.mark.parametrize('finish',['length','tool_calls'])
def test_repeated_terminal_metadata_is_emitted_once(monkeypatch,async_mode,finish):
    model=install_stream(monkeypatch,[chunk({'role':'assistant','tool_calls':[{
        'index':0,'id':'call_1','function':{'name':'inspect','arguments':'{"count":3}'}}]},finish),
        chunk({},finish)])
    message=asyncio.run(model.ainvoke('test')) if async_mode else model.invoke('test')
    assert message.response_metadata['finish_reason']==finish
    assert message.response_metadata['model_name']=='z-ai/glm-5.3'
