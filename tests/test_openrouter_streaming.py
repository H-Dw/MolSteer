"""Complete provider streams, strict raw argument parsing and private continuity."""
import asyncio
import json
import pytest
from langchain_core.messages import AIMessageChunk
from langchain_openai import ChatOpenAI
from molsteer.agents.openrouter_transport import OpenRouterChat


def test_billing_receipts_preserve_native_usage_without_private_text():
    from molsteer.agents.openrouter_transport import usage_receipt
    payload={'id':'gen-receipt','model':'z-ai/glm-5.3','provider':'fixture',
        'choices':[{'message':{'content':'private','reasoning':'private'}}],
        'usage':{'prompt_tokens':100,'completion_tokens':30,'total_tokens':130,'cost':.01,
                 'prompt_tokens_details':{'cached_tokens':80,'private_text':'omit'},
                 'completion_tokens_details':{'reasoning_tokens':20},'private_text':'omit'}}
    receipt=usage_receipt(payload)
    assert receipt['usage']['cost']==.01
    assert receipt['usage']['prompt_tokens_details']=={'cached_tokens':80}
    assert receipt['usage']['completion_tokens_details']=={'reasoning_tokens':20}
    assert 'private' not in str(receipt)
    assert receipt['id']=='gen-receipt'


def test_billing_receipts_survive_stream_and_nonstream_conversion():
    from langchain_core.messages import AIMessageChunk
    model=OpenRouterChat(model='fixture',api_key='test-only',base_url='https://example.invalid/v1')
    payload={'id':'gen-fixture','model':'fixture','object':'chat.completion','created':0,
        'choices':[{'index':0,'message':{'role':'assistant','content':'OK'},'finish_reason':'stop'}],
        'usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7,'cost':.02}}
    result=model._create_chat_result(payload)
    assert result.generations[0].message.additional_kwargs['_openrouter_receipt']['usage']['cost']==.02
    payload['choices']=[];payload['object']='chat.completion.chunk'
    chunk=model._convert_chunk_to_generation_chunk(payload,AIMessageChunk,None)
    assert chunk.message.additional_kwargs['_openrouter_receipt']['id']=='gen-fixture'
    assert chunk.message.additional_kwargs['_openrouter_receipt']['usage']['total_tokens']==7
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
