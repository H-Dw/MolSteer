"""OpenAI-compatible OpenRouter transport with ephemeral reasoning continuity.

The tolerant chat-completion transport avoids an additional generated response
schema. Provider reasoning blocks round-trip in memory, never in audit records.
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from copy import deepcopy
import json


class IncompleteStreamError(RuntimeError):
    """Safe receipt metadata only, never partial argument or reasoning content."""
    def __init__(self, arguments):
        super().__init__('OpenRouter stream ended without a valid completion marker')
        self.stream_receipt = {'tool_argument_streams': len(arguments),
                               'argument_character_count': sum(len(raw) for raw in arguments.values())}


def provider_tool_schema(schema):
    """Inline provider-facing fields with a bounded expression preview.

    Full recursive validation stays in MathematicalDesign. Expanding all six
    recursive alternatives creates a large tree; unresolved provider refs also
    caused JSON-encoded object arguments in real GLM tests. Nested expression
    objects advertise the common fields and grammar; their exact operator
    variants are supplied by get_expert_contract and checked by the host.
    """
    definitions=schema.get('$defs',{})
    expression_names=('ObservableExpression','ConstantExpression','UnaryExpression',
                      'PowerExpression','BinaryExpression','ReductionExpression')
    expression_preview=None
    if all(name in definitions for name in expression_names):
        nested={'type':'object','properties':{
            'op':{'type':'string','enum':['observable','constant','relu','abs','sqrt','sin','cos','power',
                'add','subtract','multiply','divide','maximum','minimum','periodic_difference','sum','mean']},
            'id':{'type':'string'},'value':{'type':'number'},
            'unit':deepcopy(definitions['ConstantExpression']['properties']['unit']),
            'origin':{'type':'string','minLength':5},'exponent':{'type':'number','minimum':.5,'maximum':8},
            'args':{'type':'array','items':{'type':'object'}}},'required':['op'],
            'description':'Recursive expression: observable requires id; constant requires value/unit/origin; unary and power take 1 arg (power also exponent); binary takes 2; sum/mean take 1-32. Every nested node is strictly host-validated.'}
        variants=[deepcopy(definitions[name]) for name in expression_names]
        for variant in variants:
            if 'args' in variant['properties']:
                variant['properties']['args']['items']=deepcopy(nested)
        expression_preview={'oneOf':variants}

    def expand(node, parents=()):
        if isinstance(node,list):return [expand(value,parents) for value in node]
        if not isinstance(node,dict):return node
        if expression_preview and node.get('discriminator',{}).get('propertyName')=='op':
            return deepcopy(expression_preview)
        if '$ref' in node:
            reference=node['$ref']
            if not reference.startswith('#/$defs/') or reference.split('/')[-1] not in definitions:
                raise ValueError('Unresolved provider tool schema reference')
            if reference in parents:raise ValueError('Unsupported recursive provider tool schema')
            result=expand(definitions[reference.split('/')[-1]],(*parents,reference))
            result.update({key:expand(value,parents) for key,value in node.items() if key!='$ref'})
            return result
        return {key:expand(value,parents) for key,value in node.items() if key not in ('$defs','title')}
    return expand(schema)


class OpenRouterChat(ChatOpenAI):
    @staticmethod
    def _take_stream_metadata(chunk, metadata):
        # Some providers repeat terminal metadata in the usage chunk. LangChain
        # concatenates strings when merging generations, yielding lengthlength
        # or a duplicated model name. Emit these fields exactly once at EOF.
        fields=('finish_reason','model_name','system_fingerprint','service_tier')
        info=dict(chunk.generation_info or {})
        for mapping in (info, chunk.message.response_metadata):
            for field in fields:
                value=mapping.pop(field,None)
                if value is not None:
                    if field=='finish_reason' and metadata.get(field) in ('length','max_tokens'):
                        continue
                    metadata[field]=value
        chunk.generation_info=info or None

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        result=super()._convert_chunk_to_generation_chunk(chunk,default_chunk_class,base_generation_info)
        choices=chunk.get('choices',[])
        if result is not None and choices:
            delta=choices[0].get('delta') or {}
            if delta.get('reasoning_details') is not None:
                # Nested lists append without LangChain merging repeated ids,
                # type/format strings or encrypted signatures by block index.
                result.message.additional_kwargs['_openrouter_reasoning_deltas']=[deepcopy(delta['reasoning_details'])]
            elif delta.get('reasoning') or delta.get('reasoning_content'):
                result.message.additional_kwargs['_openrouter_reasoning_text_deltas']=[delta.get('reasoning') or delta['reasoning_content']]
        return result

    def _stream(self, *args, **kwargs):
        metadata={}
        arguments={}
        for chunk in super()._stream(*args,**kwargs):
            for call in getattr(chunk.message,'tool_call_chunks',[]):
                index=call.get('index')
                arguments[index]=arguments.get(index,'')+(call.get('args') or '')
            self._take_stream_metadata(chunk,metadata)
            yield chunk
        self._validate_stream_end(metadata.get('finish_reason'),arguments)
        yield ChatGenerationChunk(message=AIMessageChunk(content=''),generation_info=metadata)

    async def _astream(self, *args, **kwargs):
        metadata={}
        arguments={}
        async for chunk in super()._astream(*args,**kwargs):
            for call in getattr(chunk.message,'tool_call_chunks',[]):
                index=call.get('index')
                arguments[index]=arguments.get(index,'')+(call.get('args') or '')
            self._take_stream_metadata(chunk,metadata)
            yield chunk
        self._validate_stream_end(metadata.get('finish_reason'),arguments)
        yield ChatGenerationChunk(message=AIMessageChunk(content=''),generation_info=metadata)

    @staticmethod
    def _validate_stream_end(finish,arguments):
        if finish not in ('stop','tool_calls','length','max_tokens','function_call'):
            raise IncompleteStreamError(arguments)
        if finish not in ('length','max_tokens'):
            # LangChain's chunk parser can repair partial JSON. It must never
            # authorize tools; validate the exact concatenated argument bytes.
            for raw in arguments.values():
                parsed=json.loads(raw)
                if not isinstance(parsed,dict):
                    raise ValueError('OpenRouter tool arguments must be a complete JSON object')

    def bind_tools(self, tools, **kwargs):
        """Adapt tool schemas without changing host argument validation."""
        formatted=[]
        for item in tools:
            if isinstance(item, BaseTool):
                schema=item.tool_call_schema
                if not isinstance(schema,dict):
                    schema=schema.model_json_schema()
                formatted.append({'type':'function','function':{
                    'name':item.name,'description':item.description,'parameters':provider_tool_schema(schema)}})
            else:
                formatted.append(item)
        return super().bind_tools(formatted, **kwargs)

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        for generation, choice in zip(result.generations, payload.get('choices', [])):
            details = choice.get('message', {}).get('reasoning_details')
            if details is not None:
                generation.message.additional_kwargs['reasoning_details'] = details
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for source, target in zip(messages, payload.get('messages', [])):
            details = source.additional_kwargs.get('reasoning_details')
            if details is None and source.additional_kwargs.get('_openrouter_reasoning_deltas'):
                details=[item for delta in source.additional_kwargs['_openrouter_reasoning_deltas'] for item in delta]
            if source.type == 'ai' and details is not None:
                target['reasoning_details'] = details
            elif source.type=='ai' and source.additional_kwargs.get('_openrouter_reasoning_text_deltas'):
                target['reasoning']=''.join(source.additional_kwargs['_openrouter_reasoning_text_deltas'])
        return payload
