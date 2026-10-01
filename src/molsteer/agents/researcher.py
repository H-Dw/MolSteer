"""On-demand Researcher ReAct sessions shared by both MolThinker experts."""
from copy import deepcopy
from datetime import datetime, timezone
from langchain_core.tools import tool
from molsteer.common import digest
from molsteer.molthinker.research.corpus import MarkdownCorpus
from molsteer.molthinker.research.retrieval import LiteratureProvider, WebProvider
from .loop import run_tools
from .trace import append_trace

RESEARCHER_INSTRUCTIONS = ('You are Researcher. Resolve the caller evidence gap. Search local first, fetch '
    'relevant sources, then synthesize source-backed claims and counterevidence. Never invent '
    'formulas, potency or retrieval. Record assumptions and abstract/full-text limits. '
    'You cannot select targets or activate controls. Use submit_research even when evidence is missing.')


class ResearchService:
    def __init__(self, runtime, state):
        self.runtime, self.state = runtime, state
        self.config = runtime.config.thinker
        self.local = MarkdownCorpus(runtime.config.repo_root/'knowledge')
        self.providers = {'local': self.local}
        if self.config.external_research:
            self.providers.update(europe_pmc=LiteratureProvider('europe_pmc'), arxiv=LiteratureProvider('arxiv'))
            if runtime.search_fn:
                self.providers['web'] = WebProvider(runtime.search_fn, runtime.fetch_fn)
            self.providers.update(runtime.research_providers)
        self.cache, self.calls = {}, 0
        state['research_packets'] = []
        state['retrieval_records'] = []

    def local_search(self, direction_id, query, function_id=None):
        result = self.local.search(query, self.config.research_result_limit, function_id=function_id)
        event = dict(result, direction_id=direction_id, provider='local')
        event['retrieval_id'] = 'ret_'+digest(event)[:24]
        if not any(r['retrieval_id'] == event['retrieval_id'] for r in self.state['retrieval_records']):
            self.state['retrieval_records'].append(event)
        return deepcopy(event)

    def sources(self):
        return {s.get('observation_id',s['source_id']): s for p in self.state['research_packets'] for s in p['sources']}

    def tool_for(self, role):
        @tool
        def request_research(direction_id: str, question: str, evidence_gap: str, completion_condition: str) -> dict:
            """Ask the independent Researcher for inspected local/external evidence; never activate a reward."""
            return self.request(role, direction_id, question, evidence_gap, completion_condition)
        return request_research

    def request(self, role, direction_id, question, evidence_gap, completion_condition):
        if any(not isinstance(x, str) or not x.strip() or len(x) > 2000
               for x in (direction_id, question, evidence_gap, completion_condition)):
            raise ValueError('Bounded research question, gap and completion condition are required')
        request = dict(direction_id=direction_id, question=question, evidence_gap=evidence_gap,
                       completion_condition=completion_condition, packet_id=self.state['packet']['packet_id'],
                       corpus_hash=self.local.version)
        key = digest(request)
        if key in self.cache:
            return dict(deepcopy(self.cache[key]), cache_hit=True)
        if self.calls >= self.config.max_research_requests:
            return {'status': 'budget_exhausted', 'request': request}
        self.calls += 1
        holder, evidence, searched, observations = {}, {}, {}, []
        counts = {'searches': 0, 'local': False}

        @tool
        def search_sources(query: str, provider: str = 'local') -> dict:
            """Search local first; external providers are available only when configured. Returns source IDs."""
            if provider not in self.providers:
                return {'status':'unavailable', 'provider':provider}
            if provider != 'local' and not counts['local']:
                raise ValueError('Search the local knowledge corpus first')
            if counts['searches'] >= self.config.research_max_searches:
                return {'status':'budget_exhausted', 'provider':provider}
            counts['searches'] += 1
            counts['local'] |= provider == 'local'
            event = {'provider':provider, 'query':query, 'time_utc':datetime.now(timezone.utc).isoformat()}
            try:
                result = self.providers[provider].search(query, self.config.research_result_limit)
                rows = result['records']
                for row in rows:
                    ident = row.get('chunk_id', row['source_id'])
                    searched[ident] = (provider, deepcopy(row))
                event.update(status=result.get('status', 'ok' if rows else 'zero_hits'), records=rows)
            except Exception as exc:
                event.update(status='failed', error_type=type(exc).__name__, records=[])
            observations.append(event)
            return deepcopy(event)

        @tool
        def fetch_source(source_id: str) -> dict:
            """Inspect a returned chunk_id (local) or source_id (external). Snippets alone cannot support claims."""
            if source_id not in searched:
                raise ValueError('Fetch requires an identifier returned by this research session')
            provider, row = searched[source_id]
            try:
                result = self.providers[provider].fetch(source_id)
                if result.get('status') == 'unavailable' or not result.get('excerpt'):
                    result = dict(result, status='unavailable')
                else:
                    result = dict(result, status='ok', observation_id='obs_'+digest(result)[:24])
                    evidence[result['observation_id']] = result
            except Exception as exc:
                result = dict(status='failed', error_type=type(exc).__name__, source_id=row['source_id'])
            observations.append(dict(action='fetch', provider=provider, result=deepcopy(result)))
            return result

        @tool
        def submit_research(summary: str, claims: list[dict], gaps: list[str]) -> dict:
            """Submit claims {claim, observation_ids, limitation}; cite inspected observations, or return explicit gaps."""
            if not counts['local'] or len(summary.strip()) < 12:
                raise ValueError('Research requires a local search and substantive summary')
            for claim in claims:
                if (set(claim) != {'claim', 'observation_ids', 'limitation'} or not claim['observation_ids']
                        or not set(claim['observation_ids']) <= set(evidence)):
                    raise ValueError('Research claims require fetched evidence observations')
            if not claims and not gaps:
                raise ValueError('Missing evidence must remain an explicit gap')
            packet = dict(kind='ResearchEvidencePacket', schema_version='1.0', request=request,
                          requested_by=role, status='supported' if claims else 'unresolved',
                          summary=summary, claims=claims, gaps=gaps, sources=list(evidence.values()),
                          observations=deepcopy(observations), automatic_reward_activation=False)
            packet['packet_id'] = 'rep_'+digest(packet)[:24]
            holder['packet'] = packet
            return deepcopy(packet)

        try:
            run_tools(self.runtime._model('molthinker.researcher'), [search_sources, fetch_source, submit_research],
                      instructions=RESEARCHER_INSTRUCTIONS,
                      context={'request':request, 'providers':list(self.providers)}, state=self.state,
                      node='thinker.researcher', max_steps=self.config.research_max_steps,
                      max_repairs=self.runtime.config.runtime.max_repairs, completed=lambda:'packet' in holder)
        except Exception as exc:
            failure = dict(kind='ResearchEvidencePacket', schema_version='1.0', request=request,
                           requested_by=role, status='failed', error_type=type(exc).__name__,
                           sources=list(evidence.values()), observations=observations, claims=[],
                           gaps=['Researcher did not produce a validated submission'], automatic_reward_activation=False)
            failure['packet_id'] = 'rep_'+digest(failure)[:24]
            holder['packet'] = failure
        packet = holder['packet']
        self.state['research_packets'].append(deepcopy(packet))
        self.cache[key] = deepcopy(packet)
        append_trace(self.state, node='thinker.researcher', kind='decision',
                     summary='Research evidence returned without reward activation', output=packet)
        return deepcopy(packet)
