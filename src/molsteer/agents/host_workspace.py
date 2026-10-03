"""Deterministic artifact receipts, state-based tool routing and progress facts."""
from copy import deepcopy
import json
from langchain_core.tools import tool
from molsteer.common import digest

NO_PROGRESS = ('The current evidence and computational candidate are unchanged. Please make a targeted query '
    'about an unresolved question, or advance candidate construction or testing. If required inputs are missing, '
    'explicitly request reconsideration or defer the affected goal.')
COMPUTATIONAL = {'expression', 'observables', 'strategy', 'direction_id', 'status', 'schema_version',
                 'graph_policy', 'constraint_mode', 'priority_weights', 'fixed_atom_ids'}


def computational_content(value):
    if isinstance(value, list):
        return [computational_content(v) for v in value]
    if not isinstance(value, dict):
        return value
    def numeric_tree(item):
        if isinstance(item, list):
            return [numeric_tree(v) for v in item]
        if isinstance(item, dict):
            return {k:numeric_tree(v) for k,v in item.items() if k not in ('origin', 'justification', 'priority_basis')}
        return item
    return {k: (numeric_tree(v) if k in COMPUTATIONAL else computational_content(v)) for k, v in value.items()
            if k in COMPUTATIONAL or k == 'directions'}


class HostWorkspace:
    def __init__(self, state, scope, threshold=3):
        self.data = state.setdefault('host_workspaces', {}).setdefault(scope, dict(
            artifacts={}, seen_progress=[], idle_actions=0, latest_change='Initialized bound inputs',
            open_questions={}, validations={}, tool_group='auto', candidate_exists=False))
        self.threshold = threshold

    def save(self, identifier, content, dependencies=None, kind='candidate'):
        # Once assembled, the design is authoritative. Direction receipts point
        # into it rather than keeping another mutable copy of the same expression.
        if kind == 'candidate' and identifier != 'design' and 'design' in self.data['artifacts']:
            design = deepcopy(self.data['artifacts']['design']['content'])
            found = next((i for i,d in enumerate(design.get('directions', [])) if d.get('direction_id') == identifier), None)
            if found is not None:
                design['directions'][found] = deepcopy(content)
                self.save('design', design, dependencies)
                return self.receipt(identifier)
        full_hash = digest(content)
        computation_hash = digest(computational_content(content)) if kind == 'candidate' else full_hash
        old = self.data['artifacts'].get(identifier, {})
        if old.get('content_hash') != full_hash and 'content' in old:
            self.data.setdefault('artifact_history', {}).setdefault(identifier, {})[str(old['version'])] = deepcopy(old)
        previous_version = old.get('version', max((int(v) for v in self.data.get('artifact_history', {}).get(identifier, {})), default=0))
        revision = previous_version + (old.get('content_hash') != full_hash)
        self.data['artifacts'][identifier] = dict(id=identifier, version=revision,
            content_hash=full_hash, computational_hash=computation_hash, dependencies=deepcopy(dependencies or {}),
            kind=kind, content=deepcopy(content))
        if kind == 'candidate':
            self.data['candidate_exists'] = True
            if old.get('computational_hash') != computation_hash or old.get('dependencies') != (dependencies or {}):
                self.data['validations'].clear()
        if identifier == 'design' and isinstance(content, dict):
            for index, direction in enumerate(content.get('directions', [])):
                ident = direction['direction_id']
                self.data['artifacts'][ident] = dict(id=ident, version=revision, kind='candidate',
                    content_hash=digest(direction), computational_hash=digest(computational_content(direction)),
                    dependencies=deepcopy(dependencies or {}), alias=['design', 'directions', index])
        return self.receipt(identifier)

    def retire_candidates(self):
        for ident, artifact in list(self.data['artifacts'].items()):
            if artifact['kind'] != 'candidate':
                continue
            if 'content' in artifact:
                self.data.setdefault('artifact_history', {}).setdefault(ident, {})[str(artifact['version'])] = deepcopy(artifact)
            del self.data['artifacts'][ident]
        self.data['candidate_exists'] = False
        self.data['validations'].clear()

    def receipt(self, identifier):
        return {k: deepcopy(v) for k, v in self.data['artifacts'][identifier].items() if k != 'content'}

    def read(self, identifier, path=None, force_full=False):
        artifact = self.data['artifacts'].get(identifier)
        if artifact is None:
            return dict(status='unavailable', blocking=False, id=identifier)
        value = artifact.get('content')
        if 'alias' in artifact:
            parent, *location = artifact['alias']
            value = self.data['artifacts'][parent]['content']
            for field in location:
                value = value[field]
        try:
            for field in path or []:
                value = value[field]
        except (KeyError, IndexError, TypeError):
            return dict(status='needs_input', blocking=False, hint='Use a field in the current artifact.')
        token = digest((identifier, artifact['content_hash'], path))
        reads = self.data.setdefault('reads', [])
        if token in reads and not force_full:
            return dict(status='unchanged', **self.receipt(identifier), path=path,
                        hint='Already delivered; force_full=true explicitly reads it again.')
        reads.append(token)
        return dict(status='available', **self.receipt(identifier), value=deepcopy(value))

    def phase(self):
        if self.data['open_questions'] or self.data['tool_group'] == 'research':
            return 'research', 'Unresolved evidence questions or explicit return to research'
        if any(v.get('passed') is True for v in self.data['validations'].values()) and self.data['candidate_exists']:
            return 'validated', 'Validation facts available; submit or revise the candidate'
        if self.data['candidate_exists']:
            return 'candidate', 'A computational candidate exists'
        return 'research', 'No computational candidate yet'

    def observe(self, name, args, result):
        # Novel facts, not the count/order of actions, establish progress. Seen hashes
        # survive candidate A -> B -> A and wording-only revisions.
        facts = []
        for ident, item in self.data['artifacts'].items():
            facts.append(digest((ident, item['computational_hash'], item['dependencies'])))
        status = result.get('status') if isinstance(result, dict) else None
        if name.startswith(('inspect_', 'read_', 'search_', 'prepare_', 'list_', 'get_')) and name not in (
                'read_host_workspace', 'read_expert_workspace', 'read_candidate') and status not in ('unchanged', 'error', 'needs_input'):
            facts.append(digest((name, result)))
        if name.startswith(('test_', 'compare_', 'patch_and_test')) and status not in ('error', 'needs_input'):
            def calculation_fact(value):
                if isinstance(value, list):
                    return [calculation_fact(v) for v in value]
                if isinstance(value, dict):
                    return {k:calculation_fact(v) for k,v in value.items()
                            if k not in ('reward_id', 'elapsed_seconds', 'cached', 'justification', 'priority_basis')}
                return value
            fingerprint = digest(calculation_fact(result))
            facts.append(fingerprint)
            self.data['validations'][name] = dict(status=status, passed=result.get('passed'), result_hash=fingerprint,
                                                reward_id=result.get('reward_id'))
        if name.startswith(('request_biology_revision', 'defer_')):
            facts.append(digest((name, result)))
        known = set(self.data['seen_progress'])
        novel = set(facts) - known
        if novel:
            self.data['seen_progress'].extend(sorted(novel))
            self.data['idle_actions'] = 0
            self.data['latest_change'] = name
        else:
            self.data['idle_actions'] += 1

    def summary(self):
        phase, reason = self.phase()
        return dict(phase=phase, reason=reason, versions=[self.receipt(k) for k in self.data['artifacts']],
            latest_change=self.data['latest_change'], open_questions=self.data['open_questions'],
            latest_validation={k: v for k, v in list(self.data['validations'].items())[-1:]},
            available_tool_groups=['auto', 'research', 'candidate', 'all'],
            no_progress_notice=NO_PROGRESS if self.data['idle_actions'] >= self.threshold else None)

    def visible(self, registry):
        group = self.data['tool_group']
        if group == 'all':
            return list(registry.values())
        phase = self.phase()[0] if group == 'auto' else group
        always = ('workspace', 'tool_group', 'revision', 'defer', 'contract', 'record_')
        reference_tools = {'submit_current_candidate', 'test_current_candidate', 'patch_candidate',
                           'read_candidate', 'read_host_workspace', 'select_tool_group', 'resolve_evidence_question', 'read_archived_artifact'}
        output = []
        for name, function in registry.items():
            if name in reference_tools or any(k in name for k in always):
                output.append(function)
            elif phase == 'research' and not name.startswith(('test_', 'patch_', 'submit_mathematical', 'stage_mathematical')):
                output.append(function)
            elif phase in ('candidate', 'validated') and name.startswith(('compare_', 'patch_', 'test_staged')):
                output.append(function)
            elif phase in ('candidate', 'validated') and name in ('construct_direction_potential',):
                output.append(function)
        return output or list(registry.values())

    def tools(self):
        @tool
        def read_archived_artifact(artifact_id: str, version: int | None = None) -> dict:
            """List historical versions or read an exact archived artifact on demand."""
            versions = self.data.get('artifact_history', {}).get(artifact_id, {})
            if version is None:
                return dict(status='available', versions=[{k:v for k,v in a.items() if k != 'content'} for a in versions.values()])
            return dict(status='available' if str(version) in versions else 'unavailable', artifact=deepcopy(versions.get(str(version))))

        @tool
        def read_host_workspace() -> dict:
            """Read current phase, artifact references, unresolved questions and validation progress."""
            return self.summary()

        @tool
        def select_tool_group(group: str) -> dict:
            """Display auto, research, candidate or all tools. Known older calls remain callable."""
            if group not in ('auto', 'research', 'candidate', 'all'):
                return dict(status='needs_input', blocking=False, groups=['auto', 'research', 'candidate', 'all'])
            self.data['tool_group'] = group
            return dict(status='selected', group=group)

        @tool
        def resolve_evidence_question(question_id: str, question: str, answer: str = '', deferred: bool = False) -> dict:
            """Record a specific evidence gap, answer or explicit deferral without introducing a gate."""
            if answer or deferred:
                self.data['open_questions'].pop(question_id, None)
                self.save('question:' + question_id, dict(question=question, answer=answer, deferred=deferred), kind='evidence')
            else:
                self.data['open_questions'][question_id] = question
            return dict(status='deferred' if deferred else 'answered' if answer else 'open', question_id=question_id)
        return [read_host_workspace, select_tool_group, resolve_evidence_question, read_archived_artifact]


def record_usage(state, node, phase, response, messages, tools):
    from langchain_core.utils.function_calling import convert_to_openai_tool
    receipt = getattr(response, 'additional_kwargs', {}).get('_openrouter_receipt') or {}
    usage = receipt.get('usage') or getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', {}).get('token_usage')
    try:
        definition_size = len(json.dumps([convert_to_openai_tool(t) for t in tools], ensure_ascii=False).encode('utf-8'))
    except (ValueError, TypeError):
        definition_size = None  # Custom injected adapters need not expose provider schemas.
    state.setdefault('api_usage', []).append(dict(module=node, phase=phase,
        usage_status='reported' if usage else 'unknown', usage=deepcopy(usage) if usage else None,
        provider_receipt=deepcopy(receipt) if receipt else None,
        size_measurement='UTF-8 serialized LangChain messages and OpenAI tool definitions; not a token estimate or exact HTTP payload',
        request_bytes=len(json.dumps([m.model_dump() for m in messages], default=str, ensure_ascii=False).encode('utf-8')),
        tool_definition_bytes=definition_size,
        tool_definition_size_status='serialized' if definition_size is not None else 'unknown'))
