"""LangGraph state machine for AgentRuntime's four-node control loop."""
from __future__ import annotations
from .state import MolSteerState
try:
 from langgraph.graph import StateGraph, END
except ImportError: StateGraph=None; END='__end__'

def build_workflow(runtime):
    if StateGraph is None: raise RuntimeError('langgraph is required')
    graph=StateGraph(dict)
    graph.add_node('reader',runtime.reader); graph.add_node('thinker',runtime.thinker)
    graph.add_node('executor',runtime.executor); graph.add_node('monitor',runtime.monitoring)
    graph.set_entry_point('reader'); graph.add_edge('reader','thinker')
    graph.add_conditional_edges('thinker',lambda state:'done' if state.get('route')=='done' else 'executor',
                                {'executor':'executor','done':END})
    graph.add_edge('executor','monitor')
    def route(state): return state.get('route','done') if state.get('route')!='done' else 'done'
    graph.add_conditional_edges('monitor',route,{'reader':'reader','thinker':'thinker','executor':'executor','done':END})
    return graph.compile()

__all__=['build_workflow']
