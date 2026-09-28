"""MolThinker: evidence-bound planning and ReAct-compatible tools."""
from __future__ import annotations
import json, math
from copy import deepcopy
from typing import Any, Callable
from langchain_core.tools import tool
from molsteer.common import digest
from .state import validate_packet, validate_report

class Thinker:
    def __init__(self, *, model: Any = None, thinker_fn: Callable[..., dict[str, Any]] | None = None, knowledge_path: str | None = None, tools: list[Any] | None = None):
        if model is None and thinker_fn is None and knowledge_path is None: raise ValueError("Thinker requires an API model, thinker_fn, or reviewed knowledge path")
        self.model, self.thinker_fn, self.knowledge_path, self.tools = model, thinker_fn, knowledge_path, tools or []
        self.bound_model = model.bind_tools(self.tools) if model is not None and self.tools and hasattr(model, "bind_tools") else model
    def run(self, packet, report):
        validate_packet(packet); validate_report(report, packet)
        if self.thinker_fn is not None: spec = self.thinker_fn(packet, report)
        elif self.model is not None:
            response = self.bound_model.invoke([{"role":"system","content":"You are MolThinker. Return only a RewardSpec proposal with evidence IDs and a concise decision summary; use read-only tools and never expose private reasoning."},{"role":"user","content":json.dumps({"packet":packet,"report":report},sort_keys=True,default=str)}])
            content = getattr(response,"content",response)
            if isinstance(content,str): spec=json.loads(content)
            elif isinstance(content,dict): spec=content
            else: raise ValueError("Thinker model did not return a RewardSpec object")
        elif self.knowledge_path is not None:
            from molsteer.molthinker.planner import derive
            spec = derive(packet, report, self.knowledge_path)
        else:
            raise ValueError("Thinker has no model, function, or knowledge path")
        if not isinstance(spec,dict): raise ValueError("Thinker must return a RewardSpec object")
        from molsteer.molthinker.planner import validate_spec
        validate_spec(spec,packet,report); return spec

def thinker_tools(packet, report, knowledge_path, *, search_fn=None, compute_fn=None, feedback=None):
    from .tools import default_thinker_tools
    base = __import__('molsteer.molthinker.planner', fromlist=['derive']).derive(packet, report, knowledge_path); result={}
    tools=default_thinker_tools(packet,str(knowledge_path),approved_knowledge_root=knowledge_path.parent,search_fn=search_fn,compute_fn=compute_fn)
    @tool
    def record_task_plan(steps:list[str], evidence_ids:list[str], summary:str) -> dict:
        """Record a concise initial or revised task plan with evidence references, not private reasoning."""
        if not 1<=len(steps)<=12 or any(not s.strip() or len(s)>500 for s in steps): raise ValueError('invalid plan steps')
        if not summary.strip() or len(summary)>1000: raise ValueError('invalid decision summary')
        if not set(evidence_ids)<=set(report['evidence_index']): raise ValueError('unknown evidence ID')
        result['task_plan']={'steps':steps,'evidence_ids':evidence_ids,'summary':summary}
        return deepcopy(result['task_plan'])
    @tool
    def derive_reward_candidates() -> dict:
        """Derive supported reward primitives from validated evidence."""
        return {"candidate":deepcopy(base),"feedback":feedback or {},"limits":"Evidence-bound terms only; no invented target bounds."}
    @tool
    def submit_reward_plan(term_ids:list[str],weights:list[float],scales:list[float]) -> dict:
        """Submit candidate IDs and checked weights/scales for host validation."""
        if len(term_ids)!=len(set(term_ids)) or not len(term_ids)==len(weights)==len(scales): raise ValueError("term IDs and parameters must align")
        by_id={t["term_id"]:t for t in base["terms"]}
        if any(t not in by_id for t in term_ids): raise ValueError("unknown candidate term")
        if base["terms"] and (not term_ids or not any(w>0 for w in weights)): raise ValueError("cannot discard all diagnosed objectives")
        if any(not math.isfinite(w) or not 0<=w<=100 for w in weights): raise ValueError("invalid weight")
        if any(not math.isfinite(s) or not 1e-6<=s<=1e6 for s in scales): raise ValueError("invalid scale")
        spec=deepcopy(base); spec["terms"]=[dict(by_id[t],weight=w,scale=s) for t,w,s in zip(term_ids,weights,scales)]
        spec["reward_groups"]=[dict(g,term_ids=[t for t in g["term_ids"] if t in term_ids]) for g in base["reward_groups"]]; spec["reward_groups"]=[g for g in spec["reward_groups"] if g["term_ids"]]; spec.pop("reward_id"); spec["reward_id"]="rw_"+digest(spec)[:24]
        from molsteer.molthinker.planner import validate_spec
        validate_spec(spec,packet,report); result["spec"]=spec; return {"status":"accepted","reward_spec":spec}
    return tools+[record_task_plan,derive_reward_candidates,submit_reward_plan],result,base

def thinker_node(state, thinker):
    spec=thinker.run(state["packet"],state["diagnostic_report"]); state["reward_spec"]=spec; state["execution_request"]={**state.get("execution_request",{}),"reward_spec":spec}; state["route"]="executor"; state["status"]="executing"; state["step"]=state.get("step",0)+1
    from .trace import append_trace
    return append_trace(state,node="thinker",kind="decision",summary="Validated evidence-bound RewardSpec",evidence_ids=[t.get("reference_evidence_id") for t in spec.get("terms",[])],output={"reward_id":spec.get("reward_id"),"term_count":len(spec.get("terms",[]))})

__all__=["Thinker","thinker_node","thinker_tools"]
