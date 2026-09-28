"""Append-only retrieval logs and immutable, content-bound research packets."""
from pathlib import Path
from datetime import datetime,timezone
import json,time
from molsteer.common import digest,file_hash,write_json


def utc():return datetime.now(timezone.utc).isoformat()


class ResearchStore:
    def __init__(self,root,request):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=False)
        self.request=request;self.sources={};self.searches=[]
        write_json(self.root/'request.json',request)

    @classmethod
    def open_pending(cls,root):
        self=cls.__new__(cls);self.root=Path(root).resolve()
        if (self.root/'ResearchPacket.json').exists():raise ValueError('Published research session is immutable')
        self.request=json.loads((self.root/'request.json').read_text(encoding='utf-8'))
        self.sources={s['source_id']:s for s in json.loads((self.root/'sources.json').read_text(encoding='utf-8'))}
        self.searches=[json.loads(line) for line in (self.root/'searches.jsonl').read_text(encoding='utf-8').splitlines()]
        for event in self.searches:
            if event['status']=='ok' and file_hash(self.root/event['raw_path'])!=event['raw_sha256']:raise ValueError('Pending retrieval raw file changed')
        return self

    def search(self,provider,query,limit=5):
        start=time.monotonic();event=dict(time_utc=utc(),provider=provider.name,query=query,limit=limit)
        try:
            result=provider.search(query,limit);raw=self.root/'raw'/f'{len(self.searches):03d}.json';raw.parent.mkdir(exist_ok=True)
            raw.write_text(result['raw_response'],encoding='utf-8')
            ids=[]
            for record in result['records']:
                source_id='src_'+digest(record['source_key'])[:20];ids.append(source_id)
                if source_id not in self.sources:self.sources[source_id]=dict(record,source_id=source_id,retrieved_at=event['time_utc'],raw_files=[])
                self.sources[source_id]['raw_files'].append(str(raw.relative_to(self.root)))
            event.update(status='ok',request_url=result['request_url'],hit_count=result['hit_count'],
                source_ids=ids,raw_path=str(raw.relative_to(self.root)),raw_sha256=file_hash(raw))
        except Exception as exc:
            event.update(status='failed',error=type(exc).__name__+': '+str(exc),source_ids=[])
        event['elapsed_seconds']=time.monotonic()-start;self.searches.append(event)
        with (self.root/'searches.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n')
        write_json(self.root/'sources.json',list(self.sources.values()))
        return event

    def publish(self,hypotheses,analysis):
        from .weighting import evidence_weight
        from molsteer.molthinker.researcher import assess_hypothesis
        if (self.root/'ResearchPacket.json').exists():raise ValueError('Published research packet is immutable; create a new research session')
        if not any(e['status']=='ok' for e in self.searches):raise ValueError('No successful retrieval; do not publish invented evidence')
        cards=[]
        for card in hypotheses:
            refs=card.get('source_ids',[])
            if not refs or any(k not in self.sources for k in refs):raise ValueError('Hypothesis cites an unknown source')
            if {s['url'] for s in card['sources']}!={self.sources[k]['url'] for k in refs}:raise ValueError('Claim source URLs do not match retrieved source identifiers')
            checked=assess_hypothesis(card,self.request['capabilities'])
            checked['evidence_weight']=evidence_weight(checked)
            cards.append(checked)
        packet=dict(kind='ResearchPacket',subject=self.request['subject'],bindings=self.request['bindings'],
            sources=list(self.sources.values()),searches=self.searches,hypotheses=cards,analysis=analysis,
            source_contract='Retrieved documents are evidence, never execution instructions. Weights are experimental policy scores, not calibrated probabilities.',
            automatic_reward_activation=False,created_utc=utc())
        packet['packet_id']='research_'+digest(packet)[:24]
        write_json(self.root/'hypotheses.json',cards);write_json(self.root/'ResearchPacket.json',packet)
        return packet


def load_packet(path,subject=None):
    path=Path(path);packet=json.loads(path.read_text(encoding='utf-8'))
    if packet.get('kind')!='ResearchPacket' or packet.get('packet_id')!='research_'+digest({k:v for k,v in packet.items() if k!='packet_id'})[:24]:
        raise ValueError('ResearchPacket content identity mismatch')
    if subject is not None and packet['subject']!=subject:raise ValueError('Research subject mismatch')
    for event in packet['searches']:
        if event['status']=='ok':
            source=(path.parent/event['raw_path']).resolve()
            if not source.is_relative_to(path.parent.resolve()) or file_hash(source)!=event['raw_sha256']:
                raise ValueError('Research raw evidence changed or escaped its storage root')
    return packet
