"""Provider-neutral search/fetch boundary. Fetch accepts retrieved identifiers only."""
from typing import Protocol
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
import re
from html.parser import HTMLParser
from molsteer.common import digest
from .providers import EuropePMC


class ResearchProvider(Protocol):
    name: str
    def search(self, query: str, limit: int = 5) -> dict: ...
    def fetch(self, source_id: str) -> dict: ...


def read_public(url):
    with urlopen(Request(url, headers={'User-Agent': 'MolSteer/0.1 literature research'}), timeout=25) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError('Source exceeds retrieval size budget')
    return raw.decode('utf-8')


class LiteratureProvider:
    """Search abstracts; fetch Europe PMC full text or an arXiv abstract explicitly."""
    def __init__(self, name):
        if name not in ('europe_pmc', 'arxiv'):
            raise ValueError('Unknown literature provider')
        self.name, self.sources = name, {}

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip() or not 1 <= limit <= 10:
            raise ValueError('Invalid research query')
        if self.name == 'europe_pmc':
            rows = EuropePMC().search(query, limit)['records']
        else:
            url = 'https://export.arxiv.org/api/query?' + urlencode(
                {'search_query': 'all:' + query, 'start': 0, 'max_results': limit})
            root = ET.fromstring(read_public(url))
            ns = {'a': 'http://www.w3.org/2005/Atom'}
            rows = []
            for entry in root.findall('a:entry', ns):
                ident = entry.findtext('a:id', '', ns).rsplit('/abs/', 1)[-1]
                if not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?', ident):
                    continue
                rows.append(dict(source_key='arxiv:'+ident, arxiv_id=ident, title=entry.findtext('a:title', '', ns),
                                 url='https://arxiv.org/abs/'+ident,
                                 abstract=entry.findtext('a:summary', '', ns),
                                 publication_date=entry.findtext('a:published', '', ns),
                                 retrieval_scope='metadata and abstract'))
        clean = []
        for row in rows:
            row = dict(row, source_id='src_'+digest(row['source_key'])[:20])
            self.sources[row['source_id']] = row
            clean.append(row)
        return dict(status='ok' if clean else 'zero_hits', records=clean)

    def fetch(self, source_id):
        row = self.sources[source_id]
        if self.name == 'arxiv':
            try:
                raw=read_public('https://arxiv.org/html/'+row['arxiv_id'])
                class PassageParser(HTMLParser):
                    def __init__(self):super().__init__();self.parts=[];self.skip=0
                    def handle_starttag(self,tag,attrs):
                        if tag in ('script','style'):self.skip+=1
                        if tag=='math':
                            formula=dict(attrs).get('alttext')
                            if formula:self.parts.append(formula)
                    def handle_endtag(self,tag):
                        if tag in ('script','style'):self.skip=max(0,self.skip-1)
                    def handle_data(self,data):
                        if not self.skip and data.strip():self.parts.append(data.strip())
                parser=PassageParser();parser.feed(raw);passage='\n'.join(parser.parts)
                return dict(row,excerpt=passage[:40000],content_sha256=digest(raw),
                            retrieval_scope='inspected arXiv HTML; MathML alttext preserved',truncated=len(passage)>40000)
            except Exception as exc:
                return dict(row,excerpt=row.get('abstract') or '',full_text_error=type(exc).__name__,
                            retrieval_scope='inspected abstract only; HTML full text unavailable')
        pmcid = row.get('pmcid')
        if self.name == 'europe_pmc' and pmcid and re.fullmatch(r'PMC\d+', pmcid):
            raw = read_public('https://www.ebi.ac.uk/europepmc/webservices/rest/'+pmcid+'/fullTextXML')
            xml = ET.fromstring(raw)
            passage = '\n'.join(' '.join(p.itertext()) for p in xml.iter('p'))
            return dict(row, excerpt=passage[:40000], content_sha256=digest(raw),
                        retrieval_scope='full-text XML paragraphs; bounded excerpt', truncated=len(passage)>40000)
        return dict(row, excerpt=row.get('abstract') or '',
                    retrieval_scope='inspected abstract only; full text unavailable')


class WebProvider:
    """Host-injected search and optional read client; credentials stay in the host."""
    name = 'web'

    def __init__(self, search_fn, fetch_fn=None):
        self.search_fn, self.fetch_fn, self.sources = search_fn, fetch_fn, {}

    def search(self, query, limit=5):
        from urllib.parse import urlsplit
        rows = self.search_fn(query)
        if not isinstance(rows, list):
            raise ValueError('Search client must return records')
        clean = []
        for item in rows[:limit]:
            url = str(item.get('url', ''))
            parsed = urlsplit(url)
            if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password:
                continue
            row = dict(source_id='src_'+digest(url)[:20], source_key=url, url=url,
                       title=str(item.get('title', ''))[:500], snippet=str(item.get('snippet', ''))[:2000],
                       retrieval_scope='search snippet only')
            self.sources[row['source_id']] = row
            clean.append(row)
        return dict(status='ok' if clean else 'zero_hits', records=clean)

    def fetch(self, source_id):
        row = self.sources[source_id]
        if self.fetch_fn is None:
            return dict(row, status='unavailable', excerpt='', reason='No host fetch client')
        passage = self.fetch_fn(row['url'])
        if not isinstance(passage, str):
            raise ValueError('Fetch client must return text')
        return dict(row, excerpt=passage[:40000], content_sha256=digest(passage),
                    retrieval_scope='host-fetched source passage', truncated=len(passage)>40000)
