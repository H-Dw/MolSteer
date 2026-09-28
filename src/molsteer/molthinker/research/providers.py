"""Pluggable public-literature retrieval. Network errors remain explicit."""
from urllib.request import Request,urlopen
from urllib.parse import urlencode
import json


class EuropePMC:
    name='EuropePMC REST'
    def search(self,query,limit=5):
        if not isinstance(query,str) or not query.strip() or not 1<=limit<=25:
            raise ValueError('Nonempty query and bounded page size required')
        url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode(dict(query=query,format='json',resultType='core',pageSize=limit))
        request=Request(url,headers={'User-Agent':'MolSteer Researcher (literature evidence retrieval)'})
        with urlopen(request,timeout=25) as response:raw=response.read()
        data=json.loads(raw);records=[]
        for row in data.get('resultList',{}).get('result',[]):
            pmid=row.get('pmid');doi=row.get('doi')
            url_source=('https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/') if pmid else ('https://doi.org/'+doi if doi else 'https://europepmc.org/article/'+row['source']+'/'+row['id'])
            records.append(dict(source_key=('doi:'+doi.lower()) if doi else row['source']+':'+row['id'],
                title=row.get('title'),url=url_source,doi=doi,pmid=pmid,pmcid=row.get('pmcid'),
                authors=row.get('authorString'),publication_date=row.get('firstPublicationDate'),
                abstract=row.get('abstractText'),publication_types=row.get('pubTypeList',{}).get('pubType',[]),
                retrieval_scope='metadata and abstract; full text was not fetched by this provider'))
        return dict(request_url=url,raw_response=raw.decode('utf-8'),hit_count=data.get('hitCount'),records=records)
