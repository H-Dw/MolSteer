import argparse,json
from molsteer.reporting.research_report import build

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    print(json.dumps(build(p.parse_args().root)))
