"""Render existing verified artifacts without rerunning metrics or optimization."""
import argparse
import json
from pathlib import Path
from molreader.localized_report import validate_localized_report
from molsteer.molreader.reporting import render_diagnostic
from molsteer.molthinker.reporting import render_derivation


def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',required=True);a=p.parse_args()
    for path in Path(a.reports).glob('*/*/*/DiagnosticReport.json'):
        packet=json.loads(path.with_name('StatePacket.json').read_text(encoding='utf-8'))
        report=json.loads(path.read_text(encoding='utf-8'));validate_localized_report(report,packet)
        for lang in ['en','zh']:
            path.with_name(f'DiagnosticReport.{lang}.md').write_text(render_diagnostic(report,lang),encoding='utf-8')
        reward=path.with_name('RewardSpec.json')
        if reward.exists():
            spec=json.loads(reward.read_text(encoding='utf-8'))
            monitor=json.loads(path.with_name('ExecutionMonitor.json').read_text(encoding='utf-8'))
            for lang in ['en','zh']:
                path.with_name(f'RewardDerivation.{lang}.md').write_text(render_derivation(spec,monitor,lang),encoding='utf-8')


if __name__=='__main__':main()
