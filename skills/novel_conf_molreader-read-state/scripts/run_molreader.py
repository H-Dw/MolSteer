"""Run the installed MolReader package or an explicitly selected source project."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--project',type=Path,default=os.environ.get('MOLREADER_HOME'))
p.add_argument('arguments',nargs=argparse.REMAINDER)
a=p.parse_args()
arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
if not arguments:p.error('Supply a MolReader subcommand after --')
if a.project and not (Path(a.project)/'molreader'/'cli.py').is_file():p.error('Project path does not contain the MolReader package')
raise SystemExit(subprocess.run([sys.executable,'-m','molreader',*arguments],cwd=a.project).returncode)
