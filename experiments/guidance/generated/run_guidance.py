#!/usr/bin/env python3
from pathlib import Path
import json
from molsteer.molexecutor.runner import run
if __name__ == "__main__":
    run(json.loads(Path(__file__).with_name("execution.json").read_text(encoding="utf-8")))
