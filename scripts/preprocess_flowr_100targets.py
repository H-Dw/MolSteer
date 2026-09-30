#!/usr/bin/env python3
"""Run the checkout's dataset preprocessor without requiring an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from molsteer.preprocessing.flowr_dataset import main

if __name__ == "__main__":
    raise SystemExit(main())
