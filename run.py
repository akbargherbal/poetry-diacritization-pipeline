#!/usr/bin/env python3
"""
Usage:
    python run.py status
    python run.py generate
    python run.py validate
    python run.py threshold --cutoff 0.90
    python run.py all --cutoff 0.90
    python run.py export

See README.md for the full explanation of each stage.
"""
from poetry_diacritization.cli import main

if __name__ == "__main__":
    main()
