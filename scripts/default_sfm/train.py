#!/usr/bin/env python3
"""Explicit --train or disposable --smoke-steps entry point; no benchmark imports."""
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF','0')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from turbovla.training.sfm import main

if __name__=='__main__':
    main()
