#!/usr/bin/env python3
"""Verify online W&B initialization and final sync without training a model."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('USE_TF','0')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from turbovla.training.sfm import start_wandb,finish_wandb
from turbovla.training.sfm_utils import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text())
    config['validation_only']=True
    config['formal_training_started']=False
    run=start_wandb(config,args.output,job_type='setup_validation')
    run.log({'setup/online_logging_test':1,'setup/optimizer_steps':0,'setup/formal_training_started':0},step=0)
    finish_wandb(run,args.output,0,'validated')
    write_json(args.output/'result.json',dict(status='passed',run_id=run.id,url=run.url,mode='online',
                                            final_sync_verified=True,formal_training_started=False,optimizer_steps=0))
    print((args.output/'result.json').read_text())


if __name__=='__main__': main()
