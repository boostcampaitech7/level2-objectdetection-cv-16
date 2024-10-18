import argparse
import torch
import os

from mmengine.config import Config 
from mmengine.runner import Runner
from mmdet.registry import VISUALIZERS
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Train detection')
    parser.add_argument('--config', default='mmdetection_code/exp_configs.py', type=str, help='train config file path')
    parser.add_argument('--work-dir', default='work_dir/DETR', type=str, help='the dir to save logs and models')
    parser.add_argument(
    '--resume',
    nargs='?',
    type=str,
    const='auto',
    help='If specify checkpoint path, resume from it, while if not '
    'specify, try to auto resume from the latest checkpoint '
    'in the work directory.')
    parser.add_argument('--seed', type=int, default=2024, help='random_seed to be fixed')
    return parser.parse_args()
    
def main():
    args = parse_args()
    register_all_modules()
    
    cfg = Config.fromfile(args.config)
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir 
        
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
    else:
        cfg.resume = False
        cfg.load_from = None
    
    print(cfg.pretty_text)
    
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__=="__main__":
    main()