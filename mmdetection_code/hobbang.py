from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmengine.runner import Runner

register_all_modules()

cfg = Config.fromfile('/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/configs/cascade_rcnn_swin.py')

cfg.work_dir = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/cascade_rcnn_swin'

runner = Runner.from_cfg(cfg)
runner.train()