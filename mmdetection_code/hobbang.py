from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmengine.runner import Runner

register_all_modules()

cfg = Config.fromfile('/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/configs/faster-rcnn_r50_fpn_1x_coco_config.py')

cfg.work_dir = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/faster-rcnn_r50_fpn_1x_coco'

runner = Runner.from_cfg(cfg)
runner.train()