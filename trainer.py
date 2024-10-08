import pandas as pd
from typing import Union
# 필요 library들을 import합니다.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from argparse import Namespace
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from collections import deque

import shutil

class Trainer: # 변수 넣으면 바로 학습되도록
    def __init__( # 여긴 config로 나중에 빼야하는지 이걸 유지하는지
        self, 
        model: nn.Module,
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        num_models_to_save: int,
        earlystop: int, 
        resume: bool,
        resume_model_path: str,
        config_json_path: dict[str, str],
        verbose: bool
    ):
        # 클래스 초기화: 모델, 디바이스, 데이터 로더 등 설정
        self.model = model  # 훈련할 모델
        self.device = device  # 연산을 수행할 디바이스 (CPU or GPU)
        self.train_loader = train_loader  # 훈련 데이터 로더
        self.val_loader = val_loader  # 검증 데이터 로더
        self.optimizer = optimizer  # 최적화 알고리즘
        self.scheduler = scheduler # 학습률 스케줄러
        self.loss_fn = loss_fn  # 손실 함수
        self.epochs = epochs  # 총 훈련 에폭 수
        self.result_path = result_path  # 모델 저장 경로
        
        self.k = num_models_to_save
        
        self.start_epoch = 0
        self.resume = resume
        self.resume_model_path = resume_model_path
        
        self.best_models = deque() # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        
        self.earlystop = earlystop
        
        self.verbose = verbose
        
        now = datetime.now()
        self.time = now.strftime('%Y-%m-%d_%H.%M.%S')
        self.checkpoint_path = os.path.join(self.result_path, self.time)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        self.save_exp_settings(config_json_path, self.checkpoint_path)
            
    def train_epoch(self, train_loader: DataLoader) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=self.verbose)
        
        for data in progress_bar:
            self.optimizer.zero_grad()
            
            loss_dict = self.model.train_step(data, self.optimizer)
            
            losses = loss_dict['loss']
            loss_value = losses.item()
            
            losses.backward()
            self.optimizer.step()
            
            train_loss += loss_value * data['img'].data[0].shape[0]
            progress_bar.set_postfix(loss=loss_value)
        
        train_loss = train_loss / train_loader.dataset.__len__()
        return train_loss

    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        # 모델의 검증을 진행
        self.model.eval()
        val_loss = 0.0

        progress_bar = tqdm(val_loader, desc="Validating", leave=False, disable=self.verbose)
  
        global log_images
        log_images= [] #wandb 로그에 올릴 이미지 저장
        
        with torch.no_grad():
            for data in progress_bar:
                data = [{k: data[k][i] for k in data} for i in range(len(next(iter(data.values()))))]
                
                for _data in data:
                    loss_dict = self.model.val_step(_data, self.optimizer)
                    
                    losses = loss_dict['loss']
                    loss_value = losses.item()
                    
                    val_loss += loss_value * _data['img'].data[0].shape[0]
                    progress_bar.set_postfix(loss=loss_value)

        val_loss = val_loss / (val_loader.dataset.__len__() * len(data))
        return val_loss

    def train(self) -> None:
        if self.resume:
            self.load_settings()
        # 전체 훈련 과정을 관리
        count = 0
        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss = self.train_epoch(self.train_loader)
            val_loss = self.validate(self.val_loader)
            
            print(f"Epoch {epoch+1}, Train Loss : {train_loss:.8f}, val_loss : {val_loss:.8f}")
            
            if train_loss < self.best_train_loss:
                self.best_train_loss = train_loss
            
            if val_loss < self.best_val_loss:
                print(f"best validation loss updated")
                count = 0
                self.best_val_loss = val_loss
                ## checkpoint 저장 코드
                checkpoint_name = f'cp_epoch{epoch + 1}_train_loss{train_loss:.4f}_val_loss{val_loss:.4f}.pth'
                self.save_checkpoint(path=self.checkpoint_path, name=checkpoint_name, epoch=epoch+1, train_loss=train_loss)
                self.keep_top_k_checkpoints(k=self.k)
            else:
                count += 1
                print("EarlyStop : 더이상 개선이 없어 학습이 중단됩니다")
                break 
            
            if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()
        
        print(f"training for {self.epochs} epochs finished")
        print(f"best train loss : {self.best_train_loss:.5f}, best val loss : {self.best_val_loss:.5f}")
        print(f"last train loss : {train_loss}, last val loss : {val_loss}")
    
    def save_checkpoint(self, path: str, name: str, epoch: int, train_loss: float) -> None:
        '''
        required parameters
            - path : checkpoint path
            - name : name of the checkpoint
            - epoch : current epoch
            - loss : current loss
        
        what it does
            - saves a checkpoint base on received parameters
            - keeps track with top 3 model (loss, name)
        '''
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': self.best_val_loss 
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = os.path.join(path, name)
        torch.save(checkpoint, checkpoint_path)
        self.best_models.appendleft(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
                
    def save_exp_settings(self, src: str, dst: str):
        try:
            shutil.copy(src, os.path.join(dst, "config.json"))
        except:
            raise Exception(f"could not copy {src} into {dst}")
    
    def keep_top_k_checkpoints(self, k: int) -> None:
        if len(self.best_models) > k:
            rm_checkpoint = self.best_models.pop()
            os.remove(rm_checkpoint)
            print(f"checkpoint removed : {rm_checkpoint}")
    
    def load_settings(self) -> None:
        ## 학습 재개를 위한 모델, 옵티마이저, 스케줄러 가중치 및 설정을 불러옵니다.
        print("loading prev training setttings")
        try:
            setting_info = torch.load(
                self.resume_model_path,
                map_location='cpu'
            )
            self.start_epoch = setting_info['epoch']
            self.model.load_state_dict(setting_info['model_state_dict'])
            self.optimizer.load_state_dict(setting_info['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(setting_info['scheduler_state_dict'])
            self.best_val_loss = setting_info['val_loss']
            print("loading successful")
        except:
            raise Exception('학습 재개를 위한 정보를 불러오는데 문제가 발생하였습니다')
        
def collate_fn(batch):
    return tuple(zip(*batch))

# 테스트용
if __name__=='__main__':
    ## device와 seed 설정
    from util.augmentation import TransformSelector
    from util.data import CustomDataset, get_kfold_json
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from util.optimizers import get_optimizer
    from util.schedulers import get_scheduler
    from config.custom_json_parser import Custom_json_parser
    
    config_json_path = "config/train_config.json"

    config_parser = Custom_json_parser(mode="train", config_json_path=config_json_path)
    config = config_parser.get_config_from_json()
    
    device = torch.device(config['device'])
    
    kfold_annotations = get_kfold_json(random_seed=config['random_seed'], **config['kfold'])
    
    img_size = config['data']['img_size']
    
    from mmdet.datasets import build_dataset, build_dataloader
    from mmcv import Config
    from mmdet.utils import get_device, build_dp
    
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    ## mmdetection implementation
    cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
    
    root = "./dataset/"

    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + '10-fold_json/train_fold_1.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (1024,1024) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + '10-fold_json/val_fold_1.json'
    cfg.data.val.pipeline[2]['img_scale'] = (1024, 1024)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize

    cfg.data.samples_per_gpu = 8

    cfg.seed = 2024
    cfg.gpu_ids = [0]
    cfg.work_dir = 'asd'

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    
    train_datasets = build_dataset(cfg.data.train)
    val_datasets = build_dataset(cfg.data.val) #, dict(test_mode=False))
    
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']

    train_dataloader_default_args = dict(
        samples_per_gpu=8,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    
    val_dataloader_default_args = dict(
        samples_per_gpu=4,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        shuffle=False,
        runner_type=runner_type,
        persistent_workers=False)

    val_dataloader_args = {
        **val_dataloader_default_args,
        **cfg.data.get('val_dataloader', {})
    }
    
    val_dataloader = build_dataloader(val_datasets, **val_dataloader_args)
    train_dataloader = build_dataloader(train_datasets, **train_loader_cfg)
    

    from mmdet.models import build_detector
    
    model = build_detector(cfg.model)
    model.init_weights()
    
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    
    ## end mmdetection implementation
    ## Optimizer 
    optimizer = get_optimizer(config['optimizer']['name'], model, config['optimizer']['kwargs'])

    ## Loss
    loss = None
    
    ## Scheduler
    scheduler = get_scheduler(config['scheduler']['name'], optimizer, config['scheduler']['kwargs'])
    
    model.to(device)    

    ## 학습 시작
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        config_json_path=config_json_path,
        **config['trainer']
    )

    trainer.train()
    