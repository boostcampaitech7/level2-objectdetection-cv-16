import os
from argparse import Namespace

import random
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.data import CustomDataset, get_kfold_json
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from util.losses import CustomLoss
from util.schedulers import get_scheduler
from trainer import Trainer

from config.custom_json_parser import Custom_json_parser

from model.model_selection import ModelSelector

def collate_fn(batch):
    return tuple(zip(*batch))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def run_train(config_json_path, config: dict) -> None:
    
    device = torch.device(config['device'])
    
    kfold_annotations = get_kfold_json(random_seed=config['random_seed'], **config['kfold'])
    
    img_size = config['data']['img_size']
    
    ## 데이터 증강 및 세팅
    transform_selector = TransformSelector(transform_type=config['augmentation']['name'],
                                           common_transform=config['augmentation']['common_transform'])
    
    train_transform = transform_selector.get_transform(augment=True,
                                                       kwargs=config['augmentation']['train_transform'])
    val_transform = transform_selector.get_transform(augment=False, 
                                                     kwargs=config['augmentation']['val_transform'])

    train_dataset = CustomDataset(config['data']['train_json'], config['data']['data_root'], transforms=train_transform)
    val_dataset = CustomDataset(config['data']['val_json'], config['data']['data_root'], transforms=val_transform)
    
    g = torch.Generator()
    g.manual_seed(config['random_seed'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )
    
    ## 학습 모델
    
    # model = model_selector.get_model()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=config['model']['pretrained'])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config['data']['num_classes']+1)
    model.to(device)
    
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

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cjp', '--config_json_path', type=str, required=True, help="include the config.json file for training", action="store")
    
    ## 입력 받은 경로의 config.json 경로 가져오기
    config_json_path = parser.parse_args().config_json_path
    
    ## 설정 및 하이퍼파라미터 가져오기
    config_parser = Custom_json_parser(mode="train", config_json_path=config_json_path)
    config = config_parser.get_config_from_json()
    
    # cuda 적용
    if config['device'].lower() == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cpu': print("cuda를 찾을 수 없어 cpu로 학습을 진행합니다")
    else:
        device = 'cpu'

    # seed값 설정
    seed = config['random_seed']
    deterministic = True

    random.seed(seed) # random seed 고정
    np.random.seed(seed) # numpy random seed 고정
    torch.manual_seed(seed) # torch random seed 고정
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_train(config_json_path, config)
    
    