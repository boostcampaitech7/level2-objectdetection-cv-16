import os
from argparse import Namespace

from args import Custom_arguments_parser
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

from model import model_selection

from util.data import CustomDataset, HoDataLoad   # hobbang: Dataset, DataLoader 코드 하나로 합체
from util.augmentation import TransformSelector
from util.optimizers import get_optimizer
from util.losses import CustomLoss
from util.schedulers import get_scheduler
from trainer import Trainer

from model.model_selection import ModelSelector

def collate_fn(batch):
    return tuple(zip(*batch))

def run_train(args:Namespace) -> None:
    ## device와 seed 설정
    device = torch.device('cuda')

    annotation = './dataset/train.json'
    data_root = './dataset'

    transform_type = 'albumentations'
    height = 1024
    width = 1024
    
    num_classes = 1 + 10
    
    batch_size = 16
    epochs = 10
    
    result_path = './temp'
    ## 데이터 증강 및 세팅
    transform_selector = TransformSelector(transform_type=transform_type)
    
    
    train_transform = transform_selector.get_transform(augment=True, height=height, width=width)

    train_dataset = CustomDataset(annotation, data_root, transforms=train_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    ## 학습 모델
    
    # model = model_selector.get_model()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    
    ## Optimizer 
    # optimizer = get_optimizer(model, optimizer_type, lr)
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    ## Loss
    loss = None
    
    ## Scheduler
    scheduler = None
    
    model.to(device)    

    ## 학습 시작
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_dataloader,
        val_loader=None,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        epochs=epochs,
        result_path=result_path,
    )

    trainer.train()

if __name__=='__main__':
    
    ## 설정 및 하이퍼파라미터 가져오기
    train_parser = Custom_arguments_parser(mode='train')
    args = train_parser.get_parser()
    
    # cuda 적용
    if args.device.lower() == 'cuda':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == 'cuda', 'cuda로 수행하려고 하였으나 cuda를 찾을 수 없습니다.'
    else:
        device = 'cpu'

    # seed값 설정
    seed = args.seed
    deterministic = True

    random.seed(seed) # random seed 고정
    np.random.seed(seed) # numpy random seed 고정
    torch.manual_seed(seed) # torch random seed 고정
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_train(args)
    
    