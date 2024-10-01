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
        
        self.best_epochs = deque() # 가장 좋은 상위 3개 모델의 정보를 저장할 리스트
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        
        self.earlystop = None
        
        self.verbose = False
        
        now = datetime.now()
        self.time = now.strftime('%Y-%m-%d_%H.%M.%S')
        self.checkpoint_path = os.path.join(self.result_path, self.time)
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> float:
        # 한 에폭 동안의 훈련을 진행
        self.model.train()
        
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False, disable=self.verbose)
        
        for images, targets, image_ids in progress_bar:
            images = list(image.float().to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            loss_dict = self.model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            losses.backward()
            self.optimizer.step()
            
            train_loss += loss_value * train_loader.batch_size
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
            for images, targets, image_ids in progress_bar:
                images = list(image.float().to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                ## loss값을 뽑아내도록 해놓긴 했으나, eval모드에서 bbox 예측값을 가지고 처리할 수 있도록
                ## 작성 해야함 - 추후 수정 예정
                self.model.train()
                loss_dict = self.model(images, targets)
                self.model.eval()
                
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                val_loss += loss_value * val_loader.batch_size
                progress_bar.set_postfix(loss=loss_value)

        val_loss = val_loss / val_loader.dataset.__len__()
        return val_loss

    def train(self) -> None:
        # 전체 훈련 과정을 관리
        count = 0
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss = self.train_epoch(self.train_loader)
            val_loss = self.validate(self.val_loader)
            
            print(f"Epoch {epoch+1}, Train Loss : {train_loss:.8f}, val_loss : {val_loss:.8f}")
            
            if val_loss < self.best_val_loss:
                print(f"best validation loss updated")
                count = 0
                self.best_val_loss = val_loss
                ## checkpoint 저장 코드
                checkpoint_name = f'cp_epoch{epoch + 1}_train_loss{train_loss:.4f}_val_loss{val_loss:.4f}.pth'
                self.save_checkpoint(path=self.checkpoint_path, name=checkpoint_name, epoch=epoch+1, train_loss=train_loss)
                self.keep_topN_checkpoints(n=3)
            else:
                count += 1
                assert count != self.earlystop, "EarlyStop - 더이상 개선이 없어 학습이 중단됩니다"
                
        ## 최종 checkpoint 저장 코드
    
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
        self.best_epochs.appendleft([checkpoint_path])
        print(f"Checkpoint saved at {checkpoint_path}")
        
    def keep_topN_checkpoints(self, n: int) -> None:
        if len(self.best_epochs) > n:
            rm_checkpoint = self.best_epochs.pop()
            os.remove(rm_checkpoint[1])
            print(f"checkpoint removed : {rm_checkpoint[1]}")
    
    def load_settings(self) -> None:
        ## 학습 재개를 위한 모델, 옵티마이저, 스케줄러 가중치 및 설정을 불러옵니다.
        print("loading prev training setttings")
        try:
            setting_info = torch.load(
                self.weights_path,
                map_location='cpu'
            )
            self.start_epoch = setting_info['epoch']
            self.model.load_state_dict(setting_info['model_state_dict'])
            self.optimizer.load_state_dict(setting_info['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(setting_info['scheduler_state_dict'])
            print("loading successful")
        except:
            raise Exception('학습 재개를 위한 정보를 불러오는데 문제가 발생하였습니다')
        
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__=='__main__':
        ## device와 seed 설정
    from util.augmentation import TransformSelector
    from util.data import CustomDataset
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    device = torch.device('cuda')

    train_annotation = 'dataset/5-fold_json/train_fold_1.json'
    val_annotation = 'dataset/5-fold_json/val_fold_1.json'
    
    data_root = './dataset'

    transform_type = 'albumentations'
    height = 1024
    width = 1024
    
    num_classes = 1 + 10
    
    batch_size = 16
    epochs = 10
    
    result_path = './checkpoints'
    ## 데이터 증강 및 세팅
    transform_selector = TransformSelector(transform_type=transform_type)
    
    train_transform = transform_selector.get_transform(augment=True, 
                                                       height=height, width=width)
    val_transform = transform_selector.get_transform(augment=False, 
                                                     height=height, width=width)

    train_dataset = CustomDataset(train_annotation, data_root, transforms=train_transform)
    val_dataset = CustomDataset(val_annotation, data_root, transforms=val_transform)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
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
        val_loader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss,
        epochs=epochs,
        result_path=result_path,
    )

    trainer.train()