from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
import numpy as np
import cv2
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
# faster rcnn model이 포함된 library
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader, Dataset

from util.data import CustomDataset
from config.custom_json_parser import Custom_json_parser
import pandas as pd
from tqdm import tqdm

def inference_fn(test_data_loader, model, device):
    outputs = []
    for images in tqdm(test_data_loader):
        # gpu 계산을 위해 image.to(device)
        images = list(image.to(device) for image in images)
        output = model(images)
        for out in output:
            outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
    return outputs

def main(test_config: dict, train_config: dict):
    test_dataset = CustomDataset(**test_config['data']['dataset'])
    
    test_data_loader = DataLoader(
        test_dataset,
        **test_config['data']['dataloader']
    )
    device = torch.device(test_config['device'])
    
    # torchvision model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=train_config['model']['pretrained'])
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, train_config['data']['num_classes'] + 1)
    model.to(device)
    model.load_state_dict(torch.load(test_config['checkpoint']['path'])['model_state_dict'])
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(test_config['data']['dataset']['annotation'])

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > test_config['result']['score_threshold']: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(test_config['result']['output_path'], "submission.csv"), index=None)
    print(submission.head())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cjp', '--config_json_path', type=str, required=True, help="include the config.json file for training", action="store")
    
    ## 입력 받은 경로의 config.json 경로 가져오기
    config_json_path = parser.parse_args().config_json_path
    
    ## 설정 및 하이퍼파라미터 가져오기
    config_parser = Custom_json_parser(mode="test", config_json_path=config_json_path)
    config = config_parser.get_config_from_json()

    config_parser = Custom_json_parser(mode="train", config_json_path=config['checkpoint']['train_config_json'])
    train_config = config_parser.get_config_from_json()
    
    main(config, train_config)