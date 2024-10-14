import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
import argparse
# 명령줄 인자 파싱
parser = argparse.ArgumentParser(description='Inference with MMDetection model')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
parser.add_argument('--output', type=str, default='submission.csv', help='Output file name')
args = parser.parse_args()
# config file 들고오기
cfg = Config.fromfile(args.config)
# 모델 초기화
model = init_detector(cfg, args.checkpoint, device='cuda:0')

# COCO 객체 생성
coco = COCO('/data/ephemeral/home/kwak/level2-objectdetection-cv-16/dataset/test.json')
img_ids = coco.getImgIds()
prediction_strings = []
file_names = []
class_num = 10
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_names.append(img_info['file_name'])

    # 추론 수행
    img_path = os.path.join('/data/ephemeral/home/kwak/level2-objectdetection-cv-16/dataset', img_info['file_name'])
    result = inference_detector(model, img_path)

    prediction_string = ''
    if hasattr(result, 'pred_instances'):
        pred_instances = result.pred_instances.cpu().numpy()
        for i in range(len(pred_instances.bboxes)):
            label = pred_instances.labels[i]
            score = pred_instances.scores[i]
            bbox = pred_instances.bboxes[i]

            if score < 0.05: # 임계값 미만의 결과는 무시
                continue

            prediction_string += f"{label} {score:.4f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "

    prediction_strings.append(prediction_string.strip())
    
submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission_file_path = os.path.join(args.output)
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved at: {submission_file_path}")
print(submission.head())
