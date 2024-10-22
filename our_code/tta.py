import os
import json
from tqdm import tqdm
import pandas as pd
from mmdet.apis import init_detector, inference_detector
from mmcv.transforms import Compose


root_path = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/dataset/'
json_file_path = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/dataset/test.json'

config_file = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/cascade_rcnn_swin_last/cascade_rcnn_swin.py'
checkpoint_file = '/data/ephemeral/home/hobbang/level2-objectdetection-cv-16/work_dirs/cascade_rcnn_swin_last/epoch_15.pth'

csv_name = 'cascade_last.csv'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test_pipeline=None

# test_pipeline=[
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(type='Resize', scale=(1024,1024), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375,],type='Normalize')],
#             [dict(type='Resize', scale=(1333, 800), keep_ratio=True), dict(type='Resize', scale=(666, 400), keep_ratio=True), dict(type='Resize', scale=(2000, 1200), keep_ratio=True)], 
#             [dict(prob=0.0, type='RandomFlip'), dict(prob=1.0, type='RandomFlip')],
#             [dict(crop_size=[1024,1024,], type='RandomCrop'),dict(crop_size=[512,512,], type='RandomCrop')],
#             [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
#         ]
#     )
# ]

test_pipeline=[ ################ 새로 돌린거에 적용한 TTA
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1024,1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375,],type='Normalize')],
            [dict(type='Resize', scale=(1333, 800), keep_ratio=True), dict(type='Resize', scale=(666, 400), keep_ratio=True), dict(type='Resize', scale=(2000, 1200), keep_ratio=True)], 
            [dict(prob=0.0, type='RandomFlip'), dict(prob=1.0, type='RandomFlip')],
            [dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction'))]
        ]
    )
]

with open(os.path.join(root_path, json_file_path), 'r') as f:
    data = json.load(f)

prediction_strings = []
file_names = []


for image_info in tqdm(data['images']):
    prediction_string = ''
    file_name = image_info['file_name']
    
    result = inference_detector(model, os.path.join(root_path, file_name), test_pipeline=Compose(test_pipeline))
    
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    class_ids = result.pred_instances.labels.cpu().numpy()

    # 결과 처리
    for i in range(len(scores)):
        score = scores[i]
        x_min, y_min, x_max, y_max = bboxes[i]
        
        prediction_string += f'{class_ids[i]} {score:.2f} {x_min:.2f} {y_min:.2f} {x_max:.2f} {y_max:.2f} '
    prediction_strings.append(prediction_string)
    file_names.append(file_name)

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names

output_path = './default'
os.makedirs(output_path, exist_ok=True) 
submission.to_csv(os.path.join(output_path, csv_name), index=False)
submission.head()