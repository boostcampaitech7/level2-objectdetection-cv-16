import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# 1. train.json 파일 로드
with open('/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/train.json', 'r') as f:
    data = json.load(f)

# 2. 이미지 및 레이블 추출
image_ids = [img['id'] for img in data['images']]
annotations = data['annotations']

# 3. 카테고리 정보 추출
categories = [
    "General trash", "Paper", "Paper pack", "Metal", "Glass", 
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"
]
num_classes = len(categories)

# 4. 각 이미지에 해당하는 레이블 벡터 생성
# 모든 카테고리를 기준으로 다중 레이블 벡터 생성
image_labels = {img_id: [0] * num_classes for img_id in image_ids}

for ann in annotations:
    image_id = ann['image_id']
    category_id = ann['category_id']
    # category_id는 0부터 시작하도록 설정
    image_labels[image_id][category_id] = 1

# 5. X와 y 배열 생성
X = np.array(list(image_labels.keys()))  # 이미지 ID 리스트
y = np.array(list(image_labels.values()))  # 다중 레이블 리스트

# 6. MultilabelStratifiedKFold 적용
n_splits = 5  # 원하는 폴드 개수
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 7. 각 폴드에 대해 데이터 분할 및 파일 저장
for fold_idx, (train_index, val_index) in enumerate(mskf.split(X, y)):
    train_ids = X[train_index]
    val_ids = X[val_index]
    
    # train/val json 저장을 위한 데이터셋 재구성
    train_data = {
        "images": [img for img in data['images'] if img['id'] in train_ids],
        "annotations": [ann for ann in data['annotations'] if ann['image_id'] in train_ids],
        "categories": data['categories']
    }
    val_data = {
        "images": [img for img in data['images'] if img['id'] in val_ids],
        "annotations": [ann for ann in data['annotations'] if ann['image_id'] in val_ids],
        "categories": data['categories']
    }
    
    # 8. JSON 파일로 저장
    with open(f'/data/ephemeral/home/whth/level2-objectdetection-cv-16/kfold_dataset/train_fold_{fold_idx + 1}.json', 'w') as train_file:
        json.dump(train_data, train_file)
    
    with open(f'/data/ephemeral/home/whth/level2-objectdetection-cv-16/kfold_dataset/val_fold_{fold_idx + 1}.json', 'w') as val_file:
        json.dump(val_data, val_file)
    
    print(f'Fold {fold_idx + 1} saved: {len(train_ids)} train samples, {len(val_ids)} val samples')
