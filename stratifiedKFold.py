import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pycocotools.coco import COCO
import os

# 사용자 정의 함수: create_subset_json
def create_subset_json(coco, img_ids, output_json):
    new_ann = {
        "images": [],
        "annotations": [],
        "categories": coco.loadCats(coco.getCatIds())
    }

    for img_id in img_ids:
        # img_id가 COCO 데이터셋에 존재하는지 확인
        if len(coco.loadImgs(img_id)) == 0:
            print(f"Warning: Image ID {img_id} not found in COCO dataset. Skipping this ID.")
            continue
        
        img_info = coco.loadImgs(img_id)[0]
        new_ann["images"].append(img_info)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        new_ann['annotations'].extend(anns)
        
    with open(output_json, 'w') as f:
        json.dump(new_ann, f)

# 1. COCO 데이터셋 로드
coco = COCO('/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/train.json')

# 2. 이미지 ID와 레이블 정보 추출
image_ids = coco.getImgIds()
categories = coco.loadCats(coco.getCatIds())
num_classes = len(categories)

# 각 이미지에 대한 다중 레이블 벡터 생성
image_labels = {img_id: [0] * num_classes for img_id in image_ids}

# 각 이미지에 해당하는 레이블을 다중 레이블 벡터에 설정
for ann in coco.loadAnns(coco.getAnnIds()):
    img_id = ann['image_id']
    category_id = ann['category_id'] - 1  # 카테고리 ID가 1부터 시작하므로 -1로 인덱스 조정
    image_labels[img_id][category_id] = 1

# 3. MultilabelStratifiedKFold 사용을 위해 X와 y 배열 생성
X = np.array(list(image_labels.keys()))  # 이미지 ID 리스트
y = np.array(list(image_labels.values()))  # 다중 레이블 리스트

# 4. MultilabelStratifiedKFold 적용
n_splits = 5
mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 5. 폴드별 데이터셋 생성 및 저장
output_dir = "/data/ephemeral/home/whth/level2-objectdetection-cv-16/kfold_dataset"
os.makedirs(output_dir, exist_ok=True)
folds = list(mskf.split(X, y))

for fold_idx, (train_index, val_index) in enumerate(folds):
    train_img_ids = [image_ids[i] for i in train_index]
    val_img_ids = [image_ids[i] for i in val_index]
    
    # 각 폴드에 대해 train/val json 파일 생성
    train_json_path = os.path.join(output_dir, f'train_fold_{fold_idx + 1}.json')
    val_json_path = os.path.join(output_dir, f'val_fold_{fold_idx + 1}.json')
    
    # train/val JSON 파일 생성
    create_subset_json(coco, train_img_ids, train_json_path)
    create_subset_json(coco, val_img_ids, val_json_path)
    
    print(f'Fold {fold_idx + 1} saved: {len(train_img_ids)} train samples, {len(val_img_ids)} val samples')
