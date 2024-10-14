import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pycocotools.coco import COCO
import os

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
        json.dump(new_ann, f, indent=4)

def multilabelStratifiedKFold_split_json(
    k:int = 5, 
    output_dir: str = "/data/ephemeral/home/whth/level2-objectdetection-cv-16/kfold_dataset",
    annotation_file: str='/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/train.json',
    random_state: int=2024
    ) -> None:

    coco = COCO(annotation_file=annotation_file)

    image_ids = coco.getImgIds()
    categories = coco.loadCats(coco.getCatIds())
    num_classes = len(categories)

    image_labels = {img_id: [0] * num_classes for img_id in image_ids}

    for ann in coco.loadAnns(coco.getAnnIds()):
        img_id = ann['image_id']
        category_id = ann['category_id'] - 1 
        image_labels[img_id][category_id] = 1

    X = np.array(list(image_labels.keys()))  # 이미지 ID 리스트
    y = np.array(list(image_labels.values()))  # 다중 레이블 리스트

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    os.makedirs(output_dir, exist_ok=True)
    folds = list(mskf.split(X, y))

    for fold_idx, (train_index, val_index) in enumerate(folds):
        train_img_ids = [image_ids[i] for i in train_index]
        val_img_ids = [image_ids[i] for i in val_index]
        
        train_json_path = os.path.join(output_dir, f'train_fold_{fold_idx + 1}.json')
        val_json_path = os.path.join(output_dir, f'val_fold_{fold_idx + 1}.json')
        
        create_subset_json(coco, train_img_ids, train_json_path)
        create_subset_json(coco, val_img_ids, val_json_path)
        
        print(f'Fold {fold_idx + 1} saved: {len(train_img_ids)} train samples, {len(val_img_ids)} val samples')

if __name__=='__main__':
    # 여기서 k개의 폴드로 나눌지, output_dir 경로, annotation_file 경로, random_state 원하면 설정
    k = 2
    output_dir = "/data/ephemeral/home/whth/level2-objectdetection-cv-16/kfold_dataset"
    annotation_file ='/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/train.json'
    multilabelStratifiedKFold_split_json(k=k, output_dir=output_dir, annotation_file=annotation_file)