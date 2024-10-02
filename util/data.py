from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
from numpy import ndarray
import cv2
import torch
import os
import json
import random

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation: str, data_dir: str, transforms=None, is_train: bool=True):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)
        
        if is_train:
            self.predictions = {
                "images": self.coco.dataset["images"].copy(),
                "categories": self.coco.dataset["categories"].copy(),
                "annotations": None
            }
            self.transforms = transforms
            
        self.image_id = self.coco.getImgIds()
        self.is_train=is_train

    def __getitem__(self, index: int) -> tuple[ndarray, dict[ndarray]]:
        
        image_id = self.image_id[index]

        image_info = self.coco.loadImgs(image_id)[0]
        
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        if self.is_train:
            boxes = np.array([x['bbox'] for x in anns])

            # boxex (x_min, y_min, x_max, y_max)
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            # torchvision faster_rcnn은 label=0을 background로 취급
            # class_id를 1~10으로 수정 
            labels = np.array([x['category_id']+1 for x in anns]) 
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            areas = np.array([x['area'] for x in anns])
            areas = torch.as_tensor(areas, dtype=torch.float32)
                                    
            is_crowds = np.array([x['iscrowd'] for x in anns])
            is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)

            target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                    'iscrowd': is_crowds}

            # transform
            if self.transforms:
                sample = {
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                }
                sample = self.transforms(**sample)
                image = sample['image']
                target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
            
            return image, target, image_id
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
            return image
    
    def __len__(self) -> int:
        return len(self.image_id)
    
def get_kfold_json(
    k:int = 5, 
    data_path: str = './dataset',
    annotation_file: str='./dataset/train.json',
    random_state: int=2024,
    force_create: bool=False
    ) -> tuple[tuple[str, str]]:
    
    folder_path = os.path.join(data_path, f'{k}-fold_json')
    
    if not(os.path.isdir(folder_path)) or force_create:
        return tuple(kfold_split_json(k, data_path, annotation_file, random_state))
    else:
        json_paths = os.listdir(folder_path)
        train_kfold_paths = sorted(os.path.join(folder_path, path) for path in json_paths if 'train' in path)
        val_kfold_paths = sorted(os.path.join(folder_path, path) for path in json_paths if 'val' in path)
        return tuple(zip(train_kfold_paths, val_kfold_paths))
        
def kfold_split_json(
    k:int = 5, 
    data_path: str = './dataset',
    annotation_file: str='./dataset/train.json',
    random_state: int=2024
    ) -> tuple[tuple[str], tuple[str]]:
    
    coco = COCO(annotation_file=annotation_file)
    
    img_ids = list(coco.imgs.keys())
    random.shuffle(img_ids)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    folds = list(kf.split(img_ids))
    
    folder_path = os.path.join(data_path, f'{k}-fold_json')
    os.makedirs(folder_path, exist_ok=True)
    
    print(f"creating {k}-Fold json fils at {folder_path}")
    train_kfold_paths = []
    val_kfold_paths = []

    for fold, (train_idx, val_idx) in enumerate(folds):
        train_img_ids = [img_ids[i] for i in train_idx]
        val_img_ids = [img_ids[i] for i in val_idx]
        
        train_kfold_path = os.path.join(folder_path, f'train_fold_{fold+1}.json')
        val_kfold_path = os.path.join(folder_path, f'val_fold_{fold+1}.json')
        create_subset_json(coco, train_img_ids, train_kfold_path)
        create_subset_json(coco, val_img_ids, val_kfold_path)
        
        train_kfold_paths.append(train_kfold_path)
        val_kfold_paths.append(val_kfold_path)
        print(f'Fold {fold+1}:')
        print(f'Training images: {len(train_img_ids)}, Validation images: {len(val_img_ids)}')
        
    return zip(train_kfold_paths, val_kfold_paths)
    
def create_subset_json(coco, img_ids, output_json):
    new_ann = {
        "images": [],
        "annotations": [],
        "categories": coco.loadCats(coco.getCatIds())
    }
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        new_ann["images"].append(img_info)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        new_ann['annotations'].extend(anns)
        
    with open(output_json, 'w') as f:
        json.dump(new_ann, f)
        
# if __name__=="__main__":
#     random.seed(2024)
#     a = get_kfold_json(k=2)