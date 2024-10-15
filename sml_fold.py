import json
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pycocotools.coco import COCO
import os
import json

def load_coco_json(json_file):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def filter_small_images(coco_data):
    small_image_ids = set()
    
    for ann in coco_data['annotations']:
        x, y, width, height = ann['bbox']
        area = width * height
        
        if area < 32 * 32:
            small_image_ids.add(ann['image_id'])
    
    new_coco_data = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories']
    }

    for img in coco_data['images']:
        if img['id'] in small_image_ids:
            new_coco_data['images'].append(img)

    for ann in coco_data['annotations']:
        if ann['image_id'] in small_image_ids:
            new_coco_data['annotations'].append(ann)

    return new_coco_data

def filter_small_annotations(coco_data):
    new_annotations = []

    for ann in coco_data['annotations']:
        x, y, width, height = ann['bbox']
        area = width * height

        if area < 32 * 32:
            new_annotations.append(ann)

    coco_data['annotations'] = new_annotations
    return coco_data

def save_coco_json(coco_data, output_file):
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

input_file = '/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/train.json'
output_file = '/data/ephemeral/home/whth/level2-objectdetection-cv-16/dataset/size_fold'

coco_data = load_coco_json(input_file)
small_coco_data = filter_small_images(coco_data)
small_annotations_data = filter_small_annotations(small_coco_data)
save_coco_json(small_annotations_data, output_file)
