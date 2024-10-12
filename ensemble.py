import pandas as pd
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

def parse_prediction_string(pred_str):
    records = pred_str.strip().split(' ')
    boxes = []
    scores = []
    labels = []
    for i in range(0, len(records), 6):
        label = int(records[i])
        score = float(records[i + 1])
        x_min = float(records[i + 2])
        y_min = float(records[i + 3])
        x_max = float(records[i + 4])
        y_max = float(records[i + 5])
        
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)
        labels.append(label)
    
    return np.array(boxes), np.array(scores), np.array(labels)

def format_prediction_string(boxes, scores, labels):
    pred_str = ""
    for i in range(len(boxes)):
        pred_str += f"{int(labels[i])} {scores[i]:.4f} {boxes[i][0]:.2f} {boxes[i][1]:.2f} {boxes[i][2]:.2f} {boxes[i][3]:.2f} "
    return pred_str.strip()

def perform_wbf_on_files(file_paths, iou_thr=0.55, skip_box_thr=0.001, weights=None):
    all_predictions = []
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            image_id = row['image_id']
            pred_str = row['PredictionString']

            if isinstance(pred_str, str) and pred_str.strip():
                boxes, scores, labels = parse_prediction_string(pred_str)
                boxes = boxes / 1024.0
                
                all_predictions.append((image_id, boxes, scores, labels))
            else:
                print(f"Skipping invalid PredictionString for image_id: {image_id}")

    results = []
    image_ids = sorted(set([pred[0] for pred in all_predictions]))
    
    for image_id in image_ids:
        image_predictions = [pred for pred in all_predictions if pred[0] == image_id]
        
        boxes_list = [pred[1] for pred in image_predictions]
        scores_list = [pred[2] for pred in image_predictions]
        labels_list = [pred[3] for pred in image_predictions]
        
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        
        boxes = boxes * 1024.0
        
        pred_str = format_prediction_string(boxes, scores, labels)
        
        results.append({
            'PredictionString': pred_str,
            'image_id': image_id
            })
    
    result_df = pd.DataFrame(results)
    return result_df

file_paths = ['/data/ephemeral/home/sejongmin/level2-objectdetection-cv-16/output.csv', '/data/ephemeral/home/sejongmin/level2-objectdetection-cv-16/work_dirs/yolox_x_49/submission.csv']
wbf_result_df = perform_wbf_on_files(file_paths)
wbf_result_df.to_csv('wbf_ensemble_result.csv', index=False)
