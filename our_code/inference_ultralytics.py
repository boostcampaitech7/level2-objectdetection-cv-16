from ultralytics import YOLO
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

data_root = "/data/ephemeral/home/dataset"
pt_path = "ultralytics/runs/detect/train9/weights/last.pt"
output_path = "submission.csv"

model = YOLO(pt_path, verbose=True)

coco = COCO('dataset/test.json')
img_ids = coco.getImgIds()
prediction_strings = []
file_names = []
img_paths = []
class_num = 10

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_names.append(img_info['file_name'])
    
    img_path = os.path.join(data_root, img_info['file_name'])
    
    results = model(img_path)
    
    prediction_string = ''
    for r in results:
        labels = r.boxes.cls.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        bboxs = r.boxes.xyxy.cpu().numpy()
        
        pred_instances = zip(labels, scores, bboxs)
        
        for label, score, bbox in pred_instances:
            prediction_string += f"{int(label)} {score:.4f} {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "

    prediction_strings.append(prediction_string.strip())

submission = pd.DataFrame()
submission['PredictionString'] = prediction_strings
submission['image_id'] = file_names
submission_file_path = os.path.join(output_path)
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved at: {submission_file_path}")
print(submission.head())
    

# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # Show results to screen (in supported environments)
#     r.show()

#     # Save results to disk
#     r.save(filename=f"results{i}.jpg")
    
    