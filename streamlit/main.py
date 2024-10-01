import eda
import detector
import pandas as pd
import streamlit as st
from pycocotools.coco import COCO

LABEL_COLORS = ['rgb(208, 56, 78)', 'rgb(238, 100, 69)', 'rgb(250, 155, 88)', 'rgb(254, 206, 124)', 'rgb(255, 241, 168)', 'rgb(244, 250, 173)', 'rgb(209, 237, 156)', 'rgb(151, 213, 164)', 'rgb(92, 183, 170)', 'rgb(54, 130, 186)']
LABEL_COLORS_WOUT_NO_FINDING = LABEL_COLORS[:8]+LABEL_COLORS[9:]
CLASSES = ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
class_colors = {
    'General trash': (255, 0, 0), 
    'Paper': (0, 255, 0),
    'Paper pack': (0, 0, 255),
    'Metal': (255, 255, 0),
    'Glass': (0, 255, 255),
    'Plastic': (255, 0, 255),
    'Styrofoam': (128, 0, 0),
    'Plastic bag': (0, 128, 0),
    'Battery': (0, 0, 128),
    'Clothing': (128, 128, 0)
}

@st.cache_data
def load_train_df():
    coco = COCO('./dataset/train.json')
    train_df = pd.DataFrame()
    image_ids = []
    class_name = []
    class_id = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []

    for image_id in coco.getImgIds():
            
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)
            
        file_name = image_info['file_name']
            
        for ann in anns:
            image_ids.append(file_name)
            class_name.append(CLASSES[ann['category_id']])
            class_id.append(ann['category_id'])
            x_min.append(float(ann['bbox'][0]))
            y_min.append(float(ann['bbox'][1]))
            x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
            y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

    train_df['image_id'] = image_ids
    train_df['class_name'] = class_name
    train_df['class_id'] = class_id
    train_df['x_min'] = x_min
    train_df['y_min'] = y_min
    train_df['x_max'] = x_max
    train_df['y_max'] = y_max

    bbox_df = pd.DataFrame()
    bbox_df['class_id'] = train_df['class_id'].values
    bbox_df['class_name'] = train_df['class_name'].values
    bbox_df['x_min'] = train_df['x_min'].values / 1024
    bbox_df['x_max'] = train_df['x_max'].values / 1024
    bbox_df['y_min'] = train_df['y_min'].values / 1024
    bbox_df['y_max'] = train_df['y_max'].values / 1024
    bbox_df['frac_x_min'] = train_df['x_min'].values / 1024
    bbox_df['frac_x_max'] = train_df['x_max'].values / 1024
    bbox_df['frac_y_min'] = train_df['y_min'].values / 1024
    bbox_df['frac_y_max'] = train_df['y_max'].values / 1024
    return train_df, bbox_df
        
@st.cache_data
def load_test_df(csv_path):
    submission_df = pd.read_csv(csv_path)
    submission_df['PredictionString'] = submission_df['PredictionString'].fillna("").astype(str)

    test_df = pd.DataFrame()

    image_ids = []
    class_name = []
    class_id = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []

    for idx, row in submission_df.iterrows():
        image_id = row['image_id'] 
        prediction_string = row['PredictionString']
        predictions = list(map(float, prediction_string.split()))

        for i in range(0, len(predictions), 6):
            class_id_value = int(predictions[i])
            score = predictions[i + 1]
            x_min_value = predictions[i + 2]
            y_min_value = predictions[i + 3]
            x_max_value = predictions[i + 4]
            y_max_value = predictions[i + 5]

            # Append values to the corresponding lists
            image_ids.append(image_id)
            class_id.append(class_id_value)
            class_name.append(CLASSES[class_id_value])
            x_min.append(x_min_value)
            y_min.append(y_min_value)
            x_max.append(x_max_value)
            y_max.append(y_max_value)

            # Create the DataFrame from the extracted information
    test_df['image_id'] = image_ids
    test_df['class_id'] = class_id
    test_df['class_name'] = class_name
    test_df['x_min'] = x_min
    test_df['y_min'] = y_min
    test_df['x_max'] = x_max
    test_df['y_max'] = y_max

    return test_df

st.sidebar.success("CV16 오늘도 화이팅하조!")
menu = st.sidebar.radio("메뉴", ["데이터 분포 확인하기", "객체 탐지 확인하기"])

if "test_df" not in st.session_state:
    st.session_state.test_df = pd.DataFrame()

train_df, bbox_df = load_train_df()

if menu == "데이터 분포 확인하기":
    st.header("데이터 분포 확인하기")
    eda.show(train_df, bbox_df, LABEL_COLORS, LABEL_COLORS_WOUT_NO_FINDING, CLASSES)
elif menu == "객체 탐지 확인하기":
    st.header("객체 탐지 확인하기")
    option = st.sidebar.radio("옵션", ["train images", "test images"])
    if option == "train images":
        image_count = st.sidebar.slider('Select image count', 1, 8, 4)
        image_id = st.sidebar.slider('Select image id', 0, len(train_df.groupby("image_id")) - (5 * image_count), 0)
        st.sidebar.write('image_id:', image_id)
        for i in range(image_id, image_id + 5 * image_count, image_count):
            image_ids = [f"train/{str(j).zfill(4)}.jpg" for j in range(i, i + image_count)]
            detector.show(train_df, image_ids, class_colors)
    elif option == "test images":
        with st.sidebar.form(key="form"):
            csv_path = st.text_input("CSV 파일 경로")
            submit_button = st.form_submit_button("적용")
        if st.session_state.test_df.empty:
            if submit_button and csv_path:
                st.session_state.test_df = load_test_df(csv_path)
                st.sidebar.success("CSV 파일을 불러오는데 성공했습니다.")
            else:
                st.sidebar.error("CSV 파일이 비어 있거나 오류가 발생했습니다.")
                st.stop()
        image_count = st.sidebar.slider('Select image count', 1, 8, 4)
        image_id = st.sidebar.slider('Select image id', 0, len(st.session_state.test_df.groupby("image_id")) - (5 * image_count), 0)
        st.sidebar.write('image_id:', image_id)
        for i in range(image_id, image_id + 5 * image_count, image_count):
            image_ids = [f"test/{str(j).zfill(4)}.jpg" for j in range(i, i + image_count)]
            detector.show(st.session_state.test_df, image_ids, class_colors)