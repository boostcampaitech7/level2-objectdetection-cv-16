import pandas as pd
import streamlit as st
from pycocotools.coco import COCO
import eda, visualize_detection, compare_detection

class_colors = {
    'General trash': (208, 56, 78), 
    'Paper': (238, 100, 69),
    'Paper pack': (250, 155, 88),
    'Metal': (254, 206, 124),
    'Glass': (255, 241, 168),
    'Plastic': (244, 250, 173),
    'Styrofoam': (209, 237, 156),
    'Plastic bag': (151, 213, 164),
    'Battery': (92, 183, 170),
    'Clothing': (54, 130, 186)
}

@st.cache_data
def load_train_df(json_path='./dataset/train.json'):
    coco = COCO(json_path)
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
            class_name.append(list(class_colors.keys())[ann['category_id']])
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
            x_min_value = predictions[i + 2]
            y_min_value = predictions[i + 3]
            x_max_value = predictions[i + 4]
            y_max_value = predictions[i + 5]

            # Append values to the corresponding lists
            image_ids.append(image_id)
            class_id.append(class_id_value)
            class_name.append(list(class_colors.keys())[class_id_value])
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

if "test_df" not in st.session_state:
    st.session_state.test_df = pd.DataFrame({"image_id": [f"test/{str(i).zfill(4)}.jpg" for i in range(4871)]})
if "train_df" not in st.session_state and "bbox_df" not in st.session_state:
    st.session_state.train_df, st.session_state.bbox_df = load_train_df()
if "pred_df" not in st.session_state:
    st.session_state.pred_df = pd.DataFrame()
if "size" not in st.session_state and "anno" not in st.session_state:
    st.session_state.size = "All"
    st.session_state.anno = "All"

st.sidebar.success("CV16 오늘도 화이팅하조!")
with st.sidebar.form(key="json_form"):
    json_path = st.text_input("json file path")
    submit_button = st.form_submit_button("OK")
    if submit_button:
        try:
            st.session_state.train_df, st.session_state.bbox_df = load_train_df(json_path)
            st.sidebar.success("json file load successed :)")
        except Exception:
            st.session_state.train_df, st.session_state.bbox_df = load_train_df()
            st.sidebar.error("json file load failed :(")

menu = st.sidebar.radio("menu", ["데이터 분포 확인하기", "객체 탐지 확인하기"])

if menu == "데이터 분포 확인하기":
    st.markdown("<h2 style='text-align: center;'>데이터 분포 확인하기</h2>", unsafe_allow_html=True)
    eda.show(st.session_state.train_df, st.session_state.bbox_df)

elif menu == "객체 탐지 확인하기":
    st.markdown("<h2 style='text-align: center;'>객체 탐지 확인하기</h2>", unsafe_allow_html=True)
    option = st.sidebar.radio("option", ["train images", "test images"])
    if option == "train images":
        with st.sidebar.form(key="form"):
            csv_path = st.text_input("csv file path")
            st.session_state.size = st.selectbox("select bbox size", ["All", "Small", "Medium", "Large"])
            st.session_state.anno = st.selectbox("select annotation", ["All", 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'])
            submit_button = st.form_submit_button("OK")
        if submit_button:
            try:
                st.session_state.pred_df = load_test_df(csv_path)
                st.sidebar.success("csv file load successed :)")
            except Exception:
                st.session_state.pred_df = pd.DataFrame()
                st.sidebar.error("csv file load failed :(")

        if st.session_state.pred_df.empty:
            st.session_state.image_ids = [img_id for img_id in st.session_state.train_df.groupby("image_id")["image_id"].first().tolist()]
            image_count = st.sidebar.slider('Select image count', 1, 8, 4)
            image_index = st.sidebar.slider('Select image id', 0, len(st.session_state.image_ids) - (5 * image_count), 0)
            image_index_input = st.sidebar.number_input('Enter image count', min_value=0, max_value=len(st.session_state.image_ids)-(5*image_count), value=image_index, step=5*image_count)
            if image_index != image_index_input:
                image_index = image_index_input
            for i in range(image_index, image_index + 5 * image_count, image_count):
                image_ids = [st.session_state.image_ids[j] for j in range(i, i + image_count)]
                visualize_detection.show(st.session_state.train_df, image_ids, class_colors)
            
        else:
            st.session_state.image_ids = [img_id for img_id in st.session_state.pred_df.groupby("image_id")["image_id"].first().tolist()]
            image_index = st.sidebar.slider('Select image id', 0, len(st.session_state.image_ids) - 5, 0)
            image_index_input = st.sidebar.number_input('Enter image count', min_value=0, max_value=len(st.session_state.image_ids)-5, value=image_index, step=5)
            if image_index != image_index_input:
                image_index = image_index_input
            image_ids = [st.session_state.image_ids[i] for i in range(image_index, image_index + 5)]
            st.sidebar.write('image index:', image_index)
            compare_detection.show(st.session_state.train_df, st.session_state.pred_df, image_ids, class_colors, st.session_state.size, st.session_state.anno)

    elif option == "test images":
        with st.sidebar.form(key="form"):
            csv_path = st.text_input("csv file path")
            submit_button = st.form_submit_button("OK")
        if submit_button and csv_path:
            try:
                st.session_state.test_df = load_test_df(csv_path)
                st.sidebar.success("csv file load successed :)")
            except Exception:
                st.sidebar.error("csv file load failed :(")

        image_count = st.sidebar.slider('Select image count', 1, 8, 4)
        image_index = st.sidebar.slider('Select image id', 0, st.session_state.test_df["image_id"].nunique() - (5 * image_count), 0)
        image_index_input = st.sidebar.number_input('Enter image count', min_value=0, max_value=len(st.session_state.image_ids) - (5 * image_count), value=image_index)
        if image_index != image_index_input:
            image_index = image_index_input
        for i in range(image_index, image_index + 5 * image_count, image_count):
            image_ids = [f"test/{str(j).zfill(4)}.jpg" for j in range(i, i + image_count)]
            visualize_detection.show(st.session_state.test_df, image_ids, class_colors)