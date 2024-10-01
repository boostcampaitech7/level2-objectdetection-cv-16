import cv2
import streamlit as st
from PIL import Image

def show(df, image_ids, class_colors):
    cols = st.columns(len(image_ids))
    for i, image_id in enumerate(image_ids):
        image_data = df[df['image_id'] == image_id]
    
        image = cv2.imread("./dataset/" + image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        for idx, row in image_data.iterrows():
            class_name = row['class_name']
            x_min, y_min, x_max, y_max = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
            
            color = class_colors.get(class_name, (255, 255, 255))
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        with cols[i]:
            st.image(image, use_column_width=True)
            st.caption(image_id)