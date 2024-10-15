import cv2
import streamlit as st

def show(df, train_df, image_ids, class_colors, size, anno):
    cols = st.columns(2)
    cols[0].markdown("<h4 style='text-align: center;'>Target</h4>", unsafe_allow_html=True)
    cols[1].markdown("<h4 style='text-align: center;'>Prediction</h4>", unsafe_allow_html=True)
    SMALL_SIZE = 32 * 32
    LARGE_SIZE = 96 * 96

    for i, image_id in enumerate(image_ids):
        flag_size_1 = False
        flag_size_2 = False
        flag_anno_1 = False
        flag_anno_2 = False
        if size == "All":
            flag_size_1 = True
            flag_size_2 = True
        if anno == "All":
            flag_anno_1 = True
            flag_anno_2 = True
        image_data_1 = df[df['image_id'] == image_id]
        image_data_2 = train_df[train_df['image_id'] == image_id]
    
        original_image = cv2.imread("./dataset/" + image_id)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_1 = original_image.copy()
        for idx, row in image_data_1.iterrows():
            try:
                class_name = row['class_name']
                x_min, y_min, x_max, y_max = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                color = class_colors.get(class_name, (255, 255, 255))

                if not flag_size_1:
                    if size == "Small" and (x_max - x_min) * (y_max - y_min) < SMALL_SIZE:
                        flag_size_1 = True
                    elif size == "Medium" and SMALL_SIZE < (x_max - x_min) * (y_max - y_min) < LARGE_SIZE:
                        flag_size_1 = True
                    elif size == "Large" and (x_max - x_min) * (y_max - y_min) > LARGE_SIZE:
                        flag_size_1 = True
                if not flag_anno_1:
                    if anno == class_name:
                        flag_anno_1 = True

                cv2.rectangle(image_1, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(image_1, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception:
                pass

        image_2 = original_image.copy()
        for idx, row in image_data_2.iterrows():
            try:
                class_name = row['class_name']
                x_min, y_min, x_max, y_max = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                color = class_colors.get(class_name, (255, 255, 255))

                if not flag_size_2:
                    if size == "Small" and (x_max - x_min) * (y_max - y_min) < SMALL_SIZE:
                        flag_size_2 = True
                    elif size == "Medium" and SMALL_SIZE < (x_max - x_min) * (y_max - y_min) < LARGE_SIZE:
                        flag_size_2 = True
                    elif size == "Large" and (x_max - x_min) * (y_max - y_min) > LARGE_SIZE:
                        flag_size_2 = True
                if not flag_anno_2:
                    if anno == class_name:
                        flag_anno_2 = True
                
                cv2.rectangle(image_2, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(image_2, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception:
                pass
        if (flag_size_1 and flag_anno_1) or (flag_size_2 and flag_anno_2):
            with cols[0]:
                st.image(image_1, use_column_width=True)
                if flag_size_1 and flag_anno_1:
                    st.markdown(f'<p style="color:green;">{image_id}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p style="color:red;">{image_id}</p>', unsafe_allow_html=True)
            with cols[1]:
                st.image(image_2, use_column_width=True)
                if flag_size_2 and flag_anno_2:
                    st.markdown(f'<p style="color:green;">{image_id}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p style="color:red;">{image_id}</p>', unsafe_allow_html=True)
