import streamlit as st
import plotly.express as px

def show(train_df, bbox_df, LABEL_COLORS, LABEL_COLORS_WOUT_NO_FINDING, CLASSES):
    fig_hist = px.histogram(
        train_df.image_id.value_counts(), 
        log_y=True, 
        color_discrete_sequence=['indianred'], 
        opacity=0.7,
        labels={"value":"Number of Annotations Per Image"},
        title="<b>DISTRIBUTION OF # OF ANNOTATIONS PER IMAGE   " \
        "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
    )
    fig_hist.update_layout(
        showlegend=False,
        xaxis_title="<b>Number of Unique Images</b>",
        yaxis_title="<b>Count of All Object Annotations</b>",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    fig_class = px.histogram(
        train_df.groupby('image_id')["class_id"].unique().apply(lambda x: len(x)), 
        log_y=True, color_discrete_sequence=['skyblue'], opacity=0.7,
        labels={"value":"Number of Unique class"},
        title="<b>DISTRIBUTION OF # OF Unique Class PER IMAGE   " + \
            "<i><sub>(Log Scale for Y-Axis)</sub></i></b>",
    )
    fig_class.update_layout(
        showlegend=False,
        xaxis_title="<b>Number of Unique CLASS</b>",
        yaxis_title="<b>Count of Unique IMAGE</b>",
    )
    st.plotly_chart(fig_class, use_container_width=True)

    fig_label = px.bar(
        train_df.class_name.value_counts().sort_index(), 
        color=train_df.class_name.value_counts().sort_index().index, opacity=0.85,
        color_discrete_sequence=LABEL_COLORS, log_y=True,
        labels={"y":"Annotations Per Class", "x":""},
        title="<b>Annotations Per Class</b>",
    )
    fig_label.update_layout(
        legend_title=None,
        xaxis_title="",
        yaxis_title="<b>Annotations Per Class</b>"
    )
    st.plotly_chart(fig_label, use_container_width=True)

    bbox_df["frac_bbox_area"] = (bbox_df["frac_x_max"]-bbox_df["frac_x_min"])*(bbox_df["frac_y_max"]-bbox_df["frac_y_min"])

    fig_bbox = px.box(
        bbox_df.sort_values(by="class_name"), x="class_name", y="frac_bbox_area", color="class_name", 
        color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING, notched=True,
        labels={"class_name":"Class Name", "frac_bbox_area":"BBox Area (%)"},
        title="<b>DISTRIBUTION OF BBOX AREAS AS % OF SOURCE IMAGE AREA   " + \
        "<i><sub>(Some Upper Outliers Excluded For Better Visualization)</sub></i></b>"
    )

    fig_bbox.update_layout(
        showlegend=True,
        yaxis_range=[-0.025,0.4],
        legend_title_text=None,
        xaxis_title="",
        yaxis_title="<b>Bounding Box Area %</b>",
    )
    st.plotly_chart(fig_bbox, use_container_width=True)

    bbox_df["aspect_ratio"] = (bbox_df["x_max"]-bbox_df["x_min"])/(bbox_df["y_max"]-bbox_df["y_min"])
    fig_ratio = px.bar(x=CLASSES, y=bbox_df.groupby("class_id").mean()["aspect_ratio"], 
                color=CLASSES, opacity=0.85,
                color_discrete_sequence=LABEL_COLORS_WOUT_NO_FINDING, 
                labels={"x":"Class Name", "y":"Aspect Ratio (W/H)"},
                title="<b>Aspect Ratios For Bounding Boxes By Class</b>",)
    fig_ratio.update_layout(
                    yaxis_title="<b>Aspect Ratio (W/H)</b>",
                    xaxis_title=None,
                    legend_title_text=None)
    fig_ratio.add_hline(y=1, line_width=2, line_dash="dot", 
                annotation_font_size=10, 
                annotation_text="<b>SQUARE ASPECT RATIO</b>", 
                annotation_position="bottom left", 
                annotation_font_color="black")
    fig_ratio.add_hrect(y0=0, y1=0.5, line_width=0, fillcolor="red", opacity=0.125,
                annotation_text="<b>>2:1 VERTICAL RECTANGLE REGION</b>", 
                annotation_position="bottom right", 
                annotation_font_size=10,
                annotation_font_color="red")
    fig_ratio.add_hrect(y0=2, y1=3.5, line_width=0, fillcolor="green", opacity=0.04,
                annotation_text="<b>>2:1 HORIZONTAL RECTANGLE REGION</b>", 
                annotation_position="top right", 
                annotation_font_size=10,
                annotation_font_color="green")
    st.plotly_chart(fig_ratio, use_container_width=True)