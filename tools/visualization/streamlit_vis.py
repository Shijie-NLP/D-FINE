import os
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# fmt: off
mscoco_category2name = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
    40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
    65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
}
# fmt: on

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}

WINDOWS_DATASET_DIR = Path(r"D:\datasets\fiftyone\coco-2017\validation\data")

SMALL_THR = 32**2
LARGE_THR = 96**2

SIZE_COLORS = {
    "small": "#FF4444",  # red (small)
    "medium": "#FFA500",  # orange (medium)
    "large": "#4488FF",  # blue (big)
}

st.set_page_config(page_title="D-FINE Explorer", layout="wide")


def _get_size_category(x1: float, y1: float, x2: float, y2: float) -> str:
    """Categorizes bounding box sizes strictly according to COCO standards."""
    area = (x2 - x1) * (y2 - y1)
    if area < SMALL_THR:
        return "small"
    elif area < LARGE_THR:
        return "medium"
    return "large"


def _get_topk_numpy(scores_array, k=3):
    """Robust Top-K extraction compatible with NumPy arrays."""
    topk_idx = np.argsort(scores_array)[-k:][::-1]
    topk_scores = scores_array[topk_idx]
    return topk_scores, topk_idx


@st.cache_data(show_spinner="Loading Inference Cache...")
def load_cache(cache_path: str):
    if not os.path.exists(cache_path):
        st.error(f"Cache file {cache_path} not found.")
        st.stop()
    with open(cache_path, "rb") as f:
        all_results = pickle.load(f)

    name_to_idx = {os.path.basename(res["image_path"]): i for i, res in enumerate(all_results)}
    idx_to_name = [os.path.basename(res["image_path"]) for res in all_results]
    return all_results, name_to_idx, idx_to_name


@st.cache_data(show_spinner="Loading Image...", max_entries=50)
def load_image(img_path_str: str):
    try:
        img_path = WINDOWS_DATASET_DIR / Path(img_path_str).name if sys.platform == "win32" else Path(img_path_str)
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image at {img_path_str}: {e}")
        st.stop()


def build_figure(
    img,
    sample,
    img_width,
    img_height,
    selected_layer,
    available_layers,
    show_gt,
    selected_gt_indices,  # 🌟 改为传入索引列表
    score_threshold,
    top_k_limit,
    canvas_scale,
    inspect_idx,
    show_all_boxes,
):
    """Constructs the rigid-coordinate Plotly figure."""
    fig = go.Figure()

    # --- 绘制 Ground Truth ---
    if show_gt and len(selected_gt_indices) > 0:
        for i, (box, label_id) in enumerate(zip(sample["gt_boxes"], sample["gt_labels"])):
            # 🌟 实例级过滤
            if i not in selected_gt_indices:
                continue

            x1, y1, x2, y2 = box.tolist()
            label_name = mscoco_category2name[mscoco_label2category[label_id]]
            box_color = SIZE_COLORS[_get_size_category(x1, y1, x2, y2)]

            fig.add_shape(type="rect", x0=x1, y0=y1, x1=x2, y1=y2, line=dict(color=box_color, width=2), layer="above")

            # 🌟 文本标签也带上索引，方便在侧边栏对应
            fig.add_trace(
                go.Scatter(
                    x=[x1],
                    y=[y1],
                    text=[f"[{i}] {label_name}"],  # 显示如 [0] person
                    mode="text",
                    textposition="top right",
                    textfont=dict(color=box_color, size=14, weight="bold"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # --- 绘制模型预测 (Query Boxes) ---
    if selected_layer != available_layers[0]:
        boxes, full_scores = _extract_layer_data(sample, selected_layer)

        max_scores = full_scores.max(-1)
        sort_idx = max_scores.argsort()[::-1]
        boxes, full_scores, max_scores = boxes[sort_idx], full_scores[sort_idx], max_scores[sort_idx]

        valid_mask = max_scores >= score_threshold
        boxes, full_scores = boxes[valid_mask][:top_k_limit], full_scores[valid_mask][:top_k_limit]

        if len(full_scores) > 0:
            valid_x, valid_y, hover_texts, point_colors, point_indices = [], [], [], [], []

            for i, (box, score) in enumerate(zip(boxes, full_scores)):
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # 1. 尺度颜色区分 (与 GT 保持一致)
                size_cat = _get_size_category(x1, y1, x2, y2)
                base_color = SIZE_COLORS[size_cat]

                # 2. Top-K 信息用于 Hover
                topk_scores, topk_labels = _get_topk_numpy(score, k=3)
                label_names = [
                    mscoco_category2name.get(mscoco_label2category.get(lbl, -1), "N/A") for lbl in topk_labels
                ]
                label_details = "<br>".join([f"{n}: {s:.3f}" for n, s in zip(label_names, topk_scores)])

                valid_x.append(cx)
                valid_y.append(cy)
                point_colors.append(base_color)
                point_indices.append(str(i))  # 用于点中心的数字显示
                hover_texts.append(f"<b>Query #{i}</b> ({size_cat})<br>{label_details}")

                # 3. 只有特定条件下才绘制 Bounding Box
                if show_all_boxes or i == inspect_idx:
                    # 如果是被选中的点，边框加粗加亮
                    box_line_color = "rgba(0, 255, 0, 1.0)" if i == inspect_idx else base_color
                    box_width = 3 if i == inspect_idx else 1
                    fig.add_shape(
                        type="rect",
                        x0=x1,
                        y0=y1,
                        x1=x2,
                        y1=y2,
                        line=dict(color=box_line_color, width=box_width),
                        layer="above",
                    )

            # 4. 绘制中心点及其内部索引
            fig.add_trace(
                go.Scatter(
                    x=valid_x,
                    y=valid_y,
                    mode="markers",  # 🌟 移除了 text 模式，彻底清爽
                    marker=dict(
                        color=point_colors,
                        # 🌟 动态尺寸逻辑：普通点 6px (极小)，选中点 14px (醒目)
                        size=[14 if j == inspect_idx else 6 for j in range(len(full_scores))],
                        line=dict(
                            color="white",
                            # 选中点增加外白圈厚度，增强对比
                            width=[1.5 if j == inspect_idx else 0.5 for j in range(len(full_scores))],
                        ),
                        # 🌟 选中点改用特殊符号（如星形或带点圆），从几何形状上进行二次区分
                        symbol=["circle-dot" if j == inspect_idx else "circle" for j in range(len(full_scores))],
                    ),
                    hoverinfo="text",
                    hovertext=hover_texts,  # 详细信息保留在 Hover 中
                    name="Query Centers",
                    showlegend=False,
                )
            )

    # --- 统一刚性坐标系配置 ---
    fig.update_layout(
        images=[
            dict(
                source=img,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=img_width,
                sizey=img_height,
                sizing="stretch",
                layer="below",
                xanchor="left",
                yanchor="top",
            )
        ],
        xaxis=dict(range=[0, img_width], showgrid=False, visible=False, fixedrange=False),
        yaxis=dict(
            range=[img_height, 0], showgrid=False, visible=False, scaleanchor="x", scaleratio=1, fixedrange=False
        ),
        width=int(img_width * canvas_scale),
        height=int(img_height * canvas_scale),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",
        dragmode="pan",
    )
    return fig, len(boxes) if selected_layer != available_layers[0] else 0


def _extract_layer_data(sample, selected_layer):
    """Helper to route the correct tensor based on layer selection."""
    if selected_layer == "Final Output":
        return sample["final_boxes"], sample["final_scores"]
    elif selected_layer.startswith("Encoder Aux Layer"):
        return sample["enc_boxes"], sample["enc_scores"]
    elif selected_layer == "Decoder Init (Pre)":
        return sample["pre_boxes"], sample["pre_scores"]
    elif selected_layer.startswith("Decoder Aux Layer"):
        l_idx = int(selected_layer.split()[-1])
        return sample["aux_boxes"][l_idx], sample["aux_scores"][l_idx]
    raise ValueError(f"Invalid layer: {selected_layer}")


def init_session_state():
    """Initializes global state variables."""
    defaults = {"img_idx": 0, "top_k_limit": 300, "canvas_scale": 1.5, "selected_layer": "None (GT Only)"}
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main():
    init_session_state()
    st.sidebar.title("D-FINE Explorer")

    all_results, name_to_idx, idx_to_name = load_cache("inference_dense_results.pkl")
    total_images = len(all_results)
    st.sidebar.markdown(f"**Total Samples:** `{total_images}`")

    # --- Navigation UI ---
    new_idx = st.sidebar.number_input(
        "Jump to Index", min_value=0, max_value=total_images - 1, value=st.session_state.img_idx
    )
    if new_idx != st.session_state.img_idx:
        st.session_state.img_idx = new_idx
        st.rerun()

    input_name = st.sidebar.text_input("Input Filename", value=idx_to_name[st.session_state.img_idx])
    if input_name != idx_to_name[st.session_state.img_idx]:
        if input_name in name_to_idx:
            st.session_state.img_idx = name_to_idx[input_name]
            st.rerun()
        else:
            st.sidebar.error(f"❌ File '{input_name}' not found.")

    # --- Display UI ---
    sample = all_results[st.session_state.img_idx]

    st.sidebar.markdown("---")
    st.session_state.canvas_scale = st.sidebar.slider(
        "Canvas Scale", 1.0, 2.0, value=st.session_state.canvas_scale, step=0.1
    )
    score_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, value=0.0, step=0.01, key=f"score_threshold_{st.session_state.img_idx}"
    )
    top_k_limit = st.sidebar.slider(
        "Max Visible Queries", 10, 300, value=300, step=10, key=f"top_k_limit_{st.session_state.img_idx}"
    )

    st.sidebar.markdown("---")
    show_gt = st.sidebar.checkbox("Show Ground Truth (GT)", value=True, key=f"show_gt_{st.session_state.img_idx}")

    selected_gt_indices = []
    if show_gt:
        # 🌟 构建实例级描述列表： "0: person", "1: tv", ...
        gt_instances = [
            f"{i}: {mscoco_category2name[mscoco_label2category[label]]}" for i, label in enumerate(sample["gt_labels"])
        ]

        # 🌟 实例拾取器：默认全选所有实例
        selected_gt_instances = st.sidebar.multiselect(
            "Select Specific GT Instances",
            options=gt_instances,
            default=gt_instances,
            key=f"gt_instance_filter_{st.session_state.img_idx}",
        )
        # 解析出选中的索引整数
        selected_gt_indices = [int(item.split(":")[0]) for item in selected_gt_instances]

    # --- Layer Selection UI ---
    available_layers = ["None (GT Only)"]
    if "enc_boxes" in sample:
        available_layers.append("Encoder Aux Layer")
    if "pre_boxes" in sample:
        available_layers.append("Decoder Init (Pre)")
    if "aux_boxes" in sample:
        available_layers.extend([f"Decoder Aux Layer {i + 1}" for i in range(len(sample["aux_boxes"]))])
    available_layers.append("Final Output")

    st.sidebar.markdown("---")
    current_layer_idx = available_layers.index(st.session_state.selected_layer)
    st.session_state.selected_layer = st.sidebar.selectbox(
        "🔍 Select Network Layer",
        available_layers,
        index=current_layer_idx,
    )

    # --- Data Loading & Validation ---
    img = load_image(input_name)
    img_width, img_height = img.size

    # --- UI for specific Query isolation (Must happen before render) ---
    inspect_idx = 0
    show_all_boxes = False
    if st.session_state.selected_layer != available_layers[0]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎯 Query Isolation")
        show_all_boxes = st.sidebar.checkbox(
            "Force Show All Visible Boxes", value=False, key=f"show_all_boxes_{st.session_state.img_idx}"
        )
        # Dummy slider first, actual max value updated inside the render logic but Streamlit reads top-down.
        # We handle this by safely passing a default max, as Streamlit UI state updates on interaction.
        inspect_idx = st.sidebar.number_input(
            "Inspect Specific Query Index", min_value=0, value=0, step=1, key=f"inspect_idx_{st.session_state.img_idx}"
        )

    # --- Build and Render ---
    fig, num_valid = build_figure(
        img,
        sample,
        img_width,
        img_height,
        st.session_state.selected_layer,
        available_layers,
        show_gt,
        selected_gt_indices,
        score_threshold,
        top_k_limit,
        st.session_state.canvas_scale,
        inspect_idx,
        show_all_boxes,
    )

    st.plotly_chart(
        fig, width="content", config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset+autosize"}
    )


if __name__ == "__main__":
    main()
