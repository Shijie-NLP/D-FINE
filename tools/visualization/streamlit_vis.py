import os
import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


st.set_page_config(page_title="D-FINE Explorer", layout="wide")

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

SIZE_COLORS = {
    "small": "#FF4444",  # red (small)
    "medium": "#FFA500",  # orange (medium)
    "large": "#4488FF",  # blue (big)
}


def get_label_name(label_id):
    return mscoco_category2name.get(mscoco_label2category.get(label_id, -1), "N/A")


@st.cache_resource(show_spinner="Loading Inference Cache (Zero-Copy)...")
def load_cache(cache_path: str):
    """Uses cache_resource to prevent Streamlit from deep-copying the potentially massive pickle file."""
    if not os.path.exists(cache_path):
        st.error(f"Cache file {cache_path} not found.")
        st.stop()
    with open(cache_path, "rb") as f:
        all_results = pickle.load(f)

    name_to_idx = {os.path.basename(res["image_path"]): i for i, res in enumerate(all_results)}
    idx_to_name = [os.path.basename(res["image_path"]) for res in all_results]
    return all_results, name_to_idx, idx_to_name


all_results, name_to_idx, idx_to_name = load_cache("inference_dense_results.pkl")


def update_image_UI():
    def sync_from_index():
        st.session_state.img_name = idx_to_name[st.session_state.img_idx]

    def sync_from_name():
        new_name = st.session_state.img_name
        if new_name in name_to_idx:
            st.session_state.img_idx = name_to_idx[new_name]
        else:
            st.sidebar.error(f"File '{new_name}' not found!")

    st.sidebar.number_input(
        "Jump to Image Index",
        min_value=0,
        max_value=len(all_results) - 1,
        value=0,
        key="img_idx",
        on_change=sync_from_index,
    )

    st.sidebar.text_input("Input Image Filename", key="img_name", value=idx_to_name[0], on_change=sync_from_name)


def update_slider_UI():
    st.sidebar.slider("Canvas Scale", min_value=1.0, max_value=3.0, step=0.1, value=2.0, key="canvas_scale")
    st.sidebar.slider("Score Threshold", min_value=0.0, max_value=1.0, step=0.05, value=0.0, key="score_threshold")
    st.sidebar.slider("Max Queries", min_value=0, max_value=300, step=10, value=300, key="max_queries")


def update_sample_UI(sample):
    show_gt = st.sidebar.checkbox("Show Ground Truth (GT)", value=True, key=f"show_gt_{st.session_state.img_idx}")

    selected_gt_indices = []
    if show_gt:
        gt_instances = [f"{i}: {get_label_name(label)}" for i, label in enumerate(sample["gt_labels"])]

        selected_gt_instances = st.sidebar.multiselect(
            "Select Specific GT Instances",
            options=gt_instances,
            default=gt_instances,
            key=f"gt_instance_filter_{st.session_state.img_idx}",
        )
        selected_gt_indices = [int(item.split(":")[0]) for item in selected_gt_instances]
    st.session_state.selected_gt_indices = selected_gt_indices

    layers_flow = ["None (GT Only)"]
    if "enc_boxes" in sample:
        layers_flow.append("Encoder Aux Layer")
    if "pre_boxes" in sample:
        layers_flow.append("Decoder Init (Pre)")
    if "aux_boxes" in sample:
        layers_flow.extend([f"Decoder Layer {i + 1}" for i in range(len(sample["aux_boxes"]))])
    layers_flow.append("Final Output")

    st.sidebar.markdown("### Select Output Layer")
    st.sidebar.slider(
        "Output Layer Index",
        min_value=0,
        max_value=len(layers_flow) - 1,
        step=1,
        value=0,
        key="layer_idx",
    )

    def change_layer(delta):
        st.session_state.layer_idx = max(0, min(len(layers_flow) - 1, st.session_state.layer_idx + delta))

    col1, col2 = st.sidebar.columns(2)
    col1.button("⬅️ Prev", on_click=change_layer, args=(-1,))
    col2.button("Next ➡️", on_click=change_layer, args=(1,))
    st.sidebar.info(f"📍 Current: **{layers_flow[st.session_state.layer_idx]}**")
    st.session_state.selected_layer = layers_flow[st.session_state.layer_idx]


@st.cache_data(show_spinner="Loading Image...", max_entries=20)
def load_image(img_path_str: str):
    dataset_dir = (
        Path("~/Data/datasets/coco/val2017").expanduser()
        if sys.platform == "linux"
        else Path(r"D:\datasets\fiftyone\coco-2017\validation\data")
    )

    img_path = dataset_dir / Path(img_path_str).name
    try:
        return Image.open(img_path).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image at {img_path}: {e}")
        st.stop()


def _get_size_categories_vectorized(boxes):
    """Vectorized calculation of size categories for an array of boxes strictly according to COCO standards."""
    if len(boxes) == 0:
        return np.array([])

    SMALL_THR = 32**2
    LARGE_THR = 96**2

    # boxes shape: (N, 4) -> x1, y1, x2, y2
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    categories = np.full(areas.shape, "large", dtype=object)
    categories[areas < LARGE_THR] = "medium"
    categories[areas < SMALL_THR] = "small"
    return categories


def _extract_layer_data(sample, selected_layer):
    """Helper to route the correct tensor based on layer selection."""
    if selected_layer == "Final Output":
        return sample["final_boxes"], sample["final_scores"]
    elif selected_layer.startswith("Encoder Aux Layer"):
        return sample["enc_boxes"], sample["enc_scores"]
    elif selected_layer == "Decoder Init (Pre)":
        return sample["pre_boxes"], sample["pre_scores"]
    elif selected_layer.startswith("Decoder Layer"):
        l_idx = int(selected_layer.split()[-1]) - 1
        return sample["aux_boxes"][l_idx], sample["aux_scores"][l_idx]
    raise ValueError(f"Invalid layer: {selected_layer}")


def _get_topk_numpy(scores_array, k=3):
    """Robust Top-K extraction compatible with NumPy arrays."""
    topk_idx = np.argsort(scores_array)[-k:][::-1]
    topk_scores = scores_array[topk_idx]
    return topk_scores, topk_idx


def build_figure(img, sample):
    """Constructs the rigid-coordinate Plotly figure using vectorized operations."""
    fig = go.Figure()
    inspect_info = None

    def add_vectorized_boxes(boxes_array, color, line_width=1):
        """Helper function to render multiple boxes as a single Scatter trace separated by None."""
        if len(boxes_array) == 0:
            return

        x_lines, y_lines = [], []
        for box in boxes_array:
            x1, y1, x2, y2 = box.tolist()
            x_lines.extend([x1, x2, x2, x1, x1, None])
            y_lines.extend([y1, y1, y2, y2, y1, None])

        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=color, width=line_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    selected_gt_indices = st.session_state.selected_gt_indices
    if selected_gt_indices:
        # Convert to numpy array safely in case they are lists or tensors
        gt_boxes = np.array(sample["gt_boxes"])[selected_gt_indices]
        gt_labels = np.array(sample["gt_labels"])[selected_gt_indices]

        size_cats = _get_size_categories_vectorized(gt_boxes)

        for cat in ["small", "medium", "large"]:
            mask = size_cats == cat
            add_vectorized_boxes(gt_boxes[mask], SIZE_COLORS[cat], line_width=2)

        text_x, text_y, texts, colors = [], [], [], []
        for orig_idx, box, label_id, cat in zip(selected_gt_indices, gt_boxes, gt_labels, size_cats):
            text_x.append(box[0])
            text_y.append(box[1])
            texts.append(f"[{orig_idx}] {get_label_name(label_id)}")
            colors.append(SIZE_COLORS[cat])

        if texts:
            fig.add_trace(
                go.Scatter(
                    x=text_x,
                    y=text_y,
                    text=texts,
                    mode="text",
                    textposition="top right",
                    textfont=dict(color=colors, size=14, weight="bold"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # --- UI for specific Query isolation ---
    if st.session_state.layer_idx > 0:
        boxes, full_scores = _extract_layer_data(sample, st.session_state.selected_layer)

        # Ensure numpy
        boxes = np.array(boxes)
        full_scores = np.array(full_scores)

        original_indices = np.arange(len(full_scores))

        max_scores = full_scores.max(-1)
        sort_idx = max_scores.argsort()[::-1]
        boxes, full_scores, max_scores = boxes[sort_idx], full_scores[sort_idx], max_scores[sort_idx]

        valid_mask = max_scores >= st.session_state.score_threshold
        boxes = boxes[valid_mask][: st.session_state.max_queries]
        full_scores = full_scores[valid_mask][: st.session_state.max_queries]

        original_indices = original_indices[valid_mask][: st.session_state.max_queries]  # 过滤索引

        num_valid = len(boxes)
        if num_valid > 0:
            st.sidebar.markdown("### Select Query")
            show_all_boxes = st.sidebar.checkbox(
                "Force Show All Visible Boxes", value=False, key=f"show_all_boxes_{st.session_state.img_idx}"
            )

            inspect_idx = st.sidebar.number_input(
                f"Inspect Specific Query Index (Total queries: {num_valid})",
                min_value=0,
                max_value=num_valid - 1,
                value=0,
                step=1,
                key=f"inspect_idx_{st.session_state.img_idx}",
            )
            target_score = full_scores[inspect_idx]
            top5_vals, top5_ids = _get_topk_numpy(target_score, k=5)

            inspect_info = [
                {
                    "Rank": f"#{i + 1}",
                    "Class": mscoco_category2name.get(mscoco_label2category.get(lbl, -1), "N/A"),
                    "Score": round(float(val), 4),
                }
                for i, (val, lbl) in enumerate(zip(top5_vals, top5_ids))
            ]

            size_cats = _get_size_categories_vectorized(boxes)

            hover_labels = [f"Query ID: {idx}" for idx in original_indices]

            if show_all_boxes:
                bg_mask = np.arange(num_valid) != inspect_idx
                for cat in ["small", "medium", "large"]:
                    cat_mask = (size_cats == cat) & bg_mask
                    add_vectorized_boxes(boxes[cat_mask], SIZE_COLORS[cat], line_width=1)

            # Highlighting the inspected box
            add_vectorized_boxes(boxes[inspect_idx : inspect_idx + 1], "rgba(0, 255, 0, 1.0)", line_width=6)

            # Center points
            cxs = (boxes[:, 0] + boxes[:, 2]) / 2
            cys = (boxes[:, 1] + boxes[:, 3]) / 2

            pt_colors = [SIZE_COLORS[cat] for cat in size_cats]
            pt_sizes = [14 if j == inspect_idx else 6 for j in range(num_valid)]
            pt_lines = [1.5 if j == inspect_idx else 0.5 for j in range(num_valid)]
            pt_symbols = ["circle-dot" if j == inspect_idx else "circle" for j in range(num_valid)]

            pt_colors[inspect_idx] = "rgba(0, 255, 0, 1.0)"

            fig.add_trace(
                go.Scatter(
                    x=cxs,
                    y=cys,
                    mode="markers",
                    text=hover_labels,
                    hoverinfo="text",
                    marker=dict(
                        color=pt_colors,
                        size=pt_sizes,
                        line=dict(color="white", width=pt_lines),
                        symbol=pt_symbols,
                    ),
                    showlegend=False,
                )
            )

    img_width, img_height = img.size

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
        width=int(img_width * st.session_state.canvas_scale),
        height=int(img_height * st.session_state.canvas_scale),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",
        dragmode="pan",
        uirevision=st.session_state.img_idx,
    )

    return fig, inspect_info


def main():
    st.sidebar.title("D-FINE Explorer")

    total_images = len(all_results)
    st.sidebar.markdown(f"**Total Samples:** `{total_images}`")

    update_image_UI()
    update_slider_UI()

    sample = all_results[st.session_state.img_idx]

    img = load_image(sample["image_path"])

    update_sample_UI(sample)

    # --- Build and Render ---
    fig, inspect_info = build_figure(img, sample)

    if inspect_info:
        with st.sidebar:
            st.table(inspect_info)

    st.plotly_chart(
        fig, width="content", config={"scrollZoom": True, "displayModeBar": True, "doubleClick": "reset+autosize"}
    )


if __name__ == "__main__":
    main()
