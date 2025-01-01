import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure page settings
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        padding: 0.75rem;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #ff3333;
    }
    .findings-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .malignant {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #ff0000;
    }
    .benign {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00ff00;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_PATH = '../saved_model'
LABEL_MAP_PATH = '../label_map.pbtxt'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

def load_label_map(label_map_path):
    label_map = {}
    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip('"')
                label_map[label_index] = {"id": label_index, "name": label_name}
    return label_map

def plot_boxes_on_img(color_map, classes, bboxes, scores, category_index, image_origi, origi_shape):
    for idx, bbox in enumerate(bboxes):
        color = color_map[classes[idx]]
        
        # Main bounding box
        cv2.rectangle(
            image_origi,
            (int(bbox[1] * origi_shape[1]), int(bbox[0] * origi_shape[0])),
            (int(bbox[3] * origi_shape[1]), int(bbox[2] * origi_shape[0])),
            color,
            3
        )
        
        # Label background with rounded corners
        label_text = f"{category_index[classes[idx]]['name']}: {scores[idx]:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        cv2.rectangle(
            image_origi,
            (int(bbox[1] * origi_shape[1]), int(bbox[0] * origi_shape[0] - text_height - 10)),
            (int(bbox[1] * origi_shape[1] + text_width + 10), int(bbox[0] * origi_shape[0])),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            image_origi,
            label_text,
            (int(bbox[1] * origi_shape[1] + 5), int(bbox[0] * origi_shape[0] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    return image_origi

@st.cache_resource
def load_model(model_path):
    return tf.saved_model.load(model_path)

def main():
    # Page header
    st.markdown("# 🔬 Breast Cancer Detection System")
    st.markdown("### Advanced Medical Image Analysis Platform")

    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Upload Settings")
        file = st.file_uploader(
            "Select Medical Image",
            type=["jpg", "png", "jpeg"],
            help="Upload a medical image for analysis"
        )
        
        threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Adjust the minimum confidence score for detection"
        )
        
        analyze_button = st.button("🔍 Analyze Image")

    # Main content
    if file is None:
        st.info("👈 Please upload a medical image using the sidebar to begin analysis")
        return

    try:
        model = load_model(MODEL_PATH)

        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Original Image")
            test_image = Image.open(file).convert("RGB")
            st.image(test_image, use_column_width=True)

        if analyze_button:
            with st.spinner("🔄 Analyzing image..."):
                # Process image
                origi_shape = np.asarray(test_image).shape
                image_resized = np.array(test_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
                
                # Load label map and set colors
                category_index = load_label_map(LABEL_MAP_PATH)
                color_map = {
                    1: [255, 75, 75],  # Modified red for malignant
                    2: [75, 192, 75]   # Modified green for benign
                }

                # Prepare input tensor
                input_tensor = tf.convert_to_tensor(image_resized)
                input_tensor = input_tensor[tf.newaxis, ...]

                # Get model predictions
                detections_output = model(input_tensor)
                num_detections = int(detections_output.pop("num_detections"))
                detections = {
                    key: value[0, :num_detections].numpy() 
                    for key, value in detections_output.items()
                }
                detections["num_detections"] = num_detections

                # Filter predictions by threshold
                indexes = np.where(detections["detection_scores"] > threshold)
                bboxes = detections["detection_boxes"][indexes]

                if len(bboxes) == 0:
                    st.warning("No anomalies detected in the image")
                else:
                    classes = detections["detection_classes"][indexes].astype(np.int64)
                    scores = detections["detection_scores"][indexes]

                    # Create annotated image
                    image_origi = np.array(Image.fromarray(image_resized).resize(
                        (origi_shape[1], origi_shape[0])
                    ))
                    annotated_image = plot_boxes_on_img(
                        color_map, classes, bboxes, scores,
                        category_index, image_origi.copy(), origi_shape
                    )

                    # Display results
                    with col2:
                        st.markdown("### Analysis Results")
                        st.image(Image.fromarray(annotated_image), use_column_width=True)

                    # Display detailed findings
                    st.markdown("### Detailed Findings")
                    for idx, (bbox, class_id, score) in enumerate(zip(bboxes, classes, scores)):
                        finding_type = "benign" if class_id == 2 else "malignant"
                        st.markdown(
                            f"""
                            <div class="findings-box {finding_type}">
                                <h4>Finding {idx + 1}</h4>
                                <p>Classification: {category_index[class_id]['name']}</p>
                                <p>Confidence Score: {score:.2%}</p>
                                <p>Location: X: {bbox[1]:.2f}, Y: {bbox[0]:.2f}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()