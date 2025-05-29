import streamlit as st
import torch
from ultralytics import YOLO
import cv2
import numpy as np

st.title("Nepali Number Plate Recognition (NNPR) 🚗")

model = YOLO("replace/with/your/platedetection.pt/model")
if torch.cuda.is_available():
    model.to("cuda")

@st.cache_resource
def load_character_detection_model():
    """Load the YOLO model for character detection."""
    return YOLO("replace/with/your/characterdetection.pt/model")

character_model = load_character_detection_model()

character_classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'BA', 'BAGMATI', 'CHA',
    'GA', 'GANDAKI', 'HA', 'JA', 'JHA', 'KA', 'KHA', 'KO', 'LU', 'LUMBINI',
    'MA', 'MADESH', 'ME', 'NA', 'PA', 'PRA', 'PRADESH', 'RA', 'SU', 'VE', 'YA'
]

def preprocess_image(cropped_image):
    """Preprocess the cropped image for character detection."""
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    gray_image_resized = cv2.resize(gray_image, (640, 640))
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(gray_image_resized, -1, sharpening_kernel)
    equalized_image = cv2.equalizeHist(sharpened_image)
    processed_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)
    return processed_image

def calculate_dynamic_threshold(characters, num_lines, min_threshold=15, max_threshold=50):
    """Calculate dynamic threshold for character line separation."""
    if len(characters) < 2:
        return max_threshold
    horizontal_distances = [abs(characters[i]['x'] - characters[i - 1]['x']) for i in range(1, len(characters))]
    avg_distance = np.mean(horizontal_distances)
    line_adjustment = np.clip(100 / (num_lines + 1), 15, 30)
    return np.clip(avg_distance / 2 + line_adjustment, min_threshold, max_threshold)

def optimized_sort_characters(characters, dynamic_threshold=False):
    """Sort detected characters into lines based on position."""
    characters.sort(key=lambda x: x['y'])
    lines, current_line = [], []
    threshold = 25
    if dynamic_threshold:
        num_lines = len(set(round(char['y']) for char in characters))
        threshold = calculate_dynamic_threshold(characters, num_lines)
    for char in characters:
        if not current_line or abs(current_line[-1]['y'] - char['y']) < threshold:
            current_line.append(char)
        else:
            lines.append(sorted(current_line, key=lambda x: x['x']))
            current_line = [char]
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['x']))
    return sorted(lines, key=lambda line: line[0]['y'])

def detect_characters(cropped_image, dynamic_threshold=True):
    """Detect and extract characters from a cropped license plate."""
    processed_image = preprocess_image(cropped_image)
    results = character_model(processed_image, conf=0.6)
    detected_classes = results[0].boxes.cls.cpu().numpy()
    boxes = results[0].boxes.xywh.cpu().numpy()
    characters = [
        {'class': character_classes[int(cls)], 'x': box[0], 'y': box[1]}
        for cls, box in zip(detected_classes, boxes)
        if int(cls) < len(character_classes)
    ]
    sorted_characters = optimized_sort_characters(characters, dynamic_threshold)
    return ''.join([char['class'] for line in sorted_characters for char in line])

def process_image(image, confidence, dynamic_threshold=True):
    """Process an image to detect and extract license plate characters."""
    results = model(image, conf=confidence)
    annotated_image_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    st.image(annotated_image_rgb, caption="Detected License Plate(s)", use_container_width=True)
    extracted_count = 0
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            extracted_count += 1
            x1, y1, x2, y2 = box
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            st.image(cropped_image_rgb, caption=f"Extracted Plate {extracted_count}", use_container_width=True)
            characters = detect_characters(cropped_image, dynamic_threshold)
            st.write(f"**Detected Characters (Plate {extracted_count}):** {characters}")

confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    process_image(image, confidence)

st.markdown("---")
st.info("Character sorting algorithm is undergoing further robust optimization for complex Nepali script. Some results may not reflect the exact order.")
