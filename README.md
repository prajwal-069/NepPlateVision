# **Nepali Number Plate Recognition (NNPR) 🚗**

The **Nepali Number Plate Recognition (NNPR)** project provides a powerful framework for detecting and recognizing Nepali license plates. Built using the **YOLO (You Only Look Once)** object detection framework and featuring an intuitive **Streamlit-based interface**, this project is designed to accurately identify number plates and extract characters while accommodating the unique characteristics of **Nepali script and numbering conventions**.

## **Project Overview**

This repository integrates:
- **YOLOv8 & YOLOv11** models from Ultralytics for plate and character detection.
- **Custom preprocessing algorithms** to enhance image quality and character sorting for improved recognition accuracy.
- **A user-friendly Streamlit interface** for real-time inference and visualization.

> **Note**: Pre-trained model weights (e.g., platedetection.pt, characterdetection.pt) are not included in this repository. Users are required to provide their own trained models and update placeholder paths (e.g., "replace/with/your/actual/...") with the correct locations of their models and dataset configurations.

For character detection using YOLOv11, input images are converted to grayscale during preprocessing, as demonstrated in character_inference.ipynb. This step has been found to significantly enhance detection accuracy for Nepali characters.

## **Repository Structure**

- **`trainv8.py`** – Script for training the **YOLOv8** model used for license plate detection. Update the dataset configuration (`data="replace/with/your/platev8.yaml"`) accordingly.
- **`train11.py`** – Script for training the **YOLOv11** model used for character detection. Update the dataset configuration (`data="replace/with/your/characterv11.yaml"`) with your custom dataset path and class details.
- **`app.py`** – Streamlit application that performs real-time inference by integrating both **plate and character detection** for simplicity. The **sorting logic for detected characters** is included within this script, so no separate 
  sorting file is required.
- **`characterv11.yaml`** – Sample YAML configuration for YOLOv11 character detection. Modify the dataset paths, number of classes, and other training parameters as required.
- **`platev8.yaml`** – Sample YAML configuration for YOLOv8 plate detection. Adjust according to your dataset structure and class labels.
- **`requirements.txt`** – Contains a list of all Python dependencies required to run the training and inference scripts.
- **`LICENSE`** – This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** to ensure open-source compliance and usage transparency.


## **Getting Started**

### **Prerequisites**
Ensure the following are installed:
- **Python** ≥ 3.8
- **CUDA-enabled GPU** (Optional, but recommended for accelerated training and inference)
- **Required dependencies**: Install them using:

  ```bash
  pip install -r requirements.txt
  ```

### **Usage**

1. **Train Models**  
   - Update dataset paths in `platev8.yaml` and `characterv11.yaml`
     
   - Train the **plate detection model**:  
     ```bash
     python trainv8.py
     ```  
   - Train the **character detection model**:  
     ```bash
     python trainv11.py
     ```  


2. **Run Inference with Streamlit**  
   - Replace placeholder model paths in `app.py`.  
   - Launch the Streamlit app:  
     ```bash
     streamlit run app.py
     ```  

## **Live Demo**

You can test the deployed version of the **NNPR** application on **Streamlit Cloud**:  
👉 **[Nepali Number Plate Recognition - Live Demo](https://nepali-number-plate-recognition.streamlit.app/)**  

The models used in the live demo achieved the following performance metrics during training:

| Task                  | Model   | Dataset Size      | mAP Score |
|-----------------------|---------|-------------------|-----------|
| License Plate Detection | YOLOv8  | 12,276 images     | 98.8%     |
| Character Detection    | YOLOv11 | 2674 plate images  | 94.5%     |

For anyone interested in a deeper look, here’s a link to the Google Drive folder containing evaluation reports for both trained models by the developer:  
👉 **[Model Evaluation Reports - Google Drive](https://drive.google.com/drive/folders/1RQyD6-COV74pJXH_S-R3Le42cIAOK2Vh)**


## **License**

This project is released under the **AGPL-3.0** license, ensuring open-source collaboration while maintaining compliance with YOLO’s licensing terms.
