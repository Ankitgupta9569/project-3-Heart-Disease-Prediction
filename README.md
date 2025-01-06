# Face Mask Detection

![Face Mask Detection](images/project4.jpg)

## 🦠 Overview

The **Face Mask Detection** system is a machine learning-based project that uses deep learning techniques to detect whether individuals are wearing face masks or not. This project aims to contribute to safety measures, especially in public spaces, by automating the monitoring of face mask usage. The system uses a Convolutional Neural Network (CNN) trained on a custom dataset of images to classify whether a person is wearing a mask or not.

## ⚙️ Technologies Used
- **Python** 🐍
- **Deep Learning** 🤖
- **TensorFlow** & **Keras** 📚
- **OpenCV** 👁️
- **NumPy** 🔢
- **Matplotlib** 📊
- **Flask** 🚀 (for web application)
- **Transfer Learning** 🧠 (using pre-trained models like MobileNet)

## 🔍 Problem Statement

In the wake of the COVID-19 pandemic, it is crucial to ensure the proper usage of face masks in public spaces. Automated face mask detection can help businesses, schools, and government agencies monitor compliance and enforce safety protocols more effectively.

## 💡 Solution

We built a deep learning-based solution for real-time face mask detection using the following steps:
1. **Data Collection**: The dataset consists of images of people wearing and not wearing face masks.
2. **Model Architecture**: We use a CNN model to classify images as "with mask" or "without mask."
3. **Transfer Learning**: Pre-trained models like MobileNet are fine-tuned to optimize the mask detection performance.
4. **Real-Time Detection**: The model can classify images or video streams in real-time, making it ideal for monitoring public spaces.

### Example Flow:
1. Capture an image or video feed.
2. The system processes the image using OpenCV.
3. The CNN model classifies the image as either "Face with Mask" or "No Mask."
4. A warning or notification is triggered for "No Mask" cases.

## 📊 Dataset

The dataset for this project consists of images labeled as **Face with Mask** and **Face without Mask**. The images were sourced from various public domains and curated to include diverse face angles and lighting conditions.

- **Dataset Size**: 10,000 images (5,000 with masks, 5,000 without masks).
- **Source**: Custom dataset (images collected from Kaggle and other open sources).
- **Features**: Image pixels, face landmarks.

## 🔧 Steps to Run the Project

1. Clone the repository:
    ```bash
    git clone https://github.com/ankitgupta9569/face-mask-detection.git
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Train the model:
    ```bash
    python train_model.py
    ```
4. Run the mask detection in real-time:
    ```bash
    python detect_mask.py
    ```
5. (Optional) Run the Flask web application to upload and test images:
    ```bash
    python app.py
    ```

## 🎯 Results

- **Accuracy**: Achieved an accuracy of **95%** on the validation dataset.
- **Real-Time Detection**: The model can detect masks in real-time with minimal latency.
  
Example Output:
- **Face with Mask**: Green bounding box around the face, label "Mask."
- **No Mask**: Red bounding box, label "No Mask."

## 📈 Visualizations

- **Confusion Matrix** 📊
- **Model Training Curve** 📉
- **Real-Time Detection Frames** 🎥

## 💬 Discussion

This project provides an automated solution to detect whether people are wearing face masks, which can be especially helpful in environments like airports, shopping malls, and offices. However, it can be improved by:
- **Improving the Dataset**: Include images of people wearing masks in various styles (e.g., cloth masks, surgical masks).
- **Enhancing Model Robustness**: Fine-tune the model further to improve accuracy, especially in low-light conditions.
  
## 🚀 Future Enhancements

- Use **YOLO** or **SSD** for object detection to simultaneously detect faces and masks.
- Implement **Edge Detection** for deploying the system on mobile devices or embedded systems.
- Expand the system to detect other protective equipment such as face shields.

## 🌐 View the Project on GitHub

[Click here to view the code on GitHub](https://github.com/ankitgupta9569/face-mask-detection)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 📝 Author
**Ankit Gupta**  
[LinkedIn](https://www.linkedin.com/in/ankitgupta2026) | [GitHub](https://github.com/ankitgupta9569)
