Real-Time Violence Detection Using Deep Learning
Project Overview
This project presents a real-time violence detection system based on deep learning and computer vision techniques. The system analyzes surveillance videos and live camera feeds to automatically classify scenes as Violence or Non-Violence. The objective is to reduce manual monitoring effort and improve response time in security and surveillance environments.
The proposed system employs transfer learning using the MobileNetV2 architecture with a custom binary classifier, ensuring both high accuracy and computational efficiency. The model is suitable for real-time deployment on CPU-based systems.

Features
Binary classification of violent and non-violent activities
Real-time violence detection using webcam input
Violence detection from recorded video files
Lightweight CNN architecture optimized for real-time performance
Probability threshold tuning to reduce false positives
Standard evaluation using classification metrics

Datasets Used
The dataset used in this project is a custom-curated dataset constructed from the following publicly available datasets:
XD Violence Dataset
UCF-Crime Dataset
Hockey Fight Dataset

The original multi-class datasets are converted into a binary classification problem by grouping all violent activities under the Violence class and normal activities under the Non-Violence class. The dataset is split into training and testing sets using an 80:20 ratio before frame extraction to avoid data leakage.

Model Architecture and Training Details

Model Architecture: MobileNetV2 with Custom Binary Classifier

Framework: TensorFlow (Keras API)

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Learning Rate: 0.001

Batch Size: 32

Number of Epochs: 6

Hardware Used: CPU

The MobileNetV2 base model is frozen during training, and only the custom classification layers are trained.

Project Structure
Project/
├── dataset/
│   ├── train/
│   │   ├── violence/
│   │   └── non_violence/
│   └── test/
│       ├── violence/
│       └── non_violence/
│
├── Training_Scripts/
│   └── train_binary_model.py
│
├── Evaluation_Scripts/
│   ├── test_binary_model.py
│   └── confusion_matrix_binary.py
│
├── Inference_Scripts/
│   ├── realtime_violence_detection.py
│   └── video_violence_analysis.py
│
├── Model/
│   └── violence_binary_model.h5
│
└── README.md

Model Training

To train the model, execute the following command:

python train_binary_model.py


The training script performs the following tasks:

Loads the training dataset

Applies data augmentation

Trains the custom binary classifier

Saves the trained model as violence_binary_model.h5

Model Evaluation

The trained model is evaluated using unseen test data. The evaluation scripts compute the following metrics:

Accuracy
Precision
Recall
F1-score
Confusion Matrix

Run the evaluation using:

python test_binary_model.py
python confusion_matrix_binary.py

Inference
Real-Time Violence Detection

To run real-time violence detection using a webcam:

python realtime_violence_detection.py

Video-Based Violence Detection

To analyze a recorded video file:

python video_violence_analysis.py


The output displays a label-based prediction (Violence or Non-Violence) along with the corresponding confidence score.

Notes

The system performs scene-level classification and does not include object localization.

Bounding boxes are not used, as the model focuses on classifying the entire frame.

Frame extraction is performed at fixed intervals to reduce redundancy and computational cost.

Conclusion

This project demonstrates an effective and efficient approach to real-time violence detection using deep learning. By leveraging transfer learning and a lightweight CNN architecture, the system achieves reliable performance suitable for real-world surveillance applications.
