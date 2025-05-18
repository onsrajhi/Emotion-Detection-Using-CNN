# Emotion-detection-using-CNN 

This project 💞 implements a deep learning-based model for detecting human emotions 😃 😠 😕 ...from facial expressions using Convolutional Neural Networks (CNN). The model is trained on the [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains facial images categorized into seven different emotions. The project explores multiple approaches, including building a CNN architecture from scratch, enhancing the model using image augmentation, and leveraging transfer learning techniques with VGGNet and ResNet50 to improve accuracy.

## Table of Contents 💫
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [Model 1: Custom CNN](#model-1-custom-cnn)
  - [Model 2: Custom CNN with Image Augmentation](#model-2-custom-cnn-with-image-augmentation)
  - [Model 3: Transfer Learning with VGGNet](#model-3-transfer-learning-with-vggnet)
  - [Model 4: Transfer Learning with ResNet50](#model-4-transfer-learning-with-resnet50)
- [Results](#results)
- [Live Emotion Detection](#live-emotion-detection)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [Project Setup and Installation](#project-setup-and-installation)
  - [Installation Guide](#installation-guide)
  - [Running the Project](#running-the-project)

## 🎯 Introduction
The goal of this project is to develop a machine learning solution that can identify human emotions such as anger, happiness, sadness, and others from facial expressions in images. Emotion detection has applications in various domains such as healthcare, entertainment, customer service, and human-computer interaction.

This project tackles the problem using Convolutional Neural Networks (CNN) to extract features from facial images and make predictions about the displayed emotion. Different strategies were employed to improve the model's performance, including transfer learning and image augmentation. The model was further integrated with OpenCV to enable real-time emotion detection from live video feeds.

## ♥️ Dataset
The [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) was used to train and evaluate the models. This dataset contains 35,887 grayscale images of size 48x48 pixels, categorized into seven distinct emotions:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Sad**
- **Surprise**
- **Neutral**

Each image is pre-processed and labeled for classification tasks, making it suitable for training deep learning models to recognize emotions.
