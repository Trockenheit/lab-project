# lab-project
Project - CNN for Image Classification


This project uses a Convolutional Neural Network (CNN) to classify images as either cats or dogs.

## 1. Installation

To run this project, you will need the following Python libraries:

- **tensorflow**
- **numpy**
- **keras**

Install the required libraries with the following command:

```bash
pip install tensorflow numpy keras
```

---

## 2. Project Overview

The goal of this project is to train a CNN to classify images of cats and dogs using a dataset with 8000 training images and 2000 test images.

### Objectives:
1. Prepare the data for the CNN model.
2. Build and train a CNN for image classification.
3. Evaluate the model's performance on test data.
4. Enable single-image predictions.

---

## 3. Project Files

This repository contains the following files and directories:

- **`CNN_for_Image_Classification.ipynb`**: The notebook with the CNN implementation and training process.
- **`dataset/training_set/`**: Contains training images organized in `cats` and `dogs` folders.
- **`dataset/test_set/`**: Contains test images organized in the same way as the training set.
- **`dataset/single_prediction/`**: Contains example images for single prediction.

---

## 4. Workflow

### Part 1: Data Preprocessing

#### Training Set:
- Images are scaled to values between 0 and 1.
- Augmentation techniques like zoom, shear, and horizontal flips are applied.

#### Test Set:
- Images are only scaled to values between 0 and 1.

---

### Part 2: Building the CNN

1. **Convolution**:
   - A convolutional layer with 32 filters of size 3x3 and ReLU activation.

2. **Pooling**:
   - Max pooling with a 2x2 pool size to reduce dimensionality.

3. **Additional Layers**:
   - Additional convolutional and pooling layers to enhance feature extraction.

4. **Flattening**:
   - Converts the output to a single vector for the fully connected layers.

5. **Fully Connected Layers**:
   - A dense layer with 128 units and ReLU activation.
   - Dropout is applied to prevent overfitting.

6. **Output Layer**:
   - A single neuron with sigmoid activation for binary classification.

---

### Part 3: Training the CNN

- The model uses the Adam optimizer and binary cross-entropy loss.
- Training is performed over 25 epochs with accuracy evaluation.

---

### Part 4: Single Prediction

A Python function is provided to preprocess a single image and predict whether it is a cat or a dog.

---

## 5. Results

### Training Performance:
- Training accuracy: ~86.62%
- Validation accuracy: ~80.10%

### Single Image Prediction:
- **cat_or_dog_1.jpg**: Dog
- **cat_or_dog_2.jpg**: Cat

---

https://medium.com/@zxie_5538/building-a-convolutional-neural-network-to-classify-cats-and-dogs-bb45ec6d286d
