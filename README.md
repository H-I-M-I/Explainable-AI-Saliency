# Saliency Maps with InceptionV3 (TensorFlow)

This project demonstrates how to generate **saliency maps** using gradients from a pre-trained Inception V3 model to visualize which pixels most influence the model’s predictions.

## 📌 What This Project Does

- Loads a pre-trained **InceptionV3** model with a softmax activation.
- Downloads a sample image and prepares it for classification.
- Makes a prediction on the image and computes the gradient of the prediction w.r.t. the input image.
- Constructs a **saliency map** from the gradients to highlight key image regions.
- Displays the original image and its corresponding saliency map.

## 🧠 What Are Saliency Maps?

Saliency maps provide visual explanations of deep learning models by showing which **input pixels** contribute most to the model’s prediction. They are computed using the **gradient of the predicted class score with respect to the input image pixels**.

### 🧮 How Saliency Maps Are Computed

> **Saliency map = max(abs(∂(class_score)/∂(input_pixels))) over RGB channels**

#### 📊 Steps:
1. Predict class from input image using a pretrained model.
2. Compute the gradient of the predicted class score w.r.t. the input image.
3. Take the absolute maximum of the gradients across RGB channels.
4. Visualize the result as a heatmap overlay.

### 🧩 Diagram

![Saliency Map Diagram](A_flowchart_diagram_in_the_image_illustrates_the_c.png)

The diagram above illustrates the process of generating saliency maps using InceptionV3.

## 🛠️ Tools and Libraries

- TensorFlow 2.x
- TensorFlow Hub
- NumPy
- OpenCV
- Matplotlib

## 📚 References

- [InceptionV3 - TF Hub](https://tfhub.dev/google/tf2-preview/inception_v3/classification/4)
- [Original Saliency Paper](https://arxiv.org/abs/1312.6034)
- TensorFlow tutorials and documentation
