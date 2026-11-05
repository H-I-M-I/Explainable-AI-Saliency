# Explainable AI - Saliency Maps with InceptionV3

This project demonstrates a **gradient-based saliency-map technique for interpreting image classification predictions**.

A pre-trained InceptionV3 model from TensorFlow Hub is used for image classification. TensorFlow's automatic differentiation is then used to calculate how strongly individual image pixels affect the classification loss.

The resulting gradients are converted into a saliency map that highlights image regions that have a strong influence on the model's prediction.

## What the Project Does

The notebook follows this pipeline:

1. Loads a pre-trained InceptionV3 classifier from TensorFlow Hub.
2. Adds a softmax activation to the model output.
3. Downloads and loads a sample image.
4. Converts the image from BGR to RGB.
5. Resizes and normalizes the image.
6. Defines the target ImageNet class.
7. Uses `tf.GradientTape()` to track the input image.
8. Computes the classification loss.
9. Calculates gradients of the loss with respect to the input pixels.
10. Converts the gradients into a grayscale saliency map.
11. Normalizes the saliency values to the range 0–255.
12. Displays both the saliency map and an overlay on the original image.

## Model

The classifier is based on a pre-trained **InceptionV3** model obtained from TensorFlow Hub.

```python
hub.KerasLayer(
    'https://tfhub.dev/google/tf2-preview/inception_v3/classification/4',
    trainable=False
)
```

A softmax activation is added to the model output for classification.

## Image Preprocessing

The image is loaded with OpenCV and processed before being passed to the classifier.

The notebook performs:

- Image loading with `cv2.imread()`
- BGR-to-RGB conversion
- Image resizing
- Pixel normalization to the range `[0, 1]`
- Addition of a batch dimension

Example:

```python
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (299, 299)) / 255.0
image = np.expand_dims(img, axis=0)
```

## Gradient Computation

The main explainability step uses TensorFlow's `GradientTape`.

```python
with tf.GradientTape() as tape:
    inputs = tf.cast(image, tf.float32)
    tape.watch(inputs)

    predictions = model(inputs)

    loss = tf.keras.losses.categorical_crossentropy(
        expected_output,
        predictions
    )

gradients = tape.gradient(loss, inputs)
```

This calculates the gradient of the classification loss with respect to every input image pixel.

Pixels with larger gradient magnitudes have a stronger effect on the loss for the selected class.

## Creating the Saliency Map

The RGB gradients are converted into a single grayscale representation using the sum of their absolute values:

```python
grayscale_tensor = tf.reduce_sum(
    tf.abs(gradients),
    axis=-1
)
```

The resulting values are normalized to the range:

```text
0 – 255
```

to create a visual saliency map.

Brighter pixels represent image regions associated with larger gradient magnitudes.

## Saliency Overlay

The notebook also creates a color representation of the gradient map using OpenCV:

```python
cv2.applyColorMap(
    normalized_tensor.numpy(),
    cv2.COLORMAP_HOT
)
```

The heatmap is then blended with the original image:

```python
cv2.addWeighted(
    img,
    0.5,
    gradient_color,
    0.5,
    0.0
)
```

This makes it easier to see where high-gradient regions occur relative to the objects in the original image.

## Technologies Used

- Python
- TensorFlow 2.x
- TensorFlow Hub
- InceptionV3
- OpenCV
- NumPy
- Matplotlib
- Gradient-based explainability
- Saliency maps

## Key Concepts Demonstrated

- Explainable AI
- CNN interpretability
- Pre-trained image classification
- TensorFlow automatic differentiation
- Gradient-based visualization
- Saliency maps
- Image preprocessing
- Heatmap overlays

## Limitations

Saliency maps show how sensitive a prediction or loss is to changes in individual pixels. They do not provide a complete explanation of how a deep neural network reaches a decision.

Gradient-based saliency maps can also be noisy and may highlight regions that are difficult to interpret visually.

## Possible Extensions

The project could be extended by:

- Comparing saliency maps with Grad-CAM
- Testing different ImageNet classes
- Evaluating multiple input images
- Comparing different CNN architectures
- Applying smoothing techniques to reduce noisy gradients
- Comparing explanation methods quantitatively
