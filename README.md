  # RealOrFakeClassifier

A project for detecting fake images using an ensemble of MobileNetV3Small and MobileNetV3Large models.

## Introduction

This repository contains a project for detecting fake images using a dual-model ensemble approach. By leveraging the power of both MobileNetV3Small and MobileNetV3Large, the model is designed to distinguish between real and fake images with high accuracy.

## Dataset

The dataset consists of real and fake images stored in separate directories. The paths to these images and their corresponding labels are stored in a CSV file. Each image is labeled as either 'real' or 'fake'.

## Model Architecture

The model uses two pre-trained MobileNetV3 models (Small and Large) for feature extraction. The outputs of these models are concatenated and passed through a series of dense layers to make the final prediction. The architecture includes:

- **Input Layer**: Accepts images of size 224x224 with 3 color channels.
- **MobileNetV3Small**: Pre-trained on ImageNet, used for extracting features.
- **MobileNetV3Large**: Pre-trained on ImageNet, used for extracting features.
- **Concatenation Layer**: Combines the features from both MobileNetV3Small and MobileNetV3Large.
- **Dense Layers**: Includes a dense layer with ReLU activation and a Dropout layer for regularization.
- **Output Layer**: A single neuron with sigmoid activation for binary classification (real or fake).

## Results

After training, the model achieves high accuracy in distinguishing between real and fake images. The results are visualized using confusion matrices and classification reports. Below is an example of a confusion matrix and classification report:

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test data
y_pred = model.predict(test_ds)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Get true labels
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=['Real', 'Fake']))

