# Malaria Blood Smear Classifier

## Problem Statement

Malaria, a life-threatening disease primarily transmitted through the bite of an infected Anopheles mosquito, continues to have significant health and economic impacts across the globe, especially in Sub-saharan Africa. Rapid and accurate diagnosis of Malaria is crucial for effective disease management and control. The conventional method of microscopic examination of blood smears is labor-intensive and requires skilled personnel. Automating this process through machine learning can potentially increase the accuracy and efficiency of malaria diagnosis thus potentially saving lives.

## Objective

The primary objective of this project is to develop a deep learning model that can automatically classify blood smear images into two categories: infected (Parasitized) and uninfected. By leveraging convolutional neural networks (CNNs), the model aims to provide a reliable tool for assisting healthcare professionals in diagnosing malaria more rapidly and accurately.

## Data Understanding

The dataset used in this project is sourced from TensorFlow Datasets alias tfds, specifically the "malaria" dataset, which contains segmented cells from thin blood smear slide images of segmented cells. These images are labeled as either "Parasitized" or "Uninfected," making it a binary classification problem.

### Dataset Characteristics:
- Total Samples: 27,558
- Two Classes: Parasitized and Uninfected
- Split: 80% for training and 20% for testing
- Image Size: 128x128 pixels

## Exploratory Data Analysis (EDA)

### Class Distribution
The distribution of the two classes, 'Parasitized' and 'Uninfected', in both the training and test datasets was examined. This analysis is crucial to ensure that our model learns from a balanced dataset, which helps in reducing biases towards the more frequent class. The distribution charts indicate that the dataset is reasonably balanced, with both classes adequately represented in both training and test sets.

### Data Preprocessing
Data preprocessing involved normalizing the pixel values to a [0, 1] scale and resizing all images to 128x128 pixels. Normalization helps in speeding up the learning process, while consistent image sizing is crucial for the CNN architecture to process the input data efficiently. The preprocessed images were then shuffled to ensure the model does not learn any inadvertent order patterns, batched to improve training speed, and cached to optimize loading times. This preparation is vital for training deep learning models, particularly when dealing with large datasets and complex models.

## Modeling

### Model Architecture
The model utilizes a sequential CNN architecture comprised of the following layers:
- **Input Layer**: Accepts an input shape of (128, 128, 3) for RGB images.
- **Convolutional and Pooling Layers**: Multiple layers for feature extraction, including ReLU activation functions and max pooling for dimensionality reduction.
- **Dropout Layer**: To prevent overfitting by randomly setting input units to 0 at a rate of 0.5 during training.
- **Flatten and Dense Layers**: For non-linear transformation and to output the final prediction.

### Compilation
The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy

### Training
- **Epochs**: 10
- **Batch Size**: 32

### Performance
- **Test Accuracy**: 95.63%

## Results

### Confusion Matrix
The confusion matrix from testing the model on the unseen test data is as follows:

|               | Predicted Uninfected | Predicted Parasitized |
|---------------|----------------------|-----------------------|
| **True Uninfected**  | 2628                 | 152                   |
| **True Parasitized** | 89                   | 2643                  |

This shows high accuracy and a low rate of false negatives, which is critical for medical diagnostic tools.

## Conclusion

The developed model demonstrates excellent potential in classifying malaria infections from blood smear images with high accuracy. With a test accuracy of 95%, the model effectively differentiates between parasitized and uninfected samples. The high sensitivity and specificity achieved shows that this model could serve as a valuable tool in aiding medical professionals to quickly and accurately diagnose malaria, thereby facilitating timely and appropriate treatment interventions hence saving lives.

Future enhancements could include model optimization to further reduce false negatives and the integration of this model into a real-time diagnostic tool for use in medical facilities, particularly in malaria-endemic regions.
