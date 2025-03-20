# Gender Classification Using Convolutional Neural Networks (CNN)

This project utilizes Convolutional Neural Networks (CNN) for gender classification (male or female) based on images from the **UTKFace** dataset. The model is trained to classify the gender of individuals based on their facial images.

## Overview

The CNN model is built using Keras with TensorFlow backend. The dataset used in this project contains images labeled with gender information (male = 0, female = 1). The process involves:

1. Downloading and preprocessing the dataset.
2. Building a CNN model.
3. Training the model on the dataset.
4. Evaluating the model on a test set.
5. Visualizing and saving results such as accuracy, loss, confusion matrix, and sample predictions.

## Project Setup

### Prerequisites

Make sure you have the following Python libraries installed:
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn
- Kagglehub (for downloading datasets)

You can install the required dependencies by running:
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn kagglehub
```

### Dataset

The **UTKFace dataset** is used, which contains images labeled with age, gender, and ethnicity information. The gender labels are extracted for this project. You can download the dataset from Kaggle using the following command:
```python
path = kagglehub.dataset_download("jangedoo/utkface-new")
DATASET_PATH = os.path.join(path, "UTKFace")
```

### Dataset Structure

The dataset should be placed in the folder `UTKFace`, with images named in the format:
```
age_gender_ethnicity.jpg
```
Where:
- `age` is the person's age,
- `gender` is the person's gender (0 for male, 1 for female),
- `ethnicity` is the ethnicity of the person (not used in this project).

## Usage

### Step 1: Download the Dataset
The dataset is automatically downloaded from Kaggle using the `kagglehub` library:
```python
path = kagglehub.dataset_download("jangedoo/utkface-new")
DATASET_PATH = os.path.join(path, "UTKFace")
```

### Step 2: Preprocess the Data
The images are loaded and resized to a target size of 64x64 pixels. Gender labels are extracted from the filenames (0 for male, 1 for female).

### Step 3: Split the Data
The data is split into training and testing sets (80% for training, 20% for testing).

### Step 4: Data Augmentation
Data augmentation is applied to the training data using random rotations and horizontal flips. This increases the diversity of the training data and helps to prevent overfitting.

### Step 5: Build the CNN Model
The CNN architecture includes:
- 3 convolutional layers with ReLU activations
- 3 max-pooling layers
- 1 fully connected layer
- 1 output layer with a sigmoid activation function for binary classification

The model is compiled using the Adam optimizer with binary cross-entropy loss.

### Step 6: Train the Model
The model is trained using the augmented data for 10 epochs. The training progress is shown in terms of accuracy and loss.

### Step 7: Evaluate the Model
After training, the model is evaluated on the test set. The evaluation includes:
- Accuracy and loss plots
- Confusion matrix
- Classification report

### Step 8: Save the Model
The trained model is saved to disk for future use:
```python
model.save('cnn_gender_classifier.keras')
```

### Step 9: Visualize Results
- **Accuracy and Loss Plots:** Show the training and validation accuracy/loss over the epochs.
- **Confusion Matrix:** Visualizes the true vs. predicted labels.
- **Sample Predictions:** Displays 25 sample test images with predicted and actual labels.

## File Structure

- `train_model.py`: Main script for training the model, preprocessing data, and evaluating performance.
- `cnn_gender_classifier.keras`: The trained model file.
- `model_plot.png`: The architecture of the CNN model.
- `accuracy_plot.png`: Plot showing training and validation accuracy over epochs.
- `loss_plot.png`: Plot showing training and validation loss over epochs.
- `confusion_matrix.png`: Confusion matrix showing true vs. predicted labels.
- `sample_predictions.png`: Visualization of sample predictions from the test set.
Here’s the section you can add to your README file regarding the saved model `cnn_gender_classifier.keras`:

---

### **Saved Model - `cnn_gender_classifier.keras`**

The file `cnn_gender_classifier.keras` contains the trained Convolutional Neural Network (CNN) model, which is capable of classifying gender based on the input images. This file includes:

- **Model Architecture**: The structure of the neural network, including all layers, their types, and configurations.
- **Model Weights**: The learned weights from the training process that allow the model to make accurate predictions.
- **Training Configuration**: The optimizer, loss function, and evaluation metrics used during the training phase.

### **How to Use the Saved Model**

Once the model is saved, you can reload it and use it to make predictions without the need to retrain it. Here’s how you can load the model and use it:

```python
from tensorflow.keras.models import load_model

# Load the pre-trained model from the saved file
model = load_model('cnn_gender_classifier.keras')

# Example: Predict gender for new images (assuming 'new_images' is your test data)
predictions = model.predict(new_images)

# Convert predictions to gender labels (0 for Male, 1 for Female)
predicted_labels = (predictions > 0.5).astype(int)
```

### **Key Benefits of the Saved Model**

- **Time Efficiency**: You don't need to train the model again. The trained weights and architecture are saved, allowing for quick predictions.
- **Consistency**: The model will produce the same results each time it is loaded, as it has already been trained.
- **Portability**: The model can be shared or deployed on different machines, allowing others to use it without the need for training data.

### **Next Steps**
Once the model is loaded, you can use it to classify new images, fine-tune it with additional data, or deploy it for real-time gender classification tasks.

---

## Results

After running the code, you will get the following outputs:

- **Training & Validation Accuracy Plot:** Saved as `accuracy_plot.png`.
- **Training & Validation Loss Plot:** Saved as `loss_plot.png`.
- **Confusion Matrix Plot:** Saved as `confusion_matrix.png`.
- **Sample Predictions:** Displayed in the `sample_predictions.png` file, showing some example test images with their predicted and true labels.

## **Conclusion**
- This project demonstrates a simple yet effective approach to binary classification of gender using Convolutional Neural Networks. The CNN model achieves a high level of accuracy and provides useful visualizations for evaluating performance.
## Further Improvements

There are several ways to enhance the performance and extend the functionality of this model:

### 1. **Data Preprocessing**
   - **Increase Image Size:** Resizing images to a higher resolution (e.g., 128x128 or 256x256) could improve the model's ability to detect finer details.
   - **Better Data Augmentation:** Implement more advanced data augmentation techniques, such as zoom, shear, and brightness adjustments, to increase the diversity of the training set.
   
### 2. **Model Improvements**
   - **Add More Layers:** Implement deeper CNN architectures with more convolutional and fully connected layers for improved feature extraction.
   - **Transfer Learning:** Use pre-trained models like VGG16, ResNet, or Inception as the base for feature extraction, and fine-tune them on the gender classification task.
   - **Hyperparameter Tuning:** Explore other optimizer algorithms (e.g., SGD, RMSprop) and fine-tune the learning rate, batch size, and other hyperparameters.

### 3. **Handling Imbalanced Data**
   - If the dataset is imbalanced (e.g., a significantly higher number of male images than female images), techniques like oversampling, undersampling, or class weighting can be employed to improve the model's performance on the minority class.

### 4. **Evaluation Improvements**
   - **Cross-Validation:** Instead of a single training/test split, use k-fold cross-validation to ensure that the model generalizes well on different subsets of the dataset.
   - **More Metrics:** Besides accuracy, consider adding more evaluation metrics such as precision, recall, and F1-score, especially if the classes are imbalanced.

### 5. **Deploy the Model**
   - The trained model can be deployed using frameworks such as Flask or FastAPI for creating web applications or integrated with cloud platforms like AWS, Google Cloud, or Azure for scalable deployment.

### 6. **Expand the Dataset**
   - Consider using additional datasets that contain more diverse facial images, including different ethnicities and age groups, to improve the robustness and generalizability of the model.

## **Acknowledgments**
- Thanks to the authors of the UTKFace dataset.
- TensorFlow, Keras, and OpenCV for providing the tools used in building the model and processing the images.
- Kagglehub for providing a convenient way to download datasets.


---
