# Gender Classification Using Machine Learning and Deep Learning Models

## Project Description

This project utilizes the [Gender Classification Dataset](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset) from Kaggle to classify images based on gender. The dataset is processed and experimented with various machine learning and deep learning models to determine the best approach for gender classification.

### Steps to Get Started:

1. **Download the Dataset**:
   - Download the dataset from [this link](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset).
   - Extract the folders and place them in the project directory.

2. **Configure the Number of Images**:
   - Before running the experiments, choose the number of images to be trained. The default is set to 1000, but this can be modified as needed.

### Project Workflow:

#### Data Exploration and Preparation:

- **Reshape Images**:
  - Resize the RGB images to a dimension of (64, 64, 3).
- **Convert to Grayscale**:
  - Convert the RGB images to grayscale.
- **Normalize Images**:
  - Normalize each image to ensure consistent data.

#### Experiments and Results:

1. **First Experiment**:
   - **Data Splitting**:
     - Split the data into training and testing datasets (if no testing dataset is available).
   - **Train SVM Model**:
     - Train a Support Vector Machine (SVM) model on the grayscale images.
   - **Model Evaluation**:
     - Test the model and provide the confusion matrix and average F1 scores for the testing dataset.

2. **Second Experiment**:
   - **Data Splitting**:
     - Further split the training dataset into training and validation datasets (if no validation dataset is available).
   - **Build Neural Networks**:
     - Construct two different Neural Networks with varying architectures (e.g., number of hidden layers, neurons, activations).
   - **Train and Validate**:
     - Train each model on the grayscale images and plot the error and accuracy curves for the training and validation data.
   - **Save and Reload Best Model**:
     - Save the best-performing model to a file, then reload it.
   - **Model Evaluation**:
     - Test the best model and provide the confusion matrix and average F1 scores for the testing dataset.

3. **Third Experiment**:
   - **Train Convolutional Neural Networks (CNNs)**:
     - Train a CNN on the grayscale images and plot the error and accuracy curves for the training and validation data.
     - Train another CNN on the RGB images and plot the error and accuracy curves for the training and validation data.
   - **Save and Reload Best Model**:
     - Save the best-performing CNN model to a file, then reload it.
   - **Model Evaluation**:
     - Test the best model and provide the confusion matrix and average F1 scores for the testing dataset.

#### Comparison and Conclusion:

- **Compare Results**:
  - Analyze and compare the results from the SVM, Neural Networks, and CNN models.
- **Best Model Suggestion**:
  - Suggest the best model based on the evaluation metrics and performance on the testing dataset.