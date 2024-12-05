# Alphabet Soup Charity - Deep Learning Model for Funding Success Prediction

## Overview

Alphabet Soup is a nonprofit foundation seeking a machine learning solution to improve the selection of applicants for funding. The aim is to develop a binary classification model using neural networks that can predict the success of funded projects based on their characteristics. This tool will empower Alphabet Soup to allocate resources more effectively to ventures with the highest probability of success.

---

## Table of Contents

1. [Background](#background)
2. [Dataset Description](#dataset-description)
3. [Project Steps](#project-steps)
    - [Step 1: Data Preprocessing](#step-1-data-preprocessing)
    - [Step 2: Model Compilation, Training, and Evaluation](#step-2-model-compilation-training-and-evaluation)
    - [Step 3: Model Optimization](#step-3-model-optimization)
    - [Step 4: Analysis Report](#step-4-analysis-report)
4. [Model Files](#model-files)
5. [Dependencies](#dependencies)
6. [Results and Summary](#results-and-summary)

---

## Background

The provided dataset contains over 34,000 records of organizations funded by Alphabet Soup, with various features describing the applications and funding outcomes. The goal is to create a machine learning model that identifies patterns in the data to predict the likelihood of success for future applicants.

---

## Dataset Description

The dataset includes the following columns:

- **Identification Columns**: `EIN`, `NAME`
- **Application Characteristics**:
  - `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`
- **Additional Attributes**:
  - `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
- **Target Variable**:
  - `IS_SUCCESSFUL`: Indicates whether funding was used effectively (1 = successful, 0 = not successful).

---

## Project Steps

### Step 1: Data Preprocessing

1. Loaded and inspected the dataset.
2. Identified:
   - **Target variable**: `IS_SUCCESSFUL`
   - **Features**: All other relevant columns excluding identification columns.
   - Dropped `EIN` and `NAME` as they are irrelevant to the prediction.
3. Encoded categorical variables using `pd.get_dummies()`.
4. Combined rare categories into an `Other` group for columns with numerous unique values.
5. Scaled the dataset using `StandardScaler` for normalization.
6. Split the data into training and testing sets.

---

### Step 2: Model Compilation, Training, and Evaluation

1. Designed a neural network model with the following:
   - Input features derived from preprocessed data.
   - Hidden layers with appropriate activation functions (e.g., ReLU).
   - Output layer with a sigmoid activation function for binary classification.
2. Compiled the model using binary cross-entropy as the loss function and accuracy as the metric.
3. Trained the model on the training dataset while monitoring performance on the test dataset.
4. Used callbacks to save the model weights every five epochs.

---

### Step 3: Model Optimization

1. Applied multiple strategies to optimize model performance:
   - Adjusted the number of neurons and layers.
   - Experimented with different activation functions.
   - Tuned hyperparameters such as the number of epochs and learning rate.
2. Evaluated optimized models for accuracy, aiming for >75% predictive performance.

---

### Step 4: Analysis Report

Detailed the following:
- Preprocessing steps and variable selections.
- Neural network architecture, activation functions, and training process.
- Results from optimization attempts.
- Recommendations for alternative models to improve classification accuracy further.

---

## Model Files

The trained and optimized models are available in the following files:
- **Initial Model**: `AlphabetSoupCharity.h5`
- **Optimized Model**: `AlphabetSoupCharity_Optimization.h5`

---

## Dependencies

The following libraries and frameworks were used:

- Python (3.x)
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn
- Matplotlib (optional for data visualization)

---

## Results and Summary

- **Initial Accuracy**: Achieved [initial accuracy]% with the base model.
- **Optimized Accuracy**: Improved accuracy to [optimized accuracy]% through model optimization techniques.
- **Recommendation**: Further improve the model using techniques such as ensemble methods, feature engineering, or alternative machine learning models like Random Forest or XGBoost.

The final tool provides Alphabet Soup with a reliable way to predict the success of funding applications, helping allocate resources effectively.

---
