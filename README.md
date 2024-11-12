# Diabetes-Prediction-using-ML-Models

This project builds a machine learning pipeline to predict diabetes using the Pima Indians Diabetes Database. Multiple machine learning algorithms are tested and optimized using hyperparameter tuning to select the best performing model.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)

## Project Overview

The aim of this project is to develop a reliable machine learning model to predict diabetes. By evaluating several classification algorithms and optimizing hyperparameters, the project seeks to reduce false positives and false negatives for improved prediction accuracy.

## Dataset

The data used in this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) hosted on Kaggle. The target variable is `Outcome`, which indicates whether a patient has diabetes (`1`) or not (`0`).

## Project Structure

- `diabetes.csv` - Dataset used for training and testing.
- `diabetes_prediction.py` - Main code file for data processing, model training, and evaluation.
- `README.md` - Project documentation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/diabetes-prediction.git
    cd diabetes-prediction
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

   **Requirements**:
   - pandas
   - numpy
   - scikit-learn
   - seaborn
   - matplotlib

3. Ensure that you have the `diabetes.csv` dataset in the specified path or update the path accordingly in the code.

## Usage

1. Open `diabetes_prediction.py` to review the main script, which:
   - Loads and preprocesses the data.
   - Splits the data into training and testing sets.
   - Scales features using `StandardScaler`.
   - Trains multiple models using `GridSearchCV` to find the best hyperparameters.
   - Evaluates each model's performance.
   - Chooses the best model and applies it to make predictions on the test data.

2. Run the script:
    ```bash
    python diabetes_prediction.py
    ```

## Model Evaluation

This project compares the following machine learning models:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Decision Tree
- Naive Bayes

Each model undergoes hyperparameter tuning with `GridSearchCV` to optimize for F1 score, a balanced metric considering both precision and recall. The best model is then selected based on cross-validation F1 score.

## Results

The script outputs:
- Cross-validated F1 scores for each model.
- Best performing model details.
- Performance metrics for the best model on test data:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

It also generates a confusion matrix heatmap to visualize the prediction performance and identify false positives and false negatives.

## Example Output

```plaintext
Random Forest - Cross-Validated F1 Score: 0.753
Logistic Regression - Cross-Validated F1 Score: 0.722
SVM - Cross-Validated F1 Score: 0.734
Decision Tree - Cross-Validated F1 Score: 0.682
Naive Bayes - Cross-Validated F1 Score: 0.710

Best Model: RandomForestClassifier(max_depth=10, n_estimators=100)
Accuracy: 0.787
Precision: 0.759
Recall: 0.740
F1 Score: 0.749
