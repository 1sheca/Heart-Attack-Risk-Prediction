# Heart Attack Risk Prediction

This project aims to predict the risk of a heart attack using various machine learning techniques. The primary focus is on using the k-nearest neighbors (KNN) algorithm on a dataset that has been balanced using the Synthetic Minority Over-sampling Technique (SMOTE).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Selection](#feature-selection)
- [Data Balancing](#data-balancing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Heart disease is a leading cause of death globally. Early prediction and diagnosis can help in reducing the risk of severe outcomes. This project utilizes machine learning to predict the likelihood of a heart attack based on various health metrics.

## Dataset

The dataset used in this project contains the following features:
- Age
- Total Cholesterol (totChol)
- Systolic Blood Pressure (sysBP)
- Diastolic Blood Pressure (diaBP)
- Body Mass Index (BMI)
- Heart Rate
- Glucose

The target variable is `TenYearCHD`, indicating the presence of heart disease over ten years.

## Feature Selection

Feature selection was performed using the Boruta algorithm to identify the most significant features for prediction. The selected features are:
- Age
- Total Cholesterol
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Body Mass Index
- Heart Rate
- Glucose

## Data Balancing

The dataset was imbalanced, so SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance it. This helps in improving the performance of the machine learning models by providing more balanced training data.

## Model Training

The k-nearest neighbors (KNN) algorithm was used for training the model. GridSearchCV was utilized to find the best hyperparameters for the KNN model.

## Model Evaluation

The model was evaluated using accuracy and a confusion matrix to understand its performance.

## Usage

To use this project, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/heart-attack-risk-prediction.git
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook to train the model and make predictions.

### Example Predictions

- To predict the risk of a heart attack for a high-risk individual:
  ```python
  h_risk = [[65, 150, 180, 70, 26.97, 80, 77]]
  prediction_risk = knn_clf_best.predict(scaler.transform(h_risk))
  print('You are safe. 😊') if prediction_risk[0] == 0 else print('Sorry, You are at risk. 👽')
  ```

- To predict the risk of a heart attack for a low-risk individual:
  ```python
  h_safe = [[39, 195, 106, 70, 26.97, 80, 77]]
  prediction_safe = knn_clf_best.predict(scaler.transform(h_safe))
  print('You are safe. 😊') if prediction_safe[0] == 0 else print('Sorry, You are at risk. 👽')
  ```

## Dependencies

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn
- statsmodels

## Results

The KNN model achieved an accuracy of approximately **85.59%**. The confusion matrix provides further insight into the model's performance.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

