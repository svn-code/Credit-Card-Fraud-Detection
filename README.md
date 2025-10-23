
# Credit Card Fraud Detection with Streamlit Deployment 🛡️

This project is an end-to-end machine learning solution for detecting credit card fraud. It uses the [Credit Card Fraud Detection Dataset 2023 from Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) to train a high-performance classification model.

The complete workflow—from data preprocessing and model training to hyperparameter tuning and deployment—is included. The final product is an interactive web application built with Streamlit that can analyze and predict the legitimacy of a transaction in real-time.



## Features

* **Real-time Fraud Prediction:** Analyzes 29 input features to classify a transaction as "Fraudulent" or "Not Fraudulent."
* **High-Performance Model:** Utilizes a tuned `XGBRFClassifier` (a GPU-accelerated Random Forest) that achieves ~100% accuracy on this (synthetic and balanced) dataset.
* **Interactive UI:** A clean and user-friendly web interface built with Streamlit, featuring an expander to neatly contain the 28 anonymized features.
* **Prediction Confidence:** Displays the precise probabilities for both fraud and non-fraud classes.

## Tech Stack

* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (for preprocessing), XGBoost (for modeling)
* **Web App:** Streamlit
* **Model Persistence:** joblib

---

## Project Workflow

This project follows a complete machine learning pipeline from data to deployment.

### 1. Data Loading & Understanding

* **Dataset:** The Kaggle 2023 Credit Card Fraud dataset was used.
* **Features:** The dataset contains an `id`, `V1-V28` (anonymized PCA features), `Amount`, and the target `Class` (0: Not Fraud, 1: Fraud).
* **Initial Analysis:**
    * The `id` column was dropped as it provides no predictive value.
    * The dataset was checked for null values (none found).
    * A key finding was that this dataset is **perfectly balanced** (50% fraud, 50% non-fraud), which simplifies modeling and makes **accuracy** a valid metric.

### 2. Preprocessing & Feature Engineering

* **Outlier Analysis:** A bivariate analysis revealed significant outliers in the `V1-V28` features.
* **Scaling Strategy:**
    * **Decision:** We decided **not** to remove these outliers, as anomalous values are the strongest indicators of fraud.
    * **Scaler:** To handle the outliers without skewing the data, `RobustScaler` was chosen over `StandardScaler`. `RobustScaler` uses the median and Interquartile Range (IQR), making it robust to extreme values.
    * **Application:** The `RobustScaler` was applied to all 29 features (`V1-V28` and `Amount`) to put them on a comparable scale.

### 3. Model Training & Selection

* **Data Split:** The data was split into training (80%) and testing (20%) sets. `stratify=y` was used to ensure the 50/50 class balance was preserved in both sets.
* **Models Tested:** Logistic Regression, Random Forest, and XGBoost were evaluated.
* **Model Selection:** The `XGBRFClassifier` (an XGBoost implementation of a Random Forest) was selected for its high performance and, most importantly, its ability to be **accelerated on a GPU**.

### 4. Hyperparameter Tuning

* `RandomizedSearchCV` was used to efficiently find the best hyperparameters for the `XGBRFClassifier`.
* The search was run with 3-fold cross-validation (`cv=3`) and optimized for `f1-score`.
* The entire tuning process was run on a Kaggle P100 GPU for maximum speed, using the `tree_method='gpu_hist'` parameter.

### 5. Final Model & Deployment

* **Final Model:** The best parameters from the search were used to train a final model on the *entire* scaled training set.
* **Evaluation:** The final model achieved **100% precision, recall, and F1-score** on the held-out test set.
* **Saving Artifacts:** The final model (`final_fraud_detection_model.joblib`) and the fitted `RobustScaler` (`data_scaler.joblib`) were saved using `joblib`.
* **Web App:** A Streamlit app (`app.py`) was built to provide a user-friendly interface for the model. The app loads the model and scaler to make live predictions.

---

## How to Run This Project Locally

Follow these steps to set up and run the Streamlit application on your local machine.

### 1. Clone the Repository

Clone this project to your local machine:
```bash
git clone https://github.com/svn-code/Credit-Card-Fraud-Detection.git
```
### 2.Install all required libraries 
```bash
pip install streamlit pandas numpy joblib scikit-learn xgboost
```
### 3.Run the Streamlit App

1.Open your terminal  or Command Prompt.

2Navigate to the project folder where all the files are located .

3.Use the following command in your terminal to run the app:
```bash
streamlit run app.py
```
Your web browser will automatically open to the application's URL.

## How to Use the App

1.Once the app is running, your browser will open.

2.Enter the Transaction Amount in the first input field.

3.Click on the expander titled "Enter Advanced Anonymized Features (V1 - V28)".

4.Fill in all 28 V feature values for the transaction.

5.Click the "Analyze Transaction" button.

6.The app will display the prediction ("FRAUDULENT" or "NOT FRAUDULENT") along with the confidence probabilities for each class.

##License
This project is licensed under the MIT License.

