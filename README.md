Customer Churn Prediction
This project uses Artificial Neural Networks (ANNs) to predict customer churn for a bank, utilizing features such as Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Credit Card status, Active Member status, and Estimated Salary. It includes end-to-end workflow: data preprocessing, model training/hyperparameter tuning, and deploying a prediction web app.

Table of Contents
Features

Project Structure

Setup Instructions

Usage

Jupyter Notebooks

Streamlit Web App

Model Details

Notes

Contributors

Features
Data Preprocessing: Encoding categorical features, scaling, and train-test split.

Neural Network Model: Uses TensorFlow & Keras for binary classification.

Hyperparameter Tuning: GridSearchCV with scikeras wrapper for tuning ANN hyperparameters.

Prediction Pipeline: End-to-end sample prediction included in notebook.

Streamlit Web App: Interactive UI for real-time churn prediction.

Project Structure
text
├── app.py                    # Streamlit web application
├── model.h5                  # Trained ANN model
├── label_encoder_gender.pkl  # Scikit-learn LabelEncoder for gender
├── onehot_encoder_geo.pkl    # Scikit-learn OneHotEncoder for geography
├── scaler.pkl                # Scikit-learn StandardScaler for features
├── experiments.ipynb         # Data preprocessing, feature engineering
├── prediction.ipynb          # End-to-end prediction pipeline
├── hyperparametertuning.ipynb# Hyperparameter tuning using GridSearchCV
├── salaryregression.ipynb    # (Optional) Regression experiments
Setup Instructions
Clone the repository

bash
git clone <https://github.com/Sridharsahu125/Customer_Churn_Prediction>

Create Virtual Environment (optional but recommended)

bash
python -m venv myenv
source myenv/bin/activate  # Or myenv\Scripts\activate for Windows
Install Dependencies

bash
pip install -r requirements.txt
Required packages include:

tensorflow

scikit-learn

pandas

numpy

streamlit

pickle (usually part of standard library)

Usage
Jupyter Notebooks
experiments.ipynb: Data loading, exploration, preprocessing, and feature engineering.

hyperparametertuning.ipynb: Model building & tuning.

prediction.ipynb: How to use trained model for sample predictions.

Open and run any notebook using:

bash
jupyter notebook
Streamlit Web App
Ensure you have files: model.h5, label_encoder_gender.pkl, onehot_encoder_geo.pkl, scaler.pkl in your project root.

Run the app:

bash
streamlit run app.py
Input customer features in the web UI to get churn probability and risk prediction.

Model Details
Input Features:

CreditScore

Geography (France, Germany, Spain)

Gender (Male, Female)

Age

Tenure

Balance

NumOfProducts

HasCrCard (0/1)

IsActiveMember (0/1)

EstimatedSalary

Preprocessing:

LabelEncoder for Gender

OneHotEncoder for Geography

StandardScaler for numerical features

Model:

TensorFlow/Keras Sequential ANN

Hyperparameters tuned with sklearn/scikeras GridSearchCV

Notes
Make sure the pickled encoders (label_encoder_gender.pkl, onehot_encoder_geo.pkl, scaler.pkl) match those used for training.

The web app expects these files in the working directory.

You may see Keras warnings about input shape; this does not affect inference.

Contributors
SRIDHAR SAHU