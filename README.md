# Cardiovascular Disease Prediction with Explainable AI (XAI)

This project aims to develop advanced machine learning and deep learning models for predicting cardiovascular disease (CVD) and integrate Explainable AI (XAI) techniques to interpret the predictions. The project combines **Exploratory Data Analysis (EDA)**, **traditional machine learning models**, **neural networks**, and **XAI tools** like SHAP and LIME to provide insights into the factors influencing predictions.

---

## **Project Objectives**

1. Build robust machine learning and deep learning models for accurate CVD prediction.
2. Apply **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to explain model predictions.
3. Compare insights from **EDA** with model explanations to validate consistency in risk factors.
4. Address class imbalance using techniques like **SMOTE**.

---

## **Dataset**

The dataset used is the **Heart Failure Clinical Records Dataset**, which contains 299 samples and 13 features. The target variable is `DEATH_EVENT`, indicating whether a patient died during the follow-up period.

### Key Features:
- **Numeric Features**: `age`, `ejection_fraction`, `serum_sodium`, `time`, etc.
- **Binary Features**: `anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`.
- **Target Variable**: `DEATH_EVENT` (1 = Died, 0 = Survived).

---

## **Steps in the Project**

### 1. **Exploratory Data Analysis (EDA)**
- Performed detailed analysis to understand feature distributions, correlations, and relationships with the target variable.
- Visualizations include:
  - Histograms for numeric features.
  - Correlation heatmaps.
  - Boxplots and bar plots for feature-target relationships.

### 2. **Data Preprocessing**
- Standardized continuous features using `StandardScaler`.
- Log-transformed skewed features to handle outliers.
- Addressed class imbalance using **SMOTE**.

### 3. **Model Training**
#### **Traditional Machine Learning Models**
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
- Hyperparameter tuning was performed using `GridSearchCV`.

#### **Neural Network Models**
- **CNN (Convolutional Neural Network)**
- **LSTM (Long Short-Term Memory)**
- **CNN-LSTM Hybrid**
- **BiLSTM (Bidirectional LSTM)**
- **GRU (Gated Recurrent Unit)**
- **CNN with Attention Mechanism**

### 4. **Model Evaluation**
- Metrics used: Accuracy, Precision, Recall, F1-score, ROC-AUC.
- Confusion matrices and ROC curves were plotted for validation and test sets.

### 5. **Explainable AI (XAI)**
- **SHAP**:
  - Used to explain feature importance for traditional models (Logistic Regression, SVM, Random Forest) and neural networks.
  - Visualizations include summary plots and feature importance rankings.
- **LIME**:
  - Provided instance-level explanations for predictions.
  - Applied to both traditional and neural network models.

---

## **How to Run the Project**

1. **Install Dependencies**:
   ```bash
   pip install lime shap scikit-learn imblearn matplotlib seaborn pandas numpy tensorflow