# 🩺 Diabetes Prediction Web Application (Streamlit + ML)

This project is a machine learning–based web application built with **Streamlit** that predicts whether a person is diabetic based on health, lifestyle, and demographic inputs. The model is trained using a cleaned dataset with proper label encoding and class imbalance handling using **SMOTE**.

---

## 📌 Problem Statement

Early detection of diabetes is critical for prevention and treatment. However, in many regions, regular checkups are limited. This application aims to assist in predicting the likelihood of diabetes based on basic user inputs like age group, gender, lifestyle, and health parameters.

---

## 📂 Project Structure
├── app.py # Streamlit web app
├── dt_model.pkl # Trained Decision Tree Classifier model
├── feature_names.pkl # Feature list used for prediction
├── diabetes_dataset.csv # Cleaned and preprocessed dataset (optional to include)
├── README.md # Project documentation


---

## 🧠 Model Pipeline

### 🔹 1. Data Cleaning
- Filled null values (e.g., missing `Pregnancies` for males set to 0).
- Replaced outlier BMI values using IQR-based clipping.
- Removed inconsistent values like `"o"` in `RegularMedicine`.

### 🔹 2. Data Normalization
- Converted inconsistent values (`"Yes"`, `"yes"`, `"0"`, `"1"`) to clean labels (`"yes"`/`"no"`).
- Standardized categorical variables (e.g., `"High"`, `"high"` → `"high"`).

### 🔹 3. Label Encoding
All categorical features were encoded using `LabelEncoder`:
- Binary & multiclass variables (e.g., `Gender`, `JunkFood`, `Stress`, `BPLevel`) were all label-encoded into integers.

### 🔹 4. Class Balancing
Used **SMOTE (Synthetic Minority Oversampling Technique)** to fix class imbalance.

### 🔹 5. Model Training
- **Model**: `DecisionTreeClassifier`  
- **Hyperparameters**: `max_depth=14`, `min_samples_leaf=2`, `random_state=42`  
- Achieved balanced performance on test data after resampling.

---

## 🚀 How to Run the Web App

### 🛠️ Step-by-Step

1. **Clone the repository** or download the files.
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
4. streamlit run app.py

