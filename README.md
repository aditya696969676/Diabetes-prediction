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

📋 Features Used for Prediction
Feature Name	Description
Age	Categorical age groups
Gender	Male / Female
Pregnancies	Number of times pregnant
PhysicallyActive	Physical activity status
JunkFood	Junk food consumption
Stress	Self-reported stress level
UriationFreq	Frequency of urination
BMI	Body Mass Index
Family_Diabetes	Family history of diabetes
Smoking	Smoking habits
Alcohol	Alcohol consumption
RegularMedicine	Regular medication usage
Pdiabetes	Prediabetic status (self-reported)
BPLevel	Blood pressure level

🧠 Model Deployment Logic (Streamlit UI)
Dropdown inputs for all categorical variables (e.g., Age, Gender, BPLevel).

Number inputs for numerical features (Pregnancies, BMI).

On submit, input values are encoded based on the same mapping used during training.

Prediction result is displayed as either Diabetic or Not Diabetic.

👨‍💻 Author
Aditya Garg

B.Tech from Delhi Technological University

PG Diploma in Data Science

Proficient in Python, SQL, Power BI, Tableau

Built various ML and dashboarding projects
