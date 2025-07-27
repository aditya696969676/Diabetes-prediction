import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load model and feature list
model = joblib.load("dt_model.pkl")
features = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Diabetes Predictor", page_icon="💉")
st.title("🩺 Diabetes Prediction App")

# 2. Define domains exactly as in training preprocessing
domains = {
    'Age': ['less than 40', '40-49', '50-59', '60 or older'],
    'Gender': ['male', 'female'],
    'PhysicallyActive': ['less than half an hr', 'more than half an hr', 'none', 'one hr or more'],
    'JunkFood': ['occasionally', 'often', 'very often',"always"],
    'Stress': ['not at all', 'sometimes', 'very often', 'always'],
    'BPLevel': ['low', 'high', 'normal'],
    'UriationFreq': ['not much', 'quite often'],
    'Family_Diabetes': ['no', 'yes'],
    'Smoking': ['no', 'yes'],
    'Alcohol': ['no', 'yes'],
    'RegularMedicine': ['no', 'yes'],
    'Pdiabetes': ['no', 'yes']
}

# 3. Identify numeric vs categorical features
numeric_fields = ['Pregancies', 'BMI', 'Sleep', 'SoundSleep']
categorical_fields = [f for f in features if f not in numeric_fields]

# 4. Collect user inputs
user_input = {}

# Numeric inputs
for col in numeric_fields:
    user_input[col] = st.number_input(col, value=0.0)

# Categorical inputs via dropdowns
cat_values = {}
for col in categorical_fields:
    opts = domains.get(col)
    if opts:
        cat_values[col] = st.selectbox(col, opts)
    else:
        st.error(f"❌ No domain defined for feature '{col}'")
        st.stop()

# 5. Build a single‐row DataFrame with zeros
input_df = pd.DataFrame(0, index=[0], columns=features)

# 6. Fill numeric fields
for col in numeric_fields:
    input_df.at[0, col] = user_input[col]

# 7. Label‐encode all categorical fields
for col in categorical_fields:
    le = LabelEncoder().fit(domains[col])
    code = le.transform([cat_values[col]])[0]
    input_df.at[0, col] = code

# 8. Predict on button click
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.markdown(
        f"### Prediction: {'✅ Diabetic' if pred == 1 else '❌ Not Diabetic'}"
    )
