import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("fetal_health.csv")
    return df

df = load_data()

# Set up page config
st.set_page_config(
    page_title="Fetal Health Prediction",
    layout="centered",
    page_icon="🧬"
)

# Title and description
st.title("🧬 Fetal Health Prediction App")
st.markdown("""
This app predicts **fetal health** based on Cardiotocographic measurements using a **Random Forest Classifier**.
""")

# Show data
with st.expander("📊 Show Dataset Preview"):
    st.dataframe(df.head())

# Sidebar - user input
st.sidebar.header("Enter Patient Features")

def user_input_features():
    baseline_value = st.sidebar.slider('Baseline Value (FHR)', 100, 180, 120)
    accelerations = st.sidebar.slider('Accelerations', 0.0, 1.0, 0.02)
    fetal_movement = st.sidebar.slider('Fetal Movement', 0.0, 1.0, 0.05)
    uterine_contractions = st.sidebar.slider('Uterine Contractions', 0.0, 1.0, 0.01)
    light_decelerations = st.sidebar.slider('Light Decelerations', 0.0, 1.0, 0.0)
    severe_decelerations = st.sidebar.slider('Severe Decelerations', 0.0, 1.0, 0.0)
    prolongued_decelerations = st.sidebar.slider('Prolongued Decelerations', 0.0, 1.0, 0.0)
    abnormal_short_term_var = st.sidebar.slider('Abnormal Short Term Variability', 0, 10, 2)
    mean_value_of_short_term_var = st.sidebar.slider('Mean Value of Short Term Variability', 0.0, 5.0, 1.0)
    percentage_of_time_with_abnormal_long_term_var = st.sidebar.slider('Abnormal Long Term Variability %', 0.0, 100.0, 20.0)
    mean_value_of_long_term_var = st.sidebar.slider('Mean Value of Long Term Variability', 0.0, 50.0, 5.0)

    data = {
        'baseline value': baseline_value,
        'accelerations': accelerations,
        'fetal_movement': fetal_movement,
        'uterine_contractions': uterine_contractions,
        'light_decelerations': light_decelerations,
        'severe_decelerations': severe_decelerations,
        'prolongued_decelerations': prolongued_decelerations,
        'abnormal_short_term_variability': abnormal_short_term_var,
        'mean_value_of_short_term_variability': mean_value_of_short_term_var,
        'percentage_of_time_with_abnormal_long_term_variability': percentage_of_time_with_abnormal_long_term_var,
        'mean_value_of_long_term_variability': mean_value_of_long_term_var
    }

    return pd.DataFrame(data, index=[0])

# Get user input
user_input_raw = user_input_features()

# Align input with training feature order
X = df.drop("fetal_health", axis=1)
input_features = X.columns.tolist()
user_input = user_input_raw.reindex(columns=input_features, fill_value=0)

# Prepare training
y = df["fetal_health"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(user_input)
prediction_label = ["Normal", "Suspect", "Pathological"][int(prediction[0]) - 1]

# Display results
st.subheader("🧾 Prediction Result")
st.write(f"### The predicted fetal condition is: **{prediction_label}**")

# Optional: show model accuracy
with st.expander("📈 Show Model Accuracy"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy on Test Set: `{acc * 100:.2f}%`")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
