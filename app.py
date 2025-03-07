import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Load dataset
def load_data():
    file_path = "crop_yield_dataset.csv"  # Update with your actual file path
    return pd.read_csv(file_path)

st.title("🌾 Crop Yield Prediction Dashboard")

# File uploader for new datasets
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# Display dataset preview
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Summary statistics
st.subheader("📈 Summary Statistics")
st.write(df.describe())

# Select columns for visualization
numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
if numeric_columns:
    x_axis = st.selectbox("📌 Select X-axis", numeric_columns)
    y_axis = st.selectbox("📌 Select Y-axis", numeric_columns)
    
    # Plot data
    st.subheader("📌 Data Visualization")
    fig, ax = plt.subplots()
    ax.scatter(df[x_axis], df[y_axis], alpha=0.7, color='green')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title(f"Scatter Plot: {x_axis} vs {y_axis}")
    st.pyplot(fig)
else:
    st.warning("⚠ No numeric columns available for visualization.")

# Crop Yield Prediction
st.subheader("🤖 Crop Yield Prediction")
predictors = st.multiselect("✅ Select input features", numeric_columns)
target = st.selectbox("🎯 Select target (Crop Yield)", numeric_columns)

if predictors and target:
    X = df[predictors]
    y = df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"📉 Mean Absolute Error: {mae:.2f}")
    st.write(f"📊 R² Score: {r2:.2f}")
    
    # User input for prediction
    st.subheader("🔍 Make a Prediction")
    input_values = []
    
    for feature in predictors:
        value = st.number_input(f"Enter {feature} value:", float(df[feature].min()), float(df[feature].max()))
        input_values.append(value)
    
    if st.button("🚀 Predict Crop Yield"):
        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)
        st.success(f"🌾 Predicted Crop Yield: {prediction[0]:.2f}")

st.success("✅ Dashboard Ready! 🎉")
