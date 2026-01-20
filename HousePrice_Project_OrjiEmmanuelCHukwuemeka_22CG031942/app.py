import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved model
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file
model_path = os.path.join(BASE_DIR, 'model', 'house_price_model.pkl')

# Load the model
model = joblib.load(model_path)

# Set up the title and description
st.title("🏡 House Price Prediction System")
st.write("Input the house features below to get an estimated market price.")

# 2. Create input fields for the 6 features
# We use the same names used during training
st.header("Enter House Details")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
    total_bsmt_sf = st.number_input("Total Basement (sq ft)", min_value=0, max_value=5000, value=800)

with col2:
    garage_cars = st.selectbox("Garage Capacity (Cars)", [0, 1, 2, 3, 4, 5])
    year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=2000)
    # Common neighborhoods from the dataset
    neighborhood = st.selectbox("Neighborhood", [
        'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 
        'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 
        'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 
        'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'
    ])

# 3. Prediction Logic
if st.button("Predict House Price"):
    # Create a DataFrame with the user inputs
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'TotalBsmtSF': [total_bsmt_sf],
        'GarageCars': [garage_cars],
        'YearBuilt': [year_built],
        'Neighborhood': [neighborhood]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # 4. Display the result
    st.success(f"### Estimated Price: ${prediction[0]:,.2f}")
    st.balloons()

st.info("Note: This prediction is based on the Random Forest model trained in Part A.")