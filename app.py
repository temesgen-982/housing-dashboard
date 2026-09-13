import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

APP_DIR = Path(__file__).parent

# Page Configuration
st.set_page_config(page_title="Ethiopia Housing Dashboard", layout="centered")
st.title("Real Estate Valuation Dashboard")
st.write("Adjust the property specifications to see predictions from a live trained ML model.")

# Securely Load Real Trained Notebook Models
@st.cache_resource # This prevents Streamlit from slowing down by reloading the files on every click
def load_ml_assets():
    model = joblib.load(APP_DIR / 'property_model.pkl')
    scaler = joblib.load(APP_DIR / 'property_scaler.pkl')
    return model, scaler

try:
    multi_model, scaler = load_ml_assets()
    st.success("Model Connected!")
except Exception as e:
    st.error("Could not load model files. Make sure 'property_model.pkl' and 'property_scaler.pkl' are in this folder.")
    st.stop()

# Dashboard Control Panels (Sidebar UI)
st.sidebar.header("Property Characteristics")

# Get inputs for all 10 features the model expects
number_of_rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
site_area_sqm = st.sidebar.slider("Total Site Area (sqm)", min_value=100, max_value=600, value=250)
built_area_sqm = st.sidebar.slider("Livable Built Area (sqm)", min_value=40, max_value=500, value=120)
property_years = st.sidebar.number_input("Property Age (Years)", min_value=0, max_value=50, value=5)
proximity_to_cbd = st.sidebar.slider("Distance to CBD (km)", min_value=0.1, max_value=10.0, value=2.5, step=0.1)
proximity_to_schools = st.sidebar.slider("Distance to Schools (km)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)

# Handled categoricals
construction_materials_encoded = st.sidebar.selectbox("Construction Material", options=[0, 1], format_func=lambda x: "Concrete" if x == 0 else "Mud & Wood")
housing_typology_encoded = st.sidebar.selectbox("Housing Typology", options=[0, 1, 2], format_func=lambda x: ["Condominium", "Detached", "Semi-detached"][x])
land_grade = st.sidebar.selectbox("Land Value Grading", options=['Low', 'Medium', 'High'], index=2)
type_of_nearest_road_encoded = st.sidebar.selectbox("Nearest Road Type", options=[0, 1], format_func=lambda x: "Asphalt" if x == 0 else "Gravel")

# Calculator Function (available on the notebook too)
def predict_my_house_value(
    number_of_rooms, site_area_sqm, built_area_sqm, property_years, 
    proximity_to_cbd_km, proximity_to_schools_km, construction_materials_encoded, 
    housing_typology_encoded, land_grade, type_of_nearest_road_encoded
):
    grading_map = {'Low': 0, 'Medium': 1, 'High': 2}
    land_value_grading_encoded = grading_map.get(land_grade, 1)
    
    feature_names = [
        'Number_of_Rooms', 'Site_Area_sqm', 'Built_Area_sqm', 'Property_Years', 
        'Proximity_to_CBD_km', 'Proximity_to_Schools_km', 'Construction_Materials_encoded', 
        'Housing_Typology_encoded', 'Land_Value_Grading_encoded', 'Type_of_Nearest_Road_encoded'
    ]
    
    raw_inputs_df = pd.DataFrame([[
        number_of_rooms, site_area_sqm, built_area_sqm, property_years, 
        proximity_to_cbd_km, proximity_to_schools_km, construction_materials_encoded, 
        housing_typology_encoded, land_value_grading_encoded, type_of_nearest_road_encoded
    ]], columns=feature_names)
    
    scaled_inputs = scaler.transform(raw_inputs_df)
    predicted_price = multi_model.predict(scaled_inputs)
    return predicted_price[0]

# Process and Display Result
predicted_etb = predict_my_house_value(
    number_of_rooms, site_area_sqm, built_area_sqm, property_years, 
    proximity_to_cbd, proximity_to_schools, construction_materials_encoded, 
    housing_typology_encoded, land_grade, type_of_nearest_road_encoded
)

st.metric(
    label="Calculated Property Valuation", 
    value=f"ETB {predicted_etb:,.2f}"
)
