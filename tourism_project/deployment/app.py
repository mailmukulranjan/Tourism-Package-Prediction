import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="mailmukulranjan/tourism_package_model", filename="best_visit_with_us_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Visit With Us Tourism Data Input Form")
st.write("""
This application predicts the whether a user will purchase a **Visit with us tourism** package or not.""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
duration_of_pitch = st.number_input("Duration Of Pitch", min_value=1, max_value=100, value=10, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=50, value=3, step=1)
number_of_followups = st.number_input("Number Of Followups", min_value=0, max_value=20, value=3, step=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, step=1)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=50, value=1, step=1)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3, step=1)
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=20, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'product_pitched': product_pitched,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# Predict button
if st.button("Predict Product Taken"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"{prediction}")
