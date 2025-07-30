import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_model.pkl")

# Load the original dataframe to fit LabelEncoders
df = pd.read_csv('Salary Data.csv')
df.columns = ['Age', 'Gender', 'Education_Level', 'Job_Title', 'Years_of_Experience', 'Salary']
df['Salary_Class'] = df['Salary'].apply(lambda x: '>50K' if x > 50000 else '<=50K')
sal_df = df.drop_duplicates(keep='first')
sal_df.dropna(how = 'any', inplace=True)

# Fit LabelEncoders on the original data
gender_le = LabelEncoder()
sal_df['Gender'] = gender_le.fit_transform(sal_df['Gender'])

degree_le = LabelEncoder()
sal_df['Education_Level'] = degree_le.fit_transform(sal_df['Education_Level'])

job_title_le = LabelEncoder()
sal_df['Job_Title'] = job_title_le.fit_transform(sal_df['Job_Title'])


st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
education = st.sidebar.selectbox("Education Level", df['Education_Level'].unique())
job_title = st.sidebar.selectbox("Job Title", df['Job_Title'].unique())
experience = st.sidebar.slider("Years of Experience", 0.0, 40.0, 5.0)

# Encode categorical inputs
gender_encoded = gender_le.transform([gender])[0]
education_encoded = degree_le.transform([education])[0]
job_title_encoded = job_title_le.transform([job_title])[0]


# Build input DataFrame - Corrected column names
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender_encoded],
    'Degree': [education_encoded], # Corrected from 'Education_Level'
    'Job_Title': [job_title_encoded],
    'Experience_Years': [experience] # Corrected from 'Years_of_Experience'
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    # Ensure columns match and encode categorical features
    batch_data.columns = ['Age', 'Gender', 'Education_Level', 'Job_Title', 'Years_of_Experience', 'Salary']
    batch_data['Gender'] = gender_le.transform(batch_data['Gender'])
    batch_data['Education_Level'] = degree_le.transform(batch_data['Education_Level'])
    batch_data['Job_Title'] = job_title_le.transform(batch_data['Job_Title'])

    # Correct column names for batch prediction before dropping 'Salary'
    batch_data = batch_data.rename(columns={'Education_Level': 'Degree', 'Years_of_Experience': 'Experience_Years'})

    batch_preds = model.predict(batch_data.drop(columns=['Salary']))
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
