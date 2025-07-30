
#  Salary Prediction App

This project is a **Streamlit-based web application** that predicts salary class based on user input features such as age, gender, degree, job title, and years of experience. It uses several machine learning models (like Logistic Regression, Random Forest, Gradient Boosting, etc.) trained on a processed Salary dataset.

##  Features
- Interactive **Streamlit UI**
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Label encoding of categorical features
- Salary class prediction using multiple ML models
- Easy model performance comparison

##  How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/salary-prediction-app.git
cd salary-prediction-app
````

### 2. Install Dependencies

Make sure you have Python 3.10+ installed, then:

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
streamlit run app.py
```

This will open the app in your default web browser at `http://localhost:8501`.

## 📊 Dataset Source

The dataset used in this project can be found here:  
[Employee Salaries Dataset – Kaggle](https://www.kaggle.com/datasets/devchauhan1/salary-datacsv)

## 📁 Project Structure

```
salary-prediction-app/
│
├── app.py                      # Streamlit app
├── Salary Data.csv             # Input dataset used
├── Salary_Prediction.ipynb     # EDA + model training notebook
├── requirements.txt            # Required packages
├── .gitignore                  # Files to ignore during git push
└── README.md                   # Project documentation
```

