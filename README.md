# IBM-CLOUD-ML-PROJECT-DEPLOYMENT
# PMGSY Scheme Type Classifier (Streamlit App)

This Streamlit web app predicts the type of **Pradhan Mantri Gram Sadak Yojana (PMGSY)** scheme for rural road development based on user input such as location, project cost, sanctioned length, completed work, and more.

---

## 🚀 Project Objective

To assist policymakers, planners, and analysts by predicting the **PMGSY scheme type** using historical data and a trained **Random Forest Classifier**. The app provides easy-to-use dropdowns for selecting `STATE_NAME` and `DISTRICT_NAME`, and fields for entering project metrics.

---

## 📊 Dataset

The dataset used (`PMGSY_DATASET.csv`) includes the following features:

- **STATE_NAME** *(categorical dropdown)*
- **DISTRICT_NAME** *(categorical dropdown)*
- **NO_OF_ROAD_WORK_SANCTIONED**
- **LENGTH_OF_ROAD_WORK_SANCTIONED**
- **NO_OF_BRIDGES_SANCTIONED**
- **COST_OF_WORKS_SANCTIONED**
- **NO_OF_ROAD_WORKS_COMPLETED**
- **LENGTH_OF_ROAD_WORK_COMPLETED**
- **NO_OF_BRIDGES_COMPLETED**
- **EXPENDITURE_OCCURED**
- **NO_OF_ROAD_WORKS_BALANCE**
- **LENGTH_OF_ROAD_WORK_BALANCE**
- **NO_OF_BRIDGES_BALANCE**

The target variable is: **PMGSY_SCHEME**

---

## 🧠 Model Training

- A **Random Forest Classifier** was trained using `scikit-learn`.
- `LabelEncoder` was applied **separately to each categorical column**, and saved.
- All numeric features were scaled using `MinMaxScaler`.
- Trained objects saved using `joblib`:
  - `random_forest_model.pkl`
  - `label_encoders.pkl` (dictionary of encoders)
  - `minmax_scaler.pkl`

---

## 💻 Streamlit App Features

- Dropdowns for `STATE_NAME` and `DISTRICT_NAME` (label-encoded)
- Numeric inputs support **full decimal precision**
- Automatically scales and encodes inputs before prediction
- Predicts and **displays the human-readable scheme type**
- Includes a visual of rural road construction under PMGSY  
  ![PMGSY Road](https://cdnbbsr.s3waas.gov.in/s3e6c2dc3dee4a51dcec3a876aa2339a78/uploads/2025/01/20250129686928869.jpg)

git clone <your-repo-url>
cd pmgsy-scheme-classifier
