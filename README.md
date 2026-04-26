# Income Prediction Project

## 📌 Overview
This project is a Machine Learning model that predicts whether an individual's income is above or below a certain threshold using the Adult Income dataset.  
The model is built using Python and Scikit-learn.

---

## 📊 Dataset
The dataset used is the **Adult Income Dataset (Census Income Dataset)**.  
It contains demographic and financial information such as:
- Age
- Education
- Occupation
- Work class
- Hours per week
- Income level (target variable)

---

## 🤖 Model Used
This project uses the **Random Forest Classifier** algorithm.

Why Random Forest?
- High accuracy for classification tasks
- Handles both numerical and categorical data
- Reduces overfitting by combining multiple decision trees

---

## ⚙️ Workflow
1. Load dataset (adult.csv)
2. Clean missing values
3. Encode categorical variables
4. Split data into training and testing sets
5. Train Random Forest model
6. Make predictions
7. Evaluate accuracy
8. Save trained model (model.pkl)

---

## 📈 Model Performance
- Accuracy: 86%

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib

---

## 🚀 How to Run
```bash
python main.py
