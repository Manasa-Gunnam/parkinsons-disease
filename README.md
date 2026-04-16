# 🧠 Parkinson's Disease Prediction System

## 📌 Overview
This project is a Machine Learning-based web application that predicts whether a person is affected by Parkinson’s disease using biomedical voice measurements.

The system uses multiple machine learning models and selects the best-performing model based on evaluation metrics.

---

## 🎯 Objectives
- Build a reliable prediction system for Parkinson’s disease  
- Compare multiple machine learning models  
- Achieve high accuracy using optimized algorithms  
- Deploy an interactive web application  

---

## 🧪 Dataset
- Source: UCI ML Repository / Kaggle  
- Features: 22 biomedical voice measurements  
- Target:
  - `1` → Parkinson’s Disease  
  - `0` → Healthy  

---

## ⚙️ Tech Stack
- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Google Colab  
- GitHub  

---

## 🧠 Models Used
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- XGBoost Classifier  

All models were tuned using **GridSearchCV** for optimal performance.

---

## 📊 Model Performance

### 🔹 SVM
- Accuracy: **89.74%**

### 🔹 Random Forest
- Accuracy: **92.31%**

### 🔹 XGBoost
- Accuracy: **92.31%**

---

## 🏆 Best Model Selected
**Random Forest Classifier**  
- Accuracy: **92.31%**

Reason:
- Balanced precision and recall  
- Better performance on both classes  
- More stable predictions  

---

## 📈 Evaluation Metrics

### 🔹 Classification Report Summary
- High precision and recall for Parkinson’s class  
- Good balance between false positives and false negatives  

---

## 📊 Visualizations

### 🔹 ROC Curve
![ROC Curve](<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/ed5c7fd3-03ab-47db-b7d6-048e23d942d5" />
)

### 🔹 Precision-Recall Curve
![Precision Recall](<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/5f5d9724-32c9-4545-acca-73b8c2ae7ae9" />
)

---

## 🖥️ Application Features
- User-friendly Streamlit interface  
- Real-time prediction  
- Multiple model comparison  
- Graph-based analysis  
- Input validation  

---

## 📁 Project Structure
