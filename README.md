# 🧠 Parkinson's Disease Prediction System

## 📌 Overview
The Parkinson's Disease Prediction System is a Machine Learning-based web application that predicts whether a person is affected by Parkinson’s disease using biomedical voice measurements.

This project implements and compares multiple machine learning models to achieve better prediction accuracy and reliability.

---

## 🎯 Objectives
- To develop a predictive system for early detection of Parkinson’s disease  
- To compare multiple machine learning algorithms  
- To improve prediction accuracy using ensemble techniques  
- To build and deploy a user-friendly web application  

---

## 🧪 Dataset Information
- Source: UCI Machine Learning Repository / Kaggle  
- Total Features: 22 biomedical voice measurements  
- Target Variable:
  - `1` → Parkinson’s Disease  
  - `0` → Healthy  

### 🔍 Feature Categories:
- Frequency-based features (Fo, Fhi, Flo)  
- Jitter and Shimmer (voice variations)  
- Noise ratios (NHR, HNR)  
- Nonlinear measures (RPDE, DFA, PPE)  

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, XGBoost  
- **Web Framework:** Streamlit  
- **Development Platform:** Google Colab  
- **Version Control & Deployment:** GitHub  

---

## 🧠 Machine Learning Models Used

### 1. Support Vector Machine (SVM)
- Effective in high-dimensional spaces  
- Works well with smaller datasets  

### 2. Random Forest Classifier
- Ensemble learning method  
- Reduces overfitting  
- Handles non-linear relationships  

### 3. XGBoost Classifier
- Gradient boosting algorithm  
- High performance and accuracy  
- Efficient handling of complex patterns  

---

## 🔄 Machine Learning Workflow
1. Data Collection  
2. Data Preprocessing  
   - Feature scaling (StandardScaler)  
3. Train-Test Split (80:20)  
4. Model Training (SVM, Random Forest, XGBoost)  
5. Model Evaluation & Comparison  
6. Best Model Selection  
7. Model Serialization using Pickle  

---

## 📊 Model Performance Comparison

| Model            | Accuracy (Approx) |
|------------------|------------------|
| SVM              | ~85%             |
| Random Forest    | ~88%             |
| XGBoost          | ~90%             |

> ✅ XGBoost achieved the best performance and was selected for deployment.

---

## 🖥️ Application Features
- Interactive UI using Streamlit  
- Real-time disease prediction  
- Input validation and error handling  
- Fast and responsive performance  

---

## 📁 Project Structure
