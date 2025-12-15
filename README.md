# ✨ Variable Star Classifier (Machine Learning)

A machine learning classifier for identifying **different classes of variable stars** using features extracted from light curves.

This project demonstrates:
- Time-series feature engineering  
- Astroinformatics  
- ML classification  
- Data visualization  
- Clean scientific project structure  

---



## ⭐ Goal

To build a robust ML model that classifies stars into categories such as:

- RR Lyrae  
- Cepheids  
- Delta Scuti  
- Eclipsing Binaries  
- Rotational Variables  

Using:
- Summary statistics  
- Period-finding algorithms  
- Fourier features  
- Light curve shape descriptors  

---

## 📊 Machine Learning Approach

### 1️⃣ Load Light Curves  
TESS/Kepler FITS or simulated data.

### 2️⃣ Extract Features  
Planned features include:
- Period (Lomb-Scargle)  
- Amplitude  
- Standard deviation  
- Flux percentiles  
- Skewness  
- Peak-to-peak variation  
- Fourier harmonics  

### 3️⃣ Train Classification Models
Models considered:
- Random Forest  
- XGBoost  
- Logistic Regression  
- LightGBM  

### 4️⃣ Evaluate Model
- Confusion matrix  
- Accuracy  
- F1 score  
- Class-wise performance  

### 5️⃣ Interpretability (Optional)
- SHAP values  
- Feature importance  

---

## 🚀 Usage (Planned)

### Extract features:

### Train classifier:

### Run notebook:
Open:

---

## 📦 Sample Data (Will Provide)
- Simulated variable star light curves  
- Labeled dataset  
- TESS/Kepler-style curves  

---

## 🎯 Purpose
Built to strengthen astrophysics + ML research skills for PhD applications.

---
---

## 📸 Example Outputs

### 🔹 Light Curve Grid
This figure shows the 5 types of variable star light curves used in the dataset:
- RR Lyrae
- Cepheid
- Delta Scuti
- Eclipsing Binary
- Rotational Variable

![Light Curve Grid](docs/lc_grid.png)


### 🔹 Confusion Matrix
This shows the performance of the Random Forest classifier after training on extracted features.

![Confusion Matrix](docs/confusion_matrix.png)

## ✨ Author
Grace (spacey-g)
