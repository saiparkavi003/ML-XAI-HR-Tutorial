# Mastering Transparency: Explainable AI (XAI) for HR Analytics

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📌 Project Overview
This repository contains a professional tutorial on **Explainable AI (XAI)**. While most Machine Learning models are "Black Boxes," this project demonstrates how to use **SHAP (SHapley Additive exPlanations)** to interpret a Random Forest model.

We apply this technique to the **IBM HR Analytics Employee Attrition** dataset to identify the global and local drivers behind why employees leave a company.

## 🚀 Key Features
- **Technical Depth:** Implementation of Tree-based Ensembles (Random Forest) with a focus on Coalitional Game Theory (SHAP).
- **Professional Preprocessing:** Includes Label Encoding, One-Hot Encoding, and Handling of Imbalanced Classes.
- **Visual Teaching:** Use of **Beeswarm Plots** (Global drivers) and **Waterfall Plots** (Local case studies).
- **Accessibility:** High-contrast visualizations and screen-reader-friendly documentation.

## 🛠️ Installation & Setup
To reproduce the results of this tutorial, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saiparkavi003/ML-XAI-HR-Tutorial.git
   cd ML-XAI-HR-Tutorial
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Source:**
   Place the [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) CSV file inside a folder named `data/` in the root directory.

## 📊 Visual Results
The tutorial generates two primary visualizations:
- **Beeswarm Plot:** Shows the top 15 features (like Overtime and Income) and their impact on attrition.
- **Waterfall Plot:** Explains the specific logic behind a single employee's prediction.

### 📚 References

- **IBM HR Analytics:** Kaggle. (2017). *IBM HR Analytics Employee Attrition & Performance*. [Kaggle Dataset Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)