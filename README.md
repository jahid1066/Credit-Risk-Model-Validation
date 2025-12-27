# Credit Risk Modeling: Model Validation & Comparison

## Project Overview
This project focuses on **credit risk modeling** using supervised machine learning techniques to predict **loan default probability**.  
The main objective is to **compare multiple classification models** using rigorous validation strategies and select the most reliable model based on statistical and business-relevant performance metrics.

The project emphasizes:
- Proper **data preprocessing**
- Robust **model validation**
- **Comparative evaluation** using ROC-AUC, Precision, Recall, and F1-score
- Clear, reproducible, and interpretable results

---


---

## Dataset Description
The dataset contains anonymized borrower information and loan characteristics used for default prediction.

**Target variable:**
- `loan_status`  
  - `1` → Default  
  - `0` → Non-default  

**Key feature groups:**
- Demographics: age, income, employment length  
- Financial profile: loan amount, interest rate, credit history  
- Behavioral risk indicators  

---

## Exploratory Data Analysis
Key findings from EDA:
- Moderate **class imbalance** between default and non-default loans
- Strong correlation between **loan amount**, **income**, and **default risk**
- Presence of **outliers** in employment length and income variables

---

## Models Trained
The following models were trained and evaluated using consistent preprocessing and validation pipelines:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- XGBoost

---

## Cross-Validation Results (ROC-AUC)

| Model               | ROC-AUC |
|--------------------|---------|
| **XGBoost**        | **0.9465** |
| Random Forest      | 0.9339 |
| Gradient Boosting  | 0.9254 |
| SVM                | 0.8981 |
| Logistic Regression| 0.8712 |

> **XGBoost achieved the highest cross-validated ROC-AUC**, indicating superior ranking ability for default risk.

---

## Final Test Set Performance

### Logistic Regression
- ROC-AUC: **0.8614**
- Accuracy: 0.8603
- Precision: 0.7537
- Recall: 0.5278
- F1-score: 0.6209

### Random Forest
- ROC-AUC: **0.9280**
- Accuracy: 0.9282
- Precision: 0.9570
- Recall: 0.7002
- F1-score: 0.8087

### Gradient Boosting
- ROC-AUC: **0.9232**
- Accuracy: 0.9190
- Precision: 0.9218
- Recall: 0.6841
- F1-score: 0.7854

### Support Vector Machine (SVM)
- ROC-AUC: **0.8893**
- Accuracy: 0.9064
- Precision: 0.9273
- Recall: 0.6164
- F1-score: 0.7406

### XGBoost
- ROC-AUC: **0.9435**
- Accuracy: 0.9305
- Precision: 0.9432
- Recall: 0.7228
- F1-score: 0.8184

---

## Model Comparison Summary

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|------|--------|----------|-----------|--------|----|
| **XGBoost** | **0.9435** | **0.9305** | 0.9432 | **0.7228** | **0.8184** |
| Random Forest | 0.9280 | 0.9282 | **0.9570** | 0.7002 | 0.8087 |
| Gradient Boosting | 0.9232 | 0.9190 | 0.9218 | 0.6841 | 0.7854 |
| SVM | 0.8893 | 0.9064 | 0.9273 | 0.6164 | 0.7406 |
| Logistic Regression | 0.8614 | 0.8603 | 0.7537 | 0.5278 | 0.6209 |

---

## Key Conclusions
- **XGBoost** provides the best overall performance and generalization
- **Random Forest** offers strong precision, suitable when false positives are costly
- **Logistic Regression** remains valuable for interpretability and regulatory contexts
- Proper **model validation significantly changes perceived model performance**

---

## Skills Demonstrated
- Statistical modeling & hypothesis-driven validation
- Credit risk & classification modeling
- Cross-validation & metric-based comparison
- Python, scikit-learn, XGBoost
- Reproducible ML pipelines
- Business-oriented model evaluation

---


## Getting Started

### 1️. Clone the repository
git clone https://github.com/jahid1066/Credit-Risk-Model-Validation.git
cd Credit-Risk-Model-Validation

## 2. Install dependencies
- pip install -r requirements.txt

## 3. Run Notebooks
jupyter notebook

- exploratory_data_analysis.ipynb

- model_training.ipynb

- model_validation_comparison.ipynb


---

## Future Improvements
- SHAP-based explainability
- Cost-sensitive learning
- Probability calibration
- Regulatory-compliant reporting (Basel / IFRS-style)

---




## Author
**Md Jahidul Islam**  


---

## License
This project is for academic, educational, and portfolio purposes.


