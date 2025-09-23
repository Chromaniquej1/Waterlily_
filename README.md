# Diabetes Readmission Prediction

This project uses the **UCI Diabetes 130-US hospitals dataset (1999–2008)** to predict whether a patient will be readmitted to the hospital within 30 days of discharge.  

The goal is to provide **actionable insights** that help healthcare providers reduce avoidable readmissions, improve patient outcomes, and optimize resource allocation.  

## Repo Structure
- **`notebooks/`** → main training pipeline (EDA → cleaning → preprocessing → modeling → save best model).    
- **`app.py`** → lightweight API for serving predictions.  
- **`models/`** → serialized model artifacts for reuse.  
- **`data/`** → cleaned dataset persisted in SQLite for reproducibility. 
---

## 📊 EDA Findings & Assumptions

### Dataset Overview
- **Size:** 101,766 encounters × 50 columns  
- **Unit of analysis:** Each row = one hospital encounter for a patient with diabetes  
- **Target variable:** `readmitted`  
  - NO → 53.9%  
  - >30 days → 34.9%  
  - <30 days → 11.2% (**positive class**)  

### Key Observations
- **Imbalance:** Only ~11% of cases are 30-day readmissions → metrics like ROC-AUC, PR-AUC, precision, and recall are more meaningful than accuracy.  
- **Missingness:**  
  - `weight` (97%), `max_glu_serum` (95%), and `A1Cresult` (83%) → mostly unusable, may be dropped or treated as “not measured.”  
  - `medical_specialty` (~49%) and `payer_code` (~40%) → partially useful, need imputation or grouping.  
- **High cardinality:** Diagnosis codes (`diag_1–3`) have 700+ unique values each. → Must group into broader Categories.  
- **Patient-level leakage risk:** Patients (`patient_nbr`) appear multiple times → must **split data grouped by patient** to avoid inflating results.  
- **Usable features at discharge:** demographics, admission/discharge info, length of stay, utilization counts, lab test results, and medication use.

### Assumptions
- Only features available at or before discharge are used for prediction.  
- Sparse fields with >90% missingness are excluded.  
- Grouped train/test splits ensure fair evaluation.  

---

## 🤖 Core Modeling Approach

### Preprocessing
- **Imputation:**  
  - Numeric → median  
  - Categorical → most frequent value  
- **Encoding:**  
  - One-hot encoding for categorical variables  
  - Handle unknown categories gracefully at inference  
- **Pipeline:** Preprocessing + model wrapped in a scikit-learn `Pipeline` for reproducibility.  

### Models Compared
- **Logistic Regression** → simple, interpretable baseline.  
- **Random Forest** → robust, non-linear model for tabular data.  
- **Gradient Boosting** → sequential tree boosting, often top performer.  

### Evaluation
- Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1.  
- PR Curves: Visualized to show precision–recall trade-offs.  
- **Best model:** Chosen based on ROC-AUC (primary metric under imbalance).  

### Deployment
- The best pipeline is **saved as a `.joblib` artifact**.  
- Exposed via a **FastAPI service** with:  
  - `GET /health` → returns status  
  - `POST /predict` → accepts JSON records, returns predictions + probabilities  

---

## 💡 Additional Questions & Why They Matter


### 🏥 Business Impact & Extensions

### 1. "Netflix for Patient Care" – Personalized Treatment Recommendations
- **What it does:**  
  Just like Netflix groups users (“people who like action movies”), this system groups patients and suggests personalized care plans.  
- **Examples:**  
  - Elderly patients with simple diabetes → **standard medication plan**  
  - Younger patients with complications → **intensive monitoring plan**  
  - Frequent hospital visitors with multiple conditions → **specialized care team**  
- **Why it matters:**  
  - Better care: treatment isn’t one-size-fits-all.  
  - Cost savings: right intensity of care for each patient.  
  - Doctor support: learns from what worked for similar patients.

---

### 2. "Medication Safety Guard" – Prescription Spellchecker
- **What it does:**  
  Acts like a smart pharmacist. Diabetic patients often take 10+ medications; this system checks for dangerous drug combinations.  
- **Example:**  
  - Patient already on 8 medications  
  - Doctor wants to add a 9th  
  - System warns: *“This combo increases readmission risk by 40%.”*  
  - Suggests safer alternatives that work equally well.  
- **Why it matters:**  
  - Prevents dangerous drug reactions (patient safety)  
  - Avoids waste from bad medication combos (cost savings)  
  - Simplifies care by reducing unnecessary meds (better outcomes)

---

## 🖥 How to Run the Service Locally

### 1. Install dependencies
```bash
conda create -n diabetes python=3.10 -y
conda activate diabetes
pip install -r requirements.txt

uvicorn app:app --reload --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health

Visit http://127.0.0.1:8000/docs in your browser, expand POST /predict, and try a sample JSON input.
