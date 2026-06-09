# 🏦 Loan Risk Prediction & Fraud Detection

An end-to-end Machine Learning project to predict loan default risks (`Risk_Flag`). This project evaluates multiple classification models, addresses severe class imbalance, and optimizes business-critical metrics like **Recall** and **Precision** to prevent banking losses.

---

## 📊 Dataset Overview
The dataset contains information about loan applicants, including demographics, income, profession, and historical data. 
* **Target Variable:** `Risk_Flag` (0 = Non-Defaulter/Safe, 1 = Defaulter/Risk)
* **Class Imbalance:** The dataset is highly imbalanced with **44,147 Safe Users (Class 0)** and only **6,253 Defaulters (Class 1)**.

---

## 🛠️ Data Preprocessing & Engineering
To prepare the data for Machine Learning models, the following steps were performed:
1. **Feature Engineering:** Extracted top 15 categories for high-cardinality features like `Profession` and grouped the remaining into 'Other'.
2. **Categorical Encoding:** Converted categorical variables (`Married/Single`, `Car_Ownership`) using `LabelEncoder` and `OneHotEncoder` (Pandas `get_dummies`) for `STATE`.
3. **Feature Scaling:** Applied `StandardScaler` on continuous features (`Income`, `Age`, `Experience`, `CURRENT_JOB_YRS`, `CURRENT_HOUSE_YRS`).
4. **Data Slicing:** Dropped unhelpful columns like `Id` and `CITY`, then split the data into 80% Training and 20% Testing sets.

---

## ⚙️ Model Evaluation Journey & Strategies

Because the data was heavily imbalanced, standard validation could crash the memory or yield misleading results. We adapted our evaluation strategies accordingly:

### 1. Baseline Model: Logistic Regression
* Started with a simple baseline using `class_weight='balanced'`.

### 2. K-Nearest Neighbors (KNN) - Batch Evaluation
* **Challenge:** KNN is lazy and memory-heavy. Predicting all 50,400 test samples at once caused potential session crashes.
* **Solution:** Implemented a custom **Batch Processing Loop (1,000 samples per chunk)** to calculate batch accuracies and compute a realistic final average accuracy without overloading the RAM.
* **Result:** Achieved ~88.8% accuracy, but Class 1 Recall was low (45%).

### 3. Advanced Evaluation with `cross_val_predict`
* Moved away from manual loops to an industry-standard pipeline using `cross_val_predict` from `sklearn` with `cv=5` and `n_jobs=-1` to multi-thread the evaluation safely.

---

## 📈 Model Performance Comparison

Here is how different models performed on the minority class (**Class 1 - Defaulters**):

| Model | Overall Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **KNN Classifier** | 88% | 0.53 | 0.45 | 0.49 |
| **Decision Tree** `(balanced)` | 88% | 0.51 | **0.78** | 0.61 |
| **Random Forest** `(balanced)` | **90%** 🏆 | **0.56** 🏆 | 0.77 | **0.64** 🏆 |
| **XGBoost Classifier** `(weighted)` | 83% | 0.40 | **0.80** 🚀 | 0.53 |

---

## 🧠 Core Insights & Trade-Offs (The "Why")

### Why is Precision lower for Class 1?
Since the dataset is heavily skewed toward Safe Users (Class 0), even a tiny misclassification percentage of Safe Users creates a large number of **False Positives** (False Alarms). By mathematically forcing models to prioritize Defaulters (using `class_weight='balanced'` or `scale_pos_weight=7`), the models aggressively tag suspicious profiles as "Risk". This maximizes **Recall** but naturally drops **Precision**.

### Business Verdict:
1. **The Risk-Averse Choice (XGBoost):** Captures the highest number of defaulters (**80% Recall**). Best if the bank's absolute priority is avoiding financial loss from bad loans.
2. **The Balanced Champion (Random Forest):** Achieves the best overall balance (**90% Accuracy, 0.64 F1-Score, 77% Recall**). It prevents high financial damage while minimizing false alarms, ensuring good customers aren't unnecessarily rejected.

---

## 🚀 How to Run the Project

1. Clone this repository:
```bash
   git clone [https://github.com/KoushalVaswani/Loan-Risk-Prediction-And-Fraud-Detection.git](https://github.com/KoushalVaswani/Loan-Risk-Prediction-And-Fraud-Detection.git)
