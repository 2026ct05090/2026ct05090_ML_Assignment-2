# Thyroid Disease Classification: ML Assignment 2

**Name:** Dhruv
**BITS ID:** 2026CT05090
**Programme:** M.Tech (AIML), Machine Learning, Assignment 2

---

## 1. Problem Statement

The goal is to predict whether a patient has a thyroid condition or not. The inputs are age, sex, medical history, medicines and blood test values (TSH, T3, TT4, T4U, FTI).

This is a binary classification problem. Five classification models are trained on the same dataset and compared using six metrics: Accuracy, AUC, Precision, Recall, F1 and MCC.

---

## 2. Dataset Description

**Source:** [Thyroid Disease Data, Kaggle](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data/data) (file used: `thyroidDF.csv`)

**Why I selected this dataset:**
It is real medical data and it meets all the conditions of the assignment. It has 9,172 rows (minimum needed is 1,000) and 29 input features (minimum needed is 12). It has a lot of missing values that need handling, and the classes are imbalanced, so the class distribution has to be examined carefully. It also has numeric, categorical and yes/no features, so preprocessing is required.

**Target variable:** `target`
The original column has 32 diagnosis codes (`-` means no thyroid problem, the others are different conditions). Most codes have very few rows, so they cannot be modelled one by one. I converted it into a binary target:

- `0` = Normal (`-`)
- `1` = Thyroid condition (any other code)

**Feature types (27 features used after cleaning):**

| Type | Count | Columns |
| --- | --- | --- |
| Numeric | 6 | age, TSH, T3, TT4, T4U, FTI |
| Categorical | 2 | sex, referral_source |
| Yes/No flags | 19 | on_thyroxine, pregnant, goitre, TSH_measured, etc. |

**Class distribution:**

| Stage | Normal (0) | Thyroid condition (1) | Thyroid % |
| --- | --- | --- | --- |
| Raw data (9,172 rows) | 6,771 | 2,401 | 26.18% |
| After cleaning (9,070 rows) | 6,690 | 2,380 | 26.24% |

The imbalance is about 2.8 : 1. Because of this, I used a stratified split and looked at Recall, F1 and MCC along with Accuracy.

### Data Preprocessing

| Step | What I did |
| --- | --- |
| Missing values | `TBG` is 96.19% missing, so I dropped it together with `TBG_measured`. `T3` (28.39%), `TSH` (9.18%), `T4U` (8.82%), `FTI` (8.74%), `TT4` (4.82%) and `age` were filled with the median. `sex` (3.35%) was filled with the most frequent value. |
| Wrong values | 4 impossible ages (455, 65511, 65512, 65526) were set to missing and then filled with the median. |
| Duplicates | 102 duplicate rows were removed (found after dropping `patient_id`). |
| Dropped columns | `patient_id` (just an identifier), `TBG`, `TBG_measured`. |
| Encoding | t/f columns changed to 1/0. `sex` and `referral_source` are one-hot encoded. |
| Scaling | `StandardScaler` on the numeric columns. |
| Train-test split | 80:20, stratified, random seed 42. Train = 7,256 rows (5,352 normal / 1,904 thyroid). Test = 1,814 rows (1,338 normal / 476 thyroid). |
| No data leakage | Imputer, scaler and encoder are inside a scikit-learn `Pipeline`, so they are fitted on training data only. |

**Assumption:** every code other than `-` (including `K`, "concurrent non-thyroidal illness") is counted as "Thyroid condition".

---

## 3. GitHub Repository Link

https://github.com/2026ct05090/2026ct05090_ML_Assignment-2

**Repository structure:**

```
2026ct05090_ML_Assignment-2/
├── README.md
├── requirements.txt
├── test_data.csv
└── model/
    ├── 2026CT05090_ml_assignment.ipynb
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── (graph images used in this README)
```

`test_data.csv` is the 20% test split (1,814 rows, 27 features + `target`) used in the experiments.

---

## 4. Models Used

All five models were trained on the same training data and tested on the same test data.

| Model | Configuration |
| --- | --- |
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Decision Tree | `max_depth=8`, `random_state=42` |
| kNN | `n_neighbors=7` |
| Naive Bayes | `GaussianNB`, default settings |
| Random Forest (Ensemble) | `n_estimators=200`, `random_state=42` |

These settings were chosen by me as reasonable values and were not tuned. Precision, Recall and F1 are calculated for the Thyroid condition class. AUC is calculated from predicted probabilities.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | 0.8264 | 0.8109 | 0.8207 | 0.4328 | 0.5667 | 0.5086 |
| **Decision Tree** | 0.9487 | 0.9469 | 0.9265 | 0.8739 | 0.8995 | 0.8658 |
| **kNN** | 0.8517 | 0.8902 | 0.8127 | 0.5651 | 0.6667 | 0.5909 |
| **Naive Bayes** | 0.8104 | 0.8309 | 0.6886 | 0.5063 | 0.5835 | 0.4736 |
| **Random Forest (Ensemble)** | 0.9465 | 0.9883 | 0.8813 | 0.9202 | 0.9003 | 0.8642 |

### Graph Representation of the Metrics

**Bar chart of all metrics:**

![Metrics comparison](model/metrics_comparison.png)

**Heatmap of all metrics:**

![Metrics heatmap](model/metrics_heatmap.png)

**Confusion matrices:**

![Confusion matrices](model/confusion_matrices.png)

| Model | TN | FP | FN | TP |
| --- | --- | --- | --- | --- |
| Logistic Regression | 1293 | 45 | 270 | 206 |
| Decision Tree | 1305 | 33 | 60 | 416 |
| kNN | 1276 | 62 | 207 | 269 |
| Naive Bayes | 1229 | 109 | 235 | 241 |
| Random Forest (Ensemble) | 1279 | 59 | 38 | 438 |

**ROC curves:**

![ROC curves](model/roc_curves.png)

---

## 5. Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| **Logistic Regression** | Accuracy (0.8264) looks fine, but recall is only 0.4328. It missed 270 of 476 thyroid patients (56.7%). Precision (0.8207) is decent, so it is mostly right when it says "thyroid", but it says it too rarely. A straight-line boundary probably cannot separate the hormone patterns well. |
| **Decision Tree** | Best accuracy (0.9487), precision (0.9265) and MCC (0.8658). It missed 60 patients (12.6%) with only 33 false alarms. Its AUC (0.9469) is lower than Random Forest's because a single tree gives rougher probabilities. |
| **kNN** | Accuracy is 0.8517 but recall is only 0.5651 (missed 207 patients, 43.5%). Its AUC (0.8902) is better than its accuracy suggests, so it ranks patients reasonably but the default cutoff misses many. The many yes/no and one-hot columns may affect the distances. |
| **Naive Bayes** | Lowest accuracy (0.8104), MCC (0.4736) and precision (0.6886), and the most false alarms (109). It assumes features are independent, but TT4, T4U and FTI are related to each other. |
| **Random Forest (Ensemble)** | Best AUC (0.9883), recall (0.9202) and F1 (0.9003). It missed only 38 of 476 patients (8.0%). It had more false alarms than the Decision Tree (59 vs 33), so its precision (0.8813) is lower. Combining many trees reduces the errors of a single tree. |
| **Overall Winner for your dataset?** | **Random Forest.** It has the best AUC, recall and F1, and the fewest missed patients. The Decision Tree is very close (accuracy higher by 0.0022, MCC higher by 0.0016, F1 lower by 0.0008). In medical screening, missing a sick patient is worse than a false alarm, so recall decides it. |

**Limitation:** some features (for example `on_thyroxine` and `thyroid_surgery`) are closely linked to the diagnosis, so the scores may be higher than in a real screening situation.

---

## 6. How to Run

1. Open `2026CT05090_ml_assignment.ipynb` in Google Colab or Jupyter Notebook.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Run all cells from top to bottom. The dataset is downloaded from Kaggle using `kagglehub`, so internet access is needed.
4. At the end, the notebook prints the results table, creates the graphs, saves `test_data.csv` and saves the trained models in the `model/` folder.

The random seed is fixed (`SEED = 42`), so results are the same on every run in the same environment.
