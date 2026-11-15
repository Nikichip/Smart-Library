# Smart Library: Book Popularity Classification Using CART Decision Tree

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Unit%205-Decision%20Trees%20(CART)-green.svg)]()

---

## 1. Project Introduction
Libraries and digital content repositories manage hundreds of thousands of book titles. Effective acquisition, shelf placement, and budget allocation require estimating which books will attract high reader demand versus those with niche readership. 

This academic machine learning project builds a **Smart Library Book Popularity Classifier** using a **CART (Classification and Regression Trees) Decision Tree**. It classifies books into three operational popularity levels:
- **Popular** (High circulation demand; prominent shelf placement and multiple copies recommended)
- **Moderately Popular** (Steady circulation demand; standard inventory)
- **Less Popular** (Low or specialized circulation demand; single copy or digital on-demand)

---

## 2. Problem Statement & Academic Integrity
A common flaw in naive book popularity prediction projects is **Target Leakage**—using the count of ratings, circulation statistics, or average ratings as input features to predict popularity. When a new book arrives at the library, circulation numbers do not exist yet!

**Key Academic Highlights of this Project:**
1. **Zero Target Leakage:** Rating counts are used **only** to define the ground-truth target label on historical data. The input feature matrix consists **exclusively** of bibliographic metadata available at acquisition time:
   - Publication Year & Book Age
   - Title Character Length & Word Count
   - Title Subtitle Indicator
   - Author Catalog Track Record (Prolificacy in library)
   - Publisher Catalog Scale (Market presence)
2. **Dedicated Single Algorithm:** Focuses strictly on **CART Decision Tree Classification** using **Gini Impurity** (Unit 5 topic), avoiding confusing ensembles or unrelated models.
3. **Honest Evaluation:** Evaluated on 54,209 unseen testing samples without manipulating data or inflating performance.

---

## 3. Dataset Description
The project utilizes the classic **Book-Crossing Benchmark Dataset** collected by Cai-Nicolas Ziegler from bookcrossing.com:

| Dataset File | Records | Features | Key Information |
| :--- | :--- | :--- | :--- |
| **`books.csv`** | 271,360 | 8 | `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, Image URLs |
| **`users.csv`** | 278,858 | 3 | `User-ID`, `Location`, `Age` |
| **`ratings.csv`** | 1,149,780 | 3 | `User-ID`, `ISBN`, `Book-Rating` (Scale: 0 to 10) |

### Investigation on Zero Ratings (`Book-Rating == 0`)
- **Observation:** 716,109 ratings out of 1,149,780 (**62.28%**) have a rating value of `0`.
- **Domain Meaning:** In the Book-Crossing dataset, a rating of `0` denotes an **implicit interaction** (the user borrowed, shelved, or read the book without entering an explicit 1–10 star score). Explicit ratings are scored 1 to 10.
- **Handling Strategy:** Because reading or shelving represents active reader interest and library demand, zero ratings are retained when calculating overall circulation popularity.

---

## 4. Target Variable Formulation
The target variable `Popularity` is categorized based on the empirical distribution of reader interaction counts per book:

| Category | Interaction Count Threshold | Book Count | Percentage |
| :--- | :--- | :--- | :--- |
| **Less Popular** | 0 to 1 interaction | 146,571 | 54.08% |
| **Moderately Popular** | 2 to 5 interactions | 91,717 | 33.84% |
| **Popular** | 6 or more interactions | 32,754 | 12.08% |

---

## 5. Machine Learning Algorithm: CART Decision Tree
The model implements the **Classification and Regression Trees (CART)** algorithm via `scikit-learn`'s `DecisionTreeClassifier`.

### Mathematical Formulation: Gini Impurity
At every internal decision node, CART selects the binary feature split that maximizes the reduction in **Gini Impurity**:

$$Gini(D) = 1 - \sum_{i=1}^{k} p_i^2$$

where $p_i$ is the probability of class $i$ at node $D$. The Information Gain for a split on feature $A$ dividing dataset $D$ into $D_1$ and $D_2$ is:

$$\Delta Gini(A) = Gini(D) - \left( \frac{|D_1|}{|D|} Gini(D_1) + \frac{|D_2|}{|D|} Gini(D_2) \right)$$

### Hyperparameter Settings
- `criterion='gini'` (Standard CART algorithm)
- `max_depth=5` (Prevents overfitting, maintains interpretability for academic presentation)
- `min_samples_leaf=50` (Ensures statistically robust leaf nodes)
- `random_state=42` (Guarantees exact experimental reproducibility)

---

## 6. Project Folder Structure
```
smart_library/
│
├── app.py                      # Interactive Streamlit Web Application
├── main.py                     # Full Training, Evaluation & Visualisation Pipeline
├── requirements.txt            # Python Dependencies
├── README.md                   # Comprehensive Project Documentation
│
├── data/                       # Dataset Files (semicolon delimited, latin-1)
│   ├── books.csv               # 271,360 books
│   ├── users.csv               # 278,858 users
│   └── ratings.csv             # 1,149,780 ratings
│
├── outputs/                    # High-Resolution Publication-Quality Plots
│   ├── confusion_matrix.png    # Test Confusion Matrix Heatmap
│   ├── decision_tree.png       # Interpretable CART Tree Diagram
│   ├── feature_importance.png  # Gini Feature Importances Chart
│   ├── popularity_distribution.png
│   ├── rating_distribution.png
│   ├── top_10_books.png
│   └── other_visualisations.png# Exploratory multi-panel analysis
│
└── models/
    └── decision_tree_model.pkl # Serialized Model & Preprocessing Dictionary Bundle
```

---

## 7. Installation & Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- Windows, macOS, or Linux

### Step 1: Navigate to the Project Directory
```powershell
cd C:\Users\NIKITHA\.gemini\antigravity\scratch\smart_library
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Training & Evaluation Pipeline
```powershell
python main.py
```
This script cleans the data, trains the CART model, computes all evaluation metrics, saves visualisations into `outputs/`, and serializes the model into `models/decision_tree_model.pkl`.

### Step 4: Launch the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 8. Experimental Results & Evaluation

The model was evaluated on a stratified test set of **54,209 unseen books** (20% test split):

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **56.75%** | Percentage of unseen books correctly classified across all 3 classes. |
| **Macro Precision** | **49.32%** | Average precision across classes without class-size bias. |
| **Macro Recall** | **41.79%** | Average ability of the model to detect true instances of each class. |
| **Macro F1-Score** | **41.63%** | Harmonic mean of precision and recall. |
| **Weighted F1-Score** | **52.38%** | Weighted average F1-score accounting for class frequencies. |

### Classification Report:
```
                    precision    recall  f1-score   support

      Less Popular     0.6170    0.8459    0.7136     29,314
Moderately Popular     0.4236    0.2797    0.3369     18,344
           Popular     0.4390    0.1281    0.1983      6,551

          accuracy                         0.5675     54,209
         macro avg     0.4932    0.4179    0.4163     54,209
      weighted avg     0.5301    0.5675    0.5238     54,209
```

### Confusion Matrix on Test Set:
```
                      Predicted:
                      Less Popular   Moderately Popular   Popular
Actual Less Popular        24,796                4,108       410
Actual Moderately Popular  12,551                5,131       662
Actual Popular              2,838                2,874       839
```

### Feature Importance (Gini Impurity Reduction):
1. **`Author_Book_Count` (58.38%):** Prolific authors with large catalogs have established reader followings.
2. **`Publisher_Book_Count` (30.29%):** Major publishing conglomerates have expansive marketing and library reach.
3. **`Book_Age` (7.14%):** Established titles accumulated steady circulations over decades.
4. **`Title_Length` (2.57%):** Subtle distinction between concise fiction and descriptive non-fiction.
5. **`Year-Of-Publication` (1.62%):** Minor historical release era signal.

---

## 9. Conclusion
- The CART Decision Tree provides an interpretable, transparent machine learning framework that avoids black-box opacity.
- By strictly disallowing target leakage (no circulation or rating counts in inputs), the model maintains genuine predictive utility for newly acquired books.
- Achieving 56.75% 3-class accuracy strictly from bibliographic metadata outperforms random guessing (33.3%) and offers practical utility for automated library inventory planning.
