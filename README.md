# Task 1: Data Cleaning & Preprocessing
**Intern:** Leena Chekuri  
**Organization:** Synent Technologies – Data Science Internship  
**Repository:** `synent-task1-datacleaning-leenachekuri`

---

## 📌 Problem Statement
Raw datasets are often messy — filled with missing values, wrong data types, duplicate rows, and inconsistent column names. This project cleans and prepares the Titanic dataset so it is fully ready for analysis and modeling.

---

## 📊 Dataset Details
- **Name:** Titanic Dataset
- **Source:** [Kaggle – Titanic Dataset (Leena_task1)](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **File:** `Titanic-Dataset.csv`
- **Rows:** 891 | **Columns:** 12
- **Key Columns:** PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

---

## 🔍 Approach

### 1. Data Exploration
- Loaded raw dataset and inspected shape, dtypes, and statistics.

### 2. Missing Value Handling
| Column | Strategy | Reason |
|---|---|---|
| `Age` | Filled with **median** | Skewed distribution, median is robust |
| `Embarked` | Filled with **mode** | Categorical, most frequent port used |
| `Cabin` | Replaced with `'Unknown'` | ~77% missing, too high to impute |
| `Deck` | Extracted from `Cabin` | First letter of cabin, or `'Unknown'` |

### 3. Duplicate Removal
- Checked and removed exact duplicate rows.

### 4. Data Type Conversion
- `Survived` → `bool`
- `Pclass`, `Sex`, `Embarked`, `Deck` → `category`
- `Age` → rounded float
- `Fare` → rounded to 2 decimal places

### 5. Column Renaming
- All columns renamed to readable, professional names (e.g., `sibsp` → `SiblingsSpouses`)

---

## 📈 Results & Key Insights
- **891 rows** cleaned with **0 missing values** remaining
- **38.4%** of passengers survived
- **Females had much higher survival rate** than males
- **1st Class passengers** survived at a higher rate than 3rd Class
- **Age median** = 28 years (used to fill 177 missing age values)
- Output saved to `outputs/titanic_clean.csv`

---

## 🛠️ Tools & Libraries
- Python 3.x
- Pandas, NumPy
- Matplotlib, Seaborn

---

## ▶️ How to Run

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the script
python task1_data_cleaning.py
```

**Outputs Generated:**
- `outputs/titanic_clean.csv` – Clean, analysis-ready dataset
- `outputs/task1_visualizations.png` – 9-panel visualization chart

---

## 📁 Project Structure
```
synent-task1-datacleaning-leenachekuri/
│
├── task1_data_cleaning.py    # Main Python script
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── outputs/
    ├── titanic_clean.csv      # Cleaned dataset
    └── task1_visualizations.png  # Charts
```
