"""
=============================================================
  Synent Technologies - Data Science Internship
  Task 1: Data Cleaning & Preprocessing
  Dataset: Titanic-Dataset.csv (Kaggle)
  Intern: Leena Chekuri
=============================================================
  Internship Workflow:
    Step 1 >> Data Cleaning
    Step 2 >> Exploratory Data Analysis
    Step 3 >> Visualization
    Step 4 >> Insights & Export
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATH FIX: Always locate dataset relative to THIS script file
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "Titanic-Dataset.csv")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  SYNENT INTERNSHIP PROCESS BANNER
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("       SYNENT TECHNOLOGIES - DATA SCIENCE INTERNSHIP")
print("=" * 65)
print("  Intern    : Leena Chekuri")
print("  Task      : Task 1 - Data Cleaning & Preprocessing")
print("  Dataset   : Titanic Dataset (Kaggle)")
print("  Tool      : Python (VS Code)")
print("=" * 65)
print()
print("  INTERNSHIP WORKFLOW:")
print("  [1] Data Cleaning        <-- YOU ARE HERE")
print("  [2] Exploratory Data Analysis")
print("  [3] Visualization")
print("  [4] Modeling (if applicable)")
print("=" * 65)
print()

# ═══════════════════════════════════════════════════════════════
#  PHASE 1: LOAD DATASET
# ═══════════════════════════════════════════════════════════════
print(">>> PHASE 1: LOADING DATASET")
print("-" * 65)
print(f"  Source : {DATASET_PATH}")

df_raw = pd.read_csv(DATASET_PATH)
df     = df_raw.copy()

print(f"  Shape  : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print()
print("  FIRST 5 ROWS (RAW DATA):")
print(df.head().to_string(index=False))

# ═══════════════════════════════════════════════════════════════
#  PHASE 2: MISSING VALUE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 2: MISSING VALUE ANALYSIS")
print("-" * 65)

missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df  = pd.DataFrame({
    "Column":        missing.index,
    "Missing Count": missing.values,
    "Missing %":     missing_pct.values
}).query("`Missing Count` > 0")

print(f"  Total missing values BEFORE cleaning: {df.isnull().sum().sum()}")
print()
print(f"  {'Column':<15} {'Missing Count':>15} {'Missing %':>12}")
print(f"  {'-'*15} {'-'*15} {'-'*12}")
for _, row in missing_df.iterrows():
    print(f"  {row['Column']:<15} {int(row['Missing Count']):>15} {row['Missing %']:>11.2f}%")

# ═══════════════════════════════════════════════════════════════
#  PHASE 3: HANDLE MISSING VALUES
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 3: HANDLING MISSING VALUES")
print("-" * 65)

# Age → median fill
age_missing = df["Age"].isnull().sum()
median_age  = round(df["Age"].median(), 1)
df["Age"]   = df["Age"].fillna(median_age)
print(f"  [FIXED] Age      -> {age_missing} missing values filled with MEDIAN = {median_age}")

# Embarked → mode fill
emb_missing   = df["Embarked"].isnull().sum()
mode_embarked = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
print(f"  [FIXED] Embarked -> {emb_missing} missing values filled with MODE = '{mode_embarked}'")

# Cabin → mark as Unknown + extract deck letter
cab_missing  = df["Cabin"].isnull().sum()
df["Cabin"]  = df["Cabin"].fillna("Unknown")
df["Deck"]   = df["Cabin"].apply(lambda x: x[0] if x != "Unknown" else "Unknown")
print(f"  [FIXED] Cabin    -> {cab_missing} missing values marked as 'Unknown'")
print(f"          Deck     -> Extracted first letter from Cabin (A/B/C.../Unknown)")

print()
print(f"  Total missing values AFTER cleaning : {df.isnull().sum().sum()} (All clean!)")

# ═══════════════════════════════════════════════════════════════
#  PHASE 4: REMOVE DUPLICATES
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 4: REMOVING DUPLICATES")
print("-" * 65)

before = len(df)
df.drop_duplicates(inplace=True)
after  = len(df)
print(f"  Rows before : {before}")
print(f"  Rows after  : {after}")
print(f"  Removed     : {before - after} duplicates")

# ═══════════════════════════════════════════════════════════════
#  PHASE 5: CONVERT DATA TYPES
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 5: CONVERTING DATA TYPES")
print("-" * 65)

df["Survived"] = df["Survived"].astype(bool)
print("  [OK] Survived  -> int  converted to BOOL")

df["Pclass"] = df["Pclass"].astype("category")
print("  [OK] Pclass    -> int  converted to CATEGORY")

df["Sex"] = df["Sex"].astype("category")
print("  [OK] Sex       -> obj  converted to CATEGORY")

df["Embarked"] = df["Embarked"].astype("category")
print("  [OK] Embarked  -> obj  converted to CATEGORY")

df["Deck"] = df["Deck"].astype("category")
print("  [OK] Deck      -> obj  converted to CATEGORY")

df["Age"]  = df["Age"].astype(float).round(1)
print("  [OK] Age       -> Rounded to 1 decimal place")

df["Fare"] = df["Fare"].round(2)
print("  [OK] Fare      -> Rounded to 2 decimal places")

# ═══════════════════════════════════════════════════════════════
#  PHASE 6: RENAME COLUMNS
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 6: RENAMING COLUMNS")
print("-" * 65)

rename_map = {
    "PassengerId": "PassengerID",
    "Pclass":      "PassengerClass",
    "Name":        "FullName",
    "Sex":         "Gender",
    "SibSp":       "SiblingsSpouses",
    "Parch":       "ParentsChildren",
    "Ticket":      "TicketNumber",
    "Embarked":    "EmbarkedPort",
}
df.rename(columns=rename_map, inplace=True)
print(f"  {'Old Name':<16} -> {'New Name'}")
print(f"  {'-'*16}    {'-'*20}")
for old, new in rename_map.items():
    print(f"  {old:<16} -> {new}")

# ═══════════════════════════════════════════════════════════════
#  PHASE 7: DERIVED FEATURES
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 7: ADDING DERIVED FEATURES")
print("-" * 65)

df["FamilySize"] = df["SiblingsSpouses"] + df["ParentsChildren"] + 1
print("  [NEW] FamilySize = SiblingsSpouses + ParentsChildren + 1")

df["IsAlone"] = (df["FamilySize"] == 1).astype(bool)
print("  [NEW] IsAlone    = True if FamilySize == 1")

def age_group(a):
    if a <= 12:   return "Child"
    elif a <= 17: return "Teenager"
    elif a <= 60: return "Adult"
    else:         return "Senior"

df["AgeGroup"] = df["Age"].apply(age_group).astype("category")
print("  [NEW] AgeGroup   = Child / Teenager / Adult / Senior")

# ═══════════════════════════════════════════════════════════════
#  PHASE 8: DATA STATISTICS AFTER CLEANING
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 8: CLEAN DATA STATISTICS")
print("-" * 65)

print(f"  Final Shape     : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Missing Values  : {df.isnull().sum().sum()}")
print()

num_cols = ["Age", "Fare", "FamilySize"]
print(f"  {'Column':<14} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10}")
print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for col in num_cols:
    print(f"  {col:<14} {df[col].min():>10.2f} {df[col].max():>10.2f}"
          f" {df[col].mean():>10.2f} {df[col].median():>10.2f}")

# Survival rate
surv_rate = df["Survived"].mean() * 100
print()
print(f"  Survival Rate      : {surv_rate:.1f}%")
print(f"  Survived           : {df['Survived'].sum()} passengers")
print(f"  Did Not Survive    : {(~df['Survived']).sum()} passengers")
print()

# Age insights
print(f"  Youngest Passenger : {df['Age'].min()} years")
print(f"  Oldest Passenger   : {df['Age'].max()} years")
print(f"  Highest Fare Paid  : GBP {df['Fare'].max():.2f}")
print(f"  Lowest Fare Paid   : GBP {df['Fare'].min():.2f}")
print(f"  Largest Family     : {int(df['FamilySize'].max())} members")

# ═══════════════════════════════════════════════════════════════
#  PHASE 9: VISUALIZATIONS (5 Key Charts)
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 9: GENERATING VISUALIZATIONS (5 Charts)")
print("-" * 65)

plt.style.use("seaborn-v0_8-whitegrid")
COLORS  = ["#6C63FF", "#FF6584", "#43B97F", "#FFB347", "#56CCF2"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
fig.suptitle(
    "Task 1: Titanic Data Cleaning - Key Visual Insights\n"
    "Intern: Leena Chekuri  |  Synent Technologies",
    fontsize=13, fontweight="bold", y=0.97
)
# Remove the 6th subplot (2x3 grid, only 5 charts needed)
fig.delaxes(axes[1, 2])

# ── Chart 1: Missing Values Before vs After ──────────────────
ax1  = axes[0, 0]
cols = ["Age (177)", "Cabin (687)", "Embarked (2)"]
before_vals = [177, 687, 2]
after_vals  = [0, 0, 0]
x = np.arange(len(cols))
w = 0.35
b1 = ax1.bar(x - w/2, before_vals, w, label="Before", color=COLORS[1], edgecolor="white")
b2 = ax1.bar(x + w/2, after_vals,  w, label="After",  color=COLORS[2], edgecolor="white")
ax1.set_title("Missing Values: Before vs After", fontweight="bold", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(cols, fontsize=9)
ax1.set_ylabel("Count")
ax1.legend()
for bar, val in zip(b1, before_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f"MAX: {val}", ha="center", fontsize=8, color=COLORS[1], fontweight="bold")
ax1.text(0.5, 0.5, "AFTER: 0 Missing", transform=ax1.transAxes,
         ha="center", va="center", fontsize=10, color=COLORS[2],
         fontweight="bold", alpha=0.4)
print("  [1/5] Missing Values Chart - DONE")

# ── Chart 2: Survival Distribution ───────────────────────────
ax2 = axes[0, 1]
surv_vals   = [df["Survived"].sum(), (~df["Survived"]).sum()]
surv_labels = [f"Survived\n({surv_vals[0]})", f"Did Not\nSurvive\n({surv_vals[1]})"]
wedges, texts, autotexts = ax2.pie(
    surv_vals, labels=surv_labels,
    colors=[COLORS[2], COLORS[1]],
    autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    textprops={"fontsize": 10}
)
for at in autotexts: at.set_fontweight("bold")
ax2.set_title("Survival Distribution", fontweight="bold", fontsize=12)
max_label = surv_labels[0] if surv_vals[0] > surv_vals[1] else surv_labels[1]
print(f"  [2/5] Survival Chart  - DONE  | HIGHEST: Did Not Survive ({surv_vals[1]})")

# ── Chart 3: Age Distribution ────────────────────────────────
ax3 = axes[0, 2]
ax3.hist(df["Age"], bins=28, color=COLORS[0], edgecolor="white", linewidth=0.5, alpha=0.85)
ax3.axvline(df["Age"].min(),    color="orange", linestyle="--", linewidth=1.5,
            label=f"Min Age: {df['Age'].min()}")
ax3.axvline(df["Age"].median(), color=COLORS[1], linestyle="--", linewidth=1.8,
            label=f"Median: {df['Age'].median()}")
ax3.axvline(df["Age"].max(),    color="green",  linestyle="--", linewidth=1.5,
            label=f"Max Age: {df['Age'].max()}")
ax3.set_title("Age Distribution (After Filling)", fontweight="bold", fontsize=12)
ax3.set_xlabel("Age (years)")
ax3.set_ylabel("Count")
ax3.legend(fontsize=9)
# Annotate highest bar
counts, edges = np.histogram(df["Age"], bins=28)
max_bin_idx   = counts.argmax()
ax3.annotate(f"HIGHEST\n{counts[max_bin_idx]} passengers",
             xy=((edges[max_bin_idx]+edges[max_bin_idx+1])/2, counts[max_bin_idx]),
             xytext=(55, counts[max_bin_idx] - 20),
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=8, color="black", fontweight="bold")
print(f"  [3/5] Age Distribution - DONE  | Min: {df['Age'].min()}, Max: {df['Age'].max()}")

# ── Chart 4: Survival by Gender ──────────────────────────────
ax4 = axes[1, 0]
gender_surv = df.groupby(["Gender", "Survived"]).size().unstack(fill_value=0)
x4 = np.arange(len(gender_surv.index))
bars_no  = ax4.bar(x4 - 0.2, gender_surv[False], 0.4, label="Did Not Survive",
                   color=COLORS[1], edgecolor="white")
bars_yes = ax4.bar(x4 + 0.2, gender_surv[True],  0.4, label="Survived",
                   color=COLORS[2], edgecolor="white")
ax4.set_title("Survival by Gender", fontweight="bold", fontsize=12)
ax4.set_xticks(x4)
ax4.set_xticklabels(gender_surv.index, fontsize=11)
ax4.set_ylabel("Count")
ax4.legend()
# Annotate max/min
for bars in [bars_no, bars_yes]:
    vals = [b.get_height() for b in bars]
    for bar in bars:
        h = bar.get_height()
        tag = "MAX" if h == max(vals) else "MIN"
        col = "red" if tag == "MAX" else "gray"
        ax4.text(bar.get_x() + bar.get_width()/2, h + 2,
                 f"{tag}\n{h}", ha="center", fontsize=8,
                 color=col, fontweight="bold")
print("  [4/5] Survival by Gender  - DONE")

# ── Chart 5: Fare Distribution ───────────────────────────────
ax5 = axes[1, 1]
fare_data = df[df["Fare"] < 200]["Fare"]
ax5.hist(fare_data, bins=35, color=COLORS[4], edgecolor="white", linewidth=0.5, alpha=0.9)
ax5.axvline(df["Fare"].min(),    color="green",  linestyle="--", linewidth=1.5,
            label=f"Min Fare: GBP {df['Fare'].min():.2f}")
ax5.axvline(df["Fare"].median(), color=COLORS[1], linestyle="--", linewidth=1.8,
            label=f"Median: GBP {df['Fare'].median():.2f}")
ax5.axvline(df["Fare"].max(),    color="red",    linestyle="--", linewidth=1.5,
            label=f"Max Fare: GBP {df['Fare'].max():.2f}")
ax5.set_title("Fare Distribution (Fare < GBP 200)", fontweight="bold", fontsize=12)
ax5.set_xlabel("Fare (GBP)")
ax5.set_ylabel("Count")
ax5.legend(fontsize=9)
# Annotate highest count bar
fare_counts, fare_edges = np.histogram(fare_data, bins=35)
fare_max_idx = fare_counts.argmax()
ax5.annotate(f"HIGHEST\n{fare_counts[fare_max_idx]} passengers",
             xy=((fare_edges[fare_max_idx]+fare_edges[fare_max_idx+1])/2,
                 fare_counts[fare_max_idx]),
             xytext=(60, fare_counts[fare_max_idx] - 30),
             arrowprops=dict(arrowstyle="->", color="black"),
             fontsize=8, color="black", fontweight="bold")
print(f"  [5/5] Fare Distribution   - DONE  | Min: GBP {df['Fare'].min():.2f}, Max: GBP {df['Fare'].max():.2f}")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_png = os.path.join(OUTPUT_DIR, "task1_visualizations.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
plt.show()
print(f"\n  Visualization saved -> {out_png}")

# ═══════════════════════════════════════════════════════════════
#  PHASE 10: EXPORT CLEAN DATASET
# ═══════════════════════════════════════════════════════════════
print()
print(">>> PHASE 10: EXPORTING CLEAN DATASET")
print("-" * 65)

out_csv = os.path.join(OUTPUT_DIR, "titanic_clean.csv")
df.to_csv(out_csv, index=False)
print(f"  [SAVED] Clean CSV -> {out_csv}")
print(f"  Final Columns     : {list(df.columns)}")

# ═══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  TASK 1 - FINAL SUMMARY REPORT")
print("=" * 65)
print(f"  Intern             : Leena Chekuri")
print(f"  Organization       : Synent Technologies")
print(f"  Task               : Task 1 - Data Cleaning & Preprocessing")
print(f"  Dataset            : Titanic-Dataset.csv (Kaggle)")
print("-" * 65)
print(f"  Original Shape     : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
print(f"  Clean Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"  Missing Fixed      : {df_raw.isnull().sum().sum()} values handled")
print(f"  Duplicates Removed : {before - after}")
print(f"  Columns Renamed    : {len(rename_map)}")
print(f"  New Features Added : Deck, FamilySize, IsAlone, AgeGroup")
print("-" * 65)
print(f"  KEY INSIGHTS:")
print(f"  - Survival Rate        : {surv_rate:.1f}% ({df['Survived'].sum()} survived)")
print(f"  - Youngest Passenger   : {df['Age'].min()} years old")
print(f"  - Oldest Passenger     : {df['Age'].max()} years old")
print(f"  - Highest Fare (MAX)   : GBP {df['Fare'].max():.2f}")
print(f"  - Lowest Fare  (MIN)   : GBP {df['Fare'].min():.2f}")
print(f"  - Largest Family Size  : {int(df['FamilySize'].max())} members")
print(f"  - Most passengers were : {df['Gender'].value_counts().idxmax()} ({df['Gender'].value_counts().max()})")
print(f"  - Most common class    : Class {df['PassengerClass'].value_counts().idxmax()} ({df['PassengerClass'].value_counts().max()} passengers)")
print("-" * 65)
print(f"  OUTPUT FILES:")
print(f"  - outputs/titanic_clean.csv")
print(f"  - outputs/task1_visualizations.png")
print("=" * 65)
print()
print("  [DONE] Task 1 Complete! Dataset is clean & ready for analysis.")

print()
