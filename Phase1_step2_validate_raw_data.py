# =============================================================================
# phase1_step2_validate_raw_data.py
# PURPOSE : Load the raw CSV and perform the very first data validation checks
# RUN     : python phase1_step2_validate_raw_data.py
# DEPENDS : Customer-Churn-Records.csv must be in data/raw/ folder
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads Customer-Churn-Records.csv from data/raw/
#   2. Confirms the file loaded correctly (row count, column count)
#   3. Prints all column names and data types
#   4. Checks for missing / null values in every column
#   5. Checks for duplicate rows and duplicate CustomerIds
#   6. Prints the churn class distribution
#   7. Prints a full numeric summary (min, max, mean, std)
#   8. Saves a text validation report to docs/phase1_validation_report.txt
#
# ETL PARALLEL:
#   In Informatica this is equivalent to the Source Qualifier transformation
#   where you preview data, check column mappings, and validate the source
#   before building the mapping logic. Always validate before transforming.
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, REPORTS_DIR


# =============================================================================
# HELPER — pretty section header
# =============================================================================

def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# =============================================================================
# STEP 1 — Load raw CSV
# =============================================================================

def load_raw_data(path):
    """
    Loads the raw BankChurners CSV file.
    Exits with a clear message if the file is missing.
    """
    section("STEP 1 — Loading Raw Dataset")

    if not os.path.exists(path):
        print(f"\n  ERROR: File not found at:\n  {path}")
        print("\n  Fix:")
        print("  1. Download BankChurners.csv from Kaggle")
        print("  2. Place it inside:  data/raw/BankChurners.csv")
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"\n  File loaded: {path}")
    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")
    print(f"  Size    : {os.path.getsize(path) / 1024:.1f} KB")
    return df


# =============================================================================
# STEP 2 — Column inventory
# =============================================================================

def check_columns(df):
    """
    Prints all column names, data types, and sample values.
    """
    section("STEP 2 — Column Inventory")

    print(f"\n  {'#':<4} {'Column':<22} {'Dtype':<12} {'Sample Value'}")
    print("  " + "-" * 56)
    for i, col in enumerate(df.columns, 1):
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        print(f"  {i:<4} {col:<22} {str(df[col].dtype):<12} {sample}")

    # Flag: are there any unexpected columns?
    expected_cols = [
        "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
        "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    unexpected = [c for c in df.columns if c not in expected_cols]
    if unexpected:
        print(f"\n  WARNING: Unexpected columns found: {unexpected}")
    else:
        print("\n  All expected columns are present.")


# =============================================================================
# STEP 3 — Null / missing value check
# =============================================================================

def check_nulls(df):
    """
    Reports null values per column.
    In ETL terms: this is the 'reject file analysis' — finding dirty records
    before they enter the pipeline.
    """
    section("STEP 3 — Null Value Analysis")

    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)

    print(f"\n  {'Column':<22} {'Null Count':>12} {'Null %':>10}  Status")
    print("  " + "-" * 58)

    any_nulls = False
    for col in df.columns:
        count = null_counts[col]
        pct   = null_pct[col]
        status = "OK" if count == 0 else "NEEDS TREATMENT"
        if count > 0:
            any_nulls = True
        print(f"  {col:<22} {count:>12,} {pct:>9.2f}%  {status}")

    if not any_nulls:
        print("\n  No null values found. Dataset is complete.")
    else:
        print("\n  Null values found — will be handled in Phase 3 (Staging).")

    return null_counts


# =============================================================================
# STEP 4 — Duplicate check
# =============================================================================

def check_duplicates(df):
    """
    Checks for full row duplicates and duplicate business keys (CustomerId).
    ETL parallel: In Informatica we use Sorter + Aggregator to deduplicate.
    Here we identify the scope of the problem before applying the fix.
    """
    section("STEP 4 — Duplicate Row Analysis")

    dup_rows      = df.duplicated().sum()
    dup_customers = df["CustomerId"].duplicated().sum()

    print(f"\n  Full duplicate rows     : {dup_rows:,}")
    print(f"  Duplicate CustomerIds   : {dup_customers:,}")

    if dup_rows > 0:
        print("\n  Duplicate rows found:")
        print(df[df.duplicated(keep=False)].to_string())
    else:
        print("\n  No duplicate rows.")

    if dup_customers > 0:
        dups = df[df["CustomerId"].duplicated(keep=False)].sort_values("CustomerId")
        print(f"\n  Customers with duplicate IDs:\n{dups[['CustomerId','Surname']].to_string()}")
    else:
        print("  No duplicate CustomerIds — primary key is unique.")

    return dup_rows, dup_customers


# =============================================================================
# STEP 5 — Churn distribution
# =============================================================================

def check_churn_distribution(df):
    """
    Analyses the target variable (Exited / churn flag).
    Class imbalance is important to understand before any analysis.
    """
    section("STEP 5 — Churn Distribution (Target Variable)")

    total    = len(df)
    churned  = df["Exited"].sum()
    active   = total - churned

    print(f"\n  Total customers : {total:,}")
    print(f"  Active  (0)     : {active:,}   ({active/total*100:.1f}%)")
    print(f"  Churned (1)     : {churned:,}   ({churned/total*100:.1f}%)")

    # Class imbalance assessment
    ratio = churned / active
    print(f"\n  Churn-to-active ratio : {ratio:.2f}")
    if ratio < 0.25:
        print("  Class imbalance detected (minority class < 25%).")
        print("  Note: For analytics purposes this is fine.")
        print("  For ML models, you would need SMOTE or class weighting.")
    else:
        print("  Class distribution is relatively balanced.")

    # Churn by Geography
    print(f"\n  Churn rate by Geography:")
    geo_churn = df.groupby("Geography")["Exited"].agg(["count","sum","mean"])
    geo_churn.columns = ["Total", "Churned", "Churn Rate"]
    geo_churn["Churn Rate"] = (geo_churn["Churn Rate"] * 100).round(1)
    print(geo_churn.to_string())

    # Churn by Gender
    print(f"\n  Churn rate by Gender:")
    gen_churn = df.groupby("Gender")["Exited"].agg(["count","sum","mean"])
    gen_churn.columns = ["Total", "Churned", "Churn Rate"]
    gen_churn["Churn Rate"] = (gen_churn["Churn Rate"] * 100).round(1)
    print(gen_churn.to_string())


# =============================================================================
# STEP 6 — Numeric summary
# =============================================================================

def check_numeric_summary(df):
    """
    Prints min, max, mean, std and flags potential outliers.
    ETL parallel: In Informatica we use the Aggregator transformation
    to compute column-level stats before routing data to targets.
    """
    section("STEP 6 — Numeric Column Summary")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].describe().round(2).T
    summary.columns = ["Count","Mean","Std","Min","25%","50%","75%","Max"]

    print(f"\n  {summary.to_string()}")

    # Outlier detection using IQR method
    print(f"\n  --- Outlier Detection (IQR Method) ---")
    for col in ["CreditScore", "Age", "Balance", "EstimatedSalary"]:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"\n  {col}:")
        print(f"    Valid range : {lower:.0f} to {upper:.0f}")
        print(f"    Outliers    : {len(outliers):,} rows ({len(outliers)/len(df)*100:.1f}%)")
        if len(outliers) > 0:
            print(f"    Min outlier : {df[col].min():.0f}  |  Max outlier: {df[col].max():.0f}")


# =============================================================================
# STEP 7 — Categorical value distributions
# =============================================================================

def check_categorical_distributions(df):
    """
    Checks unique values in categorical columns.
    Catches unexpected values (e.g. typos, encoding issues) early.
    """
    section("STEP 7 — Categorical Column Distributions")

    cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]

    for col in cat_cols:
        print(f"\n  {col}:")
        val_counts = df[col].value_counts()
        for val, cnt in val_counts.items():
            pct = cnt / len(df) * 100
            bar = "#" * int(pct / 2)
            print(f"    {str(val):<15} {cnt:>6,}  ({pct:5.1f}%)  {bar}")


# =============================================================================
# STEP 8 — Save validation report
# =============================================================================

def save_report(df, null_counts, dup_rows, dup_customers):
    """
    Writes a summary validation report to docs/phase1_validation_report.txt
    """
    section("STEP 8 — Saving Validation Report")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "phase1_validation_report.txt")

    churned = df["Exited"].sum()
    total   = len(df)

    with open(report_path, "w") as f:
        f.write("Banking Churn Analytics — Phase 1 Validation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: BankChurners.csv\n")
        f.write(f"Total rows         : {total:,}\n")
        f.write(f"Total columns      : {df.shape[1]}\n")
        f.write(f"Duplicate rows     : {dup_rows:,}\n")
        f.write(f"Duplicate Cust IDs : {dup_customers:,}\n")
        f.write(f"Columns with nulls : {(null_counts > 0).sum()}\n")
        f.write(f"Churn rate         : {churned/total*100:.1f}%\n\n")
        f.write("Column null summary:\n")
        for col in df.columns:
            f.write(f"  {col:<22}: {null_counts[col]} nulls\n")
        f.write("\nSTATUS: Raw data validated. Ready for Phase 2 (EDA).\n")

    print(f"\n  Report saved to: {report_path}")


# =============================================================================
# MAIN — Run all steps in sequence
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 60)
    print("#  PHASE 1 — Raw Data Validation")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 60)

    # Run every validation step
    df = load_raw_data(RAW_DATA_PATH)
    check_columns(df)
    null_counts = check_nulls(df)
    dup_rows, dup_customers = check_duplicates(df)
    check_churn_distribution(df)
    check_numeric_summary(df)
    check_categorical_distributions(df)
    save_report(df, null_counts, dup_rows, dup_customers)

    section("PHASE 1 COMPLETE")
    print("\n  All validation checks finished.")
    print("  Next step: Run phase2_step1_eda_univariate.py")
    print("\n" + "=" * 60 + "\n")