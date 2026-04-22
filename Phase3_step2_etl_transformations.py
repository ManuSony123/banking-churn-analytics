# =============================================================================
# phase3_step2_etl_transformations.py
# PURPOSE : Apply all ETL transformations to the raw customer data and
#           produce clean, enriched staging tables ready for warehouse load
# RUN     : python phase3_step2_etl_transformations.py
# DEPENDS : BankChurners.csv in data/raw/
# OUTPUT  : stg_customers.csv in data/staging/
#           data_quality_report.txt in docs/
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Null value treatment (fill / impute / flag)
#   2. Deduplication (exact rows + duplicate business keys)
#   3. Data type standardization
#   4. Derived column creation (age groups, credit tiers, tenure bands, flags)
#   5. Outlier capping (Winsorization)
#   6. SCD Type 2 columns (effective dates, is_current, surrogate key)
#   7. Risk score calculation (multi-factor churn risk)
#   8. Data quality checks (before vs after comparison)
#   9. Saves clean staging CSV + quality report
#
# ETL PARALLEL — FULL INFORMATICA MAPPING WALKTHROUGH:
#   Source Qualifier     → load_raw_data()
#   Expression           → derive_columns()
#   Router               → flag_high_value_customers()
#   Aggregator           → No aggregation at staging (happens in warehouse)
#   Sorter               → deduplication (sort by key before checking)
#   Filter               → outlier_treatment() caps extreme values
#   SCD Type 2 mapping   → apply_scd2_columns()
#   Target Definition    → save_staging_output()
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, STAGING_DIR, STG_CUSTOMERS_PATH, REPORTS_DIR,
    CREDIT_BINS, CREDIT_LABELS, AGE_BINS, AGE_LABELS, TENURE_BINS, TENURE_LABELS,
    SCD2_TRACK_COLUMNS
)

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TODAY = date.today().strftime("%Y-%m-%d")


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def log_counts(df, step_name, before=None):
    """Logs row counts before and after each transformation step."""
    rows = len(df)
    if before is not None:
        diff = before - rows
        flag = f"  (-{diff} rows removed)" if diff > 0 else "  (no rows removed)"
    else:
        flag = ""
    print(f"  [{step_name}] Rows after step: {rows:,}{flag}")
    return rows


# =============================================================================
# T1 — Load Raw Data (Source Qualifier equivalent)
# =============================================================================

def load_raw_data():
    """
    Loads the raw CSV exactly as-is.
    No transformations at this stage — just reading and reporting.
    ETL PARALLEL: Source Qualifier transformation in Informatica.
    """
    section("T1 — Source Qualifier: Load Raw Data")

    df = pd.read_csv(RAW_DATA_PATH)
    original_count = len(df)

    print(f"\n  Source file  : {RAW_DATA_PATH}")
    print(f"  Rows loaded  : {original_count:,}")
    print(f"  Columns      : {df.shape[1]}")
    print(f"\n  Raw dtypes:")
    for col in df.columns:
        print(f"    {col:<22}: {df[col].dtype}")

    return df, original_count


# =============================================================================
# T2 — Null Value Treatment (Expression Transformer equivalent)
# =============================================================================

def treat_nulls(df):
    """
    Handles missing values column by column with a documented strategy.

    ETL PARALLEL:
    In Informatica you use IIF() inside an Expression Transformation:
      IIF(ISNULL(Balance), 0, Balance)
      IIF(ISNULL(CreditScore), MEDIAN(CreditScore), CreditScore)
    This function replicates that exact logic in Python.
    """
    section("T2 — Expression: Null Value Treatment")

    before = len(df)
    null_before = df.isnull().sum()

    print(f"\n  Null counts BEFORE treatment:")
    for col in df.columns:
        if null_before[col] > 0:
            print(f"    {col:<22}: {null_before[col]} nulls")

    # Strategy for each column:
    # Balance       → fill with 0 (dormant account assumption)
    df["Balance"] = df["Balance"].fillna(0)

    # CreditScore   → fill with median (preserve distribution)
    median_cs = df["CreditScore"].median()
    df["CreditScore"] = df["CreditScore"].fillna(median_cs)
    print(f"\n  CreditScore nulls filled with median: {median_cs:.0f}")

    # EstimatedSalary → fill with median (preserve distribution)
    median_sal = df["EstimatedSalary"].median()
    df["EstimatedSalary"] = df["EstimatedSalary"].fillna(median_sal)
    print(f"  EstimatedSalary nulls filled with median: {median_sal:.0f}")

    # Age, Tenure → fill with median
    df["Age"]    = df["Age"].fillna(int(df["Age"].median()))
    df["Tenure"] = df["Tenure"].fillna(int(df["Tenure"].median()))

    # Binary flags → fill with mode (most common value)
    for col in ["HasCrCard", "IsActiveMember"]:
        mode_val = df[col].mode()[0]
        df[col]  = df[col].fillna(mode_val)
        print(f"  {col} nulls filled with mode: {int(mode_val)}")

    # Geography, Gender → fill with mode
    for col in ["Geography", "Gender"]:
        mode_val = df[col].mode()[0]
        df[col]  = df[col].fillna(mode_val)
        print(f"  {col} nulls filled with mode: {mode_val}")

    null_after = df.isnull().sum().sum()
    print(f"\n  Total nulls AFTER treatment: {null_after}")
    if null_after == 0:
        print("  All nulls treated successfully.")

    log_counts(df, "Null Treatment", before)
    return df


# =============================================================================
# T3 — Deduplication (Sorter + Aggregator equivalent)
# =============================================================================

def deduplicate(df):
    """
    Removes full duplicate rows first, then handles duplicate business keys.

    ETL PARALLEL:
    In Informatica:
    - Full row dedup: Sorter (all columns) → Aggregator (FIRST record per group)
    - Business key dedup: Sorter (CustomerId) → custom Update Strategy
    We keep the LAST occurrence of a duplicate CustomerId
    (assumes latest record is most up-to-date).
    """
    section("T3 — Sorter + Aggregator: Deduplication")

    before = len(df)
    print(f"\n  Rows before dedup : {before:,}")

    # Step 1: Remove exact duplicate rows
    df.drop_duplicates(inplace=True)
    after_full_dedup = len(df)
    print(f"  Full row duplicates removed : {before - after_full_dedup}")

    # Step 2: Remove duplicate CustomerIds — keep last (most recent) occurrence
    dup_cust_ids = df["CustomerId"].duplicated(keep="last").sum()
    df.drop_duplicates(subset="CustomerId", keep="last", inplace=True)
    after_key_dedup = len(df)
    print(f"  Duplicate CustomerIds removed: {after_full_dedup - after_key_dedup}")

    # Step 3: Remove 'RowNumber' — it's a source system artifact, not a business key
    if "RowNumber" in df.columns:
        df.drop(columns=["RowNumber"], inplace=True)
        print(f"  'RowNumber' column dropped (source artifact, not needed in warehouse)")

    print(f"\n  Rows after dedup : {len(df):,}")
    log_counts(df, "Deduplication", before)
    return df.reset_index(drop=True)


# =============================================================================
# T4 — Data Type Standardization
# =============================================================================

def standardize_dtypes(df):
    """
    Casts every column to the correct Python/SQL-compatible data type.

    ETL PARALLEL:
    In Informatica, the Target Definition specifies data types and
    the Expression Transformation uses TO_INTEGER(), TO_FLOAT(), etc.
    This is the Python equivalent.
    """
    section("T4 — Expression: Data Type Standardization")

    # Integer columns
    for col in ["CustomerId", "CreditScore", "Age", "Tenure",
                "NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]:
        df[col] = df[col].astype(int)

    # Float columns (2 decimal places)
    for col in ["Balance", "EstimatedSalary"]:
        df[col] = df[col].round(2).astype(float)

    # String columns — strip whitespace and title-case
    df["Geography"] = df["Geography"].str.strip().str.title()
    df["Gender"]    = df["Gender"].str.strip().str.title()
    df["Surname"]   = df["Surname"].str.strip().str.title()

    print(f"\n  Data types after standardization:")
    for col in df.columns:
        print(f"    {col:<22}: {df[col].dtype}")

    return df


# =============================================================================
# T5 — Outlier Capping (Filter + Expression equivalent)
# =============================================================================

def cap_outliers(df):
    """
    Caps extreme outliers using the Winsorization method (IQR-based).
    We CAP rather than REMOVE outliers to preserve row count.

    ETL PARALLEL:
    In Informatica: Filter Transformation routes extreme values to a
    separate target or Expression Transformation caps them inline:
      IIF(CreditScore < 300, 300, IIF(CreditScore > 850, 850, CreditScore))
    """
    section("T5 — Filter + Expression: Outlier Capping")

    print(f"\n  Using Winsorization (IQR method) — capping, not removing.")
    print(f"  {'Column':<20} {'Lower Cap':>12} {'Upper Cap':>12} {'Values Capped':>15}")
    print("  " + "-" * 65)

    for col in ["CreditScore", "Balance", "EstimatedSalary"]:
        Q1  = df[col].quantile(0.01)   # 1st percentile
        Q3  = df[col].quantile(0.99)   # 99th percentile

        below = (df[col] < Q1).sum()
        above = (df[col] > Q3).sum()

        df[col] = df[col].clip(lower=Q1, upper=Q3)
        print(f"  {col:<20} {Q1:>12.0f} {Q3:>12.0f} {below+above:>15,}")

    # Hard-coded domain knowledge caps (bank-specific rules)
    # Age: valid range 18-100 in banking context
    age_bad = ((df["Age"] < 18) | (df["Age"] > 100)).sum()
    df["Age"] = df["Age"].clip(lower=18, upper=100)
    print(f"  {'Age (domain rule)':<20} {'18':>12} {'100':>12} {age_bad:>15,}")

    # Tenure: valid range 0-10 (dataset-specific)
    ten_bad = ((df["Tenure"] < 0) | (df["Tenure"] > 10)).sum()
    df["Tenure"] = df["Tenure"].clip(lower=0, upper=10)
    print(f"  {'Tenure (domain rule)':<20} {'0':>12} {'10':>12} {ten_bad:>15,}")

    return df


# =============================================================================
# T6 — Derived Column Creation (Expression Transformer)
# =============================================================================

def derive_columns(df):
    """
    Creates all business-logic derived columns.

    ETL PARALLEL:
    This is the most important Expression Transformation in the mapping.
    Every IIF(), DECODE(), ROUND(), and calculated field from Informatica
    is replicated here as a Python expression.

    Columns created:
      AgeGroup         — age bucket for segmentation
      CreditTier       — credit quality bucket
      TenureBand       — tenure bucket
      HighValueFlag    — balance threshold flag
      EstimatedLTV     — simple lifetime value estimate
      ChurnLabel       — human-readable churn label
      BalanceSalaryRatio — derived ratio (financial health indicator)
      ProductEngagement — engagement score (0-10)
    """
    section("T6 — Expression: Derived Column Creation")

    # --- Age Group (same as Informatica DECODE or nested IIF) ---
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    ).astype(str)

    # --- Credit Tier ---
    df["CreditTier"] = pd.cut(
        df["CreditScore"],
        bins=CREDIT_BINS,
        labels=CREDIT_LABELS
    ).astype(str)

    # --- Tenure Band ---
    df["TenureBand"] = pd.cut(
        df["Tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS
    ).astype(str)

    # --- High Value Flag (Informatica: IIF(Balance > 100000, 'Y', 'N')) ---
    df["HighValueFlag"] = np.where(df["Balance"] > 100000, "Y", "N")

    # --- Estimated LTV (simple formula) ---
    # LTV = (Balance * 0.02) + (EstimatedSalary * 0.001 * Tenure)
    # This simulates a basic LTV model used by banks
    df["EstimatedLTV"] = (
        (df["Balance"] * 0.02) +
        (df["EstimatedSalary"] * 0.001 * df["Tenure"])
    ).round(2)

    # --- Churn Label ---
    df["ChurnLabel"] = df["Exited"].map({0: "Active", 1: "Churned"})

    # --- Balance to Salary Ratio (financial health indicator) ---
    df["BalanceSalaryRatio"] = np.where(
        df["EstimatedSalary"] > 0,
        (df["Balance"] / df["EstimatedSalary"]).round(4),
        0.0
    )

    # --- Product Engagement Score (0 = disengaged, 4 = highly engaged) ---
    # Combines: NumOfProducts + IsActiveMember + HasCrCard
    df["ProductEngagement"] = (
        df["NumOfProducts"] +
        df["IsActiveMember"] * 2 +
        df["HasCrCard"]
    ).clip(0, 10)

    # Print summary of new columns
    new_cols = ["AgeGroup","CreditTier","TenureBand","HighValueFlag",
                "EstimatedLTV","ChurnLabel","BalanceSalaryRatio","ProductEngagement"]

    print(f"\n  Derived columns created: {len(new_cols)}")
    for col in new_cols:
        sample_vals = df[col].unique()[:4]
        print(f"    {col:<25}: {list(sample_vals)}")

    return df


# =============================================================================
# T7 — Churn Risk Score (Router + Expression equivalent)
# =============================================================================

def calculate_risk_score(df):
    """
    Builds a multi-factor churn risk score (0-10 scale).
    This is the equivalent of a complex IIF-based Expression Transformation
    in Informatica that assigns routing codes to customers.

    Risk factors and weights:
      - NumOfProducts >= 3 : +3 (strongest signal from EDA)
      - IsActiveMember = 0 : +3 (strong signal)
      - Age >= 51          : +2 (older customers churn more)
      - Tenure <= 2        : +1 (new customers are at risk)
      - HighValueFlag = Y  : +1 (high-balance paradox)
    """
    section("T7 — Router + Expression: Churn Risk Score")

    df["RiskScore"] = (
        np.where(df["NumOfProducts"] >= 3, 3, 0) +
        np.where(df["IsActiveMember"] == 0, 3, 0) +
        np.where(df["Age"] >= 51, 2, 0) +
        np.where(df["Tenure"] <= 2, 1, 0) +
        np.where(df["HighValueFlag"] == "Y", 1, 0)
    ).clip(0, 10)

    df["RiskCategory"] = pd.cut(
        df["RiskScore"],
        bins=[-1, 2, 4, 6, 10],
        labels=["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    ).astype(str)

    print(f"\n  Risk category distribution:")
    risk_dist = df.groupby("RiskCategory")["Exited"].agg(["count","sum","mean"])
    risk_dist.columns = ["Total", "Churned", "Actual Churn Rate"]
    risk_dist["Actual Churn Rate"] = (risk_dist["Actual Churn Rate"] * 100).round(1)
    print(risk_dist.to_string())

    print(f"\n  Risk score correlation with actual churn: {df['RiskScore'].corr(df['Exited']):.3f}")

    return df


# =============================================================================
# T8 — SCD Type 2 Columns (SCD Type 2 Mapping equivalent)
# =============================================================================

def apply_scd2_columns(df):
    """
    Adds the three SCD Type 2 control columns to the staging layer.

    ETL PARALLEL:
    In Informatica you use the SCD Wizard or a manual mapping with:
      - Router: split INSERTS from UPDATES
      - Expression: set EffectiveStartDate, EffectiveEndDate
      - Update Strategy: mark old rows for UPDATE, new rows for INSERT
      - Sequence Generator: assign surrogate keys

    In this project we simulate the INITIAL load (all rows are NEW).
    In Phase 3 Step 3 we will implement the INCREMENTAL load with
    actual change detection — that's where the real SCD2 logic runs.

    SCD Type 2 pattern:
      - customer_sk         : surrogate key (auto-increment, replaces business key in FK)
      - effective_start_date: date this version of the record became active
      - effective_end_date  : date this version expired (9999-12-31 = current)
      - is_current          : 1 = current version, 0 = historical version
    """
    section("T8 — SCD Type 2: Adding Version Control Columns")

    # Surrogate key — sequential integer (in DB this is AUTO_INCREMENT)
    df = df.reset_index(drop=True)
    df.insert(0, "customer_sk", df.index + 1)

    # SCD Type 2 columns — initial load: all rows are current version 1
    df["effective_start_date"] = TODAY
    df["effective_end_date"]   = "9999-12-31"
    df["is_current"]           = 1
    df["record_version"]       = 1

    # Rename source columns to snake_case for warehouse compatibility
    rename_map = {
        "CustomerId"     : "customer_id",
        "Surname"        : "surname",
        "CreditScore"    : "credit_score",
        "Geography"      : "geography",
        "Gender"         : "gender",
        "Age"            : "age",
        "Tenure"         : "tenure",
        "Balance"        : "balance",
        "NumOfProducts"  : "num_of_products",
        "HasCrCard"      : "has_cr_card",
        "IsActiveMember" : "is_active_member",
        "EstimatedSalary": "estimated_salary",
        "Exited"         : "churn_flag",
        # Derived columns
        "AgeGroup"       : "age_group",
        "CreditTier"     : "credit_tier",
        "TenureBand"     : "tenure_band",
        "HighValueFlag"  : "high_value_flag",
        "EstimatedLTV"   : "estimated_ltv",
        "ChurnLabel"     : "churn_label",
        "BalanceSalaryRatio": "balance_salary_ratio",
        "ProductEngagement" : "product_engagement",
        "RiskScore"      : "risk_score",
        "RiskCategory"   : "risk_category",
    }
    df.rename(columns=rename_map, inplace=True)

    print(f"\n  SCD Type 2 columns added:")
    print(f"    customer_sk           : {df['customer_sk'].min()} to {df['customer_sk'].max()}")
    print(f"    effective_start_date  : {df['effective_start_date'].unique()}")
    print(f"    effective_end_date    : {df['effective_end_date'].unique()}")
    print(f"    is_current            : {df['is_current'].unique()}")
    print(f"\n  Total columns in staging output: {df.shape[1]}")
    print(f"\n  Final column list:")
    for col in df.columns:
        print(f"    {col}")

    return df


# =============================================================================
# T9 — Data Quality Validation (before saving)
# =============================================================================

def validate_output(df, original_count):
    """
    Final data quality gate before saving to staging.

    ETL PARALLEL:
    In Informatica this is the equivalent of the 'Session Log' validation
    and pre/post-session SQL checks. You verify that:
    1. No unexpected row loss
    2. No nulls in key columns
    3. Business rules are satisfied
    """
    section("T9 — Data Quality Validation")

    issues = []

    # Check 1: Row count reasonable (should not lose > 1% of rows)
    retention_pct = len(df) / original_count * 100
    print(f"\n  Row retention: {len(df):,} / {original_count:,} = {retention_pct:.1f}%")
    if retention_pct < 99:
        issues.append(f"Row retention {retention_pct:.1f}% < 99% — check dedup logic")

    # Check 2: No nulls in mandatory columns
    mandatory_cols = ["customer_sk", "customer_id", "credit_score", "geography",
                      "gender", "age", "tenure", "churn_flag", "is_current"]
    for col in mandatory_cols:
        null_count = df[col].isnull().sum()
        status = "OK" if null_count == 0 else f"FAIL ({null_count} nulls)"
        print(f"  {col:<30}: {status}")
        if null_count > 0:
            issues.append(f"NULL values in mandatory column: {col}")

    # Check 3: Surrogate keys are unique
    sk_dupes = df["customer_sk"].duplicated().sum()
    print(f"  {'Surrogate key uniqueness':<30}: {'OK' if sk_dupes==0 else f'FAIL ({sk_dupes} dupes)'}")
    if sk_dupes > 0:
        issues.append("Duplicate surrogate keys found")

    # Check 4: SCD columns are correct
    all_current = (df["is_current"] == 1).all()
    print(f"  {'All rows is_current=1':<30}: {'OK' if all_current else 'FAIL'}")

    # Check 5: Churn flag is binary
    valid_churn = df["churn_flag"].isin([0, 1]).all()
    print(f"  {'Churn flag is binary (0/1)':<30}: {'OK' if valid_churn else 'FAIL'}")

    if not issues:
        print(f"\n  All quality checks PASSED.")
    else:
        print(f"\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")

    return len(issues) == 0


# =============================================================================
# T10 — Save Staging Output + Quality Report
# =============================================================================

def save_staging_output(df):
    section("T10 — Saving Staging Output")

    df.to_csv(STG_CUSTOMERS_PATH, index=False)
    size_kb = os.path.getsize(STG_CUSTOMERS_PATH) / 1024
    print(f"\n  Saved to  : {STG_CUSTOMERS_PATH}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Columns   : {df.shape[1]}")
    print(f"  File size : {size_kb:.1f} KB")
    print(f"\n  Sample output (first 3 rows, selected columns):")
    sample_cols = ["customer_sk","customer_id","geography","age_group",
                   "credit_tier","churn_flag","risk_category","is_current"]
    print(df[sample_cols].head(3).to_string(index=False))


def save_quality_report(df, original_count, quality_passed):
    report_path = os.path.join(REPORTS_DIR, "phase3_etl_quality_report.txt")
    with open(report_path, "w") as f:
        f.write("Banking Churn Analytics — Phase 3 ETL Quality Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source rows        : {original_count:,}\n")
        f.write(f"Staging rows       : {len(df):,}\n")
        f.write(f"Retention          : {len(df)/original_count*100:.1f}%\n")
        f.write(f"Staging columns    : {df.shape[1]}\n")
        f.write(f"Quality gate       : {'PASSED' if quality_passed else 'FAILED'}\n\n")
        f.write("SCD Type 2 setup:\n")
        f.write(f"  effective_start_date : {df['effective_start_date'].iloc[0]}\n")
        f.write(f"  effective_end_date   : {df['effective_end_date'].iloc[0]}\n")
        f.write(f"  is_current           : all 1 (initial load)\n\n")
        f.write("Churn distribution in staging:\n")
        churn_pct = df['churn_flag'].mean() * 100
        f.write(f"  Churned : {df['churn_flag'].sum():,}  ({churn_pct:.1f}%)\n")
        f.write(f"  Active  : {(df['churn_flag']==0).sum():,}  ({100-churn_pct:.1f}%)\n\n")
        f.write("Risk distribution:\n")
        for risk, cnt in df['risk_category'].value_counts().items():
            f.write(f"  {risk:<15} : {cnt:,}\n")
        f.write("\nSTATUS: Ready for Phase 4 (MySQL warehouse load).\n")
    print(f"\n  Quality report saved to: {report_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 3 STEP 2 — ETL Transformations")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)

    df, original_count = load_raw_data()
    df = treat_nulls(df)
    df = deduplicate(df)
    df = standardize_dtypes(df)
    df = cap_outliers(df)
    df = derive_columns(df)
    df = calculate_risk_score(df)
    df = apply_scd2_columns(df)
    quality_ok = validate_output(df, original_count)
    save_staging_output(df)
    save_quality_report(df, original_count, quality_ok)

    section("PHASE 3 STEP 2 COMPLETE")
    print(f"\n  ETL transformations applied and staging file saved.")
    print(f"  Next: Run phase3_step3_scd2_incremental.py")
    print("\n" + "=" * 65 + "\n")