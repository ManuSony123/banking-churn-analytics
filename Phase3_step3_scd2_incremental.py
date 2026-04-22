# =============================================================================
# phase3_step3_scd2_incremental.py
# PURPOSE : Demonstrate SCD Type 2 incremental load logic — the most
#           important ETL concept you can showcase in interviews
# RUN     : python phase3_step3_scd2_incremental.py
# DEPENDS : stg_customers.csv in data/staging/
# OUTPUT  : stg_customers_with_history.csv (SCD2 history simulation)
#           scd2_change_log.csv (audit trail of all changes detected)
# =============================================================================
#
# WHAT IS SCD TYPE 2?
#   Slowly Changing Dimension Type 2 (SCD2) is a warehouse pattern that
#   preserves FULL HISTORY of how a dimension record changed over time.
#   Instead of overwriting old values, we:
#     1. EXPIRE the old row (set effective_end_date = today, is_current = 0)
#     2. INSERT a new row (new values, effective_start_date = today, is_current = 1)
#   This means one customer can have MULTIPLE rows in dim_customer —
#   one for each version of their data.
#
# REAL EXAMPLE:
#   Maria moved from France to Germany in 2023.
#   With SCD2, dim_customer has TWO rows for Maria:
#     Row 1: Geography=France, start=2022-01-01, end=2023-06-14, is_current=0
#     Row 2: Geography=Germany, start=2023-06-15, end=9999-12-31, is_current=1
#   Fact table rows from 2022 join to Row 1 (France) — correct history!
#   Fact table rows from 2023 join to Row 2 (Germany) — correct current!
#
# ETL PARALLEL — INFORMATICA SCD TYPE 2 MAPPING:
#   1. Source Qualifier: read incoming (new) data
#   2. Lookup: join to existing dim_customer on customer_id WHERE is_current=1
#   3. Expression: compare tracked columns — detect if any changed
#   4. Router:
#      - No change found → discard (no action)
#      - Change found    → route to two targets:
#          Target 1: UPDATE old row (expire it)
#          Target 2: INSERT new row (insert new version)
#      - New customer    → route to INSERT target only
#   5. Update Strategy:
#      - DD_UPDATE for expiry path
#      - DD_INSERT for new version and new customer paths
#   6. Sequence Generator: assign new surrogate key for inserted rows
#
# THIS SCRIPT REPLICATES THAT EXACT LOGIC IN PYTHON
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STAGING_DIR, STG_CUSTOMERS_PATH, REPORTS_DIR, SCD2_TRACK_COLUMNS

os.makedirs(STAGING_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Simulate "today" as the incremental load run date
LOAD_DATE = date.today().strftime("%Y-%m-%d")
# Simulate "previous load" as 6 months ago (for generating realistic change data)
PREV_LOAD_DATE = (date.today() - timedelta(days=180)).strftime("%Y-%m-%d")

SCD2_HISTORY_PATH = os.path.join(STAGING_DIR, "stg_customers_with_history.csv")
CHANGE_LOG_PATH   = os.path.join(STAGING_DIR, "scd2_change_log.csv")


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load existing dimension (simulates "current warehouse state")
# =============================================================================

def load_existing_dimension():
    """
    Loads the staging customer file as the "existing dimension" state.
    In a real project this would be a SELECT from dim_customer WHERE is_current=1.
    """
    section("STEP 1 — Load Existing Dimension (Current State)")

    df = pd.read_csv(STG_CUSTOMERS_PATH)
    current = df[df["is_current"] == 1].copy()

    print(f"\n  Total rows in dimension : {len(df):,}")
    print(f"  Current rows (v=1)      : {len(current):,}")
    print(f"  Historical rows (v=0)   : {(df['is_current']==0).sum():,}")
    print(f"\n  Columns tracked for SCD2: {SCD2_TRACK_COLUMNS}")
    print(f"\n  Sample current records:")
    print(current[["customer_sk","customer_id","geography","num_of_products",
                   "is_active_member","is_current","effective_start_date"]].head(5).to_string(index=False))

    return df, current


# =============================================================================
# STEP 2 — Simulate incoming data with changes
# =============================================================================

def simulate_incoming_changes(current_df):
    """
    Creates a realistic "incoming delta" dataset that represents:
    1. Customers who moved country (Geography change)
    2. Customers who added a product (NumOfProducts change)
    3. Customers who became inactive (IsActiveMember change)
    4. New customers who joined the bank
    5. Customers with no changes (majority — no action needed)

    In a real project this incoming data would come from:
    - A nightly extract from the core banking system
    - A CDC (Change Data Capture) feed from the operational DB
    - An incremental file from the source system with $$LastRunDate filter
    """
    section("STEP 2 — Simulating Incoming Delta Data")

    incoming = current_df.copy()

    # Simulate changes for 2% of customers — realistic for a 6-month delta
    total = len(incoming)
    n_changes_per_type = max(10, int(total * 0.02))

    # Change Type 1: Geography changes (customers who moved)
    geo_change_idx = np.random.choice(incoming.index, size=n_changes_per_type, replace=False)
    original_geos  = incoming.loc[geo_change_idx, "geography"].tolist()
    all_geos = ["France", "Germany", "Spain"]
    for idx in geo_change_idx:
        current_geo = incoming.loc[idx, "geography"]
        new_geo = np.random.choice([g for g in all_geos if g != current_geo])
        incoming.loc[idx, "geography"] = new_geo

    print(f"\n  Geography changes simulated : {n_changes_per_type}")
    print(f"    Example: Customer {incoming.loc[geo_change_idx[0], 'customer_id']} "
          f"moved from {original_geos[0]} to {incoming.loc[geo_change_idx[0], 'geography']}")

    # Change Type 2: NumOfProducts changes
    prod_change_idx = np.random.choice(
        incoming[incoming["num_of_products"] < 3].index,
        size=n_changes_per_type, replace=False
    )
    for idx in prod_change_idx:
        incoming.loc[idx, "num_of_products"] = incoming.loc[idx, "num_of_products"] + 1

    print(f"  Product count changes       : {n_changes_per_type}")

    # Change Type 3: IsActiveMember changes (became inactive)
    active_idx = incoming[incoming["is_active_member"] == 1].index
    inactive_change_idx = np.random.choice(active_idx, size=n_changes_per_type, replace=False)
    incoming.loc[inactive_change_idx, "is_active_member"] = 0
    print(f"  Active→Inactive changes     : {n_changes_per_type}")

    # Change Type 4: New customers (5 new records)
    n_new = 5
    max_sk = incoming["customer_sk"].max()
    max_id = incoming["customer_id"].max()

    new_rows = []
    for i in range(n_new):
        new_rows.append({
            "customer_sk"         : max_sk + i + 1,
            "customer_id"         : max_id + i + 1,
            "surname"             : f"NewCustomer{i+1}",
            "credit_score"        : np.random.randint(550, 800),
            "geography"           : np.random.choice(["France","Germany","Spain"]),
            "gender"              : np.random.choice(["Male","Female"]),
            "age"                 : np.random.randint(25, 55),
            "tenure"              : 0,
            "balance"             : round(np.random.uniform(0, 80000), 2),
            "num_of_products"     : 1,
            "has_cr_card"         : np.random.randint(0, 2),
            "is_active_member"    : 1,
            "estimated_salary"    : round(np.random.uniform(30000, 120000), 2),
            "churn_flag"          : 0,
            "age_group"           : "25-35",
            "credit_tier"         : "Good",
            "tenure_band"         : "New (0-2 yrs)",
            "high_value_flag"     : "N",
            "estimated_ltv"       : 0.0,
            "churn_label"         : "Active",
            "balance_salary_ratio": 0.0,
            "product_engagement"  : 3,
            "risk_score"          : 1,
            "risk_category"       : "Low Risk",
            "effective_start_date": LOAD_DATE,
            "effective_end_date"  : "9999-12-31",
            "is_current"          : 1,
            "record_version"      : 1,
        })

    new_df = pd.DataFrame(new_rows)
    print(f"  New customers to insert     : {n_new}")
    print(f"\n  Total incoming records      : {len(incoming) + n_new:,}")

    return incoming, new_df


# =============================================================================
# STEP 3 — Core SCD Type 2 Logic
# =============================================================================

def apply_scd2_logic(existing_df, current_df, incoming_df, new_customers_df):
    """
    THE HEART OF THE ETL — SCD Type 2 change detection and processing.

    For each incoming record:
      - Compare tracked columns to the current dimension row
      - If CHANGED: expire old row + insert new version
      - If UNCHANGED: do nothing
      - If NEW: insert directly

    This exactly replicates the Informatica SCD Type 2 mapping logic.
    """
    section("STEP 3 — SCD Type 2 Change Detection and Processing")

    print(f"\n  Tracked columns: {SCD2_TRACK_COLUMNS}")
    print(f"  Load date      : {LOAD_DATE}")

    # Build lookup dict for fast access: customer_id → current row
    lookup = current_df.set_index("customer_id").to_dict("index")

    changes_detected = []
    rows_to_expire   = []   # customer_sks to mark is_current=0
    rows_to_insert   = []   # new version rows to insert

    unchanged_count  = 0
    changed_count    = 0

    # --- Compare incoming to existing ---
    for _, new_row in incoming_df.iterrows():
        cust_id = new_row["customer_id"]

        if cust_id not in lookup:
            # Brand new customer — pure insert (handled separately)
            continue

        old_row = lookup[cust_id]

        # Compare each tracked column
        changed_cols = []
        for col in SCD2_TRACK_COLUMNS:
            old_val = old_row.get(col)
            new_val = new_row.get(col)
            if str(old_val) != str(new_val):
                changed_cols.append({
                    "column"   : col,
                    "old_value": old_val,
                    "new_value": new_val
                })

        if not changed_cols:
            # No tracked columns changed — NO ACTION
            unchanged_count += 1
            continue

        # --- Changes detected ---
        changed_count += 1

        # Log the change
        for change in changed_cols:
            changes_detected.append({
                "customer_id"  : cust_id,
                "column_changed": change["column"],
                "old_value"    : change["old_value"],
                "new_value"    : change["new_value"],
                "load_date"    : LOAD_DATE,
            })

        # Mark old row for expiry
        rows_to_expire.append(old_row["customer_sk"])

        # Build new version row
        new_version = new_row.to_dict()
        new_version["customer_sk"]          = existing_df["customer_sk"].max() + len(rows_to_insert) + 1
        new_version["effective_start_date"] = LOAD_DATE
        new_version["effective_end_date"]   = "9999-12-31"
        new_version["is_current"]           = 1
        new_version["record_version"]       = old_row.get("record_version", 1) + 1
        rows_to_insert.append(new_version)

    print(f"\n  Comparison Results:")
    print(f"    Unchanged records : {unchanged_count:,}")
    print(f"    Changed records   : {changed_count:,}")
    print(f"    Records to expire : {len(rows_to_expire):,}")
    print(f"    New versions to insert: {len(rows_to_insert):,}")
    print(f"    Brand new customers   : {len(new_customers_df):,}")

    # --- Apply changes ---

    # Expire old rows
    existing_df.loc[existing_df["customer_sk"].isin(rows_to_expire), "is_current"] = 0
    existing_df.loc[existing_df["customer_sk"].isin(rows_to_expire), "effective_end_date"] = (
        (datetime.strptime(LOAD_DATE, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    )

    # Insert new versions
    if rows_to_insert:
        new_versions_df = pd.DataFrame(rows_to_insert)
        existing_df = pd.concat([existing_df, new_versions_df], ignore_index=True)

    # Insert new customers
    existing_df = pd.concat([existing_df, new_customers_df], ignore_index=True)

    return existing_df, pd.DataFrame(changes_detected)


# =============================================================================
# STEP 4 — Verify SCD2 result
# =============================================================================

def verify_scd2_result(df):
    section("STEP 4 — Verifying SCD2 Result")

    current_rows  = (df["is_current"] == 1).sum()
    historical_rows = (df["is_current"] == 0).sum()
    multi_version = df[df["is_current"] == 0]["customer_id"].nunique()

    print(f"\n  Total rows in updated dimension : {len(df):,}")
    print(f"  Current rows (is_current=1)     : {current_rows:,}")
    print(f"  Historical rows (is_current=0)  : {historical_rows:,}")
    print(f"  Customers with >1 version       : {multi_version:,}")

    # Show an example of a customer with 2 versions
    if multi_version > 0:
        sample_cid = df[df["is_current"] == 0]["customer_id"].iloc[0]
        versions   = df[df["customer_id"] == sample_cid].sort_values("record_version")
        print(f"\n  Example — Customer {sample_cid} history (SCD2 versions):")
        print(versions[[
            "customer_id","geography","num_of_products","is_active_member",
            "effective_start_date","effective_end_date","is_current","record_version"
        ]].to_string(index=False))
        print(f"\n  This is exactly how Informatica SCD Type 2 stores dimension history.")


# =============================================================================
# STEP 5 — Save outputs
# =============================================================================

def save_outputs(final_df, change_log_df):
    section("STEP 5 — Saving SCD2 Outputs")

    final_df.to_csv(SCD2_HISTORY_PATH, index=False)
    print(f"\n  SCD2 history dimension : {SCD2_HISTORY_PATH}")
    print(f"  Rows: {len(final_df):,}")

    if not change_log_df.empty:
        change_log_df.to_csv(CHANGE_LOG_PATH, index=False)
        print(f"\n  Change log (audit trail) : {CHANGE_LOG_PATH}")
        print(f"  Changes recorded: {len(change_log_df):,}")
        print(f"\n  Sample change log:")
        print(change_log_df.head(10).to_string(index=False))
    else:
        print("\n  No changes detected — change log is empty.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 3 STEP 3 — SCD Type 2 Incremental Load")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)
    print(f"\n  This script demonstrates the most important ETL concept:")
    print(f"  Slowly Changing Dimension Type 2 — full history preservation.")
    print(f"  This is the exact logic used in Informatica SCD mappings,")
    print(f"  replicated here in Python so you can explain it in interviews.")

    existing_df, current_df = load_existing_dimension()
    incoming_df, new_customers_df = simulate_incoming_changes(current_df)
    final_df, change_log_df = apply_scd2_logic(existing_df, current_df, incoming_df, new_customers_df)
    verify_scd2_result(final_df)
    save_outputs(final_df, change_log_df)

    section("PHASE 3 STEP 3 COMPLETE")
    print(f"\n  SCD Type 2 incremental load demonstrated successfully.")
    print(f"  You can now explain this pattern confidently in interviews.")
    print(f"\n  KEY INTERVIEW TALKING POINT:")
    print(f"  'I implemented SCD Type 2 by comparing incoming records against")
    print(f"  the current dimension row. When tracked columns changed, I expired")
    print(f"  the old row by setting effective_end_date and is_current=0, then")
    print(f"  inserted a new version with a new surrogate key — exactly the")
    print(f"  same pattern as the Informatica SCD wizard, just in Python.'")
    print(f"\n  Next: Run phase4_step1_create_mysql_schema.py")
    print("\n" + "=" * 65 + "\n")