# =============================================================================
# phase3_step1_generate_supporting_tables.py
# PURPOSE : Simulate the Transactions, Product, and Date tables that don't
#           exist in the raw Kaggle dataset but are needed for a real
#           star-schema warehouse
# RUN     : python phase3_step1_generate_supporting_tables.py
# DEPENDS : BankChurners.csv in data/raw/
# OUTPUT  : stg_transactions.csv, stg_products.csv, stg_date.csv
#           saved to data/staging/
# =============================================================================
#
# WHY THIS STEP EXISTS:
#   The Kaggle dataset only has customer-level data.
#   A real bank has transaction-level data (every deposit, withdrawal, etc.)
#   and a product catalog. This script simulates those tables realistically
#   so we can build a proper star schema with a fact table.
#
# ETL PARALLEL:
#   In Informatica this is like creating Source Definitions for multiple
#   source systems (core banking, product master, calendar dimension)
#   before building the mapping that joins them all.
#   Each table generated here maps to a Source Qualifier in Informatica.
#
# WHAT THIS SCRIPT DOES:
#   1. Generates a transactions table (one row per customer per month)
#   2. Generates a product dimension table (4 bank products)
#   3. Generates a full date dimension table (2022-2024, 3 years)
#   4. Generates a churn reason dimension (exit survey simulation)
#   5. Saves all as CSVs to data/staging/
#   6. Prints row counts and sample rows for every table
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, timedelta
import calendar
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RAW_DATA_PATH, STAGING_DIR,
    STG_TRANSACTIONS_PATH, STG_PRODUCTS_PATH,
    STG_DATE_PATH, STG_CHURN_REASON_PATH,
    PRODUCT_CATALOG
)

# Fix random seed for reproducibility
np.random.seed(42)
os.makedirs(STAGING_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


# =============================================================================
# STEP 1 — Load raw customer data (base for all generation)
# =============================================================================

def load_customers():
    section("STEP 1 — Loading Customer Base Data")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"\n  Customers loaded: {len(df):,}")
    print(f"  Churned: {df['Exited'].sum():,}  |  Active: {(df['Exited']==0).sum():,}")
    return df


# =============================================================================
# STEP 2 — Generate Transactions Table
# =============================================================================

def generate_transactions(df):
    """
    Simulates a realistic transaction history table.

    Logic:
    - Each customer gets between 3 and 36 transactions (realistic spread)
    - Churned customers have transactions only up to their estimated exit date
    - Transaction amounts are correlated with account balance (realistic)
    - Channel mix reflects real-world banking (mobile-heavy)
    - Product assignment weighted by NumOfProducts column

    ETL PARALLEL:
    In Informatica this is like a Source Qualifier reading from
    the core banking transaction system (T24, Temenos, etc.)
    """
    section("STEP 2 — Generating Transactions Table")

    records = []
    txn_id  = 1

    # Date range: 2022-01-01 to 2024-12-31
    start_date = date(2022, 1, 1)
    end_date   = date(2024, 12, 31)
    date_range = (end_date - start_date).days

    txn_types = ["Deposit", "Withdrawal", "Transfer Out", "Transfer In",
                 "Loan EMI", "Credit Card Payment", "Interest Credit"]

    # Channel weights — mobile-heavy, realistic for 2022-2024
    channels       = ["Mobile App", "Internet Banking", "ATM", "Branch", "Phone Banking"]
    channel_weights = [0.40,         0.25,              0.20,  0.10,     0.05]

    # Transaction type weights
    txn_weights = [0.30, 0.25, 0.15, 0.15, 0.08, 0.05, 0.02]

    for _, row in df.iterrows():
        cust_id   = row["CustomerId"]
        balance   = row["Balance"]
        num_prods = row["NumOfProducts"]
        is_churned= row["Exited"]
        tenure    = row["Tenure"]

        # Number of transactions: more products → more transactions
        base_txns = int(tenure * 3)            # avg 3 txns per year
        n_txns    = max(3, min(base_txns + np.random.randint(-3, 8), 60))

        # Churned customers have transactions ending before end of period
        if is_churned:
            # Simulate exit roughly 6-18 months before dataset cutoff
            months_before_exit = np.random.randint(6, 18)
            cust_end_days = max(30, date_range - months_before_exit * 30)
        else:
            cust_end_days = date_range

        # Generate transaction dates (sorted)
        txn_day_offsets = sorted(
            np.random.choice(cust_end_days, size=n_txns, replace=False)
        )

        # Transaction amount correlated with balance
        if balance > 0:
            amount_mean = balance * 0.05    # avg 5% of balance per txn
            amount_std  = balance * 0.08
        else:
            amount_mean = 1500
            amount_std  = 800

        for day_offset in txn_day_offsets:
            txn_date = start_date + timedelta(days=int(day_offset))

            # Product used in this transaction
            # Customers with more products use different ones
            prod_choices = list(PRODUCT_CATALOG.keys())[:num_prods]
            product_id   = int(np.random.choice(prod_choices))

            amount = max(10, round(
                np.random.normal(amount_mean, amount_std), 2
            ))

            txn_type = np.random.choice(txn_types, p=txn_weights)
            channel  = np.random.choice(channels, p=channel_weights)

            records.append({
                "transaction_id"  : txn_id,
                "customer_id"     : cust_id,
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "transaction_year": txn_date.year,
                "transaction_month": txn_date.month,
                "transaction_type": txn_type,
                "channel"         : channel,
                "product_id"      : product_id,
                "amount"          : amount,
                "churn_flag"      : int(is_churned),
            })
            txn_id += 1

    txn_df = pd.DataFrame(records)

    print(f"\n  Total transactions generated : {len(txn_df):,}")
    print(f"  Date range                   : {txn_df['transaction_date'].min()} to {txn_df['transaction_date'].max()}")
    print(f"  Unique customers             : {txn_df['customer_id'].nunique():,}")
    print(f"\n  Transaction type distribution:")
    print(txn_df["transaction_type"].value_counts().to_string())
    print(f"\n  Channel distribution:")
    print(txn_df["channel"].value_counts().to_string())
    print(f"\n  Amount stats:")
    print(txn_df["amount"].describe().round(2).to_string())
    print(f"\n  Sample rows:")
    print(txn_df.head(5).to_string(index=False))

    txn_df.to_csv(STG_TRANSACTIONS_PATH, index=False)
    print(f"\n  Saved to: {STG_TRANSACTIONS_PATH}")
    return txn_df


# =============================================================================
# STEP 3 — Generate Product Dimension
# =============================================================================

def generate_products():
    """
    Creates the product dimension table.
    This represents the bank's product catalog.

    ETL PARALLEL:
    In Informatica this comes from a Product Master source system
    (often a flat file or a slowly-changing lookup table).
    """
    section("STEP 3 — Generating Product Dimension")

    rows = []
    for prod_id, details in PRODUCT_CATALOG.items():
        rows.append({
            "product_id"      : prod_id,
            "product_name"    : details["name"],
            "product_category": details["category"],
            "interest_rate_pct": details["interest_rate"],
            "is_active"       : 1,
            "launch_year"     : 2015,
        })

    prod_df = pd.DataFrame(rows)

    print(f"\n  Products created: {len(prod_df)}")
    print(f"\n  Product catalog:")
    print(prod_df.to_string(index=False))

    prod_df.to_csv(STG_PRODUCTS_PATH, index=False)
    print(f"\n  Saved to: {STG_PRODUCTS_PATH}")
    return prod_df


# =============================================================================
# STEP 4 — Generate Date Dimension
# =============================================================================

def generate_date_dimension():
    """
    Generates a complete date dimension table for 2022-01-01 to 2024-12-31.

    A date dimension is a fundamental pattern in every data warehouse.
    It allows slicing any fact table by year, quarter, month, week, day,
    weekend flag etc. — all pre-computed so dashboard queries are fast.

    ETL PARALLEL:
    In Informatica the date dimension is often pre-loaded separately
    using a sequence generator + expression transformation.
    The date_sk (surrogate key) is always in YYYYMMDD integer format
    so it can be used as a foreign key without a full date join.
    """
    section("STEP 4 — Generating Date Dimension")

    start = date(2022, 1, 1)
    end   = date(2024, 12, 31)

    rows = []
    current = start

    while current <= end:
        week_no  = current.isocalendar()[1]
        day_name = current.strftime("%A")
        is_weekend = 1 if current.weekday() >= 5 else 0

        rows.append({
            "date_sk"       : int(current.strftime("%Y%m%d")),  # e.g. 20220115
            "full_date"     : current.strftime("%Y-%m-%d"),
            "year"          : current.year,
            "quarter"       : (current.month - 1) // 3 + 1,
            "month"         : current.month,
            "month_name"    : current.strftime("%B"),
            "month_short"   : current.strftime("%b"),
            "week_no"       : week_no,
            "day_of_month"  : current.day,
            "day_of_week_no": current.weekday() + 1,       # 1=Mon, 7=Sun
            "day_name"      : day_name,
            "is_weekend"    : is_weekend,
            "is_month_start": 1 if current.day == 1 else 0,
            "is_month_end"  : 1 if current.day == calendar.monthrange(current.year, current.month)[1] else 0,
            "year_month"    : current.strftime("%Y-%m"),    # e.g. 2022-01
            "year_quarter"  : f"{current.year}-Q{(current.month-1)//3+1}",
        })

        current += timedelta(days=1)

    date_df = pd.DataFrame(rows)

    print(f"\n  Date rows generated : {len(date_df):,}")
    print(f"  Date range          : {date_df['full_date'].min()} to {date_df['full_date'].max()}")
    print(f"  Years covered       : {sorted(date_df['year'].unique())}")
    print(f"  Weekend days        : {date_df['is_weekend'].sum():,}")
    print(f"  Weekdays            : {(date_df['is_weekend']==0).sum():,}")
    print(f"\n  Sample rows:")
    print(date_df.head(5).to_string(index=False))

    date_df.to_csv(STG_DATE_PATH, index=False)
    print(f"\n  Saved to: {STG_DATE_PATH}")
    return date_df


# =============================================================================
# STEP 5 — Generate Churn Reason Dimension
# =============================================================================

def generate_churn_reason(df):
    """
    Simulates an exit survey / churn reason dimension.
    Only churned customers have a reason — active customers get 'N/A'.

    In real banking, this comes from exit interview forms or
    CRM cancellation codes. We simulate it here with realistic
    proportions based on industry churn research.

    ETL PARALLEL:
    In Informatica this would be a Lookup transformation joining
    the customer table to the exit survey system by CustomerId.
    Non-matching (active) customers route to a default 'N/A' row
    via the Lookup's default port.
    """
    section("STEP 5 — Generating Churn Reason Dimension")

    reasons = [
        ("Better rates elsewhere",    0.28),
        ("Poor customer service",     0.20),
        ("Moved to another country",  0.15),
        ("Too many fees",             0.12),
        ("Digital experience issues", 0.10),
        ("Product not suitable",      0.08),
        ("Peer recommendation",       0.07),
    ]
    reason_labels = [r[0] for r in reasons]
    reason_weights= [r[1] for r in reasons]

    churned = df[df["Exited"] == 1]
    active  = df[df["Exited"] == 0]

    churn_records = []

    # Churned customers get a random reason
    for _, row in churned.iterrows():
        churn_records.append({
            "customer_id"  : row["CustomerId"],
            "churn_flag"   : 1,
            "churn_reason" : np.random.choice(reason_labels, p=reason_weights),
            "exit_month"   : np.random.choice(["2023-01","2023-04","2023-07",
                                                "2023-10","2024-01","2024-04"]),
        })

    # Active customers get N/A
    for _, row in active.iterrows():
        churn_records.append({
            "customer_id"  : row["CustomerId"],
            "churn_flag"   : 0,
            "churn_reason" : "N/A - Active Customer",
            "exit_month"   : None,
        })

    churn_df = pd.DataFrame(churn_records)

    print(f"\n  Total rows: {len(churn_df):,}")
    print(f"\n  Churn reasons distribution (churned customers only):")
    churned_only = churn_df[churn_df["churn_flag"] == 1]
    reason_counts = churned_only["churn_reason"].value_counts()
    for reason, cnt in reason_counts.items():
        print(f"    {reason:<35} : {cnt:,}  ({cnt/len(churned_only)*100:.1f}%)")

    churn_df.to_csv(STG_CHURN_REASON_PATH, index=False)
    print(f"\n  Saved to: {STG_CHURN_REASON_PATH}")
    return churn_df


# =============================================================================
# STEP 6 — Final verification of all staging files
# =============================================================================

def verify_staging_files():
    section("STEP 6 — Verifying All Staging Files")

    files = {
        "Transactions" : STG_TRANSACTIONS_PATH,
        "Products"     : STG_PRODUCTS_PATH,
        "Date Dimension": STG_DATE_PATH,
        "Churn Reasons": STG_CHURN_REASON_PATH,
    }

    all_ok = True
    print(f"\n  {'Table':<20} {'File':<40} {'Rows':>8}  Status")
    print("  " + "-" * 78)

    for table, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {table:<20} {os.path.basename(path):<40} {len(df):>8,}  OK  ({size_kb:.0f} KB)")
        else:
            print(f"  {table:<20} {os.path.basename(path):<40} {'MISSING':>8}  FAILED")
            all_ok = False

    if all_ok:
        print(f"\n  All staging files created successfully.")
        print(f"  Location: {STAGING_DIR}")
    else:
        print(f"\n  Some files are missing — re-run this script.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 3 STEP 1 — Generate Supporting Tables")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)

    customers = load_customers()
    generate_transactions(customers)
    generate_products()
    generate_date_dimension()
    generate_churn_reason(customers)
    verify_staging_files()

    section("STEP 1 COMPLETE")
    print("\n  All supporting tables generated and saved to data/staging/")
    print("  Next: Run phase3_step2_etl_transformations.py")
    print("\n" + "=" * 65 + "\n")