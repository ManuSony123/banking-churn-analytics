# =============================================================================
# config.py
# Central configuration file for Banking Customer Churn Analytics Project
# Import this file in every script to keep credentials and paths consistent
# =============================================================================

import os
import sqlalchemy

# =============================================================================
# DATABASE CONFIGURATION
# Update DB_PASSWORD with your actual MySQL root password
# =============================================================================

DB_USER     = "root"
DB_PASSWORD = "manohar"       # <-- Update this with your MySQL password
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_NAME     = "banking_churn_db"

# =============================================================================
# FILE PATHS
# All paths are relative to the project root folder
# =============================================================================

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH   = os.path.join(BASE_DIR, "data", "raw", "Customer-Churn-Records.csv")
STAGING_DIR     = os.path.join(BASE_DIR, "data", "staging")
WAREHOUSE_DIR   = os.path.join(BASE_DIR, "data", "warehouse")
REPORTS_DIR     = os.path.join(BASE_DIR, "docs")

# =============================================================================
# STAGING FILE PATHS (output files written by Phase 2 and 3 scripts)
# =============================================================================

STG_CUSTOMERS_PATH    = os.path.join(STAGING_DIR, "stg_customers.csv")
STG_TRANSACTIONS_PATH = os.path.join(STAGING_DIR, "stg_transactions.csv")
STG_PRODUCTS_PATH     = os.path.join(STAGING_DIR, "stg_products.csv")
STG_DATE_PATH         = os.path.join(STAGING_DIR, "stg_date.csv")
STG_CHURN_REASON_PATH = os.path.join(STAGING_DIR, "stg_churn_reason.csv")

# =============================================================================
# DATABASE ENGINE FACTORY
# Call get_engine() in any script that needs a DB connection
# =============================================================================

def get_engine():
    """
    Returns a SQLAlchemy engine connected to banking_churn_db.
    Usage:
        from config import get_engine
        engine = get_engine()
    """
    connection_url = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = sqlalchemy.create_engine(
        connection_url,
        pool_pre_ping=True,       # checks connection health before using it
        pool_recycle=3600         # recycles connections every 1 hour
    )
    return engine

# =============================================================================
# PROJECT CONSTANTS
# =============================================================================

# Columns we track for SCD Type 2 change detection
SCD2_TRACK_COLUMNS = ["Geography", "NumOfProducts", "IsActiveMember", "HasCrCard"]

# Churn label mapping
CHURN_LABEL_MAP = {0: "Active", 1: "Churned"}

# Credit score tier thresholds
CREDIT_BINS   = [0, 579, 669, 739, 799, 900]
CREDIT_LABELS = ["Poor", "Fair", "Good", "Very Good", "Excellent"]

# Age group thresholds
AGE_BINS   = [0, 25, 35, 50, 65, 100]
AGE_LABELS = ["Under 25", "25-35", "36-50", "51-65", "65+"]

# Tenure band thresholds
TENURE_BINS   = [0, 2, 5, 8, 10]
TENURE_LABELS = ["New (0-2 yrs)", "Growing (3-5 yrs)", "Mature (6-8 yrs)", "Loyal (9-10 yrs)"]

# Product IDs for the simulated product dimension
PRODUCT_CATALOG = {
    101: {"name": "Savings Account",  "category": "Deposit", "interest_rate": 3.5},
    102: {"name": "Fixed Deposit",    "category": "Deposit", "interest_rate": 6.8},
    103: {"name": "Personal Loan",    "category": "Loan",    "interest_rate": 12.5},
    104: {"name": "Credit Card",      "category": "Credit",  "interest_rate": 18.0},
}

if __name__ == "__main__":
    print("=" * 60)
    print("Banking Churn Analytics — Configuration Summary")
    print("=" * 60)
    print(f"Database : {DB_NAME} @ {DB_HOST}:{DB_PORT}")
    print(f"Raw Data : {RAW_DATA_PATH}")
    print(f"Staging  : {STAGING_DIR}")
    print(f"Warehouse: {WAREHOUSE_DIR}")
    print("\nTesting database connection...")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            print("MySQL connection: SUCCESS")
    except Exception as e:
        print(f"MySQL connection: FAILED\nError: {e}")
        print("Check your DB_PASSWORD in config.py")