# =============================================================================
# phase4_load_warehouse.py
# PURPOSE : Create the star schema in MySQL and load all staging data
# RUN     : python phase4_load_warehouse.py
# DEPENDS : All files in data/staging/ must exist (run Phase 3 first)
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Creates the MySQL database (if not exists)
#   2. Creates all dimension tables (dim_customer, dim_product, dim_date, dim_churn_reason)
#   3. Creates the fact table (fact_transactions)
#   4. Loads all staging CSVs into MySQL tables
#   5. Adds indexes for query performance
#   6. Runs verification queries to confirm row counts and FK integrity
#   7. Prints a warehouse build summary report
#
# STAR SCHEMA DESIGN:
#
#   dim_date ─────────────────────────────────────────────────────┐
#   dim_customer (SCD2) ──────────────────────────────────────────┤
#                                                    fact_transactions
#   dim_product ──────────────────────────────────────────────────┤
#   dim_churn_reason ─────────────────────────────────────────────┘
#
# ETL PARALLEL:
#   In Informatica, this phase maps to the Target Load Order,
#   Target Definition setup, and Session-level pre/post SQL.
#   Dimension tables MUST be loaded before the fact table
#   (same as in Informatica's target load order configuration).
# =============================================================================

import sys
import os
import pandas as pd
from sqlalchemy import text
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    get_engine, DB_NAME,
    STG_CUSTOMERS_PATH, STG_TRANSACTIONS_PATH,
    STG_PRODUCTS_PATH, STG_DATE_PATH, STG_CHURN_REASON_PATH,
    STAGING_DIR
)


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def execute_sql(conn, sql_statement, description=""):
    """Executes a single SQL statement and prints result."""
    try:
        conn.execute(text(sql_statement))
        conn.commit()
        if description:
            print(f"  OK: {description}")
    except Exception as e:
        print(f"  ERROR in {description}: {e}")
        raise


# =============================================================================
# STEP 1 — Create Database
# =============================================================================

def create_database(engine):
    section("STEP 1 — Create Database")

    with engine.connect() as conn:
        execute_sql(
            conn,
            f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci",
            f"Database '{DB_NAME}' created/verified"
        )
        execute_sql(conn, f"USE {DB_NAME}", f"Switched to database '{DB_NAME}'")


# =============================================================================
# STEP 2 — Create Dimension Tables
# =============================================================================

def create_dimension_tables(conn):
    section("STEP 2 — Creating Dimension Tables")

    # -----------------------------------------------------------------
    # dim_customer — SCD Type 2 pattern
    # Note: customer_sk is the SURROGATE key (warehouse key)
    #       customer_id is the NATURAL key (source system key)
    #       Multiple rows per customer_id when SCD2 changes occur
    # -----------------------------------------------------------------
    execute_sql(conn, "DROP TABLE IF EXISTS dim_customer", "Dropped dim_customer (if exists)")
    execute_sql(conn, """
        CREATE TABLE dim_customer (
            customer_sk           INT            NOT NULL AUTO_INCREMENT,
            customer_id           INT            NOT NULL,
            surname               VARCHAR(100),
            credit_score          INT,
            geography             VARCHAR(50),
            gender                VARCHAR(10),
            age                   INT,
            tenure                INT,
            balance               DECIMAL(15,2),
            num_of_products       INT,
            has_cr_card           TINYINT(1),
            is_active_member      TINYINT(1),
            estimated_salary      DECIMAL(15,2),
            churn_flag            TINYINT(1)     NOT NULL DEFAULT 0,
            age_group             VARCHAR(30),
            credit_tier           VARCHAR(20),
            tenure_band           VARCHAR(30),
            high_value_flag       CHAR(1)        DEFAULT 'N',
            estimated_ltv         DECIMAL(15,2),
            churn_label           VARCHAR(10),
            balance_salary_ratio  DECIMAL(10,4),
            product_engagement    INT,
            risk_score            INT,
            risk_category         VARCHAR(20),
            effective_start_date  DATE           NOT NULL,
            effective_end_date    DATE           NOT NULL DEFAULT '9999-12-31',
            is_current            TINYINT(1)     NOT NULL DEFAULT 1,
            record_version        INT            DEFAULT 1,
            created_at            TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
            Complain              INT,
            `Satisfaction Score`  INT,
            `Card Type`           VARCHAR(50),
            `Point Earned`        INT,
            PRIMARY KEY (customer_sk)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """, "Created dim_customer (SCD Type 2)")

    # -----------------------------------------------------------------
    # dim_product
    # -----------------------------------------------------------------
    execute_sql(conn, "DROP TABLE IF EXISTS dim_product", "Dropped dim_product")
    execute_sql(conn, """
        CREATE TABLE dim_product (
            product_sk        INT           NOT NULL AUTO_INCREMENT,
            product_id        INT           NOT NULL UNIQUE,
            product_name      VARCHAR(100),
            product_category  VARCHAR(50),
            interest_rate_pct DECIMAL(5,2),
            is_active         TINYINT(1)    DEFAULT 1,
            launch_year       INT,
            PRIMARY KEY (product_sk)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """, "Created dim_product")

    # -----------------------------------------------------------------
    # dim_date — full date spine
    # date_sk format: YYYYMMDD integer (e.g. 20220115)
    # This format allows fast date range filtering without string parsing
    # -----------------------------------------------------------------
    execute_sql(conn, "DROP TABLE IF EXISTS dim_date", "Dropped dim_date")
    execute_sql(conn, """
        CREATE TABLE dim_date (
            date_sk          INT           NOT NULL,
            full_date        DATE          NOT NULL,
            year             SMALLINT,
            quarter          TINYINT,
            month            TINYINT,
            month_name       VARCHAR(15),
            month_short      CHAR(3),
            week_no          TINYINT,
            day_of_month     TINYINT,
            day_of_week_no   TINYINT,
            day_name         VARCHAR(10),
            is_weekend       TINYINT(1),
            is_month_start   TINYINT(1),
            is_month_end     TINYINT(1),
            `year_month`       CHAR(7),
            year_quarter     VARCHAR(8),
            PRIMARY KEY (date_sk)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """, "Created dim_date")

    # -----------------------------------------------------------------
    # dim_churn_reason — exit survey dimension
    # -----------------------------------------------------------------
    execute_sql(conn, "DROP TABLE IF EXISTS dim_churn_reason", "Dropped dim_churn_reason")
    execute_sql(conn, """
        CREATE TABLE dim_churn_reason (
            churn_reason_sk  INT           NOT NULL AUTO_INCREMENT,
            customer_id      INT           NOT NULL,
            churn_flag       TINYINT(1),
            churn_reason     VARCHAR(100),
            exit_month       VARCHAR(8),
            PRIMARY KEY (churn_reason_sk)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """, "Created dim_churn_reason")


# =============================================================================
# STEP 3 — Create Fact Table
# =============================================================================

def create_fact_table(conn):
    section("STEP 3 — Creating Fact Table")

    # fact_transactions — one row per transaction
    # Foreign keys reference dimension surrogate keys
    # NOTE: We add FKs after loading data for performance (standard practice)
    execute_sql(conn, "DROP TABLE IF EXISTS fact_transactions", "Dropped fact_transactions")
    execute_sql(conn, """
        CREATE TABLE fact_transactions (
            transaction_sk     BIGINT         NOT NULL AUTO_INCREMENT,
            transaction_id     INT            NOT NULL,
            customer_sk        INT,
            product_sk         INT,
            date_sk            INT,
            customer_id        INT,
            transaction_date   DATE,
            transaction_year   SMALLINT,
            transaction_month  TINYINT,
            transaction_type   VARCHAR(50),
            channel            VARCHAR(50),
            amount             DECIMAL(15,2),
            churn_flag         TINYINT(1)     DEFAULT 0,
            PRIMARY KEY (transaction_sk)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """, "Created fact_transactions")


# =============================================================================
# STEP 4 — Load Dimension Tables
# =============================================================================

def load_dimensions(engine):
    section("STEP 4 — Loading Dimension Tables")

    # Helper: load a CSV to a MySQL table using pandas to_sql
    def load_table(csv_path, table_name, drop_cols=None, rename_cols=None):
        df = pd.read_csv(csv_path)
        if drop_cols:
            df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        if rename_cols:
            df.rename(columns=rename_cols, inplace=True)
        df.to_sql(table_name, engine, if_exists="append", index=False, chunksize=500)
        print(f"  Loaded {len(df):,} rows into {table_name}")
        return len(df)

    # Load dim_customer (from staging)
    stg_cust = pd.read_csv(STG_CUSTOMERS_PATH)
    stg_cust.to_sql("dim_customer", engine, if_exists="append", index=False, chunksize=500)
    print(f"  Loaded {len(stg_cust):,} rows into dim_customer")

    # Load dim_product
    load_table(STG_PRODUCTS_PATH, "dim_product")

    # Load dim_date
    load_table(STG_DATE_PATH, "dim_date")

    # Load dim_churn_reason
    load_table(STG_CHURN_REASON_PATH, "dim_churn_reason")


# =============================================================================
# STEP 5 — Load Fact Table (with dimension key lookups)
# =============================================================================

def load_fact_table(engine):
    section("STEP 5 — Loading Fact Table")

    # Load transactions CSV
    txn_df   = pd.read_csv(STG_TRANSACTIONS_PATH)
    prod_df  = pd.read_csv(STG_PRODUCTS_PATH)
    date_df  = pd.read_csv(STG_DATE_PATH)
    cust_df  = pd.read_csv(STG_CUSTOMERS_PATH)[["customer_id","customer_sk"]].drop_duplicates()

    print(f"\n  Transactions loaded: {len(txn_df):,}")

    # --- Resolve customer_sk from customer_id ---
    txn_df = txn_df.merge(
        cust_df[cust_df.duplicated("customer_id", keep="last") == False],
        on="customer_id", how="left"
    )
    print(f"  customer_sk resolved for: {txn_df['customer_sk'].notna().sum():,} rows")

    # --- Resolve product_sk from product_id ---
    prod_lookup = prod_df[["product_id"]].copy()
    prod_lookup["product_sk"] = prod_lookup.index + 1
    txn_df = txn_df.merge(prod_lookup, on="product_id", how="left")
    print(f"  product_sk resolved")

    # --- Resolve date_sk from transaction_date ---
    # date_sk format: YYYYMMDD integer
    txn_df["date_sk"] = pd.to_datetime(txn_df["transaction_date"]).dt.strftime("%Y%m%d").astype(int)

    # Select and rename for fact table
    fact_df = txn_df[[
        "transaction_id", "customer_sk", "product_sk", "date_sk",
        "customer_id", "transaction_date", "transaction_year",
        "transaction_month", "transaction_type", "channel",
        "amount", "churn_flag"
    ]].copy()

    fact_df.to_sql("fact_transactions", engine, if_exists="append",
                   index=False, chunksize=1000)
    print(f"  Loaded {len(fact_df):,} rows into fact_transactions")
    return len(fact_df)


# =============================================================================
# STEP 6 — Add Indexes for Performance
# =============================================================================

def add_indexes(conn):
    section("STEP 6 — Adding Indexes for Query Performance")

    indexes = [
        # dim_customer indexes
        ("idx_cust_customer_id",  "dim_customer",     "customer_id"),
        ("idx_cust_is_current",   "dim_customer",     "is_current"),
        ("idx_cust_churn_flag",   "dim_customer",     "churn_flag"),
        ("idx_cust_geography",    "dim_customer",     "geography"),
        ("idx_cust_risk_cat",     "dim_customer",     "risk_category"),
        # fact_transactions indexes
        ("idx_fact_customer_sk",  "fact_transactions","customer_sk"),
        ("idx_fact_product_sk",   "fact_transactions","product_sk"),
        ("idx_fact_date_sk",      "fact_transactions","date_sk"),
        ("idx_fact_churn_flag",   "fact_transactions","churn_flag"),
        ("idx_fact_txn_date",     "fact_transactions","transaction_date"),
        # dim_date index
        ("idx_date_year_month",   "dim_date",         "year, month"),
    ]

    for idx_name, table, columns in indexes:
        try:
            conn.execute(text(
                f"CREATE INDEX {idx_name} ON {table} ({columns})"
            ))
            conn.commit()
            print(f"  Created: {idx_name} ON {table}({columns})")
        except Exception as e:
            print(f"  Skipped {idx_name}: {e}")


# =============================================================================
# STEP 7 — Verification Queries
# =============================================================================

def verify_warehouse(conn):
    section("STEP 7 — Warehouse Verification")

    tables = [
        "dim_customer", "dim_product", "dim_date",
        "dim_churn_reason", "fact_transactions"
    ]

    print(f"\n  {'Table':<25} {'Row Count':>12}")
    print("  " + "-" * 40)
    for table in tables:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
        print(f"  {table:<25} {result[0]:>12,}")

    # Verify fact table FK joins
    print(f"\n  Foreign Key Join Verification:")

    fk_checks = [
        ("customer_sk",  "dim_customer",  "customer_sk"),
        ("product_sk",   "dim_product",   "product_sk"),
        ("date_sk",      "dim_date",      "date_sk"),
    ]

    for fk_col, dim_table, dim_col in fk_checks:
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM fact_transactions f
            LEFT JOIN {dim_table} d ON f.{fk_col} = d.{dim_col}
            WHERE d.{dim_col} IS NULL
        """)).fetchone()
        orphans = result[0]
        status = "OK (no orphans)" if orphans == 0 else f"WARNING: {orphans:,} orphan rows"
        print(f"  fact.{fk_col:<15} -> {dim_table:<20} : {status}")

    # Key business metric check
    churn_result = conn.execute(text("""
        SELECT
            SUM(churn_flag) AS churned,
            COUNT(*) - SUM(churn_flag) AS active,
            ROUND(SUM(churn_flag) * 100.0 / COUNT(*), 2) AS churn_rate
        FROM dim_customer
        WHERE is_current = 1
    """)).fetchone()
    print(f"\n  Business Metric Check (dim_customer, is_current=1):")
    print(f"    Churned customers : {churn_result[0]:,}")
    print(f"    Active customers  : {churn_result[1]:,}")
    print(f"    Churn rate        : {churn_result[2]}%")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 4 — MySQL Warehouse Build + Data Load")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)

    # Verify staging files exist before starting
    required_files = [
        STG_CUSTOMERS_PATH, STG_TRANSACTIONS_PATH,
        STG_PRODUCTS_PATH, STG_DATE_PATH, STG_CHURN_REASON_PATH
    ]
    for f in required_files:
        if not os.path.exists(f):
            print(f"\n  ERROR: Missing file: {f}")
            print(f"  Run Phase 3 scripts first.")
            sys.exit(1)
    print(f"\n  All staging files verified.")

    try:
        engine = get_engine()

        create_database(engine)

        with engine.connect() as conn:
            conn.execute(text(f"USE {DB_NAME}"))
            create_dimension_tables(conn)
            create_fact_table(conn)

        load_dimensions(engine)
        load_fact_table(engine)

        with engine.connect() as conn:
            conn.execute(text(f"USE {DB_NAME}"))
            add_indexes(conn)
            verify_warehouse(conn)

        section("PHASE 4 COMPLETE")
        print(f"\n  Star schema built and loaded successfully in MySQL.")
        print(f"  Database: {DB_NAME}")
        print(f"\n  Tables created:")
        print(f"    dim_customer      (SCD Type 2)")
        print(f"    dim_product")
        print(f"    dim_date")
        print(f"    dim_churn_reason")
        print(f"    fact_transactions (central fact table)")
        print(f"\n  Next: Run phase5_analytics_queries.py")
        print("\n" + "=" * 65 + "\n")

    except Exception as e:
        print(f"\n  FAILED: {e}")
        print(f"  Check your MySQL connection settings in config.py")
        sys.exit(1)