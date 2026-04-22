# =============================================================================
# phase1_step1_test_connection.py
# PURPOSE : Verify that Python can connect to your MySQL database
# RUN     : python phase1_step1_test_connection.py
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Imports the database engine from config.py
#   2. Attempts to open a connection to MySQL
#   3. Runs a simple test query (SELECT VERSION())
#   4. Prints success or a clear error message with fix instructions
#
# ETL PARALLEL:
#   In Informatica, before running any mapping you would validate the
#   ODBC/JDBC connection in the Source/Target Analyzer. This script
#   is exactly that — a connection validation step.
# =============================================================================

import sys
import os

# Add project root to path so we can import config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_engine, DB_NAME, DB_HOST, DB_PORT, DB_USER


def test_mysql_connection():
    """
    Tests the MySQL connection and prints a detailed status report.
    Returns True if connection succeeds, False otherwise.
    """

    print("=" * 60)
    print("  PHASE 1 — MySQL Connection Test")
    print("=" * 60)
    print(f"  Host     : {DB_HOST}:{DB_PORT}")
    print(f"  User     : {DB_USER}")
    print(f"  Database : {DB_NAME}")
    print("-" * 60)

    try:
        # Step 1: Create the engine (this does NOT open a connection yet)
        print("\n[1/4] Creating SQLAlchemy engine...")
        engine = get_engine()
        print("      Engine created successfully.")

        # Step 2: Open an actual connection to MySQL
        print("\n[2/4] Opening connection to MySQL...")
        with engine.connect() as conn:
            print("      Connection opened successfully.")

            # Step 3: Run a simple query to confirm MySQL is responding
            print("\n[3/4] Running test query (SELECT VERSION())...")
            from sqlalchemy import text
            result = conn.execute(text("SELECT VERSION()"))
            version = result.fetchone()[0]
            print(f"      MySQL version: {version}")

            # Step 4: Confirm the target database exists
            print(f"\n[4/4] Checking database '{DB_NAME}' exists...")
            result = conn.execute(text(f"SHOW DATABASES LIKE '{DB_NAME}'"))
            db_exists = result.fetchone()

            if db_exists:
                print(f"      Database '{DB_NAME}' found.")
            else:
                print(f"      Database '{DB_NAME}' NOT found.")
                print(f"      Please run this in MySQL Workbench:")
                print(f"      CREATE DATABASE {DB_NAME};")
                return False

        print("\n" + "=" * 60)
        print("  CONNECTION TEST: PASSED")
        print("  You are ready to proceed to Phase 1 Step 2.")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("  CONNECTION TEST: FAILED")
        print("=" * 60)
        print(f"\n  Error: {e}")
        print("\n  Common fixes:")
        print("  1. Check DB_PASSWORD in config.py")
        print("  2. Make sure MySQL service is running")
        print("     Windows: Open Services -> MySQL80 -> Start")
        print("     Mac    : brew services start mysql")
        print("  3. Make sure MySQL port 3306 is not blocked")
        print("  4. Try connecting manually in MySQL Workbench first")
        return False


if __name__ == "__main__":
    success = test_mysql_connection()
    sys.exit(0 if success else 1)