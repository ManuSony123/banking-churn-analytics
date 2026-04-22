# =============================================================================
# run_all_phases.py
# PURPOSE : Master script to run all project phases in the correct sequence
# RUN     : python run_all_phases.py
#
# This script runs every phase in order, with clear status reporting.
# Use this when you want to rebuild the entire project from scratch.
# Or run individual phase scripts for specific steps.
# =============================================================================

import subprocess
import sys
import os
import time

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

PHASES = [
    ("Phase 1 Step 1", "phase1_step1_test_connection.py",           "MySQL connection test"),
    ("Phase 1 Step 2", "phase1_step2_validate_raw_data.py",         "Raw data validation"),
    ("Phase 2 Step 1", "phase2_step1_eda_univariate.py",            "Univariate EDA"),
    ("Phase 2 Step 2", "phase2_step2_eda_bivariate.py",             "Bivariate EDA"),
    ("Phase 3 Step 1", "phase3_step1_generate_supporting_tables.py","Generate supporting tables"),
    ("Phase 3 Step 2", "phase3_step2_etl_transformations.py",       "ETL transformations"),
    ("Phase 3 Step 3", "phase3_step3_scd2_incremental.py",          "SCD Type 2 incremental"),
    ("Phase 4",        "phase4_load_warehouse.py",                  "MySQL warehouse build"),
    ("Phase 5",        "phase5_analytics_queries.py",               "Analytics queries"),
]


def run_phase(label, script_file, description):
    script_path = os.path.join(SCRIPTS_DIR, script_file)

    print(f"\n{'='*65}")
    print(f"  RUNNING: {label} — {description}")
    print(f"{'='*65}")

    if not os.path.exists(script_path):
        print(f"  SKIP: Script not found: {script_path}")
        return "SKIP"

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  DONE: {label} completed in {elapsed:.1f}s")
        return "PASS"
    else:
        print(f"\n  FAILED: {label} exited with code {result.returncode}")
        return "FAIL"


if __name__ == "__main__":

    print("\n" + "#"*65)
    print("#  BANKING CHURN ANALYTICS — FULL PROJECT RUN")
    print("#  Running all phases in sequence")
    print("#"*65)

    results = []
    for label, script, description in PHASES:
        status = run_phase(label, script, description)
        results.append((label, description, status))
        if status == "FAIL":
            print(f"\n  Stopping run — {label} failed.")
            print(f"  Fix the issue above and re-run, or run that script individually.")
            break

    print("\n\n" + "="*65)
    print("  FINAL RUN SUMMARY")
    print("="*65)
    print(f"\n  {'Phase':<18} {'Description':<40} {'Status'}")
    print("  " + "-"*65)
    for label, desc, status in results:
        icon = "OK  " if status == "PASS" else "FAIL" if status == "FAIL" else "SKIP"
        print(f"  {label:<18} {desc:<40} {icon}")
    print()