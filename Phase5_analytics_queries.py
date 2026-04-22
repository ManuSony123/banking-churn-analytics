# =============================================================================
# phase5_analytics_queries.py
# PURPOSE : Run all business analytics SQL queries against the warehouse
#           and save results to CSV for Power BI import
# RUN     : python phase5_analytics_queries.py
# DEPENDS : Phase 4 must be complete (MySQL warehouse loaded)
# OUTPUT  : CSV files in data/warehouse/ for each query result
# =============================================================================
#
# QUERIES IN THIS SCRIPT:
#   Q1  — Overall churn rate by Geography and Gender
#   Q2  — Revenue at risk (balance of churned customers)
#   Q3  — Customer LTV segmentation using NTILE window function
#   Q4  — Month-over-month churn trend using LAG window function
#   Q5  — Product-wise revenue and churn using DENSE_RANK
#   Q6  — High-value at-risk customers (actionable CRM list)
#   Q7  — Credit tier churn analysis
#   Q8  — Active vs inactive member churn comparison
#   Q9  — Tenure band churn rates
#   Q10 — Executive summary KPIs (single-row summary)
#
# EACH QUERY INCLUDES:
#   - Business question it answers
#   - SQL explanation (what each clause does)
#   - Window function explanation where used
#   - Interview talking point
#   - CSV output for Power BI
# =============================================================================

import sys
import os
import pandas as pd
from sqlalchemy import text
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_engine, DB_NAME, WAREHOUSE_DIR

os.makedirs(WAREHOUSE_DIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_query(conn, sql, query_name, output_file=None):
    """
    Executes a SQL query, prints results, and saves to CSV.
    """
    print(f"\n  Running: {query_name}")
    try:
        df = pd.read_sql(text(sql), conn)
        print(f"  Rows returned: {len(df):,}")
        print(f"\n  Results:")
        print(df.to_string(index=False))

        if output_file:
            path = os.path.join(WAREHOUSE_DIR, output_file)
            df.to_csv(path, index=False)
            print(f"\n  Saved to: {path}")

        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


# =============================================================================
# Q1 — Churn Rate by Geography and Gender
# =============================================================================

SQL_Q1 = """
    SELECT
        c.geography,
        c.gender,
        COUNT(DISTINCT c.customer_id)                                    AS total_customers,
        SUM(c.churn_flag)                                                AS churned_customers,
        COUNT(DISTINCT c.customer_id) - SUM(c.churn_flag)               AS active_customers,
        ROUND(SUM(c.churn_flag) * 100.0 / COUNT(DISTINCT c.customer_id), 2) AS churn_rate_pct
    FROM dim_customer c
    WHERE c.is_current = 1
    GROUP BY c.geography, c.gender
    ORDER BY churn_rate_pct DESC
"""

# =============================================================================
# Q2 — Revenue at Risk (Balance of Churned Customers)
# =============================================================================

SQL_Q2 = """
    SELECT
        c.geography,
        c.age_group,
        c.credit_tier,
        COUNT(DISTINCT c.customer_id)               AS churned_customers,
        ROUND(SUM(c.balance), 2)                    AS total_balance_at_risk,
        ROUND(AVG(c.balance), 2)                    AS avg_balance_per_churner,
        ROUND(
            SUM(c.balance) * 100.0 /
            SUM(SUM(c.balance)) OVER (),
        2)                                          AS pct_of_total_at_risk
    FROM dim_customer c
    WHERE c.is_current = 1
      AND c.churn_flag = 1
    GROUP BY c.geography, c.age_group, c.credit_tier
    ORDER BY total_balance_at_risk DESC
    LIMIT 20
"""

# =============================================================================
# Q3 — LTV Segmentation using NTILE Window Function
#
# WINDOW FUNCTION EXPLANATION:
#   NTILE(4) divides rows into 4 equal buckets (quartiles) based on
#   estimated_ltv. The highest LTV customers are Quartile 1 (Platinum).
#   This is equivalent to a percentile ranking in Excel, but computed
#   directly in SQL without any extra steps.
# =============================================================================

SQL_Q3 = """
    WITH customer_ltv AS (
        SELECT
            c.customer_id,
            c.geography,
            c.age_group,
            c.tenure_band,
            c.churn_flag,
            c.estimated_ltv,
            c.balance,
            c.risk_category,
            NTILE(4) OVER (ORDER BY c.estimated_ltv DESC) AS ltv_quartile
        FROM dim_customer c
        WHERE c.is_current = 1
    )
    SELECT
        ltv_quartile,
        CASE ltv_quartile
            WHEN 1 THEN 'Platinum (Top 25%)'
            WHEN 2 THEN 'Gold (26-50%)'
            WHEN 3 THEN 'Silver (51-75%)'
            ELSE        'Bronze (Bottom 25%)'
        END                                           AS segment_label,
        COUNT(*)                                      AS customer_count,
        ROUND(MIN(estimated_ltv), 2)                  AS min_ltv,
        ROUND(MAX(estimated_ltv), 2)                  AS max_ltv,
        ROUND(AVG(estimated_ltv), 2)                  AS avg_ltv,
        SUM(churn_flag)                               AS churned_in_segment,
        ROUND(SUM(churn_flag) * 100.0 / COUNT(*), 2) AS segment_churn_rate_pct
    FROM customer_ltv
    GROUP BY ltv_quartile
    ORDER BY ltv_quartile
"""

# =============================================================================
# Q4 — Month-over-Month Churn Trend using LAG Window Function
#
# WINDOW FUNCTION EXPLANATION:
#   LAG(churn_rate) OVER (ORDER BY year, month) retrieves the churn_rate
#   value from the PREVIOUS row in the result set (previous month).
#   This lets us compute month-over-month change in a single query
#   without self-joins. In older SQL dialects this required joining the
#   table to itself — LAG makes it clean and efficient.
# =============================================================================

SQL_Q4 = """
    WITH monthly_churn AS (
        SELECT
            d.year,
            d.month,
            d.month_name,
            d.year_month,
            COUNT(DISTINCT f.customer_sk)                          AS total_transactions,
            SUM(f.churn_flag)                                      AS churned_txns,
            ROUND(SUM(f.churn_flag) * 100.0 / COUNT(*), 2)        AS churn_rate
        FROM fact_transactions f
        JOIN dim_date d ON f.date_sk = d.date_sk
        GROUP BY d.year, d.month, d.month_name, d.year_month
    )
    SELECT
        year,
        month,
        month_name,
        year_month,
        total_transactions,
        churned_txns,
        churn_rate,
        LAG(churn_rate) OVER (ORDER BY year, month)               AS prev_month_churn_rate,
        ROUND(
            churn_rate - LAG(churn_rate) OVER (ORDER BY year, month),
        2)                                                         AS mom_churn_change,
        CASE
            WHEN churn_rate > LAG(churn_rate) OVER (ORDER BY year, month) THEN 'Increasing'
            WHEN churn_rate < LAG(churn_rate) OVER (ORDER BY year, month) THEN 'Decreasing'
            ELSE 'Stable'
        END                                                        AS trend_direction
    FROM monthly_churn
    ORDER BY year, month
"""

# =============================================================================
# Q5 — Product Revenue and Churn Ranking using DENSE_RANK
#
# WINDOW FUNCTION EXPLANATION:
#   DENSE_RANK() assigns rank 1 to the highest revenue product, 2 to next, etc.
#   Unlike RANK(), DENSE_RANK() does not skip numbers for ties.
#   This lets the business see revenue rankings without writing an ORDER BY
#   on the entire result — rankings can coexist with groupings.
# =============================================================================

SQL_Q5 = """
    SELECT
        p.product_name,
        p.product_category,
        p.interest_rate_pct,
        COUNT(DISTINCT f.customer_sk)                         AS customer_count,
        COUNT(f.transaction_sk)                               AS transaction_count,
        ROUND(SUM(f.amount), 2)                               AS total_revenue,
        ROUND(AVG(f.amount), 2)                               AS avg_transaction_value,
        SUM(f.churn_flag)                                     AS churned_transactions,
        ROUND(SUM(f.churn_flag) * 100.0 / COUNT(*), 2)       AS product_churn_rate_pct,
        DENSE_RANK() OVER (ORDER BY SUM(f.amount) DESC)       AS revenue_rank
    FROM fact_transactions f
    JOIN dim_product p ON f.product_sk = p.product_sk
    GROUP BY p.product_name, p.product_category, p.interest_rate_pct
    ORDER BY total_revenue DESC
"""

# =============================================================================
# Q6 — High-Value At-Risk Customers (CRM Action List)
#
# This is the most ACTIONABLE query — it generates a ranked list of
# customers who have NOT yet churned but show strong churn signals.
# This is the list the CRM team uses for retention outreach.
# =============================================================================

SQL_Q6 = """
    SELECT
        c.customer_id,
        c.surname,
        c.geography,
        c.age,
        c.age_group,
        c.credit_tier,
        c.tenure,
        c.tenure_band,
        ROUND(c.balance, 2)       AS balance,
        c.num_of_products,
        c.is_active_member,
        c.risk_score,
        c.risk_category,
        ROUND(c.estimated_ltv, 2) AS estimated_ltv,
        CASE
            WHEN c.risk_score >= 7 THEN 'Immediate outreach required'
            WHEN c.risk_score >= 5 THEN 'Schedule retention call'
            WHEN c.risk_score >= 3 THEN 'Send engagement email'
            ELSE 'Monitor'
        END                       AS recommended_action
    FROM dim_customer c
    WHERE c.is_current   = 1
      AND c.churn_flag   = 0
      AND c.risk_score   >= 3
    ORDER BY c.risk_score DESC, c.balance DESC
    LIMIT 100
"""

# =============================================================================
# Q7 — Credit Tier Churn Analysis
# =============================================================================

SQL_Q7 = """
    SELECT
        c.credit_tier,
        c.high_value_flag,
        COUNT(*)                                              AS total_customers,
        SUM(c.churn_flag)                                     AS churned,
        ROUND(SUM(c.churn_flag) * 100.0 / COUNT(*), 2)       AS churn_rate_pct,
        ROUND(AVG(c.balance), 2)                              AS avg_balance,
        ROUND(AVG(c.estimated_ltv), 2)                        AS avg_ltv
    FROM dim_customer c
    WHERE c.is_current = 1
    GROUP BY c.credit_tier, c.high_value_flag
    ORDER BY c.credit_tier, c.high_value_flag
"""

# =============================================================================
# Q8 — Active vs Inactive Member Churn (ROW_NUMBER example)
#
# ROW_NUMBER() assigns a unique sequential number to each row within a partition.
# Here we use it to rank customers within each active/inactive group by balance.
# =============================================================================

SQL_Q8 = """
    WITH ranked AS (
        SELECT
            c.customer_id,
            c.is_active_member,
            c.balance,
            c.churn_flag,
            c.geography,
            ROW_NUMBER() OVER (
                PARTITION BY c.is_active_member
                ORDER BY c.balance DESC
            ) AS balance_rank_within_group
        FROM dim_customer c
        WHERE c.is_current = 1
    )
    SELECT
        CASE is_active_member WHEN 1 THEN 'Active Member' ELSE 'Inactive Member' END AS member_status,
        COUNT(*)                                                 AS total_customers,
        SUM(churn_flag)                                          AS churned,
        ROUND(SUM(churn_flag) * 100.0 / COUNT(*), 2)            AS churn_rate_pct,
        ROUND(AVG(balance), 2)                                   AS avg_balance
    FROM ranked
    GROUP BY is_active_member
    ORDER BY is_active_member DESC
"""

# =============================================================================
# Q9 — Tenure Band Churn Rates
# =============================================================================

SQL_Q9 = """
    SELECT
        c.tenure_band,
        c.geography,
        COUNT(*)                                              AS total_customers,
        SUM(c.churn_flag)                                     AS churned,
        ROUND(SUM(c.churn_flag) * 100.0 / COUNT(*), 2)       AS churn_rate_pct,
        ROUND(AVG(c.balance), 2)                              AS avg_balance,
        ROUND(AVG(c.risk_score), 2)                           AS avg_risk_score
    FROM dim_customer c
    WHERE c.is_current = 1
    GROUP BY c.tenure_band, c.geography
    ORDER BY churn_rate_pct DESC
"""

# =============================================================================
# Q10 — Executive Summary KPIs (single-row summary for dashboard header)
# =============================================================================

SQL_Q10 = """
    SELECT
        COUNT(DISTINCT c.customer_id)                              AS total_customers,
        SUM(c.churn_flag)                                          AS total_churned,
        COUNT(DISTINCT c.customer_id) - SUM(c.churn_flag)         AS total_active,
        ROUND(SUM(c.churn_flag) * 100.0 / COUNT(*), 2)            AS overall_churn_rate_pct,
        ROUND(SUM(CASE WHEN c.churn_flag=1 THEN c.balance ELSE 0 END), 2)
                                                                   AS total_revenue_at_risk,
        ROUND(AVG(CASE WHEN c.churn_flag=1 THEN c.balance END), 2)
                                                                   AS avg_churner_balance,
        SUM(CASE WHEN c.risk_category = 'Critical Risk'
                  AND c.churn_flag = 0 THEN 1 ELSE 0 END)         AS critical_risk_active_count,
        SUM(CASE WHEN c.risk_category = 'High Risk'
                  AND c.churn_flag = 0 THEN 1 ELSE 0 END)         AS high_risk_active_count,
        ROUND(AVG(c.estimated_ltv), 2)                            AS avg_customer_ltv,
        ROUND(SUM(c.estimated_ltv), 2)                            AS total_portfolio_ltv
    FROM dim_customer c
    WHERE c.is_current = 1
"""


# =============================================================================
# MAIN — Run all queries
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 5 — Analytics Queries")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)

    try:
        engine = get_engine()

        with engine.connect() as conn:
            conn.execute(text(f"USE {DB_NAME}"))

            section("Q1 — Churn Rate by Geography and Gender")
            print("  Business question: Which regions and demographics churn most?")
            run_query(conn, SQL_Q1, "Geo x Gender Churn", "q1_geo_gender_churn.csv")

            section("Q2 — Revenue at Risk by Segment")
            print("  Business question: How much balance are we losing and from which segments?")
            print("  Window function: SUM() OVER () for % of total calculation")
            run_query(conn, SQL_Q2, "Revenue at Risk", "q2_revenue_at_risk.csv")

            section("Q3 — LTV Segmentation (NTILE Window Function)")
            print("  Business question: Which value tier has the highest churn problem?")
            print("  Window function: NTILE(4) divides customers into 4 equal value buckets")
            run_query(conn, SQL_Q3, "LTV Segmentation", "q3_ltv_segmentation.csv")

            section("Q4 — Month-over-Month Churn Trend (LAG Window Function)")
            print("  Business question: Is churn improving or worsening month by month?")
            print("  Window function: LAG() retrieves the previous month's churn rate")
            run_query(conn, SQL_Q4, "MoM Churn Trend", "q4_mom_churn_trend.csv")

            section("Q5 — Product Revenue and Churn (DENSE_RANK Window Function)")
            print("  Business question: Which products drive revenue and which have churn risk?")
            print("  Window function: DENSE_RANK() ranks products by revenue without gaps")
            run_query(conn, SQL_Q5, "Product Analysis", "q5_product_analysis.csv")

            section("Q6 — High-Value At-Risk Customers (CRM Action List)")
            print("  Business question: Which active customers should we call this week?")
            run_query(conn, SQL_Q6, "CRM At-Risk List", "q6_crm_at_risk.csv")

            section("Q7 — Credit Tier x High Value Churn")
            print("  Business question: Do high-value customers churn more regardless of credit?")
            run_query(conn, SQL_Q7, "Credit Tier Churn", "q7_credit_tier_churn.csv")

            section("Q8 — Active vs Inactive Member Churn (ROW_NUMBER Window Function)")
            print("  Business question: How much does inactivity drive churn?")
            print("  Window function: ROW_NUMBER() PARTITION BY ranks within each group")
            run_query(conn, SQL_Q8, "Active vs Inactive", "q8_active_vs_inactive.csv")

            section("Q9 — Tenure Band Churn by Geography")
            print("  Business question: Are newer customers more at risk in specific regions?")
            run_query(conn, SQL_Q9, "Tenure Band Churn", "q9_tenure_band_churn.csv")

            section("Q10 — Executive Summary KPIs")
            print("  Business question: What are the headline numbers for leadership?")
            run_query(conn, SQL_Q10, "Executive KPIs", "q10_executive_kpis.csv")

        section("PHASE 5 COMPLETE")
        print(f"\n  All 10 analytics queries executed successfully.")
        print(f"  CSV results saved to: {WAREHOUSE_DIR}")
        print(f"\n  These CSVs are ready to import directly into Power BI.")
        print(f"  In Power BI: Get Data → Text/CSV → select each file.")
        print(f"\n  Alternatively, connect Power BI directly to MySQL:")
        print(f"  Get Data → MySQL Database → localhost → {DB_NAME}")
        print(f"\n  Next: Build the Power BI dashboard (Phase 6).")
        print("\n" + "=" * 65 + "\n")

    except Exception as e:
        print(f"\n  FAILED: {e}")
        print(f"  Make sure Phase 4 (warehouse build) completed successfully.")
        sys.exit(1)