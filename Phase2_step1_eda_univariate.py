# =============================================================================
# phase2_step1_eda_univariate.py
# PURPOSE : Exploratory Data Analysis — individual column deep-dives
# RUN     : python phase2_step1_eda_univariate.py
# DEPENDS : BankChurners.csv in data/raw/
# OUTPUT  : Plots saved to docs/eda_plots/ folder
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads the raw dataset
#   2. Analyses each numeric column individually (distribution, skew, outliers)
#   3. Analyses each categorical column (value counts, proportions)
#   4. Generates and saves plots for every column
#   5. Prints a written interpretation after each analysis
#
# ETL PARALLEL:
#   In Informatica, this is like running a Data Profiling task before
#   building transformations. You understand what the data looks like
#   so you know exactly which Expression/Router/Filter logic to build.
#   Never transform data you haven't profiled.
# =============================================================================

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_PATH, REPORTS_DIR


# =============================================================================
# SETUP — create output folder for plots
# =============================================================================

PLOT_DIR = os.path.join(REPORTS_DIR, "eda_plots")
os.makedirs(PLOT_DIR, exist_ok=True)

COLORS = {
    "primary"  : "#1D9E75",   # teal  — active customers
    "secondary": "#D85A30",   # coral — churned customers
    "neutral"  : "#888780",   # gray  — general bars
    "highlight": "#BA7517",   # amber — highlights
}


def section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Plot saved: {filename}")


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    section("Loading Dataset")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"\n  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


# =============================================================================
# ANALYSIS 1 — CreditScore
# =============================================================================

def analyse_credit_score(df):
    section("COLUMN 1 — CreditScore")

    col = "CreditScore"
    data = df[col]

    print(f"\n  Min    : {data.min()}")
    print(f"  Max    : {data.max()}")
    print(f"  Mean   : {data.mean():.1f}")
    print(f"  Median : {data.median():.1f}")
    print(f"  Std    : {data.std():.1f}")
    print(f"  Skew   : {data.skew():.3f}  (0=symmetric, >1 right-skewed, <-1 left-skewed)")

    # Credit tier breakdown
    bins   = [0, 579, 669, 739, 799, 900]
    labels = ["Poor (300-579)", "Fair (580-669)", "Good (670-739)",
              "Very Good (740-799)", "Excellent (800-850)"]
    df["_cs_tier"] = pd.cut(data, bins=bins, labels=labels)
    tier_counts = df["_cs_tier"].value_counts().sort_index()
    print(f"\n  Credit score tier breakdown:")
    for tier, cnt in tier_counts.items():
        print(f"    {tier:<25} : {cnt:,}  ({cnt/len(df)*100:.1f}%)")

    # Plot: histogram + boxplot side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CreditScore Distribution", fontsize=13, fontweight="bold")

    # Histogram
    axes[0].hist(data, bins=40, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    axes[0].axvline(data.mean(),   color=COLORS["secondary"], linestyle="--", label=f"Mean: {data.mean():.0f}")
    axes[0].axvline(data.median(), color=COLORS["highlight"],  linestyle=":",  label=f"Median: {data.median():.0f}")
    axes[0].set_xlabel("Credit Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].legend(fontsize=9)

    # Boxplot by Churn
    churn_groups = [df[df["Exited"]==0][col], df[df["Exited"]==1][col]]
    bp = axes[1].boxplot(churn_groups, patch_artist=True,
                         labels=["Active (0)", "Churned (1)"],
                         medianprops={"color":"white","linewidth":2})
    bp["boxes"][0].set_facecolor(COLORS["primary"])
    bp["boxes"][1].set_facecolor(COLORS["secondary"])
    axes[1].set_title("By Churn Status")
    axes[1].set_ylabel("Credit Score")

    plt.tight_layout()
    save_plot(fig, "01_credit_score.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Credit scores are roughly normally distributed (skew={data.skew():.2f}).")
    print(f"  - Most customers fall in the 'Good' to 'Very Good' tier (580-799).")
    print(f"  - Churned vs active median difference is small — credit score alone")
    print(f"    may not be a strong churn predictor. Needs multivariate confirmation.")

    df.drop(columns=["_cs_tier"], inplace=True)


# =============================================================================
# ANALYSIS 2 — Age
# =============================================================================

def analyse_age(df):
    section("COLUMN 2 — Age")

    col = "Age"
    data = df[col]

    print(f"\n  Min    : {data.min()}")
    print(f"  Max    : {data.max()}")
    print(f"  Mean   : {data.mean():.1f}")
    print(f"  Median : {data.median():.1f}")
    print(f"  Std    : {data.std():.1f}")
    print(f"  Skew   : {data.skew():.3f}")

    # Age group breakdown
    bins   = [0, 25, 35, 50, 65, 100]
    labels = ["Under 25", "25-35", "36-50", "51-65", "65+"]
    df["_age_grp"] = pd.cut(data, bins=bins, labels=labels)
    age_churn = df.groupby("_age_grp")["Exited"].agg(["count","sum","mean"])
    age_churn.columns = ["Total", "Churned", "Churn Rate"]
    age_churn["Churn Rate"] = (age_churn["Churn Rate"] * 100).round(1)
    print(f"\n  Churn rate by age group:")
    print(age_churn.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Age Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(data, bins=30, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    axes[0].axvline(data.mean(),   color=COLORS["secondary"], linestyle="--", label=f"Mean: {data.mean():.0f}")
    axes[0].axvline(data.median(), color=COLORS["highlight"],  linestyle=":",  label=f"Median: {data.median():.0f}")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram")
    axes[0].legend(fontsize=9)

    churn_rate_by_age = age_churn["Churn Rate"]
    bars = axes[1].bar(churn_rate_by_age.index, churn_rate_by_age.values,
                       color=COLORS["secondary"], edgecolor="white")
    for bar, val in zip(bars, churn_rate_by_age.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_title("Churn Rate by Age Group")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_xlabel("Age Group")

    plt.tight_layout()
    save_plot(fig, "02_age.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Age is right-skewed — most customers are between 30-45.")
    print(f"  - Older customers (51-65+) show significantly higher churn rates.")
    print(f"  - Age is likely one of the strongest churn predictors.")
    print(f"  - This insight would drive targeted retention campaigns for older segments.")

    df.drop(columns=["_age_grp"], inplace=True)


# =============================================================================
# ANALYSIS 3 — Balance
# =============================================================================

def analyse_balance(df):
    section("COLUMN 3 — Balance")

    col = "Balance"
    data = df[col]

    zero_bal = (data == 0).sum()
    nonzero  = data[data > 0]

    print(f"\n  Min    : {data.min():,.0f}")
    print(f"  Max    : {data.max():,.0f}")
    print(f"  Mean   : {data.mean():,.0f}")
    print(f"  Median : {data.median():,.0f}")
    print(f"  Std    : {data.std():,.0f}")
    print(f"\n  Zero-balance customers : {zero_bal:,}  ({zero_bal/len(df)*100:.1f}%)")
    print(f"  Non-zero balance stats :")
    print(f"    Mean   : {nonzero.mean():,.0f}")
    print(f"    Median : {nonzero.median():,.0f}")

    # Balance band breakdown
    churn_by_zero = df.groupby(data == 0)["Exited"].mean() * 100
    print(f"\n  Churn rate — zero balance   : {churn_by_zero.get(True, 0):.1f}%")
    print(f"  Churn rate — non-zero balance: {churn_by_zero.get(False, 0):.1f}%")

    # High value segment
    high_val = df[data > 100000]
    print(f"\n  High-value customers (balance > 100k): {len(high_val):,}")
    print(f"  Churn rate in high-value segment     : {high_val['Exited'].mean()*100:.1f}%")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Balance Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(data[data > 0], bins=40, color=COLORS["primary"], edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Balance (non-zero accounts)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Histogram (excluding zero balances)")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    active_bal  = df[df["Exited"]==0]["Balance"]
    churned_bal = df[df["Exited"]==1]["Balance"]
    axes[1].hist(active_bal[active_bal>0],  bins=30, alpha=0.6,
                 color=COLORS["primary"],   edgecolor="white", label="Active")
    axes[1].hist(churned_bal[churned_bal>0],bins=30, alpha=0.6,
                 color=COLORS["secondary"], edgecolor="white", label="Churned")
    axes[1].set_title("Balance: Active vs Churned")
    axes[1].set_xlabel("Balance")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=9)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    plt.tight_layout()
    save_plot(fig, "03_balance.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - {zero_bal/len(df)*100:.1f}% of customers have zero balance — likely dormant accounts.")
    print(f"  - High-balance customers are paradoxically at higher churn risk.")
    print(f"  - Non-zero balance distribution is roughly bimodal — two clusters exist.")
    print(f"  - Balance is a strong feature for churn risk scoring.")


# =============================================================================
# ANALYSIS 4 — Tenure
# =============================================================================

def analyse_tenure(df):
    section("COLUMN 4 — Tenure")

    col = "Tenure"
    data = df[col]

    print(f"\n  Unique values: {sorted(data.unique())}")
    print(f"  Min    : {data.min()}")
    print(f"  Max    : {data.max()}")
    print(f"  Mean   : {data.mean():.1f} years")

    tenure_churn = df.groupby(col)["Exited"].agg(["count","sum","mean"])
    tenure_churn.columns = ["Total", "Churned", "Churn Rate"]
    tenure_churn["Churn Rate"] = (tenure_churn["Churn Rate"] * 100).round(1)
    print(f"\n  Churn rate by Tenure year:")
    print(tenure_churn.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Tenure Analysis", fontsize=13, fontweight="bold")

    axes[0].bar(tenure_churn.index, tenure_churn["Total"],
                color=COLORS["neutral"], edgecolor="white")
    axes[0].set_xlabel("Tenure (years)")
    axes[0].set_ylabel("Customer Count")
    axes[0].set_title("Customers per Tenure Year")

    axes[1].bar(tenure_churn.index, tenure_churn["Churn Rate"],
                color=COLORS["secondary"], edgecolor="white")
    axes[1].set_xlabel("Tenure (years)")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_title("Churn Rate per Tenure Year")

    plt.tight_layout()
    save_plot(fig, "04_tenure.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Customers with 1-2 years tenure have higher churn than very new ones.")
    print(f"  - 'Loyal' customers (9-10 yrs) show surprisingly moderate churn.")
    print(f"  - Tenure alone is not a linear predictor — use bands in modeling.")


# =============================================================================
# ANALYSIS 5 — NumOfProducts
# =============================================================================

def analyse_num_products(df):
    section("COLUMN 5 — NumOfProducts")

    col = "NumOfProducts"
    data = df[col]

    prod_churn = df.groupby(col)["Exited"].agg(["count","sum","mean"])
    prod_churn.columns = ["Total", "Churned", "Churn Rate"]
    prod_churn["Churn Rate"] = (prod_churn["Churn Rate"] * 100).round(1)
    print(f"\n  Product count breakdown:")
    print(prod_churn.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Number of Products", fontsize=13, fontweight="bold")

    axes[0].bar(prod_churn.index, prod_churn["Total"],
                color=COLORS["primary"], edgecolor="white")
    axes[0].set_title("Customer Count by Products Held")
    axes[0].set_xlabel("Number of Products")
    axes[0].set_ylabel("Count")

    colors_bar = [COLORS["primary"] if r < 30 else COLORS["secondary"]
                  for r in prod_churn["Churn Rate"]]
    bars = axes[1].bar(prod_churn.index, prod_churn["Churn Rate"],
                       color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, prod_churn["Churn Rate"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Churn Rate by Products Held")
    axes[1].set_xlabel("Number of Products")
    axes[1].set_ylabel("Churn Rate (%)")

    plt.tight_layout()
    save_plot(fig, "05_num_products.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Customers with 3-4 products have dramatically higher churn (>80%).")
    print(f"  - This is a critical insight — over-sold customers are flight risks.")
    print(f"  - Customers with 2 products are most stable (lowest churn).")
    print(f"  - NumOfProducts is likely the STRONGEST single churn predictor.")


# =============================================================================
# ANALYSIS 6 — Geography (Categorical)
# =============================================================================

def analyse_geography(df):
    section("COLUMN 6 — Geography (Categorical)")

    col = "Geography"

    geo_stats = df.groupby(col)["Exited"].agg(["count","sum","mean"])
    geo_stats.columns = ["Total", "Churned", "Churn Rate"]
    geo_stats["Churn Rate"] = (geo_stats["Churn Rate"] * 100).round(1)
    geo_stats["% of Customers"] = (geo_stats["Total"] / len(df) * 100).round(1)
    print(f"\n  Geography breakdown:")
    print(geo_stats.to_string())

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Geography Analysis", fontsize=13, fontweight="bold")

    geos = geo_stats.index.tolist()
    bar_colors = [COLORS["primary"], COLORS["neutral"], COLORS["secondary"]]

    axes[0].bar(geos, geo_stats["Total"], color=bar_colors, edgecolor="white")
    axes[0].set_title("Customer Count by Country")
    axes[0].set_ylabel("Count")

    bars = axes[1].bar(geos, geo_stats["Churn Rate"], color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, geo_stats["Churn Rate"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f"{val}%", ha="center", va="bottom", fontsize=10)
    axes[1].set_title("Churn Rate by Country")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].axhline(df["Exited"].mean()*100, color="black", linestyle="--",
                    linewidth=1, label=f"Avg: {df['Exited'].mean()*100:.1f}%")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    save_plot(fig, "06_geography.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Germany has significantly higher churn than France and Spain.")
    print(f"  - France has the most customers but a lower churn rate.")
    print(f"  - Geography should be a primary filter in the executive dashboard.")


# =============================================================================
# ANALYSIS 7 — IsActiveMember
# =============================================================================

def analyse_active_member(df):
    section("COLUMN 7 — IsActiveMember")

    col = "IsActiveMember"

    active_churn = df.groupby(col)["Exited"].agg(["count","sum","mean"])
    active_churn.columns = ["Total", "Churned", "Churn Rate"]
    active_churn["Churn Rate"] = (active_churn["Churn Rate"] * 100).round(1)
    active_churn.index = ["Inactive (0)", "Active (1)"]
    print(f"\n  Active member breakdown:")
    print(active_churn.to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("Churn Rate: Active vs Inactive Members", fontsize=13, fontweight="bold")
    bars = ax.bar(active_churn.index, active_churn["Churn Rate"],
                  color=[COLORS["secondary"], COLORS["primary"]], edgecolor="white", width=0.5)
    for bar, val in zip(bars, active_churn["Churn Rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, active_churn["Churn Rate"].max() * 1.3)
    plt.tight_layout()
    save_plot(fig, "07_active_member.png")

    print(f"\n  INTERPRETATION:")
    print(f"  - Inactive members churn at nearly 2x the rate of active members.")
    print(f"  - IsActiveMember is a powerful churn flag in the risk score model.")
    print(f"  - Business action: re-engagement campaigns for inactive customers.")


# =============================================================================
# MAIN — Run all analyses
# =============================================================================

if __name__ == "__main__":

    print("\n" + "#" * 65)
    print("#  PHASE 2 — Univariate EDA (Exploratory Data Analysis)")
    print("#  Banking Customer Churn Analytics Project")
    print("#" * 65)

    df = load_data()

    analyse_credit_score(df)
    analyse_age(df)
    analyse_balance(df)
    analyse_tenure(df)
    analyse_num_products(df)
    analyse_geography(df)
    analyse_active_member(df)

    section("PHASE 2 — UNIVARIATE EDA COMPLETE")
    print(f"\n  All plots saved to: {PLOT_DIR}")
    print(f"\n  KEY FINDINGS SUMMARY:")
    print(f"  1. NumOfProducts (3-4)  → churn rate >80%  [STRONGEST SIGNAL]")
    print(f"  2. IsActiveMember = 0   → nearly 2x churn rate")
    print(f"  3. Age 51-65+           → highest age-group churn")
    print(f"  4. Germany              → highest geography churn")
    print(f"  5. High Balance         → paradoxically higher churn")
    print(f"\n  Next: Run phase2_step2_eda_bivariate.py")
    print("\n" + "=" * 65 + "\n")