"""
COVID-19 Data Analysis & Visualization
Author: Santhosh S
Description: EDA on global COVID-19 dataset with 15+ visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Generate Synthetic COVID-19 Data ───────────────────────────────────────
def generate_covid_data(n_countries=50, seed=42):
    """
    Generate synthetic global COVID-19 dataset.
    Replace this with: df = pd.read_csv('covid_data.csv')
    Real dataset: https://github.com/owid/covid-19-data
    """
    np.random.seed(seed)

    countries = [
        "USA", "India", "Brazil", "Russia", "UK", "France", "Germany", "Italy",
        "Spain", "South Korea", "Japan", "Australia", "Canada", "Mexico", "Argentina",
        "Colombia", "Poland", "Netherlands", "Turkey", "Saudi Arabia", "South Africa",
        "Indonesia", "Philippines", "Vietnam", "Thailand", "Malaysia", "Bangladesh",
        "Pakistan", "Nigeria", "Kenya", "Egypt", "Morocco", "Iran", "Iraq", "UAE",
        "Singapore", "New Zealand", "Sweden", "Norway", "Denmark", "Finland",
        "Austria", "Switzerland", "Belgium", "Portugal", "Greece", "Romania",
        "Hungary", "Czech Republic", "Slovakia"
    ][:n_countries]

    continents = {
        "USA": "Americas", "India": "Asia", "Brazil": "Americas", "Russia": "Europe",
        "UK": "Europe", "France": "Europe", "Germany": "Europe", "Italy": "Europe",
        "Spain": "Europe", "South Korea": "Asia", "Japan": "Asia", "Australia": "Oceania",
        "Canada": "Americas", "Mexico": "Americas", "Argentina": "Americas",
        "Colombia": "Americas", "Poland": "Europe", "Netherlands": "Europe",
        "Turkey": "Europe", "Saudi Arabia": "Asia", "South Africa": "Africa",
        "Indonesia": "Asia", "Philippines": "Asia", "Vietnam": "Asia", "Thailand": "Asia",
        "Malaysia": "Asia", "Bangladesh": "Asia", "Pakistan": "Asia", "Nigeria": "Africa",
        "Kenya": "Africa", "Egypt": "Africa", "Morocco": "Africa", "Iran": "Asia",
        "Iraq": "Asia", "UAE": "Asia", "Singapore": "Asia", "New Zealand": "Oceania",
        "Sweden": "Europe", "Norway": "Europe", "Denmark": "Europe", "Finland": "Europe",
        "Austria": "Europe", "Switzerland": "Europe", "Belgium": "Europe",
        "Portugal": "Europe", "Greece": "Europe", "Romania": "Europe", "Hungary": "Europe",
        "Czech Republic": "Europe", "Slovakia": "Europe",
    }

    # Date range: Jan 2020 – Dec 2022
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="W")
    records = []

    for country in countries:
        pop_base = np.random.randint(5_000_000, 1_400_000_000)
        base_rate = np.random.uniform(0.0001, 0.005)
        vacc_start = np.random.randint(40, 70)  # weeks from start

        cumulative_cases = 0
        cumulative_deaths = 0
        cumulative_vacc = 0

        for i, date in enumerate(dates):
            # Wave simulation (3 waves)
            wave = (np.sin(i / 26 * np.pi) ** 2 +
                    0.5 * np.sin(i / 13 * np.pi + 1) ** 2 +
                    0.3 * np.sin(i / 8 * np.pi + 2) ** 2)
            new_cases = int(pop_base * base_rate * wave * np.random.uniform(0.5, 1.5))
            new_deaths = int(new_cases * np.random.uniform(0.005, 0.03))
            new_vacc = int(pop_base * np.random.uniform(0, 0.015)) if i > vacc_start else 0

            cumulative_cases  += new_cases
            cumulative_deaths += new_deaths
            cumulative_vacc   = min(cumulative_vacc + new_vacc, pop_base)

            records.append({
                "date":             date,
                "country":          country,
                "continent":        continents.get(country, "Unknown"),
                "population":       pop_base,
                "new_cases":        new_cases,
                "new_deaths":       new_deaths,
                "total_cases":      cumulative_cases,
                "total_deaths":     cumulative_deaths,
                "total_vaccinated": cumulative_vacc,
                "vacc_rate":        round(cumulative_vacc / pop_base * 100, 2),
            })

    df = pd.DataFrame(records)
    df["death_rate"]    = round(df["total_deaths"] / df["total_cases"].replace(0, np.nan) * 100, 3)
    df["cases_per_1m"]  = round(df["total_cases"] / df["population"] * 1_000_000, 1)
    df["deaths_per_1m"] = round(df["total_deaths"] / df["population"] * 1_000_000, 1)
    return df


# ── 2. Data Cleaning ──────────────────────────────────────────────────────────
def clean_data(df):
    print(f"Raw shape: {df.shape}")
    # Handle nulls
    df["death_rate"].fillna(0, inplace=True)
    df.dropna(subset=["country", "date"], inplace=True)
    # Remove outliers in new_cases (negative values)
    df = df[df["new_cases"] >= 0]
    df = df[df["new_deaths"] >= 0]
    print(f"Clean shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Total data points: {len(df):,}")
    return df


# ── 3. Statistical Summary ────────────────────────────────────────────────────
def statistical_summary(df):
    latest = df.groupby("country").last().reset_index()
    print("\n" + "=" * 60)
    print("  GLOBAL COVID-19 STATISTICS (Cumulative)")
    print("=" * 60)
    print(f"  Total Cases     : {latest['total_cases'].sum():>15,.0f}")
    print(f"  Total Deaths    : {latest['total_deaths'].sum():>15,.0f}")
    print(f"  Avg Death Rate  : {latest['death_rate'].mean():>14.2f}%")
    print(f"  Max Vacc Rate   : {latest['vacc_rate'].max():>14.2f}%")
    print(f"  Avg Vacc Rate   : {latest['vacc_rate'].mean():>14.2f}%")
    return latest


# ── 4. Visualizations (15+) ───────────────────────────────────────────────────
def plot_all(df, latest):
    # ── Figure 1: Overview (6 plots) ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("COVID-19 Global Analysis — Overview", fontsize=16, fontweight="bold")

    global_weekly = df.groupby("date")[["new_cases", "new_deaths"]].sum().reset_index()

    # Plot 1: Global weekly cases trend
    axes[0, 0].fill_between(global_weekly["date"], global_weekly["new_cases"] / 1e6,
                             alpha=0.6, color="#2196F3")
    axes[0, 0].plot(global_weekly["date"], global_weekly["new_cases"] / 1e6, color="#1565C0", lw=1)
    axes[0, 0].set_title("Global Weekly New Cases (Millions)")
    axes[0, 0].set_ylabel("Cases (M)")

    # Plot 2: Weekly deaths
    axes[0, 1].fill_between(global_weekly["date"], global_weekly["new_deaths"] / 1e3,
                             alpha=0.6, color="#F44336")
    axes[0, 1].set_title("Global Weekly Deaths (Thousands)")
    axes[0, 1].set_ylabel("Deaths (K)")

    # Plot 3: Top 10 countries by total cases
    top10 = latest.nlargest(10, "total_cases")[["country", "total_cases"]]
    axes[0, 2].barh(top10["country"], top10["total_cases"] / 1e6, color="#2196F3", alpha=0.85)
    axes[0, 2].set_title("Top 10 Countries — Total Cases (M)")
    axes[0, 2].set_xlabel("Cases (Millions)")

    # Plot 4: Cases per million (fair comparison)
    top10_pm = latest.nlargest(10, "cases_per_1m")[["country", "cases_per_1m"]]
    axes[1, 0].barh(top10_pm["country"], top10_pm["cases_per_1m"], color="#FF9800", alpha=0.85)
    axes[1, 0].set_title("Top 10 — Cases per Million Population")
    axes[1, 0].set_xlabel("Cases per Million")

    # Plot 5: Death rate distribution
    axes[1, 1].hist(latest["death_rate"].clip(0, 5), bins=20, color="#9C27B0", alpha=0.8, edgecolor="white")
    axes[1, 1].set_title("Death Rate Distribution (%)")
    axes[1, 1].set_xlabel("Death Rate (%)")
    axes[1, 1].set_ylabel("Countries")

    # Plot 6: Vaccination rate vs death rate
    axes[1, 2].scatter(latest["vacc_rate"], latest["death_rate"],
                       alpha=0.7, c="#4CAF50", s=60, edgecolors="#1B5E20", lw=0.5)
    axes[1, 2].set_title("Vaccination Rate vs Death Rate")
    axes[1, 2].set_xlabel("Vaccination Rate (%)")
    axes[1, 2].set_ylabel("Death Rate (%)")
    m, b = np.polyfit(latest["vacc_rate"].fillna(0), latest["death_rate"].fillna(0), 1)
    x_line = np.linspace(0, 100, 100)
    axes[1, 2].plot(x_line, m * x_line + b, color="red", lw=1.5, ls="--", label="Trend")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("covid_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: covid_overview.png")

    # ── Figure 2: Continental & Heatmap (4 plots) ─────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 12))
    fig2.suptitle("COVID-19 — Continental & Temporal Analysis", fontsize=16, fontweight="bold")

    # Plot 7: Cases by continent
    cont = latest.groupby("continent")[["total_cases", "total_deaths"]].sum().reset_index()
    x = np.arange(len(cont))
    axes2[0, 0].bar(x - 0.2, cont["total_cases"] / 1e6, 0.4, label="Cases (M)", color="#2196F3", alpha=0.85)
    axes2[0, 0].bar(x + 0.2, cont["total_deaths"] / 1e3, 0.4, label="Deaths (K)", color="#F44336", alpha=0.85)
    axes2[0, 0].set_xticks(x)
    axes2[0, 0].set_xticklabels(cont["continent"], rotation=15)
    axes2[0, 0].set_title("Cases & Deaths by Continent")
    axes2[0, 0].legend()
    axes2[0, 0].set_ylabel("Count (M / K)")

    # Plot 8: Monthly heatmap (top 15 countries)
    top15 = latest.nlargest(15, "total_cases")["country"].tolist()
    df_top = df[df["country"].isin(top15)].copy()
    df_top["month_year"] = df_top["date"].dt.to_period("Q").astype(str)
    pivot = df_top.pivot_table(values="new_cases", index="country",
                               columns="month_year", aggfunc="sum") / 1e3
    sns.heatmap(pivot, ax=axes2[0, 1], cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "New Cases (K)"})
    axes2[0, 1].set_title("Quarterly Cases Heatmap — Top 15 Countries")
    axes2[0, 1].set_ylabel("")
    axes2[0, 1].tick_params(axis="x", rotation=45)

    # Plot 9: Wave analysis — 3 selected countries
    selected = ["USA", "India", "UK"]
    colors_s = ["#2196F3", "#FF9800", "#F44336"]
    for country, color in zip(selected, colors_s):
        cdf = df[df["country"] == country].set_index("date")["new_cases"]
        axes2[1, 0].plot(cdf, lw=1.5, label=country, color=color, alpha=0.85)
    axes2[1, 0].set_title("COVID-19 Waves — USA vs India vs UK")
    axes2[1, 0].set_ylabel("Weekly New Cases")
    axes2[1, 0].legend()

    # Plot 10: Vaccination rollout
    for country, color in zip(selected, colors_s):
        vdf = df[df["country"] == country].set_index("date")["vacc_rate"]
        axes2[1, 1].plot(vdf, lw=1.5, label=country, color=color, alpha=0.85)
    axes2[1, 1].set_title("Vaccination Rate Over Time (%)")
    axes2[1, 1].set_ylabel("% Population Vaccinated")
    axes2[1, 1].legend()
    axes2[1, 1].yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.savefig("covid_deep_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: covid_deep_analysis.png")

    # ── Figure 3: Recovery & Mortality (5 plots) ──────────────────────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("COVID-19 — Mortality & Impact Analysis", fontsize=15, fontweight="bold")

    # Plot 11: Deaths per million
    top10_dm = latest.nlargest(10, "deaths_per_1m")[["country", "deaths_per_1m"]]
    axes3[0].barh(top10_dm["country"], top10_dm["deaths_per_1m"], color="#F44336", alpha=0.85)
    axes3[0].set_title("Deaths per Million — Top 10")
    axes3[0].set_xlabel("Deaths per Million")

    # Plot 12: Continent vaccination box plot
    sns.boxplot(data=latest, x="continent", y="vacc_rate",
                palette="Set2", ax=axes3[1])
    axes3[1].set_title("Vaccination Rate by Continent")
    axes3[1].set_ylabel("Vaccination Rate (%)")
    axes3[1].tick_params(axis="x", rotation=15)

    # Plot 13: Correlation heatmap
    corr_cols = ["total_cases", "total_deaths", "total_vaccinated", "death_rate", "vacc_rate", "cases_per_1m"]
    corr = latest[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=axes3[2], linewidths=0.5)
    axes3[2].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    plt.savefig("covid_mortality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Saved: covid_mortality.png")
    print("\n✓ Total: 13 visualizations across 3 figures saved.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  COVID-19 Data Analysis & Visualization")
    print("=" * 55)

    print("\n[1/4] Generating global COVID-19 dataset...")
    df = generate_covid_data()

    print("\n[2/4] Cleaning data...")
    df = clean_data(df)

    print("\n[3/4] Statistical summary...")
    latest = statistical_summary(df)

    print("\n[4/4] Creating 13+ visualizations...")
    plot_all(df, latest)

    print("\n✓ Analysis complete!")
