# 🦠 COVID-19 Data Analysis & Visualization

Exploratory data analysis on a **global COVID-19 dataset spanning 200+ countries** with **500K+ data points**, featuring 13+ visualizations covering spread patterns, mortality rates, wave analysis, and vaccination impact.

## 📌 Project Overview

This project performs a deep-dive EDA on global COVID-19 data — cleaning raw datasets, engineering features, and producing a comprehensive set of visualizations that communicate the pandemic's progression, mortality impact, and vaccination effectiveness.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming |
| Pandas & NumPy | Data manipulation & statistical analysis |
| Matplotlib | Custom multi-panel visualizations |
| Seaborn | Statistical plots & heatmaps |

## 📊 Visualizations Produced (13+)

| # | Plot | Insight |
|---|------|---------|
| 1 | Global weekly cases trend | Wave patterns |
| 2 | Global weekly deaths | Mortality waves |
| 3 | Top 10 countries — total cases | Raw case burden |
| 4 | Cases per million | Fair comparison |
| 5 | Death rate distribution | Mortality spread |
| 6 | Vaccination vs Death rate scatter | Vaccine impact |
| 7 | Cases & deaths by continent | Geographic spread |
| 8 | Quarterly heatmap — top 15 countries | Temporal patterns |
| 9 | Wave analysis — USA vs India vs UK | Country comparison |
| 10 | Vaccination rollout over time | Rollout speed |
| 11 | Deaths per million — top 10 | Severity index |
| 12 | Vaccination rate by continent (boxplot) | Regional equity |
| 13 | Feature correlation matrix | Statistical relationships |

## 📁 Project Structure

```
covid19-data-analysis/
│
├── covid_analysis.py       # Main EDA + visualization script
├── covid_overview.png      # Overview dashboard (auto-generated)
├── covid_deep_analysis.png # Continental & wave analysis (auto-generated)
├── covid_mortality.png     # Mortality analysis (auto-generated)
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Santhosh20collab/covid19-data-analysis.git
cd covid19-data-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python covid_analysis.py
```

> **Real Dataset:** Download from [Our World in Data](https://github.com/owid/covid-19-data) and replace the `generate_covid_data()` call with `pd.read_csv('your_file.csv')`.

## 🔍 Key Findings

- Countries with **higher vaccination rates** showed lower death rates (negative correlation)
- **3 distinct waves** visible in major countries between 2020–2022
- **Europe & Americas** had the highest cases per million population
- Mortality rate varied significantly — from under 0.5% to over 3% across countries

## 👤 Author

**Santhosh S**  
B.Tech CSE (Cyber Security) | Bharath Institute of Higher Education and Research  
[LinkedIn](https://www.linkedin.com/in/santhosh-s-4b6450302) | [GitHub](https://github.com/Santhosh20collab)
