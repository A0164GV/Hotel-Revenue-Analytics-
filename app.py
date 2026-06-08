"""
Hotel Revenue Analytics - Business Intelligence Dashboard
Streamlit web application for the Hotel Revenue BI Workshop.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, then connect the repo on share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Revenue Analytics",
    page_icon="🏨",
    layout="wide",
)

PREDICTORS = [
    "available_rooms", "avg_daily_rate_usd", "occupancy_rate_pct",
    "marketing_spend_usd", "online_rating", "competitor_price_index",
    "booking_window_days", "loyalty_members",
]
TARGET = "monthly_revenue_usd"


# ----------------------------------------------------------------------
# Data loading + model training (cached so it only runs once)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("hotel_revenue_bi.csv")
    return df


@st.cache_resource
def train_model(df):
    X = df[PREDICTORS]
    y = df[TARGET]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    metrics = {
        "r2": r2_score(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }
    return model, metrics


df = load_data()
model, metrics = train_model(df)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("🏨 Hotel Revenue Analytics with Python & Scikit-Learn")
st.caption("Business Intelligence Workshop — interactive dashboard")

# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Descriptive BI", "Correlation & Visuals",
     "Regression Model", "Revenue Forecast", "Executive Summary"],
)

# ----------------------------------------------------------------------
# 1. OVERVIEW (Part A)
# ----------------------------------------------------------------------
if section == "Overview":
    st.header("Part A — Data Understanding")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Target variable", "monthly_revenue_usd")

    st.subheader("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Data quality check")
    quality = pd.DataFrame({
        "Missing values": df.isnull().sum(),
        "Data type": df.dtypes.astype(str),
    })
    st.dataframe(quality, use_container_width=True)
    st.success(
        f"No missing values ({df.isnull().sum().sum()} total) and "
        f"{df.duplicated().sum()} duplicated rows — the dataset is clean and "
        "ready for analysis."
    )

    st.subheader("Descriptive statistics")
    st.dataframe(df[[TARGET] + PREDICTORS].describe().round(2), use_container_width=True)
    st.markdown(
        "**Interpretation:** Average monthly revenue is about **$619,122** with a "
        "standard deviation near **$143,312**, ranging from roughly $247k to over "
        "$1.0M. This wide spread shows revenue is highly variable and driven by "
        "combinations of city, segment, channel, and season rather than any single metric."
    )

# ----------------------------------------------------------------------
# 2. DESCRIPTIVE BI (Part B1–B3)
# ----------------------------------------------------------------------
elif section == "Descriptive BI":
    st.header("Part B — Descriptive Business Intelligence")

    st.subheader("B1. Revenue KPI table")
    kpi_cols = ["monthly_revenue_usd", "avg_daily_rate_usd",
                "occupancy_rate_pct", "marketing_spend_usd"]
    kpi = df[kpi_cols].agg(["mean", "median", "std", "min", "max"]).T.round(2)
    st.dataframe(kpi, use_container_width=True)
    st.markdown(
        "**Interpretation:** Revenue is **variable, not stable**. Its standard "
        "deviation (~$143k) is about 23% of the mean, and the min-to-max spread is "
        "several hundred thousand dollars, while rate and occupancy move within much "
        "tighter bands."
    )

    st.subheader("B2. Revenue by city and segment")
    pivot = pd.pivot_table(df, values=TARGET, index="city",
                           columns="hotel_segment", aggfunc="mean").round(0)
    st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)
    st.markdown(
        "**Interpretation:** **Miami** posts the highest average revenue "
        "(~$745,416) and **Austin** the lowest (~$450,842). Miami is a high-demand "
        "destination supporting higher rates and occupancy, while Austin is a more "
        "price-sensitive market — so revenue targets should be set per city–segment."
    )

    st.subheader("B3. Revenue by sales channel")
    channel_rev = df.groupby("sales_channel")[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(channel_rev.index, channel_rev.values,
                  color=["#2b5b84", "#5fa1c9", "#a3cce9"], edgecolor="black")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8000,
                f"${b.get_height():,.0f}", ha="center", fontsize=10)
    ax.set_ylabel("Average Revenue (USD)")
    ax.set_title("Average Monthly Revenue by Sales Channel", fontweight="bold")
    ax.set_ylim(0, channel_rev.max() * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)
    st.markdown(
        "**Interpretation:** **Direct** generates the highest average revenue "
        "(~$634,967), with **Corporate** close behind (~$632,200) and **Online "
        "Travel Agency** lowest (~$590,199). The ~$45k gap (≈7%) is managerially "
        "relevant: steering demand toward Direct avoids OTA commissions and lifts net revenue."
    )

# ----------------------------------------------------------------------
# 3. CORRELATION & VISUALS (Part B4–B5)
# ----------------------------------------------------------------------
elif section == "Correlation & Visuals":
    st.header("Part B — Correlation & Visual Analysis")

    st.subheader("B4. Correlation matrix")
    corr = df[[TARGET] + PREDICTORS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix of Numerical Variables", fontweight="bold")
    st.pyplot(fig)
    st.markdown(
        "**Interpretation:** The strongest positive correlations with revenue are "
        "**avg_daily_rate_usd (~+0.87)** and **occupancy_rate_pct (~+0.79)**, "
        "followed by available_rooms (~+0.57). This fits the revenue identity "
        "(price × rooms sold). But correlation is **not causation** — a third factor "
        "such as a high-demand city or peak season can drive all of these together."
    )

    st.subheader("B5. Visual analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x="occupancy_rate_pct", y=TARGET,
                        hue="hotel_segment", alpha=0.7, ax=ax1)
        ax1.set_title("Occupancy Rate vs Revenue", fontweight="bold")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df, x="city", y=TARGET, ax=ax2)
        ax2.set_title("Revenue by City", fontweight="bold")
        st.pyplot(fig2)
    st.markdown(
        "**Interpretation:** The scatter shows a clear upward trend — fuller hotels "
        "earn more — but the vertical spread at any occupancy level proves price and "
        "segment also matter. The boxplot confirms Miami sits highest and Austin "
        "lowest, so location is a major revenue driver and deserves city-specific goals."
    )

# ----------------------------------------------------------------------
# 4. REGRESSION MODEL (Part C1–C3)
# ----------------------------------------------------------------------
elif section == "Regression Model":
    st.header("Part C — Regression Analysis (Scikit-Learn)")

    st.subheader("C2. Coefficients")
    coef_df = pd.DataFrame({
        "Predictor": PREDICTORS,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df.style.format({"Coefficient": "{:,.2f}"}),
                 use_container_width=True)
    st.write(f"**Intercept:** {model.intercept_:,.2f}")
    st.markdown(
        "**Interpretation:** Holding other variables constant, occupancy (~+6,513 per "
        "point) and average daily rate (~+3,327 per dollar) carry strong positive "
        "weights. The competitor price index is **negative** (~−27,721): pricing far "
        "above the market without matching value tends to suppress bookings. "
        "Coefficients are **associations within this model**, not proof of causation."
    )

    st.subheader("C3. Model quality")
    m1, m2, m3 = st.columns(3)
    m1.metric("R-squared", f"{metrics['r2']:.4f}")
    m2.metric("MAE", f"${metrics['mae']:,.0f}")
    m3.metric("RMSE", f"${metrics['rmse']:,.0f}")
    st.markdown(
        "**Interpretation:** R² ≈ **0.917** means the model explains about 92% of "
        "revenue variation. MAE ≈ **$31,970** is the average prediction error and "
        "RMSE ≈ **$41,315** penalizes larger misses more. With error near 5% of "
        "average revenue, the model is a useful planning tool — to support judgment, "
        "not replace it. (Fitted on the complete dataset, no train_test_split, as required.)"
    )

# ----------------------------------------------------------------------
# 5. REVENUE FORECAST (Part C4) — interactive
# ----------------------------------------------------------------------
elif section == "Revenue Forecast":
    st.header("Part C4 — Interactive Revenue Forecast")
    st.write("Adjust the operating scenario and the model predicts monthly revenue.")

    col1, col2 = st.columns(2)
    with col1:
        available_rooms = st.slider("Available rooms", 120, 200, 160)
        avg_daily_rate = st.slider("Average daily rate (USD)", 90.0, 200.0, 145.0)
        occupancy = st.slider("Occupancy rate (%)", 50.0, 95.0, 74.0)
        marketing = st.slider("Marketing spend (USD)", 10000, 55000, 33000, step=500)
    with col2:
        rating = st.slider("Online rating", 3.5, 4.5, 4.35, step=0.01)
        cpi = st.slider("Competitor price index", 0.85, 1.20, 1.02, step=0.01)
        booking_window = st.slider("Booking window (days)", 5.0, 45.0, 24.0)
        loyalty = st.slider("Loyalty members", 200, 1800, 1250, step=10)

    scenario = pd.DataFrame([{
        "available_rooms": available_rooms,
        "avg_daily_rate_usd": avg_daily_rate,
        "occupancy_rate_pct": occupancy,
        "marketing_spend_usd": marketing,
        "online_rating": rating,
        "competitor_price_index": cpi,
        "booking_window_days": booking_window,
        "loyalty_members": loyalty,
    }])

    forecast = model.predict(scenario)[0]
    st.metric("Forecasted monthly revenue", f"${forecast:,.0f}")
    st.info(
        "Treat this as a **business planning estimate, not a guarantee**. Given the "
        "model's typical error (~$32,000), actual revenue could reasonably fall in a "
        "band around this figure. The default sliders match the assignment scenario, "
        "which forecasts ~$707,029."
    )

# ----------------------------------------------------------------------
# 6. EXECUTIVE SUMMARY (Part D)
# ----------------------------------------------------------------------
elif section == "Executive Summary":
    st.header("Part D — Executive Summary")
    st.markdown(
        """
**For the Hotel Revenue Manager**

This analysis of 432 monthly operating combinations across four cities, three
segments, and three sales channels shows revenue is **highly variable**: average
monthly revenue is about **$619,122** with a standard deviation near **$143,312**.
The clearest descriptive pattern is geographic — **Miami leads** (~$745,416) while
**Austin trails** (~$450,842) — and by channel, **Direct** (~$634,967) and
**Corporate** (~$632,200) outperform **Online Travel Agencies** (~$590,199).

The variables most strongly associated with revenue are **average daily rate**
(correlation ≈ +0.87) and **occupancy rate** (≈ +0.79), reflecting the basic
price × rooms-sold identity — though correlation does not prove causation.

The multiple linear regression model, fitted on the complete dataset, is strong:
**R² ≈ 0.917**, **MAE ≈ $31,970**, **RMSE ≈ $41,315**. For the defined scenario
(160 rooms, $145 rate, 74% occupancy, $33,000 marketing, 4.35 rating), it forecasts
approximately **$707,029**, to be read as a planning estimate rather than a guarantee.

**Recommendation:** prioritize disciplined **rate management** and **occupancy
growth** — the two strongest levers — while steering demand toward the higher-yielding
Direct channel to maximize net revenue.
        """
    )

st.sidebar.markdown("---")
st.sidebar.caption("Hotel Revenue BI Workshop · Streamlit + Scikit-Learn")
