"""
Hotel Revenue Analytics — Business Intelligence Dashboard
A polished Streamlit web application for the Hotel Revenue BI Workshop.

Run locally:   streamlit run app.py
Deploy:        push app.py, requirements.txt, and hotel_revenue_bi.csv to a
               public GitHub repo, then connect it at share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Hotel Revenue Analytics",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

PREDICTORS = [
    "available_rooms", "avg_daily_rate_usd", "occupancy_rate_pct",
    "marketing_spend_usd", "online_rating", "competitor_price_index",
    "booking_window_days", "loyalty_members",
]
TARGET = "monthly_revenue_usd"

# Palette — deep ink navy + warm brass, editorial hospitality
INK = "#10243E"
INK_SOFT = "#1B3A5B"
BRASS = "#C89B3C"
BRASS_SOFT = "#E2C275"
PAPER = "#F7F4EF"
SLATE = "#5A6B7B"
TEAL = "#2E6E6A"

# ======================================================================
# GLOBAL STYLING
# ======================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,900&family=Archivo:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Archivo', sans-serif;
}}
.stApp {{
    background:
        radial-gradient(1200px 600px at 80% -10%, rgba(200,155,60,0.10), transparent 60%),
        radial-gradient(900px 500px at -10% 110%, rgba(16,36,62,0.06), transparent 55%),
        {PAPER};
}}
h1, h2, h3 {{
    font-family: 'Fraunces', serif !important;
    color: {INK};
    letter-spacing: -0.01em;
}}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}

/* Hero */
.hero {{
    background: linear-gradient(135deg, {INK} 0%, {INK_SOFT} 100%);
    border-radius: 18px;
    padding: 2.4rem 2.6rem;
    color: {PAPER};
    position: relative;
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(16,36,62,0.28);
    margin-bottom: 1.8rem;
}}
.hero::after {{
    content: "";
    position: absolute; top: -40%; right: -10%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(200,155,60,0.35), transparent 70%);
}}
.hero h1 {{ color: {PAPER} !important; font-size: 2.5rem; margin: 0 0 .35rem 0; font-weight: 900; }}
.hero .sub {{ color: {BRASS_SOFT}; font-size: 1.02rem; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; }}
.hero .by {{ color: rgba(247,244,239,0.72); font-size: .92rem; margin-top: .9rem; }}
.brass-rule {{ height: 3px; width: 64px; background: {BRASS}; border-radius: 3px; margin: .9rem 0 0 0; }}

/* Metric cards */
.metric-card {{
    background: #FFFFFF;
    border: 1px solid rgba(16,36,62,0.08);
    border-left: 4px solid {BRASS};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 6px 18px rgba(16,36,62,0.06);
    height: 100%;
}}
.metric-card .label {{ color: {SLATE}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }}
.metric-card .value {{ color: {INK}; font-family: 'Fraunces', serif; font-size: 1.85rem; font-weight: 600; margin-top: .25rem; }}
.metric-card .note {{ color: {TEAL}; font-size: .82rem; margin-top: .2rem; font-weight: 500; }}

/* Interpretation callout */
.interp {{
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(16,36,62,0.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin: .4rem 0 1.4rem 0;
    font-size: .96rem;
    color: #29384A;
    line-height: 1.6;
}}
.interp b {{ color: {INK}; }}
.interp .tag {{
    display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.08em;
    text-transform:uppercase; color:{BRASS}; margin-bottom:.4rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {INK};
}}
section[data-testid="stSidebar"] * {{ color: {PAPER} !important; }}
section[data-testid="stSidebar"] .stRadio label {{ font-size: .96rem; }}

/* Dataframe rounding */
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}

footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# Matplotlib theme to match
mpl.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#C9D2DC",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.color": "#E7ECF1",
    "grid.linewidth": 0.8,
})
BAR_COLORS = [INK, BRASS, TEAL]


# ======================================================================
# DATA + MODEL
# ======================================================================
@st.cache_data
def load_data():
    return pd.read_csv("hotel_revenue_bi.csv")


@st.cache_resource
def train_model(df):
    X, y = df[PREDICTORS], df[TARGET]
    model = LinearRegression().fit(X, y)
    yp = model.predict(X)
    return model, {
        "r2": r2_score(y, yp),
        "mae": mean_absolute_error(y, yp),
        "rmse": np.sqrt(mean_squared_error(y, yp)),
    }


def card(label, value, note=""):
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="metric-card"><div class="label">{label}</div><div class="value">{value}</div>{note_html}</div>'


def interp(tag, html):
    st.markdown(f'<div class="interp"><div class="tag">{tag}</div>{html}</div>', unsafe_allow_html=True)


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find **hotel_revenue_bi.csv**. Make sure it is uploaded to the "
        "same folder as this app (the GitHub repo root)."
    )
    st.stop()

model, metrics = train_model(df)

# ======================================================================
# HERO
# ======================================================================
st.markdown(f"""
<div class="hero">
    <div class="sub">Business Intelligence · Scikit-Learn</div>
    <h1>Hotel Revenue Analytics</h1>
    <div class="brass-rule"></div>
    <div class="by">Pablo Maelan Giteau Valdes · A01641619 &nbsp;|&nbsp; 432 monthly observations · 4 cities · 3 segments · 3 channels</div>
</div>
""", unsafe_allow_html=True)

# ======================================================================
# SIDEBAR NAV
# ======================================================================
st.sidebar.markdown("### Navigation")
section = st.sidebar.radio(
    "",
    ["Overview", "Descriptive BI", "Correlation & Visuals",
     "Regression Model", "Revenue Forecast", "Executive Summary"],
    label_visibility="collapsed",
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(f"**Model R²**  ·  {metrics['r2']:.3f}")
st.sidebar.markdown(f"**Avg revenue**  ·  ${df[TARGET].mean():,.0f}")
st.sidebar.markdown("---")
st.sidebar.caption("Hotel Revenue BI Workshop")


# ======================================================================
# 1. OVERVIEW
# ======================================================================
if section == "Overview":
    st.header("Data Understanding")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Records", f"{df.shape[0]}", "monthly rows"), unsafe_allow_html=True)
    c2.markdown(card("Variables", f"{df.shape[1]}", "columns"), unsafe_allow_html=True)
    c3.markdown(card("Avg Revenue", f"${df[TARGET].mean():,.0f}", "per month"), unsafe_allow_html=True)
    c4.markdown(card("Data Quality", "100%", "no missing / dupes"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("First 10 rows")
    st.dataframe(df.head(10), use_container_width=True, height=380)

    interp("Interpretation",
        "The dataset holds <b>432 rows and 14 columns</b>, each a monthly operating "
        "combination of city, segment, and channel. There are <b>zero missing values "
        "and zero duplicates</b>, so no cleaning is required. The target variable is "
        "<b>monthly_revenue_usd</b>.")

    st.subheader("Descriptive statistics")
    st.dataframe(df[[TARGET] + PREDICTORS].describe().round(2), use_container_width=True)
    interp("Interpretation",
        "Average monthly revenue is about <b>$619,122</b> with a standard deviation near "
        "<b>$143,312</b>, ranging from roughly $247k to over $1.0M. This wide spread shows "
        "revenue is highly variable, driven by combinations of city, segment, channel, and "
        "season rather than any single metric.")


# ======================================================================
# 2. DESCRIPTIVE BI
# ======================================================================
elif section == "Descriptive BI":
    st.header("Descriptive Business Intelligence")

    st.subheader("Revenue KPIs")
    kpi_cols = ["monthly_revenue_usd", "avg_daily_rate_usd", "occupancy_rate_pct", "marketing_spend_usd"]
    kpi = df[kpi_cols].agg(["mean", "median", "std", "min", "max"]).T.round(2)
    st.dataframe(kpi, use_container_width=True)
    interp("Interpretation",
        "Revenue is <b>variable, not stable</b>. Its standard deviation (~$143k) is about "
        "23% of the mean and the min-to-max spread exceeds $770k, while rate and occupancy "
        "move within much tighter bands. Marketing spend varies (~$7.5k std) as the company "
        "adjusts budgets across the season.")

    st.subheader("Revenue by City & Segment")
    pivot = pd.pivot_table(df, values=TARGET, index="city", columns="hotel_segment", aggfunc="mean").round(0)
    st.dataframe(pivot.style.format("${:,.0f}").background_gradient(cmap="YlOrBr"), use_container_width=True)
    interp("Interpretation",
        "The strongest cell is <b>Miami — Conference (~$849,787)</b>; the weakest is "
        "<b>Austin — Leisure (~$393,203)</b>. Miami is a high-demand destination where the "
        "Conference segment commands premium rates, while Austin's Leisure market is more "
        "price-sensitive. Revenue targets should be set per city–segment, not uniformly.")

    st.subheader("Revenue by Sales Channel")
    ch = df.groupby("sales_channel")[TARGET].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(ch.index, ch.values, color=BAR_COLORS, edgecolor="white", linewidth=1.5, zorder=3)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 7000,
                f"${b.get_height():,.0f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylabel("Average Revenue (USD)")
    ax.set_ylim(0, ch.max() * 1.15)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    interp("Interpretation",
        "<b>Direct</b> leads (~$634,967), with <b>Corporate</b> close (~$632,200) and "
        "<b>Online Travel Agency</b> lowest (~$590,199). The ~$45k gap (≈7%) is managerially "
        "relevant: Direct avoids the 15–25% OTA commission, so steering demand toward Direct "
        "lifts net revenue.")


# ======================================================================
# 3. CORRELATION & VISUALS
# ======================================================================
elif section == "Correlation & Visuals":
    st.header("Correlation & Visual Analysis")

    st.subheader("Correlation Matrix")
    corr = df[[TARGET] + PREDICTORS].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="YlOrBr", fmt=".2f", linewidths=0.6,
                linecolor="white", vmin=-1, vmax=1, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("")
    st.pyplot(fig)
    interp("Interpretation",
        "The strongest positive correlations with revenue are <b>avg_daily_rate_usd "
        "(~+0.87)</b> and <b>occupancy_rate_pct (~+0.79)</b>, fitting the price × rooms-sold "
        "identity. But correlation is <b>not causation</b> — a third factor such as a "
        "high-demand city or peak season can drive these together.")

    st.subheader("Visual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        sns.scatterplot(data=df, x="occupancy_rate_pct", y=TARGET, hue="hotel_segment",
                        palette=[INK, BRASS, TEAL], alpha=0.75, ax=ax1)
        ax1.set_title("Occupancy vs Revenue", fontweight="bold")
        ax1.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=df, x="city", y=TARGET,
                    palette=[INK, INK_SOFT, BRASS, TEAL], ax=ax2)
        ax2.set_title("Revenue by City", fontweight="bold")
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2)
    interp("Interpretation",
        "The scatter shows a clear upward trend — fuller hotels earn more — but the vertical "
        "spread at any occupancy level proves price and segment also matter. The boxplot "
        "confirms Miami sits highest and Austin lowest, so location is a major driver and "
        "deserves city-specific goals.")


# ======================================================================
# 4. REGRESSION MODEL
# ======================================================================
elif section == "Regression Model":
    st.header("Regression Analysis · Scikit-Learn")

    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({"Predictor": PREDICTORS, "Coefficient": model.coef_}).sort_values("Coefficient", ascending=False)
    cfig, cax = plt.subplots(figsize=(9, 4.6))
    colors = [BRASS if v >= 0 else "#B4543A" for v in coef_df["Coefficient"]]
    cax.barh(coef_df["Predictor"], coef_df["Coefficient"], color=colors, edgecolor="white", zorder=3)
    cax.axvline(0, color=SLATE, linewidth=1)
    cax.set_xlabel("Coefficient (USD per unit)")
    cax.invert_yaxis()
    cax.spines[["top", "right"]].set_visible(False)
    st.pyplot(cfig)
    st.caption(f"Intercept: {model.intercept_:,.2f}")
    interp("Interpretation",
        "Holding other variables constant, <b>occupancy_rate_pct (~+6,513 per point)</b> and "
        "<b>avg_daily_rate_usd (~+3,327 per dollar)</b> carry strong positive weights. "
        "<b>competitor_price_index (~−27,721)</b> is negative — pricing far above the market "
        "without matching value suppresses bookings. These are <b>associations within the "
        "model, not proof of causation</b>.")

    st.subheader("Model Quality")
    m1, m2, m3 = st.columns(3)
    m1.markdown(card("R-squared", f"{metrics['r2']:.4f}", "≈ 92% variance explained"), unsafe_allow_html=True)
    m2.markdown(card("MAE", f"${metrics['mae']:,.0f}", "avg error"), unsafe_allow_html=True)
    m3.markdown(card("RMSE", f"${metrics['rmse']:,.0f}", "penalizes big misses"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    interp("Interpretation",
        "R² ≈ <b>0.917</b> means the model explains about 92% of revenue variation. MAE ≈ "
        "<b>$31,970</b> is the average error and RMSE ≈ <b>$41,315</b> weights larger misses "
        "more heavily. With error near 5% of average revenue, it is a useful planning tool. "
        "Fitted on the complete dataset, no train_test_split, as required.")


# ======================================================================
# 5. REVENUE FORECAST
# ======================================================================
elif section == "Revenue Forecast":
    st.header("Interactive Revenue Forecast")
    st.markdown("Adjust the operating scenario; the model re-estimates monthly revenue in real time.")

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
        "available_rooms": available_rooms, "avg_daily_rate_usd": avg_daily_rate,
        "occupancy_rate_pct": occupancy, "marketing_spend_usd": marketing,
        "online_rating": rating, "competitor_price_index": cpi,
        "booking_window_days": booking_window, "loyalty_members": loyalty,
    }])
    forecast = model.predict(scenario)[0]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{INK},{INK_SOFT});border-radius:16px;
                padding:1.8rem 2rem;color:{PAPER};box-shadow:0 14px 34px rgba(16,36,62,0.25);">
        <div style="font-size:.8rem;letter-spacing:.1em;text-transform:uppercase;color:{BRASS_SOFT};font-weight:600;">
            Forecasted Monthly Revenue</div>
        <div style="font-family:'Fraunces',serif;font-size:3rem;font-weight:900;margin-top:.2rem;">
            ${forecast:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    interp("Note",
        "Treat this as a <b>planning estimate, not a guarantee</b>. Given the model's typical "
        "error (~$32,000), actual revenue could reasonably fall in a band around this figure. "
        "The default sliders match the assignment scenario, which forecasts ~$707,029.")


# ======================================================================
# 6. EXECUTIVE SUMMARY
# ======================================================================
elif section == "Executive Summary":
    st.header("Executive Summary")
    st.markdown(f"""
<div class="interp" style="font-size:1rem;">
<div class="tag">For the Hotel Revenue Manager</div>

This analysis of 432 monthly operating combinations across four cities, three segments,
and three sales channels shows revenue is <b>highly variable</b>: average monthly revenue is
about <b>$619,122</b> with a standard deviation near <b>$143,312</b>. The clearest descriptive
pattern is geographic — <b>Miami leads</b> (Miami–Conference tops the table at ~$849,787) while
<b>Austin trails</b> (Austin–Leisure lowest at ~$393,203) — and by channel, <b>Direct</b>
(~$634,967) and <b>Corporate</b> (~$632,200) outperform <b>Online Travel Agencies</b> (~$590,199).

<br><br>The variables most strongly associated with revenue are <b>average daily rate</b>
(correlation ≈ +0.87) and <b>occupancy rate</b> (≈ +0.79), reflecting the price × rooms-sold
identity — though correlation does not prove causation.

<br><br>The multiple linear regression model, fitted on the complete dataset, is strong:
<b>R² ≈ 0.917</b>, <b>MAE ≈ $31,970</b>, <b>RMSE ≈ $41,315</b>. For the defined scenario
(160 rooms, $145 rate, 74% occupancy, $33,000 marketing, 4.35 rating), it forecasts
approximately <b>$707,029</b>, to be read as a planning estimate rather than a guarantee.

<br><br><b>Recommendation:</b> prioritize disciplined <b>rate management</b> and <b>occupancy
growth</b> — the two strongest levers — while steering demand toward the higher-yielding Direct
channel to maximize net revenue.
</div>
""", unsafe_allow_html=True)
