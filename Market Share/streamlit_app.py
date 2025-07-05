# streamlit_app.py
# -----------------------------------------------------------
# Interactive market-share dashboard (Retail example)
# Now robust to missing default CSV – falls back to a file-uploader
# *Underlying data table removed per user request*
# -----------------------------------------------------------
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from typing import Union, IO

# ---------- page config ----------
st.set_page_config(
    page_title="Retail Market-Share dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- data loader ----------
BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "alpha_vantage_retail_revenues.csv"

@st.cache_data(show_spinner=False)
def load_data(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    """Read the CSV from *source* and add a Fiscal Year column."""
    df = pd.read_csv(source)
    df["Fiscal Date"] = pd.to_datetime(df["Fiscal Date"], errors="coerce")
    df["Fiscal Year"] = df["Fiscal Date"].dt.year
    return df

# ------------------------------------------------------------------
# Helper functions (matching notebook)
# ------------------------------------------------------------------

def Market_Share_Calculations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Market Share"] = df["Revenue (USD)"] / df.groupby("Fiscal Year")["Revenue (USD)"].transform("sum")
    return df


def top_companies(df: pd.DataFrame, top_n: int = 5):
    return (
        df.groupby("Ticker")["Market Share"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )


def plot_market_share_over_time(df: pd.DataFrame, focus: list[str]):
    long_df = (
        df[df["Ticker"].isin(focus)]
        .pivot_table(index="Fiscal Date", columns="Ticker", values="Market Share")
        .reset_index()
        .melt(id_vars="Fiscal Date", var_name="Ticker", value_name="Market Share")
    )
    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x="Fiscal Date:T",
            y=alt.Y("Market Share:Q", axis=alt.Axis(format=".0%")),
            color="Ticker:N",
            tooltip=["Fiscal Date:T", "Ticker", alt.Tooltip("Market Share:Q", format=".2%")],
        )
        .properties(height=450)
        .interactive()
    )


def plot_market_share_stacked_bar(df: pd.DataFrame, focus: list[str], year: int):
    year_df = df[df["Fiscal Year"] == year].copy()
    focus_mask = year_df["Ticker"].isin(focus)
    df_focus = year_df[focus_mask]
    df_other = year_df[~focus_mask]

    other_share = df_other["Market Share"].sum()
    if other_share > 0:
        df_focus = pd.concat(
            [
                df_focus,
                pd.DataFrame({"Ticker": ["Other"], "Market Share": [other_share], "Fiscal Year": [year]}),
            ],
            ignore_index=True,
        )

    order = (
        df_focus[df_focus["Ticker"] != "Other"].sort_values("Market Share", ascending=False)["Ticker"].tolist()
    )
    if "Other" in df_focus["Ticker"].values:
        order.append("Other")

    return (
        alt.Chart(df_focus)
        .mark_bar()
        .encode(
            y=alt.Y("Ticker:N", sort=order),
            x=alt.X("Market Share:Q", axis=alt.Axis(format=".0%")),
            color=alt.Color("Ticker:N", legend=None),
            tooltip=["Ticker", alt.Tooltip("Market Share:Q", format=".2%")],
        )
        .properties(height=max(300, 25 * len(df_focus)))
    )

# ------------------------------------------------------------------
# Data acquisition – try default path, else request upload
# ------------------------------------------------------------------
try:
    df_raw = load_data(DEFAULT_CSV)
except FileNotFoundError:
    st.sidebar.warning(
        f"Default data file not found at '{DEFAULT_CSV}'.\n" "Please upload the CSV exported from Alpha Vantage."
    )
    uploaded = st.sidebar.file_uploader("Upload alpha_vantage_retail_revenues.csv", type="csv")
    if uploaded is None:
        st.stop()
    df_raw = load_data(uploaded)

market = Market_Share_Calculations(df_raw)

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
all_tickers = sorted(market["Ticker"].unique())
sel_tickers = st.sidebar.multiselect(
    "Focus companies (highlight individually)",
    options=all_tickers,
    default=top_companies(market, 5) if len(all_tickers) >= 5 else all_tickers,
)

years = sorted([int(y) for y in market["Fiscal Year"].dropna().unique()])

yr_min, yr_max = st.sidebar.slider(
    "Year range (for line chart)",
    min_value=min(years),
    max_value=max(years),
    value=(min(years), max(years)),
)

chart_style = st.sidebar.radio("Chart type", ["Market share over time", "Stacked share by year"])

# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------
st.title("Retail market-share explorer")

filtered = market[(market["Fiscal Year"] >= yr_min) & (market["Fiscal Year"] <= yr_max)].copy()

if chart_style == "Market share over time":
    st.altair_chart(plot_market_share_over_time(filtered, sel_tickers), use_container_width=True)
else:
    sel_year = st.sidebar.selectbox("Year for stacked bar", years, index=years.index(yr_max))
    st.altair_chart(plot_market_share_stacked_bar(market, sel_tickers, int(sel_year)), use_container_width=True)

# ------------------------------------------------------------------
# Underlying data display removed as per request
# ------------------------------------------------------------------
