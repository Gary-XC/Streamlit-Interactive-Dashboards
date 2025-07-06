import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from typing import Union, IO

###
# Note, have to update individual Apps on StreamLit's Website
# https://share.streamlit.io/
###

# ---------- page config ----------
st.set_page_config(
    page_title="Retail Market‑Share dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- data loader ----------
BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "alpha_vantage_retail_revenues.csv"

@st.cache_data(show_spinner=False)
def load_data(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["Fiscal Date"] = pd.to_datetime(df["Fiscal Date"], errors="coerce")
    df["Fiscal Year"] = df["Fiscal Date"].dt.year
    return df

# ---------- helper functions ----------

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


def plot_yearly_stacked_bars_vertical(df: pd.DataFrame, focus: list[str], yr_min: int, yr_max: int):
    """One 100 % stacked bar **per fiscal year**, arranged vertically."""
    subset = df[(df["Fiscal Year"] >= yr_min) & (df["Fiscal Year"] <= yr_max)].copy()

    # Collapse non‑focus tickers into "Other" to keep colour legend concise
    subset["Ticker"] = subset["Ticker"].where(subset["Ticker"].isin(focus), "Other")
    subset["Bar"] = "Share"  # constant nominal field for bar height control

    base = (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            x=alt.X(
                "Market Share:Q",
                stack="normalize",
                axis=alt.Axis(format=".0%", title="Market Share"),
            ),
            y=alt.Y("Bar:N", axis=None),  # hide y‑axis — single bar row
            color=alt.Color("Ticker:N", legend=alt.Legend(title="Ticker")),
            tooltip=[
                "Fiscal Year:O",
                "Ticker",
                alt.Tooltip("Market Share:Q", format=".2%"),
            ],
        )
        .properties(height=40)  # compact row height
    )

    # Facet vertically (row) so each year stacks **on top of** the next
    facet_chart = base.facet(
        row=alt.Row(
            "Fiscal Year:O",
            sort=list(range(yr_max, yr_min - 1, -1)),  # newest at top
            header=alt.Header(labelAngle=0, labelAlign="right", title="Fiscal Year"),
        ),
        spacing=4,
    )

    return facet_chart

# ---------- load data ----------
try:
    df_raw = load_data(DEFAULT_CSV)
except FileNotFoundError:
    st.sidebar.warning(
        f"Default data file not found at '{DEFAULT_CSV}'.\nPlease upload the CSV exported from Alpha Vantage."
    )
    uploaded = st.sidebar.file_uploader("Upload alpha_vantage_retail_revenues.csv", type="csv")
    if uploaded is None:
        st.stop()
    df_raw = load_data(uploaded)

market = Market_Share_Calculations(df_raw)

# ---------- sidebar controls ----------
all_tickers = sorted(market["Ticker"].unique())
sel_tickers = st.sidebar.multiselect(
    "Focus companies (highlight individually)",
    options=all_tickers,
    default=top_companies(market, 5) if len(all_tickers) >= 5 else all_tickers,
)

years = sorted([int(y) for y in market["Fiscal Year"].dropna().unique()])

yr_min, yr_max = st.sidebar.slider(
    "Year range", min_value=min(years), max_value=max(years), value=(min(years), max(years))
)

chart_style = st.sidebar.radio(
    "Chart type",
    [
        "Market share over time",
        "Stacked bars (vertical by year)",
    ],
)

# ---------- filter data ----------
filtered = market[(market["Fiscal Year"] >= yr_min) & (market["Fiscal Year"] <= yr_max)].copy()

# ---------- main content ----------
st.title("Retail market‑share explorer")

if chart_style == "Market share over time":
    st.altair_chart(plot_market_share_over_time(filtered, sel_tickers), use_container_width=True)
else:
    st.altair_chart(
        plot_yearly_stacked_bars_vertical(market, sel_tickers, yr_min, yr_max),
        use_container_width=True,
    )
