import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
from typing import Union, IO, List, Optional

# ---------- page config ----------
st.set_page_config(
    page_title="Retail Market-Share dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h1 style='text-align:center;'>Market Share Dashboard</h1>",
    unsafe_allow_html=True,
)

# ---------- data loader ----------
BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "Data" / "StreamLit Data.csv"

@st.cache_data(show_spinner=False)
def load_data(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    df = pd.read_csv(source)
    df["Fiscal Date"] = pd.to_datetime(df["Fiscal Date"], errors="coerce")
    df["Fiscal Year"] = df["Fiscal Date"].dt.year
    return df

def top_companies(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    return (
        df.groupby("Ticker")["Market Share"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )


# ---------- Altair line chart ----------

def plot_market_share_over_time(df: pd.DataFrame, focus: List[str]):
    long_df = df[df["Ticker"].isin(focus)].copy()
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

def plot_market_share_stacked_bar(
    data: pd.DataFrame,
    years: Optional[List[int]] = None,
    include_tickers: Optional[List[str]] = None,
    value_col: str = "Market Share",
):
    """Return a matplotlib Figure with horizontal stacked bars (one per year)."""

    # ---------- filtering -------------------------------------------------
    df = data.copy()
    df["Fiscal Date"] = pd.to_datetime(df["Fiscal Date"])
    df["Year"] = df["Fiscal Date"].dt.year

    if years is not None:
        years = [years] if isinstance(years, int) else list(years)
        df = df[df["Year"].isin(years)]

    # ---------- choose focus tickers ------------------------------------
    focus_mask = df["Ticker"].isin(include_tickers) if include_tickers is not None else pd.Series(True, index=df.index)
    df_focus = df[focus_mask]
    
    if df_focus.empty:
        st.warning("No data available for the selected years and companies.")
        return None  # Skip plotting

    # ---------- aggregate ----------------------------------------------
    yearly_focus = (
        df_focus.groupby(["Year", "Ticker"], as_index=False)[value_col].mean()
    )
    pivot_focus = (
        yearly_focus.pivot(index="Year", columns="Ticker", values=value_col).fillna(0)
    )

    # ---------- add the “Other” bucket ----------------------------------
    other_share = 1.0 - pivot_focus.sum(axis=1)
    other_share = other_share.clip(lower=0)
    pivot_focus["Other"] = other_share

    # ---------- order columns -------------------------------------------
    order = (
        pivot_focus.drop(columns="Other")
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
        + ["Other"]
    )
    pivot_focus = pivot_focus[order]

    # ---------- plotting -------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_focus.plot(kind="barh", stacked=True, ax=ax, legend=False)

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Market Share (%)")
    ax.set_ylabel("Fiscal Year")
    ax.set_title("Market-Share Composition by Ticker")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # annotate segments
    for container in ax.containers:
        for patch in container:
            width = patch.get_width()
            if width <= 0:
                continue
            x_pos = patch.get_x() + width / 2
            y_pos = patch.get_y() + patch.get_height() / 2
            ax.text(
                x_pos,
                y_pos,
                f"{width*100:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax.legend(
        pivot_focus.columns,
        title="Ticker",
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
        borderaxespad=0,
    )

    plt.tight_layout()
    return fig

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

market = df_raw

# ---------- sidebar controls ----------
all_tickers = sorted(market["Ticker"].unique())
top_n = int(st.sidebar.number_input(
    "Number of top companies to highlight",
    min_value=1,
    max_value=len(all_tickers),
    value=5,
    step=1
))

sel_tickers = top_companies(market, top_n)

years_available = sorted([int(y) for y in market["Fiscal Year"].dropna().unique()])

years_selected = st.sidebar.multiselect(
    "Select Fiscal Years",
    options=years_available[::-1],
    default=years_available[-3:][::-1],
)

yr_min, yr_max = st.sidebar.slider(
    "Year range for line chart",
    min_value=min(years_available),
    max_value=max(years_available),
    value=(min(years_available), max(years_available)),
)

chart_style = st.sidebar.radio(
    "Chart type",
    [
        "Market share over time",
        "Stacked Bar",
    ],
)

# ---------- filter data for line chart ----------
filtered_range = market[(market["Fiscal Year"] >= yr_min) & (market["Fiscal Year"] <= yr_max)].copy()

# ---------- main content ----------
st.title("Retail market-share explorer")

if chart_style == "Market share over time":
    st.altair_chart(plot_market_share_over_time(filtered_range, sel_tickers), use_container_width=True)
else:
    fig = plot_market_share_stacked_bar(market, years_selected, include_tickers=sel_tickers)
    if fig:
        st.pyplot(fig, use_container_width=True)
