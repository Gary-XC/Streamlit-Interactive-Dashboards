import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.cm as cm
from pathlib import Path
from typing import Union, IO, List, Optional, cast
import seaborn as sns


st.set_page_config(
    page_title="Market-Share dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    "<h1 style='text-align:center;'>Market Share Dashboard</h1>",
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).parent
DEFAULT_CSV = BASE_DIR / "Data" / "rStreamLitData.csv"

@st.cache_data(show_spinner=False)
def load_data(source: Union[str, Path, IO[bytes]]) -> pd.DataFrame:
    df = pd.read_csv(source)
    return df

def top_companies(df: pd.DataFrame, top_n: int = 5) -> List[str]:
    return (
        df.groupby("Ticker")["Market Share"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )




def plot_market_share_over_time(df: pd.DataFrame, focus: List[str]):
    long_df = df[df["Ticker"].isin(focus)].copy()
    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x="Fiscal Year:T",
            y=alt.Y("Market Share:Q", axis=alt.Axis(format=".0%")),
            color="Ticker:N",
            tooltip=["Fiscal Year:T", "Ticker", alt.Tooltip("Market Share:Q", format=".2%")],
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

    df = data.copy()


    if years is not None:
        years = [years] if isinstance(years, int) else list(years)
        df = df[df["Fiscal Year"].isin(years)]

    focus_mask = df["Ticker"].isin(include_tickers) if include_tickers is not None else pd.Series(True, index=df.index)
    df_focus = df[focus_mask]
    
    if df_focus.empty:
        st.warning("No data available for the selected years and companies.")
        return None  # Skip plotting for years when there is no values
    

    yearly_focus = (
        df_focus.groupby(["Fiscal Year", "Ticker"], as_index=False)[value_col].mean()
    )
    pivot_focus = (
        yearly_focus.pivot(index="Fiscal Year", columns="Ticker", values=value_col).fillna(0)
    )

    other_share = 1.0 - pivot_focus.sum(axis=1)
    other_share = other_share.clip(lower=0)
    pivot_focus["Other"] = other_share

    order = (
        pivot_focus.drop(columns="Other")
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
        + ["Other"]
    )
    pivot_focus = pivot_focus[order]

    plt.style.use("dark_background")  #"ggplot", "fivethirtyeight", Dark Modes: Solarize_Light2, dark_background

    # Generating a colormap
    cmap = cm.get_cmap('tab20')  # other options 'Set3', 'Paired', 'Accent', 'Pastel1', 'tab20'
    n_colors = len(pivot_focus.columns)
    colors = [cmap(i) for i in range(n_colors)]

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot_focus.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor='none')

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Market Share (%)", fontsize=12)
    ax.set_ylabel("Fiscal Year", fontsize=12)
    ax.set_title("Market-Share Composition by Ticker", fontsize=14, fontweight="bold", pad=15)

    ax.tick_params(axis="both", labelsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for container in ax.containers:
        for patch in container:
            width = patch.get_width()
            if width <= 0.03:
                continue
            x = patch.get_x() + width / 2
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x, y, f"{width*100:.1f}%", ha='center', va='center', fontsize=9)

    ax.legend(
        pivot_focus.columns,
        title="Ticker",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        fontsize=9,
        title_fontsize=10,
    )

    plt.tight_layout()
    return fig



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

company_options = sorted(market["Ticker"].unique())
sel_company = st.sidebar.selectbox("Company", company_options)

year_options = sorted(market["Fiscal Year"].dropna().unique(), reverse=True)
sel_year = st.sidebar.selectbox("Fiscal Year", year_options)

mask = (
    (market["Ticker"] == sel_company)
    & (market["Fiscal Year"] == sel_year)
)

filtered = cast(pd.Series, market.loc[mask, "Market Share"])         

if len(filtered):                                    
    share_pct = float(filtered.iloc[0]) * 100        
    st.sidebar.metric(
        label=f"{sel_company} market share in {sel_year}",
        value=f"{share_pct:.2f} %",
    )
else:
    st.sidebar.info("No data for that company + year combination.")

# filtering the data for line chart
filtered_range = market[(market["Fiscal Year"] >= yr_min) & (market["Fiscal Year"] <= yr_max)].copy()

st.title("Market Share Dashboard")

if chart_style == "Market share over time":
    st.altair_chart(plot_market_share_over_time(filtered_range, sel_tickers), use_container_width=True)
else:
    fig = plot_market_share_stacked_bar(market, years_selected, include_tickers=sel_tickers)
    if fig:
        st.pyplot(fig, use_container_width=True)