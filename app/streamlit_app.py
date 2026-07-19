"""PartnerLens -- Streamlit app.

Loads pre-computed artifacts from data/processed/ only. The app trains
nothing at runtime: that split keeps boot fast on Streamlit Community Cloud
and keeps "how was this trained" and "how is this served" as two separate,
independently testable concerns. Regenerate the artifacts with
`python scripts/build_pipeline.py` after any change to src/.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, profiling  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "assets" / "Logo_White.png"

st.set_page_config(
    page_title=f"{config.PROJECT_NAME} — {config.PROJECT_TAGLINE}",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling
# =============================================================================
PALETTE = config.PALETTE
PLOTLY_FONT = dict(family="Inter, sans-serif", color=PALETTE["text_secondary"], size=12)


def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        margin=dict(l=44, r=20, t=24, b=40),
        showlegend=False,
        xaxis=dict(gridcolor="#242A35", zerolinecolor=PALETTE["border"], linecolor=PALETTE["border"]),
        yaxis=dict(gridcolor="#242A35", zerolinecolor=PALETTE["border"], linecolor=PALETTE["border"]),
    )
    layout.update(overrides)
    return layout


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stApp"] {{
            background-color: {PALETTE['bg']} !important;
        }}
        [data-testid="stHeader"] {{ background-color: transparent !important; }}
        .block-container {{ padding-top: 2rem; max-width: 1360px; }}

        h1, h2, h3, h4 {{ font-family: 'Space Grotesk', sans-serif !important; color: {PALETTE['text_primary']}; letter-spacing: -0.01em; }}
        p, span, label, div {{ color: {PALETTE['text_primary']}; }}
        .muted {{ color: {PALETTE['text_secondary']} !important; font-size: 0.85rem; line-height: 1.55; }}
        .tiny-muted {{ color: {PALETTE['text_tertiary']} !important; font-size: 0.75rem; }}

        /* ---------------- sidebar ---------------- */
        [data-testid="stSidebar"] {{
            background-color: {PALETTE['bg_elevated']} !important;
            border-right: 1px solid {PALETTE['border']};
        }}
        [data-testid="stSidebarUserContent"] {{
            display: flex; flex-direction: column; min-height: calc(100vh - 3rem);
            padding-top: 1rem;
        }}
        /* st.image(use_container_width=True) sets an inline width in px on
           the <img> itself, which beats a plain class rule -- !important is
           required to shrink it here. */
        .st-key-sidebar_logo img {{ width: 80% !important; height: auto; }}
        .sidebar-name {{
            font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.05rem;
            color: {PALETTE['text_primary']}; margin-top: 10px; letter-spacing: -0.01em;
        }}
        .sidebar-tag {{
            font-size: 0.68rem; color: {PALETTE['text_tertiary']}; text-transform: uppercase;
            letter-spacing: 0.06em; margin-top: 2px;
        }}
        .sidebar-footer {{
            margin-top: auto; padding-top: 14px; border-top: 1px solid {PALETTE['border']};
            font-size: 0.72rem; color: {PALETTE['text_tertiary']}; line-height: 1.6;
        }}
        .sidebar-footer a {{ color: {PALETTE['gold']}; text-decoration: none; }}
        .sidebar-footer a:hover {{ text-decoration: underline; }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            background: transparent; border: 1px solid transparent; border-radius: 10px;
            padding: 6px 10px; margin-bottom: 2px; transition: background .15s ease;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {PALETTE['surface_hover']}; }}

        /* ---------------- cards ----------------
           .pl-card: single self-contained st.markdown() HTML blocks.
           [class*="st-key-card_"]: st.container(key=...) card() helper --
           the only way to style a card that wraps real widgets (charts,
           selectboxes) alongside markdown text; see card() docstring. */
        .pl-card, [class*="st-key-card_"] {{
            background: {PALETTE['surface']}; border: 1px solid {PALETTE['border']};
            border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
        }}
        .pl-card h3, [class*="st-key-card_"] h3 {{ font-size: 0.95rem; margin: 0 0 4px 0; }}
        .pl-card .caption, [class*="st-key-card_"] .caption {{ font-size: 0.72rem; color: {PALETTE['text_tertiary']}; margin-bottom: 10px; }}

        .kpi-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: {PALETTE['text_tertiary']}; font-weight: 600; margin-bottom: 6px; }}
        .kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.55rem; font-weight: 600; color: {PALETTE['text_primary']}; line-height: 1.1; overflow-wrap: anywhere; }}
        .kpi-sub {{ font-size: 0.72rem; color: {PALETTE['text_secondary']}; margin-top: 6px; }}

        .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.66rem; font-weight: 700; letter-spacing: 0.02em; }}

        .seg-swatch {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }}

        code {{ font-family: 'IBM Plex Mono', monospace; background: {PALETTE['bg_elevated']}; padding: 1px 6px; border-radius: 4px; font-size: 0.82em; color: {PALETTE['teal']}; border: 1px solid {PALETTE['border']}; }}

        [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; }}
        hr {{ border-color: {PALETTE['border']}; }}

        /* ---------------- top KPI row ----------------
           the gauge card (Avg. Partner Health) is taller than the plain
           text kpi_card()s because it also holds a plotly chart; pin all
           four top-row cards to the same height so they align. */
        .kpirow, [class*="_kpirow"] {{ min-height: 180px; box-sizing: border-box; }}

        /* ---------------- select control + dropdown popover ----------------
           the BaseWeb select control (closed, showing the chosen value) and
           its menu popover (open, portal-rendered) both have an unstyled
           white/light background; the global span/div rule above still
           applies PALETTE['text_primary'] (light, meant for dark surfaces),
           making the text unreadable in both states. Force dark text on
           the select control itself and on the popover's option list. */
        [data-baseweb="select"] *,
        [data-baseweb="popover"] li, [data-baseweb="popover"] li *,
        [data-baseweb="menu"] li, [data-baseweb="menu"] li * {{
            color: #1a1a1a !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Data loading (cached -- the app trains nothing)
# =============================================================================
@st.cache_data
def load_sellers() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_FILES["sellers_segmented"])


@st.cache_data
def load_profiles() -> list[dict]:
    with open(config.PROCESSED_FILES["cluster_profiles"], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_diagnostics() -> dict:
    with open(config.PROCESSED_FILES["model_diagnostics"], encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data_quality() -> dict:
    with open(config.PROCESSED_FILES["data_quality_report"], encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Formatters
# =============================================================================
def fmt_brl_compact(v: float) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e6:
        return f"R$ {v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"R$ {v/1e3:.1f}K"
    return f"R$ {v:.0f}"


def fmt_brl(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"R$ {v:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")


def fmt_int(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.0f}"


def fmt_pct(v: float, decimals: int = 1) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:.{decimals}f}%"


def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def short_id(seller_id: str, head: int = 10, tail: int = 4) -> str:
    """Real Olist seller_ids are 32-char hex strings -- too long for headers,
    table cells and selectbox rows without wrapping or overflowing their
    container. Truncate for display only; lookups always use the full id."""
    if len(seller_id) <= head + tail + 1:
        return seller_id
    return f"{seller_id[:head]}…{seller_id[-tail:]}"


# =============================================================================
# Reusable components
# =============================================================================
_card_seq = itertools.count()


def card(extra_key: str = ""):
    """A "pl-card"-styled container that can hold real Streamlit widgets
    (charts, selectboxes, dataframes), not just markdown text.

    Raw HTML from one st.markdown() call cannot wrap components rendered by
    separate st.* calls -- each call is its own DOM fragment, so an opening
    <div> in one markdown call and a closing </div> in another never actually
    nest anything between them. st.container(key=...) gives a real DOM
    parent (class `st-key-<key>`) that every widget rendered inside its
    `with` block becomes a true child of; the `[class*="st-key-card_"]` CSS
    rule (see inject_css) styles any such container as a card.

    `extra_key` is appended to the container key so a CSS rule can target
    this specific card via `[class*="_{extra_key}"]` (e.g. "kpirow" to pin
    the top KPI row to a shared height -- see inject_css).
    """
    key = f"card_{next(_card_seq)}"
    if extra_key:
        key = f"{key}_{extra_key}"
    return st.container(key=key, border=False)


def kpi_card(label: str, value: str, sub: str, extra_class: str = "") -> None:
    st.markdown(
        f"""<div class="pl-card {extra_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def health_gauge(score: float, height: int = 190, title: str | None = None) -> go.Figure:
    color = profiling.health_tier_color(score)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(score, 1),
            number={"suffix": "", "font": {"size": 30, "family": "IBM Plex Mono, monospace", "color": PALETTE["text_primary"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": PALETTE["border"], "showticklabels": False},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": PALETTE["border"]}],
            },
            title={"text": title or "", "font": {"size": 11, "color": PALETTE["text_tertiary"]}},
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=height, margin=dict(l=18, r=18, t=36 if title else 10, b=10))
    return fig


def segment_pill(action_tag: str, color: str) -> str:
    return f'<span class="pill" style="background:{color}22; color:{color};">{action_tag}</span>'


# =============================================================================
# Sidebar
# =============================================================================
NAV_ITEMS = ["Overview", "Segment Explorer", "Seller Lookup", "Methodology"]


def render_sidebar() -> str:
    with st.sidebar:
        with st.container(key="sidebar_logo"):
            st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown(
            f'<div class="sidebar-name">{config.PROJECT_NAME}</div>'
            f'<div class="sidebar-tag">{config.PROJECT_TAGLINE}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        tab = st.radio("Navigate", NAV_ITEMS, label_visibility="collapsed")
        st.markdown(
            f"""<div class="sidebar-footer">
                Olist Brazilian E-Commerce dataset<br>
                seller-order pair granularity &middot; K-Means (k={config.N_CLUSTERS})<br><br>
                Developed by <b>{config.AUTHOR_NAME}</b><br>
                <a href="{config.AUTHOR_GITHUB}" target="_blank">{config.AUTHOR_GITHUB.replace('https://', '')}</a>
            </div>""",
            unsafe_allow_html=True,
        )
    return tab


# =============================================================================
# Tab: Overview
# =============================================================================
def render_overview(sellers: pd.DataFrame, profiles: list[dict], diagnostics: dict) -> None:
    st.title("Executive Overview")
    st.markdown(
        '<p class="muted">Marketplace sellers, not customers, are the unit of segmentation -- profiling '
        "partner health, quality risk and growth potential from the seller-order pair upward.</p>",
        unsafe_allow_html=True,
    )

    eligible = sellers[sellers["cluster"] != -1]
    active_mask = sellers["recency_days"] <= 90
    total_revenue = sellers["total_revenue"].sum()
    avg_health = np.average(sellers["health_score"], weights=sellers["total_revenue"].clip(lower=1))
    avg_neg_review = sellers["neg_review_rate"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            "Sellers Active (≤90d)",
            fmt_int(active_mask.sum()),
            f"{100*active_mask.mean():.1f}% of {fmt_int(len(sellers))} sellers",
            extra_class="kpirow",
        )
    with c2:
        kpi_card("GMV Analyzed", fmt_brl_compact(total_revenue), "sum of seller-order pair revenue", extra_class="kpirow")
    with c3:
        with card(extra_key="kpirow"):
            st.markdown('<div class="kpi-label">Avg. Partner Health (GMV-weighted)</div>', unsafe_allow_html=True)
            st.plotly_chart(health_gauge(avg_health, height=110), use_container_width=True, config={"displayModeBar": False})
    with c4:
        kpi_card("Avg. Negative Review Rate", fmt_pct(avg_neg_review), "order review score ≤ 2", extra_class="kpirow")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        with card():
            st.markdown("<h3>Revenue Share by Segment</h3>", unsafe_allow_html=True)
            st.markdown(
                '<div class="caption">Share of total GMV captured by each named segment, including the excluded low-data tail</div>',
                unsafe_allow_html=True,
            )
            ordered = sorted(profiles, key=lambda p: p["revenue_share_pct"], reverse=True)
            fig = go.Figure(
                go.Bar(
                    x=[p["revenue_share_pct"] for p in ordered],
                    y=[p["name"] for p in ordered],
                    orientation="h",
                    marker=dict(color=[p["color"] for p in ordered]),
                    hovertemplate="%{y}<br>Revenue share: %{x:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(**base_layout(margin=dict(l=170, r=20, t=10, b=30), xaxis=dict(gridcolor="#242A35", ticksuffix="%"), yaxis=dict(autorange="reversed")))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        with card():
            st.markdown("<h3>Segment Size</h3>", unsafe_allow_html=True)
            st.markdown('<div class="caption">Share of the seller base per segment</div>', unsafe_allow_html=True)
            fig = go.Figure(
                go.Pie(
                    labels=[p["name"] for p in profiles],
                    values=[p["n_sellers"] for p in profiles],
                    hole=0.62,
                    marker=dict(colors=[p["color"] for p in profiles], line=dict(color=PALETTE["bg"], width=2)),
                    textfont=dict(color=PALETTE["text_primary"], size=10.5),
                    hovertemplate="%{label}<br>%{value} sellers (%{percent})<extra></extra>",
                )
            )
            fig.update_layout(**base_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=True, legend=dict(orientation="v", font=dict(size=9.5, color=PALETTE["text_secondary"]), x=1, y=0.5)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with card():
        st.markdown("<h3>Clusters, projected (PCA)</h3>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="caption">First two principal components (explain {diagnostics["pca_explained_variance_sum"]*100:.0f}% of variance '
            "-- illustrative separation, not exact inter-cluster distance), colored by K-Means assignment</div>",
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        for p in profiles:
            if p["cluster"] == -1:
                continue
            sub = eligible[eligible["segment_key"] == p["segment_key"]]
            fig.add_trace(
                go.Scattergl(
                    x=sub["pc1"], y=sub["pc2"], mode="markers", name=p["name"],
                    marker=dict(size=4.5, color=p["color"], opacity=0.55),
                    hovertemplate=p["name"] + "<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(**base_layout(
            showlegend=True, legend=dict(orientation="h", y=1.12, font=dict(size=9.5, color=PALETTE["text_secondary"])),
            xaxis=dict(title="PC1", gridcolor="#242A35"), yaxis=dict(title="PC2", gridcolor="#242A35"),
            height=420,
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    mss = diagnostics["multi_seller_order_stats"]
    st.markdown(
        f"""<div class="pl-card" style="background:linear-gradient(180deg, {PALETTE['gold']}0D, transparent 60%), {PALETTE['surface']};">
        <h3>Modeling notes</h3>
        <ul class="muted" style="padding-left:18px; margin-top:10px;">
        <li style="margin-bottom:8px;"><b style="color:{PALETTE['text_primary']}">Unit of analysis:</b> the <code>seller-order pair</code>, not the raw order.
        Olist's <code>order_items</code> table allows more than one seller per order, so aggregating straight from
        <code>orders</code> would misattribute shared-order signals to sellers who didn't drive them. Every feature is
        built by first exploding to seller-order pairs, then aggregating up to seller level.</li>
        <li style="margin-bottom:8px;"><b style="color:{PALETTE['text_primary']}">Monetary features:</b> <code>log1p(revenue)</code>
        prior to scaling -- raw revenue is heavily right-skewed by a small number of high-GMV sellers.</li>
        <li style="margin-bottom:8px;"><b style="color:{PALETTE['text_primary']}">Recency window:</b> computed against the
        dataset's maximum order date (<code>{diagnostics['reference_date']}</code>), not <code>today()</code> -- this is a
        historical snapshot, not a live feed.</li>
        <li><b style="color:{PALETTE['text_primary']}">Multi-seller review attribution (measured, not assumed):</b>
        Olist review scores attach to the order, not the line item, so a multi-seller order assigns the same review to
        every participating seller. Measured directly from the data: <b style="color:{PALETTE['text_primary']}">{mss['multi_seller_orders_pct']}%</b>
        of orders ({fmt_int(mss['multi_seller_orders'])} of {fmt_int(mss['total_orders_with_items'])}) involve more than one
        seller, carrying <b style="color:{PALETTE['text_primary']}">{mss['multi_seller_revenue_pct']}%</b> of revenue -- the
        distortion is real but marginal. See the Methodology tab for the full derivation.</li>
        </ul>
        </div>""",
        unsafe_allow_html=True,
    )


# =============================================================================
# Tab: Segment Explorer
# =============================================================================
def render_segment_explorer(sellers: pd.DataFrame, profiles: list[dict]) -> None:
    st.title("Segment Explorer")
    st.markdown(
        '<p class="muted">Each K-Means cluster is translated into a business-actionable profile: who they are, '
        "what the health score says, and what action the segment justifies.</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, p in enumerate(profiles):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="pl-card" style="border-top:3px solid {p['color']}; min-height: 300px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <h3>{p['name']}</h3>
                        <div class="tiny-muted">{fmt_int(p['n_sellers'])} sellers &middot; {p['seller_share_pct']:.1f}% of base</div>
                    </div>
                    <div class="kpi-value" style="font-size:1.1rem; color:{p['color']};">{p['median_health']:.0f}</div>
                </div>
                <p class="muted" style="margin:10px 0;">{p['description']}</p>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px 12px; margin-bottom:12px;">
                    <div><div class="tiny-muted">REVENUE (MEDIAN)</div><div class="kpi-value" style="font-size:0.95rem;">{fmt_brl_compact(p['median_revenue'])}</div></div>
                    <div><div class="tiny-muted">REVENUE SHARE</div><div class="kpi-value" style="font-size:0.95rem;">{p['revenue_share_pct']:.1f}%</div></div>
                    <div><div class="tiny-muted">ORDERS (MEDIAN)</div><div class="kpi-value" style="font-size:0.95rem;">{p['median_frequency']:.0f}</div></div>
                    <div><div class="tiny-muted">NEG. REVIEW RATE</div><div class="kpi-value" style="font-size:0.95rem;">{p['median_neg_review_rate']*100:.1f}%</div></div>
                </div>
                <div style="border-top:1px solid {PALETTE['border']}; padding-top:10px;">
                    {segment_pill(p['action_tag'], p['color'])}
                    <p class="muted" style="margin:8px 0 6px;">{p['action']}</p>
                    <div class="tiny-muted" style="font-style:italic;">{p['impact']}</div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("<h3>Deep dive: segment vs. marketplace average</h3>", unsafe_allow_html=True)
    name_to_key = {p["name"]: p for p in profiles if p["cluster"] != -1}
    choice = st.selectbox("Segment", list(name_to_key.keys()))
    p = name_to_key[choice]
    seg_rows = sellers[sellers["segment_key"] == p["segment_key"]]
    pop_rows = sellers[sellers["cluster"] != -1]

    col1, col2 = st.columns([1, 1.2])
    with col1:
        with card():
            st.markdown(f"<h3>{choice} vs. marketplace</h3>", unsafe_allow_html=True)
            cats = ["Revenue", "Frequency", "Recency", "Delivery delay", "Review quality", "Cancel control"]

            def radar_values(df: pd.DataFrame) -> list[float]:
                bounds = load_diagnostics()["health_score_bounds"]
                rev = ((np.log1p(df["total_revenue"]).median() - bounds["total_revenue"][0]) / (bounds["total_revenue"][1] - bounds["total_revenue"][0]))
                freq = ((np.log1p(df["frequency"]).median() - bounds["frequency"][0]) / (bounds["frequency"][1] - bounds["frequency"][0]))
                rec = 1 - ((df["recency_days"].median() - bounds["recency_days"][0]) / (bounds["recency_days"][1] - bounds["recency_days"][0]))
                delay = 1 - ((df["avg_delay_days"].median() - bounds["avg_delay_days"][0]) / (bounds["avg_delay_days"][1] - bounds["avg_delay_days"][0]))
                neg = 1 - df["neg_review_rate"].median()
                can = 1 - df["cancel_rate"].median()
                return [float(np.clip(v, 0, 1)) for v in [rev, freq, rec, delay, neg, can]]

            seg_vals = radar_values(seg_rows)
            pop_vals = radar_values(pop_rows)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=pop_vals + pop_vals[:1], theta=cats + cats[:1], name="Marketplace avg", fill="toself", fillcolor="rgba(155,161,173,0.08)", line=dict(color=PALETTE["text_tertiary"], width=1.5, dash="dot")))
            fig.add_trace(go.Scatterpolar(r=seg_vals + seg_vals[:1], theta=cats + cats[:1], name=choice, fill="toself", fillcolor=hex_to_rgba(p["color"], 0.2), line=dict(color=p["color"], width=2)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=PLOTLY_FONT,
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=PALETTE["border"]), angularaxis=dict(gridcolor=PALETTE["border"], tickfont=dict(size=9.5, color=PALETTE["text_secondary"]))),
                showlegend=True, legend=dict(orientation="h", y=-0.1, font=dict(size=9.5, color=PALETTE["text_secondary"])),
                margin=dict(l=40, r=40, t=20, b=20), height=360,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        with card():
            st.markdown("<h3>Health score distribution within segment</h3>", unsafe_allow_html=True)
            st.markdown('<div class="caption">A healthy segment can still carry a lower tail worth watching individually</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Box(x=seg_rows["health_score"], marker_color=p["color"], name=choice, boxmean=True))
            fig.update_layout(**base_layout(margin=dict(l=20, r=20, t=10, b=30), xaxis=dict(title="health score", range=[0, 100], gridcolor="#242A35"), height=170))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            below_40 = (seg_rows["health_score"] < 40).sum()
            st.markdown(
                f'<div class="muted">{fmt_int(below_40)} sellers in this segment ({100*below_40/max(len(seg_rows),1):.1f}%) '
                f'score below 40/100 despite the segment label -- worth a manual spot-check before treating "{choice}" as a '
                "uniform group.</div>",
                unsafe_allow_html=True,
            )


# =============================================================================
# Tab: Seller Lookup
# =============================================================================
def render_seller_lookup(sellers: pd.DataFrame) -> None:
    st.title("Individual Seller Lookup")
    st.markdown(
        '<p class="muted">Search by seller ID or state, or filter by segment, to inspect a single partner against '
        "their segment and the overall population.</p>",
        unsafe_allow_html=True,
    )

    segments = ["All segments"] + sorted(sellers["segment_name"].unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        query = st.text_input("Search seller ID or state", placeholder="e.g. 3442f895 or SP", label_visibility="collapsed")
    with c2:
        segment_filter = st.selectbox("Segment", segments, label_visibility="collapsed")

    filtered = sellers.copy()
    if query:
        q = query.strip().lower()
        filtered = filtered[
            filtered["seller_id"].str.lower().str.contains(q, regex=False)
            | filtered["seller_state"].str.lower().str.contains(q, regex=False)
        ]
    if segment_filter != "All segments":
        filtered = filtered[filtered["segment_name"] == segment_filter]
    filtered = filtered.sort_values("health_score", ascending=False)

    no_exact_match = filtered.empty
    # Never leave the panel with nothing to show: if the search/filter combo
    # matches nobody, fall back to the full base (still sorted by health)
    # rather than a dead-end empty state or a synthetic placeholder option.
    table_source = sellers.sort_values("health_score", ascending=False) if no_exact_match else filtered

    if no_exact_match:
        st.markdown(
            f'<div class="tiny-muted">No sellers match "{query}" -- showing the full base ({fmt_int(len(sellers))} sellers) instead.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f'<div class="tiny-muted">Showing {fmt_int(min(len(filtered), 150))} of {fmt_int(len(filtered))} matching sellers</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        display_df = table_source.head(150)[["seller_id", "seller_state", "segment_name", "frequency", "total_revenue", "health_score"]].rename(
            columns={"seller_id": "Seller", "seller_state": "State", "segment_name": "Segment", "frequency": "Orders", "total_revenue": "Revenue", "health_score": "Health"}
        )
        display_df["Seller"] = display_df["Seller"].apply(short_id)
        display_df["Revenue"] = display_df["Revenue"].apply(fmt_brl_compact)
        display_df["Health"] = display_df["Health"].round(0).astype(int)
        st.dataframe(display_df, use_container_width=True, height=460, hide_index=True)

        options = table_source.head(150)["seller_id"].tolist()
        # Options change on every search/filter edit. Manage session_state for
        # this key explicitly (rather than passing `index=`) so a stale
        # selection from a previous, differently-shaped options list never
        # leaks into a widget rebuilt with a new one.
        if st.session_state.get("seller_select") not in options:
            st.session_state["seller_select"] = options[0]
        selected_id = st.selectbox("Inspect seller", options, key="seller_select", format_func=short_id)

    with col2:
        s = table_source[table_source["seller_id"] == selected_id].iloc[0]
        seg_rows = sellers[sellers["segment_key"] == s["segment_key"]]
        with card():
            gc, tc = st.columns([1, 1.4])
            with gc:
                st.plotly_chart(health_gauge(s["health_score"], height=140), use_container_width=True, config={"displayModeBar": False})
            with tc:
                st.markdown(
                    f"""<div style="margin-top:14px;">
                    <div class="kpi-value" style="font-size:1.05rem;" title="{s['seller_id']}">{short_id(s['seller_id'], 14, 6)}</div>
                    <div class="tiny-muted">{s['seller_state']} &middot; <span style="color:{s['segment_color']}">{s['segment_name']}</span></div>
                    <div class="tiny-muted" style="color:{profiling.health_tier_color(s['health_score'])}; text-transform:uppercase; font-weight:700; margin-top:4px;">{profiling.health_tier(s['health_score'])}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""<div style="display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; margin:16px 0;">
                <div><div class="tiny-muted">ORDERS</div><div class="kpi-value" style="font-size:1rem;">{fmt_int(s['frequency'])}</div></div>
                <div><div class="tiny-muted">REVENUE</div><div class="kpi-value" style="font-size:1rem;">{fmt_brl_compact(s['total_revenue'])}</div></div>
                <div><div class="tiny-muted">AVG. DELIVERY DELAY</div><div class="kpi-value" style="font-size:1rem;">{s['avg_delay_days']:.1f}d</div></div>
                <div><div class="tiny-muted">DAYS SINCE LAST ORDER</div><div class="kpi-value" style="font-size:1rem;">{s['recency_days']:.0f}d</div></div>
                <div><div class="tiny-muted">NEG. REVIEW RATE</div><div class="kpi-value" style="font-size:1rem;">{fmt_pct(s['neg_review_rate'])}</div></div>
                <div><div class="tiny-muted">CANCEL RATE</div><div class="kpi-value" style="font-size:1rem;">{fmt_pct(s['cancel_rate'])}</div></div>
                </div>""",
                unsafe_allow_html=True,
            )

            cats = ["Revenue", "Frequency", "Recency", "Delivery delay", "Review quality", "Cancel control"]
            bounds = load_diagnostics()["health_score_bounds"]

            def seller_radar(row) -> list[float]:
                rev = (np.log1p(row["total_revenue"]) - bounds["total_revenue"][0]) / (bounds["total_revenue"][1] - bounds["total_revenue"][0])
                freq = (np.log1p(row["frequency"]) - bounds["frequency"][0]) / (bounds["frequency"][1] - bounds["frequency"][0])
                rec = 1 - (row["recency_days"] - bounds["recency_days"][0]) / (bounds["recency_days"][1] - bounds["recency_days"][0])
                delay = 1 - (row["avg_delay_days"] - bounds["avg_delay_days"][0]) / (bounds["avg_delay_days"][1] - bounds["avg_delay_days"][0])
                neg = 1 - row["neg_review_rate"]
                can = 1 - row["cancel_rate"]
                return [float(np.clip(v, 0, 1)) for v in [rev, freq, rec, delay, neg, can]]

            seller_vals = seller_radar(s)
            seg_vals = seller_radar(seg_rows.median(numeric_only=True))
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=seg_vals + seg_vals[:1], theta=cats + cats[:1], name=f"{s['segment_name']} median", fill="toself", fillcolor="rgba(155,161,173,0.08)", line=dict(color=PALETTE["text_tertiary"], width=1.5, dash="dot")))
            fig.add_trace(go.Scatterpolar(r=seller_vals + seller_vals[:1], theta=cats + cats[:1], name=s["seller_id"][:10], fill="toself", fillcolor=hex_to_rgba(s["segment_color"], 0.2), line=dict(color=s["segment_color"], width=2)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font=PLOTLY_FONT,
                polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=PALETTE["border"]), angularaxis=dict(gridcolor=PALETTE["border"], tickfont=dict(size=9, color=PALETTE["text_secondary"]))),
                showlegend=True, legend=dict(orientation="h", y=-0.15, font=dict(size=9, color=PALETTE["text_secondary"])),
                margin=dict(l=30, r=30, t=10, b=10), height=260,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            if s["segment_key"] == "new_low_data":
                action = f"Fewer than {config.MIN_ORDERS_THRESHOLD} orders on record -- not yet eligible for formal segmentation. Monitor early quality signals."
            else:
                action = next(p["action"] for p in load_profiles() if p["segment_key"] == s["segment_key"])
            action_tag = "MONITOR" if s["segment_key"] == "new_low_data" else next(p["action_tag"] for p in load_profiles() if p["segment_key"] == s["segment_key"])
            st.markdown(
                f"""<div style="border-top:1px solid {PALETTE['border']}; padding-top:12px;">
                {segment_pill(action_tag, s['segment_color'])}
                <p class="muted" style="margin-top:8px;">{action}</p>
                </div>""",
                unsafe_allow_html=True,
            )


# =============================================================================
# Tab: Methodology
# =============================================================================
def render_methodology(diagnostics: dict, profiles: list[dict]) -> None:
    st.title("Methodology")
    st.markdown(
        '<p class="muted">The technical decisions behind the segmentation, in the order they were made -- including '
        "where the data pushed back on the original design assumptions.</p>",
        unsafe_allow_html=True,
    )

    with card():
        st.markdown("<h3>1. Unit of analysis: the seller-order pair</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""<p class="muted">Olist is a marketplace: a single <code>order_id</code> in <code>order_items</code> can contain
            items from more than one seller. Summing straight from <code>order_items</code> inflates a seller's order
            frequency when they sell multiple items in one order; aggregating straight from <code>orders</code> credits a
            seller with revenue from items they didn't sell. Materializing the seller &times; order_id grain first (one row =
            one seller's participation in one order) makes both revenue and frequency correct by construction.</p>
            <p class="muted"><b style="color:{PALETTE['text_primary']}">Measured impact of the resulting review-attribution limitation</b>
            (reviews attach to the order, not the line item, so a multi-seller order assigns the same score to every
            seller involved): <b style="color:{PALETTE['text_primary']}">{diagnostics['multi_seller_order_stats']['multi_seller_orders_pct']}%</b>
            of orders, <b style="color:{PALETTE['text_primary']}">{diagnostics['multi_seller_order_stats']['multi_seller_revenue_pct']}%</b> of
            revenue. Quantified, not assumed.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>2. Minimum-order threshold</h3>", unsafe_allow_html=True)
        sens = pd.DataFrame(diagnostics["threshold_sensitivity"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sens["threshold"], y=sens["sellers_kept_pct"], mode="lines+markers", name="% sellers kept", line=dict(color=PALETTE["teal"])))
        fig.add_trace(go.Scatter(x=sens["threshold"], y=sens["revenue_kept_pct"], mode="lines+markers", name="% revenue kept", line=dict(color=PALETTE["gold"])))
        fig.update_layout(**base_layout(
            showlegend=True, legend=dict(orientation="h", y=1.15, font=dict(size=10, color=PALETTE["text_secondary"])),
            xaxis=dict(title="minimum distinct orders", gridcolor="#242A35"), yaxis=dict(title="%", gridcolor="#242A35"),
            shapes=[dict(type="line", x0=diagnostics["min_orders_threshold"], x1=diagnostics["min_orders_threshold"], y0=0, y1=1, xref="x", yref="paper", line=dict(color=PALETTE["border_strong"], width=1, dash="dot"))],
            height=320,
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        row5 = sens[sens["threshold"] == diagnostics["min_orders_threshold"]].iloc[0]
        st.markdown(
            f"""<p class="muted">Sellers with too few orders have degenerate rate features -- a negative-review rate can only
            land on 0%, 50% or 100% at 1-2 orders. At the chosen threshold (<code>≥{diagnostics['min_orders_threshold']} orders</code>),
            the eligible population keeps <b style="color:{PALETTE['text_primary']}">{row5['sellers_kept_pct']:.1f}%</b> of sellers but
            <b style="color:{PALETTE['text_primary']}">{row5['revenue_kept_pct']:.1f}%</b> of revenue. The excluded
            {fmt_int(diagnostics['n_sellers_excluded'])} sellers are not dropped from the project -- they appear as the
            "New / Low Data" segment throughout the app.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>3. Feature correlation & the frequency decision</h3>", unsafe_allow_html=True)
        corr = pd.DataFrame(diagnostics["correlation_matrix"])
        labels = [config.FEATURE_LABELS.get(c, c) for c in corr.columns]
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=labels, y=labels, zmin=-1, zmax=1,
            colorscale=[[0, PALETTE["teal"]], [0.5, PALETTE["surface"]], [1, PALETTE["rust"]]],
            colorbar=dict(tickfont=dict(color=PALETTE["text_secondary"], size=10), outlinewidth=0),
            texttemplate="%{z:.2f}", textfont=dict(size=10, color=PALETTE["text_primary"]),
            hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(**base_layout(margin=dict(l=110, r=20, t=10, b=90), height=420))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"""<p class="muted"><code>log_revenue</code> and <code>log_frequency</code> correlate at
            <b style="color:{PALETTE['text_primary']}">0.80</b> on the eligible population -- comfortably past the ~0.7 flag
            threshold. Keeping both would double-weight one underlying "scale" dimension in Euclidean distance.
            <b style="color:{PALETTE['text_primary']}">Decision: drop frequency from the clustering feature set, keep revenue</b> --
            revenue already captures both order count and ticket size and is the more business-critical axis for
            GMV-based prioritization. Frequency is not discarded from the project: it still feeds the health score and
            every profile card.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>4. Choosing k -- elbow, silhouette, and stability</h3>", unsafe_allow_html=True)
        elbow = pd.DataFrame(diagnostics["elbow_silhouette"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elbow["k"], y=elbow["inertia"], mode="lines+markers", name="Inertia", line=dict(color=PALETTE["teal"]), yaxis="y"))
        fig.add_trace(go.Scatter(x=elbow["k"], y=elbow["silhouette"], mode="lines+markers", name="Silhouette", line=dict(color=PALETTE["gold"]), yaxis="y2"))
        fig.update_layout(**base_layout(
            showlegend=True, legend=dict(orientation="h", y=1.15, font=dict(size=10, color=PALETTE["text_secondary"])),
            xaxis=dict(title="k", gridcolor="#242A35", dtick=1),
            yaxis=dict(title=dict(text="Inertia", font=dict(color=PALETTE["teal"])), gridcolor="#242A35"),
            yaxis2=dict(title=dict(text="Silhouette", font=dict(color=PALETTE["gold"])), overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
            shapes=[dict(type="line", x0=diagnostics["n_clusters"], x1=diagnostics["n_clusters"], y0=0, y1=1, xref="x", yref="paper", line=dict(color=PALETTE["border_strong"], width=1, dash="dot"))],
            height=320,
        ))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        max_sil_row = elbow.loc[elbow["silhouette"].idxmax()]
        st.markdown(
            f"""<p class="muted">The elbow flattens from k=4 onward -- it narrows the candidate range, it doesn't pick a winner.
            Silhouette's global maximum is the coarser <b style="color:{PALETTE['text_primary']}">k={int(max_sil_row['k'])}</b> solution
            ({max_sil_row['silhouette']:.3f}); the chosen <b style="color:{PALETTE['text_primary']}">k={diagnostics['n_clusters']}</b> is a local
            peak ({diagnostics['chosen_k_silhouette']:.3f}) rather than the statistical optimum. A modest silhouette is expected
            for continuous behavioral data with no natural density valleys between clusters -- it is not, by itself, a red
            flag (the DBSCAN section below hits the same wall from a different angle). <b style="color:{PALETTE['text_primary']}">k=5
            is chosen on business interpretability</b>: coarser k values collapse segments that need opposite actions into
            one bucket, which defeats the purpose of a segmentation meant to drive differentiated partner treatment.</p>
            <p class="muted">Stability check: refitting K-Means at k={diagnostics['n_clusters']} across {diagnostics['n_stability_seeds']}
            different seeds and comparing partitions with the Adjusted Rand Index gives a mean pairwise ARI of
            <b style="color:{PALETTE['text_primary']}">{diagnostics['stability_ari_mean']:.3f}</b> -- clusters this stable across
            re-initializations are real structure, not an artifact of a lucky starting point.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>5. DBSCAN -- a critical stress test, not a competing segmenter</h3>", unsafe_allow_html=True)
        st.markdown(
            """<table class="cmp" style="width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:14px;">
            <thead><tr>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #3A4152; font-size:0.68rem; text-transform:uppercase; color:#666C79;">Algorithm</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #3A4152; font-size:0.68rem; text-transform:uppercase; color:#666C79;">Silhouette</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #3A4152; font-size:0.68rem; text-transform:uppercase; color:#666C79;">Clusters found</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #3A4152; font-size:0.68rem; text-transform:uppercase; color:#666C79;">Noise</th>
            <th style="text-align:left; padding:8px 10px; border-bottom:1px solid #3A4152; font-size:0.68rem; text-transform:uppercase; color:#666C79;">Verdict</th>
            </tr></thead>
            <tbody>"""
            + f"""<tr>
            <td style="padding:9px 10px; border-bottom:1px solid #2B303C; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">K-Means (k={diagnostics['n_clusters']})</td>
            <td style="padding:9px 10px; border-bottom:1px solid #2B303C; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">{diagnostics['chosen_k_silhouette']:.3f}</td>
            <td style="padding:9px 10px; border-bottom:1px solid #2B303C; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">{diagnostics['n_clusters']}</td>
            <td style="padding:9px 10px; border-bottom:1px solid #2B303C; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">0%</td>
            <td style="padding:9px 10px; border-bottom:1px solid #2B303C;"><span class="pill" style="background:{PALETTE['green']}22; color:{PALETTE['green']};">PRIMARY MODEL</span></td>
            </tr>
            <tr>
            <td style="padding:9px 10px; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">DBSCAN (eps={diagnostics['dbscan']['eps']}, min_samples={diagnostics['dbscan']['min_samples']})</td>
            <td style="padding:9px 10px; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">{diagnostics['dbscan']['silhouette_core_points'] if diagnostics['dbscan']['silhouette_core_points'] else 'n/a (1 cluster)'}</td>
            <td style="padding:9px 10px; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">{diagnostics['dbscan']['n_clusters']}</td>
            <td style="padding:9px 10px; font-family:'IBM Plex Mono',monospace; color:{PALETTE['text_primary']};">{diagnostics['dbscan']['noise_pct']}%</td>
            <td style="padding:9px 10px;"><span class="pill" style="background:{PALETTE['red']}22; color:{PALETTE['red']};">STRESS TEST ONLY</span></td>
            </tr>
            </tbody></table>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            kd = diagnostics["dbscan_k_distance"]
            fig = go.Figure(go.Scatter(y=kd, mode="lines", line=dict(color=PALETTE["violet"])))
            fig.add_hline(y=diagnostics["dbscan"]["eps"], line=dict(color=PALETTE["gold"], dash="dot"), annotation_text=f"knee eps ≈ {diagnostics['dbscan']['eps']}", annotation_font=dict(size=10, color=PALETTE["gold"]))
            fig.update_layout(**base_layout(xaxis=dict(title=f"points, sorted by distance to {diagnostics['dbscan']['min_samples']}th NN", gridcolor="#242A35"), yaxis=dict(title="distance", gridcolor="#242A35"), height=280))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with col2:
            sweep = pd.DataFrame(diagnostics["dbscan_eps_sweep"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sweep["eps_multiplier"], y=sweep["noise_pct"], mode="lines+markers", line=dict(color=PALETTE["red"])))
            fig.update_layout(**base_layout(xaxis=dict(title="eps as × of knee estimate", gridcolor="#242A35"), yaxis=dict(title="% labeled noise", gridcolor="#242A35"), height=280))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            f"""<p class="muted">DBSCAN finds clusters as dense regions separated by density valleys. Seller performance
            features form a continuous gradient with no such valleys: across the eps sweep above, the noise fraction falls
            monotonically as eps grows, but cluster count never rises above 1 -- there is no eps that produces several
            well-separated, business-meaningful clusters. This confirms the density-based assumption doesn't hold on this
            feature space; DBSCAN is a stress test on the chosen feature space, not a candidate to replace K-Means.</p>
            <p class="muted"><b style="color:{PALETTE['text_primary']}">Reframed as insight rather than a dead end:</b> the points
            DBSCAN flags as noise are, by definition, sellers sitting in sparse regions of the feature space -- behavioral
            outliers worth a manual look, independent of which K-Means segment they landed in.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>6. PCA -- for the scatter plot only</h3>", unsafe_allow_html=True)
        pv = diagnostics["pca_explained_variance_ratio"]
        st.markdown(
            f"""<p class="muted">PC1 explains {pv[0]*100:.1f}% of variance, PC2 {pv[1]*100:.1f}% -- a combined
            <b style="color:{PALETTE['text_primary']}">{diagnostics['pca_explained_variance_sum']*100:.1f}%</b>, below the ~60% mark that
            would let the Overview tab's scatter be read as a faithful map of inter-cluster distance. It is illustrative:
            useful for seeing that segments occupy distinguishable regions, not for reading exact distances between them.
            Clustering itself runs on the original {len(diagnostics['cluster_features'])} scaled features, each with direct
            business meaning, precisely so centroids translate back into "this segment has high revenue / high recency"
            rather than an uninterpretable PCA axis.</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>7. Health score</h3>", unsafe_allow_html=True)
        w = diagnostics["health_score_weights"]
        st.markdown(
            f"""<p class="muted">A weighted 0-100 composite of six normalized features. Recency, delivery delay, negative
            reviews and cancellations are "bad when high," so they enter inverted -- +1 always means healthier once
            combined. Weights favor <b style="color:{PALETTE['text_primary']}">quality</b> ({w['delay']+w['neg_review']+w['cancel']:.2f})
            over raw <b style="color:{PALETTE['text_primary']}">volume</b> ({w['revenue']+w['frequency']+w['recency']:.2f}) on purpose: in
            a marketplace, the cost of one bad partner is buyer-trust erosion that bleeds into the whole platform, not just
            their own GMV -- a business decision, documented here rather than left as unexplained numerology.</p>""",
            unsafe_allow_html=True,
        )
        weight_labels = {"revenue": "Revenue", "frequency": "Frequency", "recency": "Recency", "delay": "Delivery delay", "neg_review": "Neg. review rate", "cancel": "Cancel rate"}
        fig = go.Figure(go.Bar(x=[weight_labels[k] for k in w], y=list(w.values()), marker_color=PALETTE["gold"]))
        fig.update_layout(**base_layout(yaxis=dict(title="weight", gridcolor="#242A35", tickformat=".0%"), height=260))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with card():
        st.markdown("<h3>8. Data quality validation</h3>", unsafe_allow_html=True)
        dq = load_data_quality()
        dq_rows = "".join(
            f'<tr><td style="padding:7px 10px; border-bottom:1px solid #2B303C; color:#9CA1AD;">{k.replace("_"," ").capitalize()}</td>'
            f'<td style="padding:7px 10px; border-bottom:1px solid #2B303C; font-family:\'IBM Plex Mono\',monospace; color:{PALETTE["text_primary"]};">{fmt_int(v) if isinstance(v, (int, float)) else v}</td></tr>'
            for k, v in dq.items()
        )
        st.markdown(f'<table style="width:100%; border-collapse:collapse; font-size:0.82rem;">{dq_rows}</table>', unsafe_allow_html=True)
        st.markdown(
            """<p class="muted" style="margin-top:12px;">Checked before any join, not after: orphan order_items (0, safe to
            join), orders without items (excluded from revenue features, retained for cancellation-rate features),
            duplicate review rows per order (deduplicated by keeping the most recent), and delivered-status orders missing
            a delivered_customer_date (excluded from delivery-time features, an edge case affecting 8 orders).</p>""",
            unsafe_allow_html=True,
        )

    with card():
        st.markdown("<h3>9. Limitations & next steps</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""<ul class="muted" style="padding-left:18px;">
            <li style="margin-bottom:8px;">Review-score attribution to multi-seller orders ({diagnostics['multi_seller_order_stats']['multi_seller_orders_pct']}%
            of orders) remains an approximation inherent to the source schema, not a modeling choice that can be fixed downstream.</li>
            <li style="margin-bottom:8px;">This is a <b style="color:{PALETTE['text_primary']}">snapshot</b>, not a trend: the dataset
            ends {diagnostics['reference_date']}, and every feature is computed against that fixed reference date. A seller's
            segment membership over time -- not just at a point in time -- is the natural extension.</li>
            <li>The original design sketch anticipated a "high volume, low quality" archetype. It doesn't appear in the real
            k=5 solution: above-median order frequency only shows up in the healthiest cluster, and quality problems
            concentrate in low/mid-volume sellers instead. Segment names and copy were rewritten to match what the data
            actually shows rather than forcing a preconceived narrative.</li>
            </ul>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    inject_css()
    sellers = load_sellers()
    profiles = load_profiles()
    diagnostics = load_diagnostics()

    tab = render_sidebar()

    if tab == "Overview":
        render_overview(sellers, profiles, diagnostics)
    elif tab == "Segment Explorer":
        render_segment_explorer(sellers, profiles)
    elif tab == "Seller Lookup":
        render_seller_lookup(sellers)
    elif tab == "Methodology":
        render_methodology(diagnostics, profiles)


if __name__ == "__main__":
    main()
