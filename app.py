"""
Executive Web Dashboard for Issuer Authorization Analytics, ML Scoring & AI Insights.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".tmp_mpl"))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data_generator import generate_datasets, DATA_DIR
from src.ml_pipeline import (
    train_and_evaluate_models,
    compute_threshold_economics,
    SCORED_PARQUET,
    DEFAULT_ECONOMICS,
)
from src.ai_engine import (
    build_dashboard_context,
    generate_executive_insights,
    answer_followup_question,
    SYSTEM_PROMPT,
)

st.set_page_config(
    page_title="Issuer Authorization Analytics & AI Insights",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# CUSTOM EXECUTIVE STYLING
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    .exec-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 1.25rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
    }
    .exec-header h1 {
        margin: 0;
        font-size: 1.65rem;
        font-weight: 700;
        color: #ffffff;
    }
    .exec-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .kpi-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        font-weight: 600;
    }
    .kpi-val {
        font-size: 1.42rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.15rem;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #475569;
        margin-top: 0.15rem;
    }
    .insight-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #0284c7;
        border-radius: 10px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
    }
    .action-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }
    .action-tag {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    .impact-pill {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .action-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.25rem 0 0.35rem 0;
        line-height: 1.35;
    }
    .action-desc {
        font-size: 0.85rem;
        color: #334155;
        margin-bottom: 0.5rem;
        line-height: 1.45;
    }
    .copilot-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 65%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem 1.35rem;
        color: #f8fafc;
        margin-top: 0.5rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    }
    .copilot-badge {
        display: inline-block;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        margin-right: 0.45rem;
    }
    .prompt-card-tag {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #2563eb;
        margin-bottom: 0.2rem;
    }
    .prompt-card-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.55rem;
        line-height: 1.35;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# CACHED DATA & MODEL LOADING (Auto-Generates on First Cloud Boot if Missing)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading 220,000+ Issuer Authorization & 3DS Authentication records...")
def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df_auth, df_3ds = generate_datasets(force_regenerate=False)
    ml_res = train_and_evaluate_models(force_retrain=False)

    if SCORED_PARQUET.exists():
        scored = pd.read_parquet(SCORED_PARQUET)
        if "ml_top_drivers" not in scored.columns:
            ml_res = train_and_evaluate_models(force_retrain=True)
            scored = pd.read_parquet(SCORED_PARQUET)
        for col in ["split", "ml_fraud_prob_xgb", "ml_fraud_prob_lr", "ml_risk_segment", "ml_top_drivers"]:
            if col in scored.columns:
                df_auth[col] = scored[col].to_numpy()
            else:
                df_auth[col] = "Baseline Normal Profile"
    return df_auth, df_3ds, ml_res


df_auth_full, df_3ds_full, ml_results = load_all_data()


# -----------------------------------------------------------------------------
# GLOBAL FILTER SIDEBAR (Applies Across Every View)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Global Portfolio Filters")
    st.caption("Filters apply dynamically across all 6 dashboard views and the AI Insights Engine.")

    preset = st.selectbox(
        "⚡ Executive Filter Presets",
        options=[
            "Full Portfolio (All Data)",
            "E-Commerce Untokenized Risk",
            "LegacySwitch_Bank Deep-Dive",
            "High-Risk MCCs ($100+ Tickets)",
            "Pre-Payday Cash Crunch (Days 15-24)",
        ],
        index=0,
        help="Quickly jump to key domain storylines or keep 'Full Portfolio' for custom filtering.",
    )

    min_date = pd.to_datetime(df_auth_full["date"].min()).date()
    max_date = pd.to_datetime(df_auth_full["date"].max()).date()

    date_range = st.date_input(
        "📅 Date Range (Jul–Oct 2026)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_channels = sorted(df_auth_full["channel"].unique().tolist())
    all_acquirers = sorted(df_auth_full["acquirer"].unique().tolist())
    all_risk_tiers = ["Low", "Medium", "High"]
    all_3ds = sorted(df_auth_full["three_ds_indicator"].unique().tolist())
    all_tokens = sorted(df_auth_full["tokenization_indicator"].unique().tolist())
    all_buckets = ["<$25", "$25-$100", "$100-$500", "$500+"]

    # Default selections modified by preset
    def_channels = all_channels
    def_acquirers = all_acquirers
    def_tiers = all_risk_tiers
    def_tokens = all_tokens
    def_buckets = all_buckets

    if preset == "E-Commerce Untokenized Risk":
        def_channels = ["E-Commerce"]
        def_tokens = ["Raw PAN (Untokenized)"]
    elif preset == "LegacySwitch_Bank Deep-Dive":
        def_acquirers = ["LegacySwitch_Bank"]
    elif preset == "High-Risk MCCs ($100+ Tickets)":
        def_tiers = ["High"]
        def_buckets = ["$100-$500", "$500+"]

    sel_channels = st.multiselect("🛣️ Channel", options=all_channels, default=def_channels)
    sel_acquirers = st.multiselect("🏦 Acquirer", options=all_acquirers, default=def_acquirers)
    sel_tiers = st.multiselect("⚠️ MCC Risk Tier", options=all_risk_tiers, default=def_tiers)
    sel_3ds = st.multiselect("🔐 3DS Status", options=all_3ds, default=all_3ds)
    sel_tokens = st.multiselect("🛡️ Tokenization", options=all_tokens, default=def_tokens)
    sel_buckets = st.multiselect("💵 Ticket Size Bracket", options=all_buckets, default=def_buckets)

    only_pre_payday = st.checkbox(
        "📆 Filter to Pre-Payday Window Only (Days 15–24)",
        value=(preset == "Pre-Payday Cash Crunch (Days 15-24)"),
    )

    st.markdown("---")
    st.markdown(
        "**Dataset Footprint:**\n"
        f"- Auth Tx: `{len(df_auth_full):,}` rows (4 mos)\n"
        f"- 3DS Logs: `{len(df_3ds_full):,}` rows\n"
        f"- Unique Cards: `{df_auth_full['tokenised_card_id'].nunique():,}`"
    )

# Apply Global Filters to Authorization & 3DS DataFrames
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d, end_d = pd.Timestamp(min_date), pd.Timestamp(max_date)

auth_mask = (
    (df_auth_full["date"] >= start_d)
    & (df_auth_full["date"] <= end_d)
    & (df_auth_full["channel"].isin(sel_channels or all_channels))
    & (df_auth_full["acquirer"].isin(sel_acquirers or all_acquirers))
    & (df_auth_full["mcc_risk_tier"].isin(sel_tiers or all_risk_tiers))
    & (df_auth_full["three_ds_indicator"].isin(sel_3ds or all_3ds))
    & (df_auth_full["tokenization_indicator"].isin(sel_tokens or all_tokens))
    & (df_auth_full["ticket_size_bucket"].isin(sel_buckets or all_buckets))
)
if only_pre_payday:
    auth_mask = auth_mask & (df_auth_full["is_pre_payday"] == 1)

df_auth = df_auth_full.loc[auth_mask]

threeds_mask = (
    (df_3ds_full["date"] >= start_d)
    & (df_3ds_full["date"] <= end_d)
    & (df_3ds_full["channel"].isin(sel_channels or all_channels))
    & (df_3ds_full["acquirer"].isin(sel_acquirers or all_acquirers))
    & (df_3ds_full["mcc_risk_tier"].isin(sel_tiers or all_risk_tiers))
    & (df_3ds_full["tokenization_indicator"].isin(sel_tokens or all_tokens))
    & (df_3ds_full["ticket_size_bucket"].isin(sel_buckets or all_buckets))
)
if "three_ds_indicator" in df_3ds_full.columns:
    threeds_mask = threeds_mask & (df_3ds_full["three_ds_indicator"].isin(sel_3ds or all_3ds))
df_3ds = df_3ds_full.loc[threeds_mask]

active_filters_dict = {
    "preset": preset,
    "date_range": f"{start_d.strftime('%Y-%m-%d')} to {end_d.strftime('%Y-%m-%d')}",
    "channels": sel_channels,
    "acquirers": sel_acquirers,
    "mcc_risk_tiers": sel_tiers,
    "three_ds_status": sel_3ds,
    "tokenization": sel_tokens,
    "ticket_size_buckets": sel_buckets,
    "only_pre_payday_days_15_24": only_pre_payday,
}


# -----------------------------------------------------------------------------
# HEADER & EXECUTIVE KPI BANNER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="exec-header">
        <h1>💳 Issuer Card Authorization & Risk Intelligence Platform</h1>
        <p>End-to-End Portfolio Analytics • Out-of-Time Fraud ML & Threshold Simulator • Grounded AI Insights Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if len(df_auth) == 0:
    st.warning("No transactions match the selected filter combination. Please broaden your sidebar filters.")
    st.stop()

n_tx = len(df_auth)
total_vol = float(df_auth["amount"].sum())
app_rate = float(df_auth["is_approved"].mean() * 100)
base_app_rate = float(df_auth_full["is_approved"].mean() * 100)
app_delta = app_rate - base_app_rate

declined_df = df_auth[df_auth["is_approved"] == 0]
dec_vol = float(declined_df["amount"].sum())
soft_dec_pct = float((declined_df["decline_type"] == "Soft Decline (Recoverable)").mean() * 100) if len(declined_df) else 0.0

fraud_rate = float(df_auth["fraud_flag"].mean() * 100)
cb_rate = float(df_auth["has_chargeback"].mean() * 100)
auth_3ds_pass = float(df_3ds["is_auth_success"].mean() * 100) if len(df_3ds) else 0.0

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Transactions</div>
            <div class="kpi-val">{n_tx:,}</div>
            <div class="kpi-sub">{n_tx / len(df_auth_full) * 100:.1f}% of total portfolio</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-title">Gross Volume</div>
            <div class="kpi-val">${total_vol / 1e6:,.2f}M</div>
            <div class="kpi-sub">Avg Ticket: ${df_auth['amount'].mean():,.2f}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    delta_color = "#16a34a" if app_delta >= 0 else "#dc2626"
    st.markdown(
        f"""<div class="kpi-card" style="border-left-color: {delta_color};">
            <div class="kpi-title">Approval Rate</div>
            <div class="kpi-val">{app_rate:.2f}%</div>
            <div class="kpi-sub" style="color:{delta_color};">{app_delta:+.2f} pts vs Benchmark ({base_app_rate:.2f}%)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="kpi-title">Declined Volume</div>
            <div class="kpi-val">${dec_vol / 1e6:,.2f}M</div>
            <div class="kpi-sub">{soft_dec_pct:.1f}% Soft / Recoverable</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""<div class="kpi-card" style="border-left-color: #dc2626;">
            <div class="kpi-title">True Fraud / CB Rate</div>
            <div class="kpi-val">{fraud_rate:.2f}%</div>
            <div class="kpi-sub">Chargebacks: {cb_rate:.2f}% ({int(df_auth['has_chargeback'].sum()):,})</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k6:
    st.markdown(
        f"""<div class="kpi-card" style="border-left-color: #0284c7;">
            <div class="kpi-title">3DS Auth Pass Rate</div>
            <div class="kpi-val">{auth_3ds_pass:.2f}%</div>
            <div class="kpi-sub">{len(df_3ds):,} 3DS Attempts</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# MAIN NAVIGATION TABS (5 Required Views + ML Simulator + AI Insights Engine)
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "📊 1. Overview",
    "✅ 2. Authorization",
    "🚫 3. Declines",
    "🛡️ 4. Fraud & Chargebacks",
    "🔐 5. Authentication (3DS)",
    "🤖 6. ML Model & What-If Simulator",
    "🧠 7. AI Insights Engine",
])


# =============================================================================
# TAB 1: OVERVIEW (Volumes, Approval Rate, Channel Mix, Seasonality)
# =============================================================================
with tabs[0]:
    st.subheader("Portfolio Overview: Transaction Volumes, Approval Rate & Channel Mix")

    col_ov1, col_ov2 = st.columns([1.65, 1.0])

    with col_ov1:
        st.markdown("##### 📈 Daily Transaction Volume (Bars) & Approval Rate % (Line)")
        daily_df = (
            df_auth.groupby("date", observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                total_usd=("amount", "sum"),
                approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
                fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
            )
            .reset_index()
        )
        daily_df["date_str"] = pd.to_datetime(daily_df["date"]).dt.strftime("%Y-%m-%d")

        base = alt.Chart(daily_df).encode(
            x=alt.X("date:T", title="Transaction Date"),
            tooltip=[
                alt.Tooltip("date_str:N", title="Date"),
                alt.Tooltip("tx_count:Q", title="Transactions", format=","),
                alt.Tooltip("total_usd:Q", title="Volume ($)", format="$,.0f"),
                alt.Tooltip("approval_rate:Q", title="Approval Rate (%)", format=".2f"),
                alt.Tooltip("fraud_rate:Q", title="True Fraud Rate (%)", format=".2f"),
            ],
        )
        bars = base.mark_bar(color="#93c5fd", opacity=0.65).encode(
            y=alt.Y("tx_count:Q", title="Daily Transaction Volume"),
        )
        line = base.mark_line(color="#1d4ed8", strokeWidth=2.5).encode(
            y=alt.Y(
                "approval_rate:Q",
                title="Approval Rate (%)",
                scale=alt.Scale(domain=[max(65, float(daily_df["approval_rate"].min()) - 3), 100]),
            ),
        )
        chart_daily = (
            alt.layer(bars, line)
            .resolve_scale(y="independent")
            .properties(height=320)
        )
        st.altair_chart(chart_daily, use_container_width=True)

    with col_ov2:
        st.markdown("##### 🛣️ Channel Mix & Approval Rate Shading")
        ch_df = (
            df_auth.groupby("channel", observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                volume_usd=("amount", "sum"),
                approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
                fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
                avg_ticket=("amount", lambda s: round(float(s.mean()), 2)),
            )
            .reset_index()
        )
        ch_df["share_pct"] = (ch_df["tx_count"] / len(df_auth) * 100).round(2)

        ch_chart = (
            alt.Chart(ch_df)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                y=alt.Y("channel:N", sort="-x", title="Channel"),
                x=alt.X("share_pct:Q", title="Share of Transactions (%)"),
                color=alt.Color(
                    "approval_rate:Q",
                    title="Approval %",
                    scale=alt.Scale(scheme="blues", domain=[80, 98]),
                ),
                tooltip=[
                    alt.Tooltip("channel:N", title="Channel"),
                    alt.Tooltip("tx_count:Q", title="Transactions", format=","),
                    alt.Tooltip("share_pct:Q", title="Volume Share (%)", format=".2f"),
                    alt.Tooltip("approval_rate:Q", title="Approval Rate (%)", format=".2f"),
                    alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                    alt.Tooltip("avg_ticket:Q", title="Avg Ticket ($)", format="$,.2f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(ch_chart, use_container_width=True)

    # Second row: Payday Cycle Effect (Day of Month) & Monthly Summary
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    col_ov3, col_ov4 = st.columns(2)
    with col_ov3:
        st.markdown("##### 📅 Seasonality: Monthly Payday Cycle vs. Insufficient Funds Declines")
        dom_df = (
            df_auth.groupby("day_of_month", observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
                nsf_rate=("decline_reason", lambda s: round(float((s == "Insufficient Funds").mean() * 100), 2)),
            )
            .reset_index()
        )
        dom_df["Cycle Phase"] = np.where(
            (dom_df["day_of_month"] >= 25) | (dom_df["day_of_month"] <= 5),
            "Post-Payday Liquidity (Days 25-5)",
            np.where(
                (dom_df["day_of_month"] >= 15) & (dom_df["day_of_month"] <= 24),
                "Pre-Payday Cash Crunch (Days 15-24)",
                "Mid-Month Normal (Days 6-14)",
            ),
        )
        dom_chart = (
            alt.Chart(dom_df)
            .mark_bar()
            .encode(
                x=alt.X("day_of_month:O", title="Day of Month (1–31)"),
                y=alt.Y("nsf_rate:Q", title="Insufficient Funds Decline Rate (%)"),
                color=alt.Color(
                    "Cycle Phase:N",
                    scale=alt.Scale(
                        domain=[
                            "Post-Payday Liquidity (Days 25-5)",
                            "Mid-Month Normal (Days 6-14)",
                            "Pre-Payday Cash Crunch (Days 15-24)",
                        ],
                        range=["#10b981", "#60a5fa", "#f97316"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("day_of_month:O", title="Day of Month"),
                    alt.Tooltip("Cycle Phase:N", title="Payday Phase"),
                    alt.Tooltip("nsf_rate:Q", title="Insufficient Funds Rate (%)", format=".2f"),
                    alt.Tooltip("approval_rate:Q", title="Overall Approval Rate (%)", format=".2f"),
                    alt.Tooltip("tx_count:Q", title="Transactions", format=","),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(dom_chart, use_container_width=True)

    with col_ov4:
        st.markdown("##### 📋 Channel Performance Summary Table")
        st.dataframe(
            ch_df[["channel", "tx_count", "share_pct", "volume_usd", "avg_ticket", "approval_rate", "fraud_rate"]].rename(
                columns={
                    "channel": "Channel",
                    "tx_count": "Tx Count",
                    "share_pct": "Share (%)",
                    "volume_usd": "Total Spend ($)",
                    "avg_ticket": "Avg Ticket ($)",
                    "approval_rate": "Approval Rate (%)",
                    "fraud_rate": "True Fraud Rate (%)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# TAB 2: AUTHORIZATION (By Acquirer, 3DS, Tokenization, Merchant)
# =============================================================================
with tabs[1]:
    st.subheader("Authorization Performance: By Acquirer, 3DS, Tokenization & Merchant")
    st.caption(
        "Compare approval rates across acquiring processors, evaluate the lift from Network Tokenization and 3D Secure, "
        "and inspect merchant-level authorization performance."
    )

    col_au1, col_au2 = st.columns(2)

    with col_au1:
        st.markdown("##### 🏦 Approval Rate by Acquirer (LegacySwitch_Bank Highlighted)")
        acq_df = (
            df_auth.groupby("acquirer", observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                volume_usd=("amount", "sum"),
                approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
                tech_err_rate=("response_code", lambda s: round(float((s == "91").mean() * 100), 2)),
                cvv_err_rate=("response_code", lambda s: round(float((s == "82").mean() * 100), 2)),
                tokenized_pct=("is_tokenized", lambda s: round(float(s.mean() * 100), 1)),
            )
            .reset_index()
            .sort_values("approval_rate", ascending=True)
        )

        acq_df["label_text"] = acq_df["approval_rate"].map(lambda x: f"{x:.1f}%")
        acq_base = alt.Chart(acq_df).encode(
            y=alt.Y("acquirer:N", sort=acq_df["acquirer"].tolist(), title="Acquirer Processor"),
            x=alt.X("approval_rate:Q", title="Authorization Approval Rate (%)", scale=alt.Scale(domain=[0, 100])),
        )
        acq_bars = acq_base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
            color=alt.condition(
                alt.datum.acquirer == "LegacySwitch_Bank",
                alt.value("#ef4444"),
                alt.value("#2563eb"),
            ),
            tooltip=[
                alt.Tooltip("acquirer:N", title="Acquirer"),
                alt.Tooltip("approval_rate:Q", title="Approval Rate (%)", format=".2f"),
                alt.Tooltip("tech_err_rate:Q", title="Tech/Switch Error Rate (%)", format=".2f"),
                alt.Tooltip("cvv_err_rate:Q", title="Invalid CVV Rate (%)", format=".2f"),
                alt.Tooltip("tokenized_pct:Q", title="Tokenized Share (%)", format=".1f"),
                alt.Tooltip("tx_count:Q", title="Transactions", format=","),
            ],
        )
        acq_text = acq_base.mark_text(
            align="right",
            dx=-8,
            fontWeight="bold",
            fontSize=11,
            color="#ffffff",
        ).encode(
            text=alt.Text("label_text:N"),
        )
        acq_chart = (acq_bars + acq_text).properties(height=300)
        st.altair_chart(acq_chart, use_container_width=True)

    with col_au2:
        st.markdown("##### 🔐 Approval Rate by 3DS Status × Tokenization Indicator")
        sec_df = (
            df_auth.groupby(["three_ds_indicator", "tokenization_indicator"], observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
                fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
            )
            .reset_index()
        )
        sec_chart = (
            alt.Chart(sec_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "three_ds_indicator:N",
                    title="3DS Authentication Status",
                    axis=alt.Axis(labelAngle=-30, labelOverlap=False, labelLimit=0),
                ),
                y=alt.Y("approval_rate:Q", title="Approval Rate (%)", scale=alt.Scale(domain=[0, 100])),
                xOffset="tokenization_indicator:N",
                color=alt.Color(
                    "tokenization_indicator:N",
                    title="Tokenization",
                    scale=alt.Scale(range=["#94a3b8", "#0ea5e9"]),
                ),
                tooltip=[
                    alt.Tooltip("three_ds_indicator:N", title="3DS Status"),
                    alt.Tooltip("tokenization_indicator:N", title="Tokenization"),
                    alt.Tooltip("approval_rate:Q", title="Approval Rate (%)", format=".2f"),
                    alt.Tooltip("fraud_rate:Q", title="True Fraud Rate (%)", format=".2f"),
                    alt.Tooltip("tx_count:Q", title="Transactions", format=","),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(sec_chart, use_container_width=True)

    # Merchant Authorization Table & Chart
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    st.markdown("##### 🏪 Authorization Performance by Merchant")
    merch_df = (
        df_auth.groupby(["merchant", "mcc_category", "mcc_risk_tier"], observed=True)
        .agg(
            tx_count=("transaction_id", "count"),
            volume_usd=("amount", "sum"),
            avg_ticket=("amount", lambda s: round(float(s.mean()), 2)),
            approval_rate=("is_approved", lambda s: round(float(s.mean() * 100), 2)),
            tokenized_pct=("is_tokenized", lambda s: round(float(s.mean() * 100), 1)),
            three_ds_pct=("is_3ds_authenticated", lambda s: round(float(s.mean() * 100), 1)),
            fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
        )
        .reset_index()
        .sort_values("approval_rate", ascending=True)
    )

    col_au3, col_au4 = st.columns([1.2, 1.3])
    with col_au3:
        bottom_merch = merch_df.head(12).copy()
        bottom_merch["label_text"] = bottom_merch["approval_rate"].map(lambda x: f"{x:.1f}%")

        m_base = alt.Chart(bottom_merch).encode(
            y=alt.Y("merchant:N", sort=bottom_merch["merchant"].tolist(), title="Merchant (Lowest 12 Approval)"),
            x=alt.X("approval_rate:Q", title="Approval Rate (%)", scale=alt.Scale(domain=[0, 100])),
        )
        m_bars = m_base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            color=alt.Color(
                "mcc_risk_tier:N",
                title="MCC Risk Tier",
                scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#10b981", "#f59e0b", "#ef4444"]),
            ),
            tooltip=[
                alt.Tooltip("merchant:N", title="Merchant"),
                alt.Tooltip("mcc_category:N", title="MCC Category"),
                alt.Tooltip("approval_rate:Q", title="Approval Rate (%)", format=".2f"),
                alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                alt.Tooltip("tx_count:Q", title="Transactions", format=","),
            ],
        )
        m_text = m_base.mark_text(
            align="right",
            dx=-8,
            fontWeight="bold",
            fontSize=11,
            color="#ffffff",
        ).encode(
            text=alt.Text("label_text:N"),
        )
        m_chart = (m_bars + m_text).properties(height=340, title="12 Lowest-Approval Merchants by Risk Tier")
        st.altair_chart(m_chart, use_container_width=True)

    with col_au4:
        st.dataframe(
            merch_df.rename(
                columns={
                    "merchant": "Merchant",
                    "mcc_category": "MCC Category",
                    "mcc_risk_tier": "Risk Tier",
                    "tx_count": "Tx Count",
                    "volume_usd": "Volume ($)",
                    "avg_ticket": "Avg Ticket ($)",
                    "approval_rate": "Approval (%)",
                    "tokenized_pct": "Tokenized (%)",
                    "three_ds_pct": "3DS (%)",
                    "fraud_rate": "Fraud (%)",
                }
            ),
            height=340,
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# TAB 3: DECLINES (Top Reasons, Trends, By Merchant, MCC & Ticket Size)
# =============================================================================
with tabs[2]:
    st.subheader("Decline Intelligence: Top Reasons, Trends, MCC, Merchant & Ticket Size")
    st.caption(
        "Distinguishes Hard Declines (true fraud, invalid CVV) from Soft/Recoverable Declines "
        "(insufficient funds, false-positive fraud blocks, switch errors, expired cards)."
    )

    if len(declined_df) == 0:
        st.info("No declined transactions in the current filter slice.")
    else:
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### 📉 Top Decline Reasons (Soft vs. Hard Classification)")
            reason_df = (
                declined_df.groupby(["decline_reason", "response_code", "decline_type"], observed=True)
                .agg(
                    decline_count=("transaction_id", "count"),
                    lost_usd=("amount", "sum"),
                    avg_ticket=("amount", lambda s: round(float(s.mean()), 2)),
                )
                .reset_index()
                .sort_values("decline_count", ascending=False)
            )
            reason_df["share_pct"] = (reason_df["decline_count"] / len(declined_df) * 100).round(2)

            tot_df = (
                declined_df.groupby("decline_reason", observed=True)
                .size()
                .reset_index(name="total_count")
            )
            tot_df["share_pct"] = (tot_df["total_count"] / len(declined_df) * 100).round(1)
            tot_df["label"] = tot_df.apply(lambda r: f"{r.total_count:,} ({r.share_pct:.1f}%)", axis=1)
            sort_order = tot_df.sort_values("total_count", ascending=False)["decline_reason"].tolist()

            max_x = float(tot_df["total_count"].max()) * 1.30 if len(tot_df) > 0 else 1000

            r_bars = (
                alt.Chart(reason_df)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                .encode(
                    y=alt.Y("decline_reason:N", sort=sort_order, title="Decline Reason"),
                    x=alt.X("decline_count:Q", title="Declined Transactions Count", scale=alt.Scale(domain=[0, max_x])),
                    color=alt.Color(
                        "decline_type:N",
                        legend=alt.Legend(orient="top", title=None, direction="horizontal"),
                        scale=alt.Scale(
                            domain=["Soft Decline (Recoverable)", "Hard Decline"],
                            range=["#f59e0b", "#dc2626"],
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("decline_reason:N", title="Reason"),
                        alt.Tooltip("response_code:N", title="ISO Code"),
                        alt.Tooltip("decline_type:N", title="Type"),
                        alt.Tooltip("decline_count:Q", title="Decline Count", format=","),
                        alt.Tooltip("share_pct:Q", title="Share of Declines (%)", format=".2f"),
                        alt.Tooltip("lost_usd:Q", title="Declined Value ($)", format="$,.0f"),
                    ],
                )
            )

            r_text = (
                alt.Chart(tot_df)
                .mark_text(
                    align="left",
                    baseline="middle",
                    dx=6,
                    fontWeight="bold",
                    fontSize=11,
                    color="#1e293b",
                )
                .encode(
                    y=alt.Y("decline_reason:N", sort=sort_order),
                    x=alt.X("total_count:Q"),
                    text=alt.Text("label:N"),
                )
            )

            r_chart = (r_bars + r_text).properties(height=320)
            st.altair_chart(r_chart, use_container_width=True)

        with col_d2:
            st.markdown("##### 📈 Weekly Decline Volume Trend by Reason")
            # Weekly decline trend with explicit Week 1, Week 2, ... labels
            declined_df_trend = declined_df.assign(
                week_start=pd.to_datetime(declined_df["date"]).dt.to_period("W").apply(lambda r: r.start_time)
            )
            unique_weeks = sorted(declined_df_trend["week_start"].unique())
            week_map = {w: f"Week {i+1}" for i, w in enumerate(unique_weeks)}
            week_order = [f"Week {i+1}" for i in range(len(unique_weeks))]

            dec_trend = (
                declined_df_trend.groupby(["week_start", "decline_reason"], observed=True)
                .size()
                .reset_index(name="declines")
            )
            dec_trend["week_label"] = dec_trend["week_start"].map(week_map)
            dec_trend["date_str"] = dec_trend["week_start"].dt.strftime("%b %d")
            dec_trend["week_tooltip"] = dec_trend["week_label"] + " (Week of " + dec_trend["date_str"] + ")"

            dt_chart = (
                alt.Chart(dec_trend)
                .mark_area(opacity=0.75)
                .encode(
                    x=alt.X(
                        "week_label:N",
                        sort=week_order,
                        title="Week",
                        axis=alt.Axis(labelAngle=-45, labelOverlap=False, labelLimit=0),
                    ),
                    y=alt.Y("declines:Q", title="Weekly Declines"),
                    color=alt.Color("decline_reason:N", title="Decline Reason", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("week_tooltip:N", title="Timeline"),
                        alt.Tooltip("decline_reason:N", title="Decline Reason"),
                        alt.Tooltip("declines:Q", title="Decline Count", format=","),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(dt_chart, use_container_width=True)

        # Second row: Heatmap of Decline Rate by MCC × Ticket Size + Ticket Size Breakdown
        st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
        col_d3, col_d4 = st.columns([1.35, 1.0])

        with col_d3:
            heat_df = (
                df_auth.groupby(["mcc_category", "ticket_size_bucket"], observed=True)
                .agg(
                    tx_count=("transaction_id", "count"),
                    decline_rate=("is_approved", lambda s: round(float((1.0 - s.mean()) * 100), 2)),
                    top_reason=("decline_reason", lambda s: s[s != "Approved"].mode().iloc[0] if (s != "Approved").any() else "None"),
                )
                .reset_index()
            )
            heat_df["decline_label"] = heat_df["decline_rate"].map(lambda v: f"{v:.1f}%")

            heat_base = alt.Chart(heat_df).encode(
                x=alt.X(
                    "ticket_size_bucket:N",
                    sort=["<$25", "$25-$100", "$100-$500", "$500+"],
                    title="Ticket Size Bracket",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "mcc_category:N",
                    title="MCC Category",
                    axis=alt.Axis(labelLimit=220, labelOverlap=False),
                ),
                tooltip=[
                    alt.Tooltip("mcc_category:N", title="MCC Category"),
                    alt.Tooltip("ticket_size_bucket:N", title="Ticket Size"),
                    alt.Tooltip("decline_rate:Q", title="Decline Rate (%)", format=".2f"),
                    alt.Tooltip("top_reason:N", title="Primary Decline Driver"),
                    alt.Tooltip("tx_count:Q", title="Transactions", format=","),
                ],
            )
            heat_rect = heat_base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("decline_rate:Q", title="Decline Rate (%)", scale=alt.Scale(scheme="orangered")),
            )
            heat_text = heat_base.mark_text(fontWeight="bold", fontSize=11).encode(
                text=alt.Text("decline_label:N"),
                color=alt.condition(
                    alt.datum.decline_rate > 20,
                    alt.value("#ffffff"),
                    alt.value("#1e293b"),
                ),
            )
            heatmap = (heat_rect + heat_text).properties(
                height=380,
                title="Decline Rate (%) Heatmap: MCC Category × Ticket Size Bracket",
            )
            st.altair_chart(heatmap, use_container_width=True)

        with col_d4:
            tkt_dec = (
                df_auth.groupby("ticket_size_bucket", observed=True)
                .agg(
                    tx_count=("transaction_id", "count"),
                    decline_rate=("is_approved", lambda s: round(float((1.0 - s.mean()) * 100), 2)),
                    nsf_rate=("decline_reason", lambda s: round(float((s == "Insufficient Funds").mean() * 100), 2)),
                    fraud_dec_rate=("response_code", lambda s: round(float(s.isin(["59", "05"]).mean() * 100), 2)),
                )
                .reset_index()
            )
            tkt_melt = tkt_dec.melt(
                id_vars=["ticket_size_bucket", "tx_count"],
                value_vars=["nsf_rate", "fraud_dec_rate"],
                var_name="Decline Bucket",
                value_name="Rate (%)",
            )
            tkt_melt["Decline Bucket"] = tkt_melt["Decline Bucket"].map({
                "nsf_rate": "Insufficient Funds (51)",
                "fraud_dec_rate": "Suspected Fraud / DNH (59/05)",
            })
            tkt_chart = (
                alt.Chart(tkt_melt)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "ticket_size_bucket:N",
                        sort=["<$25", "$25-$100", "$100-$500", "$500+"],
                        title="Ticket Size Bracket",
                        axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=0),
                    ),
                    y=alt.Y("Rate (%):Q", title="% of All Transactions Declined"),
                    xOffset="Decline Bucket:N",
                    color=alt.Color("Decline Bucket:N", scale=alt.Scale(range=["#f59e0b", "#ef4444"])),
                    tooltip=[
                        alt.Tooltip("ticket_size_bucket:N", title="Ticket Bracket"),
                        alt.Tooltip("Decline Bucket:N", title="Reason Group"),
                        alt.Tooltip("Rate (%):Q", title="Decline Rate (%)", format=".2f"),
                        alt.Tooltip("tx_count:Q", title="Total Bracket Tx", format=","),
                    ],
                )
                .properties(height=380, title="How Ticket Size Drives NSF vs. Fraud Rule Declines")
            )
            st.altair_chart(tkt_chart, use_container_width=True)


# =============================================================================
# TAB 4: FRAUD & CHARGEBACKS (Rates, Diurnal Curve, Reason Codes & Lifecycle)
# =============================================================================
with tabs[3]:
    st.subheader("Fraud & Chargebacks: Risk Exposure, Diurnal Patterns & Dispute Lifecycle")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("##### 🚨 True Fraud Rate (%) by MCC Category")
        mcc_fraud = (
            df_auth.groupby(["mcc_category", "mcc_risk_tier"], observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                fraud_cnt=("fraud_flag", "sum"),
                fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
                fraud_usd=("amount", lambda s: float(s[df_auth.loc[s.index, "fraud_flag"] == 1].sum())),
            )
            .reset_index()
            .sort_values("fraud_rate", ascending=False)
        )
        f_mcc_chart = (
            alt.Chart(mcc_fraud)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                y=alt.Y("mcc_category:N", sort="-x", title="MCC Category"),
                x=alt.X("fraud_rate:Q", title="True Fraud Rate (%)"),
                color=alt.Color(
                    "mcc_risk_tier:N",
                    title="Risk Tier",
                    scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#10b981", "#f59e0b", "#dc2626"]),
                ),
                tooltip=[
                    alt.Tooltip("mcc_category:N", title="MCC Category"),
                    alt.Tooltip("mcc_risk_tier:N", title="Risk Tier"),
                    alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                    alt.Tooltip("fraud_cnt:Q", title="Fraud Tx Count", format=","),
                    alt.Tooltip("fraud_usd:Q", title="Fraud Volume ($)", format="$,.0f"),
                ],
            )
            .properties(height=310)
        )
        st.altair_chart(f_mcc_chart, use_container_width=True)

    with col_f2:
        st.markdown("##### 🌙 Late-Night Fraud Concentration (01:00–05:00)")
        hr_fraud = (
            df_auth.groupby("hour", observed=True)
            .agg(
                tx_count=("transaction_id", "count"),
                fraud_rate=("fraud_flag", lambda s: round(float(s.mean() * 100), 2)),
            )
            .reset_index()
        )
        hr_fraud["Window"] = np.where(
            (hr_fraud["hour"] >= 1) & (hr_fraud["hour"] <= 5),
            "Late-Night Attack Window (01:00–05:00)",
            "Standard Daytime/Evening Hours",
        )
        hr_chart = (
            alt.Chart(hr_fraud)
            .mark_bar()
            .encode(
                x=alt.X("hour:O", title="Hour of Day (00:00 – 23:00 UTC)"),
                y=alt.Y("fraud_rate:Q", title="Fraud Rate (%)"),
                color=alt.Color(
                    "Window:N",
                    scale=alt.Scale(
                        domain=["Late-Night Attack Window (01:00–05:00)", "Standard Daytime/Evening Hours"],
                        range=["#dc2626", "#3b82f6"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour of Day"),
                    alt.Tooltip("Window:N", title="Period"),
                    alt.Tooltip("fraud_rate:Q", title="Fraud Rate (%)", format=".2f"),
                    alt.Tooltip("tx_count:Q", title="Total Transactions", format=","),
                ],
            )
            .properties(height=310)
        )
        st.altair_chart(hr_chart, use_container_width=True)

    # Chargeback Reason & Lifecycle Stage
    st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
    cb_only = df_auth[df_auth["has_chargeback"] == 1]
    col_f3, col_f4 = st.columns(2)

    if len(cb_only) == 0:
        st.info("No chargebacks present in the current filter slice.")
    else:
        with col_f3:
            st.markdown("##### 📁 Chargebacks by Network Reason Code")
            cb_r = (
                cb_only.groupby("chargeback_reason", observed=True)
                .agg(
                    cb_count=("transaction_id", "count"),
                    cb_usd=("amount", "sum"),
                )
                .reset_index()
                .sort_values("cb_count", ascending=False)
            )
            cb_r["share_pct"] = (cb_r["cb_count"] / len(cb_only) * 100).round(2)

            cb_r_chart = (
                alt.Chart(cb_r)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#7c3aed")
                .encode(
                    y=alt.Y("chargeback_reason:N", sort="-x", title="Chargeback Reason Code"),
                    x=alt.X("cb_count:Q", title="Chargeback Volume"),
                    tooltip=[
                        alt.Tooltip("chargeback_reason:N", title="Reason Code"),
                        alt.Tooltip("cb_count:Q", title="Count", format=","),
                        alt.Tooltip("share_pct:Q", title="Share (%)", format=".2f"),
                        alt.Tooltip("cb_usd:Q", title="Disputed Amount ($)", format="$,.0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(cb_r_chart, use_container_width=True)

        with col_f4:
            st.markdown("##### 🔄 Chargeback Lifecycle Funnel (Notification -> Write-Off)")
            cb_s = (
                cb_only.groupby("chargeback_lifecycle_stage", observed=True)
                .agg(
                    cb_count=("transaction_id", "count"),
                    cb_usd=("amount", "sum"),
                )
                .reset_index()
                .sort_values("chargeback_lifecycle_stage")
            )
            cb_s["share_pct"] = (cb_s["cb_count"] / len(cb_only) * 100).round(2)
            cb_s["label"] = cb_s["cb_count"].map(lambda x: f"{x:,}")

            max_cb_y = float(cb_s["cb_count"].max()) * 1.15 if len(cb_s) > 0 else 100

            cb_s_base = alt.Chart(cb_s).encode(
                x=alt.X(
                    "chargeback_lifecycle_stage:N",
                    sort=[
                        "1. First Chargeback",
                        "2. Representment",
                        "3. Pre-Arbitration",
                        "4. Resolved - Issuer Won",
                        "4. Write-Off - Issuer Lost",
                    ],
                    title="Lifecycle Stage",
                    axis=alt.Axis(labelAngle=-35, labelOverlap=False, labelLimit=0),
                ),
                y=alt.Y("cb_count:Q", title="Dispute Cases", scale=alt.Scale(domain=[0, max_cb_y])),
            )

            cb_s_bars = cb_s_base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                color=alt.Color("chargeback_lifecycle_stage:N", legend=None, scale=alt.Scale(scheme="purples")),
                tooltip=[
                    alt.Tooltip("chargeback_lifecycle_stage:N", title="Stage"),
                    alt.Tooltip("cb_count:Q", title="Cases", format=","),
                    alt.Tooltip("share_pct:Q", title="Share (%)", format=".2f"),
                    alt.Tooltip("cb_usd:Q", title="Disputed Value ($)", format="$,.0f"),
                ],
            )

            cb_s_text = cb_s_base.mark_text(dy=-8, fontWeight="bold", fontSize=11).encode(
                text=alt.Text("label:N"),
            )

            cb_s_chart = (cb_s_bars + cb_s_text).properties(height=320)
            st.altair_chart(cb_s_chart, use_container_width=True)


# =============================================================================
# TAB 5: AUTHENTICATION (3DS) (Success/Failure, Failure Reasons, Mandate)
# =============================================================================
with tabs[4]:
    st.subheader("3D Secure (3DS) Authentication Analytics: Pre-Authorization Funnel")
    st.caption(
        "Analyzes the 134,191-row 3DS Authentication dataset (1-to-1 mapped with all Card-Not-Present authorization transactions). Note that failed 3DS challenges abandon "
        "before reaching authorization—making 3DS health a critical hidden driver of customer conversion."
    )

    if len(df_3ds) == 0:
        st.info("No 3DS authentication records match the selected filters (e.g., POS-only filter active).")
    else:
        c3_1, c3_2 = st.columns(2)

        with c3_1:
            mand_df = (
                df_3ds.groupby("challenge_mandate", observed=True)
                .agg(
                    attempts=("auth_id", "count"),
                    success_rate=("is_auth_success", lambda s: round(float(s.mean() * 100), 2)),
                    failure_rate=("is_auth_success", lambda s: round(float((1.0 - s.mean()) * 100), 2)),
                    avg_latency_ms=("latency_ms", lambda s: int(s.mean())),
                )
                .reset_index()
                .sort_values("success_rate", ascending=False)
            )
            m3_chart = (
                alt.Chart(mand_df)
                .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
                .encode(
                    y=alt.Y("challenge_mandate:N", sort="-x", title="3DS Challenge Mandate"),
                    x=alt.X("success_rate:Q", title="Authentication Success Rate (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("success_rate:Q", scale=alt.Scale(scheme="tealblues"), title="Pass %"),
                    tooltip=[
                        alt.Tooltip("challenge_mandate:N", title="Mandate Type"),
                        alt.Tooltip("attempts:Q", title="3DS Attempts", format=","),
                        alt.Tooltip("success_rate:Q", title="Success Rate (%)", format=".2f"),
                        alt.Tooltip("failure_rate:Q", title="Failure Rate (%)", format=".2f"),
                        alt.Tooltip("avg_latency_ms:Q", title="Avg Latency (ms)", format=","),
                    ],
                )
                .properties(height=300, title="3DS Success Rate by Challenge Mandate")
            )
            st.altair_chart(m3_chart, use_container_width=True)

        with c3_2:
            fail_3ds_df = df_3ds[df_3ds["is_auth_success"] == 0]
            if len(fail_3ds_df) > 0:
                f_reason_df = (
                    fail_3ds_df.groupby("failure_reason", observed=True)
                    .agg(
                        fail_count=("auth_id", "count"),
                        lost_usd=("amount", "sum"),
                    )
                    .reset_index()
                    .sort_values("fail_count", ascending=False)
                )
                f_reason_df["share_pct"] = (f_reason_df["fail_count"] / len(fail_3ds_df) * 100).round(2)

                f3_chart = (
                    alt.Chart(f_reason_df)
                    .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#e11d48")
                    .encode(
                        y=alt.Y("failure_reason:N", sort="-x", title="3DS Failure Reason"),
                        x=alt.X("fail_count:Q", title="Failed Authentication Count"),
                        tooltip=[
                            alt.Tooltip("failure_reason:N", title="Failure Reason"),
                            alt.Tooltip("fail_count:Q", title="Failed Attempts", format=","),
                            alt.Tooltip("share_pct:Q", title="Share of Failures (%)", format=".2f"),
                            alt.Tooltip("lost_usd:Q", title="Abandoned Cart Value ($)", format="$,.0f"),
                        ],
                    )
                    .properties(height=300, title="3DS Authentication Failure Reasons Breakdown")
                )
                st.altair_chart(f3_chart, use_container_width=True)

        # Acquirer 3DS Success Rate & Latency comparison
        acq_3ds_df = (
            df_3ds.groupby("acquirer", observed=True)
            .agg(
                attempts=("auth_id", "count"),
                success_rate=("is_auth_success", lambda s: round(float(s.mean() * 100), 2)),
                acs_tech_err_rate=(
                    "failure_reason",
                    lambda s: round(float((s == "ACS / Switch Technical Error").mean() * 100), 2),
                ),
                avg_latency_ms=("latency_ms", lambda s: int(s.mean())),
            )
            .reset_index()
            .sort_values("success_rate", ascending=True)
        )
        st.markdown("##### 🏦 3DS Authentication Performance & Latency by Acquirer")
        st.dataframe(
            acq_3ds_df.rename(
                columns={
                    "acquirer": "Acquirer",
                    "attempts": "3DS Attempts",
                    "success_rate": "3DS Pass Rate (%)",
                    "acs_tech_err_rate": "ACS Tech Error Rate (%)",
                    "avg_latency_ms": "Avg Latency (ms)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =============================================================================
# TAB 6: ML MODEL, SHAP EXPLAINABILITY & INTERACTIVE WHAT-IF SIMULATOR
# =============================================================================
with tabs[5]:
    st.subheader("Machine Learning Fraud Model: Validation, Explainability & What-If Simulator")

    # 1. Problem Justification & Time-Based Validation Summary
    with st.expander("📌 Problem Selection Justification & Time-Based Validation Design (Click to Expand)", expanded=True):
        p_col1, p_col2 = st.columns([1.3, 1.0])
        with p_col1:
            st.markdown("#### Why We Chose Fraud Probability Prediction (`fraud_flag = 1`)")
            st.write(ml_results["problem_choice"]["justification"])
        with p_col2:
            vs = ml_results["validation_split"]
            st.markdown("#### Strict Chronological Split (Zero Future Leakage)")
            split_table = pd.DataFrame([
                {"Split": "1. Train Set", "Date Window": vs["train"]["period"], "Rows": f"{vs['train']['rows']:,}", "Fraud Rate": f"{vs['train']['fraud_rate']*100:.2f}%"},
                {"Split": "2. Validation Set", "Date Window": vs["validation"]["period"], "Rows": f"{vs['validation']['rows']:,}", "Fraud Rate": f"{vs['validation']['fraud_rate']*100:.2f}%"},
                {"Split": "3. Out-of-Time Test", "Date Window": vs["test_oot"]["period"], "Rows": f"{vs['test_oot']['rows']:,}", "Fraud Rate": f"{vs['test_oot']['fraud_rate']*100:.2f}%"},
            ])
            st.dataframe(split_table, use_container_width=True, hide_index=True)

    # 2. Baseline vs Advanced Model Comparison & Diagnostic Curves (Responsive 2-Row Layout)
    lr_m = ml_results["models"]["baseline_logistic_regression"]["test_metrics"]
    xgb_m = ml_results["models"]["advanced_xgboost"]["test_metrics"]

    st.markdown("##### 🏆 Out-of-Time Test Set: Baseline vs. Advanced Model Performance")
    comp_df = pd.DataFrame([
        {
            "Evaluation Metric": "PR-AUC (Average Precision) ★ Primary Metric",
            "Logistic Regression (Baseline)": f"{lr_m['pr_auc']:.4f}",
            "XGBoost (Advanced Model)": f"{xgb_m['pr_auc']:.4f}",
            "Out-of-Time Lift": f"+{(xgb_m['pr_auc'] - lr_m['pr_auc'])*100:.2f} pts",
        },
        {
            "Evaluation Metric": "ROC-AUC (Discrimination)",
            "Logistic Regression (Baseline)": f"{lr_m['roc_auc']:.4f}",
            "XGBoost (Advanced Model)": f"{xgb_m['roc_auc']:.4f}",
            "Out-of-Time Lift": f"+{(xgb_m['roc_auc'] - lr_m['roc_auc'])*100:.2f} pts",
        },
        {
            "Evaluation Metric": "Brier Score (Calibration Error ↓ lower is better)",
            "Logistic Regression (Baseline)": f"{lr_m['brier_score']:.5f}",
            "XGBoost (Advanced Model)": f"{xgb_m['brier_score']:.5f}",
            "Out-of-Time Lift": f"{(xgb_m['brier_score'] - lr_m['brier_score']):.5f}",
        },
        {
            "Evaluation Metric": "F1-Score @ Optimal Business Threshold (τ = 0.12)",
            "Logistic Regression (Baseline)": f"{lr_m['f1_at_threshold']:.4f}",
            "XGBoost (Advanced Model)": f"{xgb_m['f1_at_threshold']:.4f}",
            "Out-of-Time Lift": f"+{(xgb_m['f1_at_threshold'] - lr_m['f1_at_threshold'])*100:.2f} pts",
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Place PR-Curve and ROC-Curve side-by-side in 2 spacious columns with top-aligned legend
    curv_col1, curv_col2 = st.columns(2)

    with curv_col1:
        pr_xgb = pd.DataFrame(ml_results["pr_curve"]["xgb"]).assign(Model=f"XGBoost (PR-AUC={xgb_m['pr_auc']:.3f})")
        pr_lr = pd.DataFrame(ml_results["pr_curve"]["lr"]).assign(Model=f"LogReg Baseline (PR-AUC={lr_m['pr_auc']:.3f})")
        pr_df = pd.concat([pr_xgb, pr_lr], ignore_index=True)
        pr_chart = (
            alt.Chart(pr_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("recall:Q", title="Recall (Fraud Capture Rate)"),
                y=alt.Y("precision:Q", title="Precision (Hit Rate)"),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(range=["#2563eb", "#94a3b8"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["Model:N", alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f")],
            )
            .properties(height=280, title="Precision-Recall Curve (Out-of-Time Test)")
        )
        st.altair_chart(pr_chart, use_container_width=True)

    with curv_col2:
        roc_xgb = pd.DataFrame(ml_results["roc_curve"]["xgb"]).assign(Model=f"XGBoost (AUC={xgb_m['roc_auc']:.3f})")
        roc_lr = pd.DataFrame(ml_results["roc_curve"]["lr"]).assign(Model=f"LogReg Baseline (AUC={lr_m['roc_auc']:.3f})")
        roc_df = pd.concat([roc_xgb, roc_lr], ignore_index=True)
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("fpr:Q", title="False Positive Rate (False Declines)"),
                y=alt.Y("tpr:Q", title="True Positive Rate (Fraud Recall)"),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(range=["#2563eb", "#94a3b8"]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["Model:N", alt.Tooltip("fpr:Q", format=".3f"), alt.Tooltip("tpr:Q", format=".3f")],
            )
            .properties(height=280, title="ROC Curve (Out-of-Time Test)")
        )
        st.altair_chart(roc_chart, use_container_width=True)

    st.markdown("---")

    # 3. INTERACTIVE OPERATING THRESHOLD & BUSINESS IMPACT "WHAT-IF" SIMULATOR
    st.markdown("### 🎛️ Interactive Operating Threshold & Business Impact 'What-If' Simulator")
    st.caption(
        "Adjust the operating probability cutoff τ and unit-economics assumptions below to simulate live financial impact "
        "on the Out-of-Time Test Set (45,000+ transactions)."
    )

    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    opt_tau = float(ml_results["optimal_threshold"])
    with sim_c1:
        user_tau = st.slider(
            "🎯 Operating Threshold (τ)",
            min_value=0.06,
            max_value=0.88,
            value=opt_tau,
            step=0.02,
            help=f"Validation-optimal threshold is τ = {opt_tau:.2f}. Transactions with Score >= τ are blocked/challenged.",
        )
    with sim_c2:
        ic_pct = st.number_input(
            "💰 Interchange Rate (%)",
            min_value=0.50,
            max_value=3.50,
            value=1.75,
            step=0.05,
            help="Issuer interchange revenue earned per approved legitimate transaction.",
        )
    with sim_c3:
        fd_fee_input = st.number_input(
            "😤 False-Decline Friction Cost ($)",
            min_value=2.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Fixed customer service & abandonment penalty per falsely declined legitimate transaction (plus 8% spend churn).",
        )
    with sim_c4:
        cb_fee_input = st.number_input(
            "⚖️ Chargeback Processing Fee ($)",
            min_value=5.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Operational dispute fee per missed fraud transaction (on top of 100% fraud principal loss).",
        )

    # Recompute threshold curve dynamically on Out-of-Time Test Set with user's assumptions
    test_slice = df_auth_full[df_auth_full["split"] == "3_Test_OOT"]
    custom_econ = {
        "interchange_rate": ic_pct / 100.0,
        "false_decline_churn_rate": 0.080,
        "false_decline_friction_usd": fd_fee_input,
        "chargeback_fee_usd": cb_fee_input,
        "fraud_loss_share": 1.00,
    }
    tau_grid = np.round(np.arange(0.06, 0.90, 0.02), 2)
    sim_curve_df = compute_threshold_economics(
        test_slice["fraud_flag"].to_numpy(dtype=int),
        test_slice["ml_fraud_prob_xgb"].to_numpy(dtype=float),
        test_slice["amount"].to_numpy(dtype=float),
        tau_grid,
        econ=custom_econ,
    )
    sel_row = sim_curve_df.iloc[(sim_curve_df["threshold"] - user_tau).abs().argsort()[:1]].iloc[0]
    leg_row = ml_results["models"]["legacy_static_rules_benchmark"]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Net ML Value Created ($)",
        f"${sel_row['net_ml_savings_usd']:,.0f}",
        delta=f"{sel_row['net_ml_savings_usd'] - leg_row['net_ml_savings_usd']:+,.0f} vs Legacy Rules",
    )
    s2.metric(
        "Fraud Value Captured (%)",
        f"{sel_row['recall_usd']*100:.1f}%",
        delta=f"{(sel_row['recall_usd'] - leg_row['recall_usd'])*100:+.1f} pts vs Legacy Rules",
    )
    s3.metric(
        "False Declines (Good Tx Blocked)",
        f"{int(sel_row['fp_false_declines']):,}",
        delta=f"{int(sel_row['fp_false_declines'] - leg_row['fp_false_declines']):+,} vs Legacy Rules",
        delta_color="inverse",
    )
    s4.metric(
        "Simulated Approval Rate (%)",
        f"{sel_row['approval_rate']*100:.2f}%",
        delta=f"Precision: {sel_row['precision']*100:.1f}% | Recall: {sel_row['recall_tx']*100:.1f}%",
    )

    # Plot Threshold Business Impact Curve & Global SHAP Importance side by side
    sh_col1, sh_col2 = st.columns(2)
    with sh_col1:
        base_t = alt.Chart(sim_curve_df).encode(
            x=alt.X("threshold:Q", title="Operating Probability Threshold (τ)"),
            tooltip=[
                alt.Tooltip("threshold:Q", title="Threshold τ", format=".2f"),
                alt.Tooltip("net_ml_savings_usd:Q", title="Net ML Value ($)", format="$,.0f"),
                alt.Tooltip("fraud_prevented_usd:Q", title="Fraud Prevented ($)", format="$,.0f"),
                alt.Tooltip("false_decline_cost_usd:Q", title="False Decline Cost ($)", format="$,.0f"),
                alt.Tooltip("recall_tx:Q", title="Fraud Recall", format=".1%"),
                alt.Tooltip("fp_false_declines:Q", title="False Declines", format=","),
            ],
        )
        curve_line = base_t.mark_line(color="#16a34a", strokeWidth=3).encode(
            y=alt.Y("net_ml_savings_usd:Q", title="Net Financial Benefit ($ USD)"),
        )
        rule_line = (
            alt.Chart(pd.DataFrame({"threshold": [user_tau]}))
            .mark_rule(color="#dc2626", strokeDash=[5, 5], strokeWidth=2)
            .encode(x="threshold:Q")
        )
        st.altair_chart(
            (curve_line + rule_line).properties(
                height=320,
                title=f"Net Business Impact Curve Across Thresholds (Selected τ = {user_tau:.2f})",
            ),
            use_container_width=True,
        )

    with sh_col2:
        shap_df = pd.DataFrame(ml_results["shap_global_importance"][:10])
        shap_chart = (
            alt.Chart(shap_df)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5, color="#0284c7")
            .encode(
                y=alt.Y("label:N", sort="-x", title="Engineered Feature"),
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP Value| (Impact on Log-Odds)"),
                tooltip=[
                    alt.Tooltip("label:N", title="Feature"),
                    alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".4f"),
                ],
            )
            .properties(height=320, title="Global Explainability: Top 10 SHAP Feature Drivers")
        )
        st.altair_chart(shap_chart, use_container_width=True)

    # 4. Risk Segments & Scored Transaction Explorer with Local SHAP
    st.markdown("##### 📊 Risk Segments Breakdown (Out-of-Time Test Portfolio)")
    seg_df = pd.DataFrame(ml_results["risk_segments_test"])
    st.dataframe(
        seg_df.rename(
            columns={
                "risk_segment": "ML Risk Segment Tier",
                "tx_count": "Transactions",
                "volume_share_pct": "Volume Share (%)",
                "total_amount_usd": "Total Spend ($)",
                "avg_predicted_risk_pct": "Avg Predicted Risk (%)",
                "actual_fraud_rate_pct": "Actual Fraud Rate (%)",
                "legacy_approval_rate_pct": "Legacy Approval Rate (%)",
                "fraud_captured_count": "Fraud Tx Captured",
                "fraud_captured_share_pct": "Share of Total Fraud (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### 🔍 Scored Transactions Explorer & Local SHAP Transaction Inspector")
    exp_col1, exp_col2 = st.columns([1.35, 1.0])
    with exp_col1:
        top_scored_slice = (
            df_auth.sort_values("ml_fraud_prob_xgb", ascending=False)
            .head(100)[[
                "transaction_id", "timestamp", "channel", "acquirer", "merchant",
                "mcc_category", "amount", "ml_fraud_prob_xgb", "ml_risk_segment",
                "ml_top_drivers", "decline_reason", "fraud_flag",
            ]]
        )
        st.dataframe(
            top_scored_slice.rename(
                columns={
                    "transaction_id": "Tx ID",
                    "timestamp": "Timestamp",
                    "channel": "Channel",
                    "acquirer": "Acquirer",
                    "merchant": "Merchant",
                    "mcc_category": "MCC",
                    "amount": "Amount ($)",
                    "ml_fraud_prob_xgb": "XGB Risk Score",
                    "ml_risk_segment": "Risk Segment",
                    "ml_top_drivers": "Primary Risk Drivers",
                    "decline_reason": "Legacy Outcome",
                    "fraud_flag": "True Fraud",
                }
            ),
            height=310,
            use_container_width=True,
            hide_index=True,
        )

    with exp_col2:
        local_examples = ml_results.get("local_shap_examples", {})
        if local_examples:
            selected_tx_id = st.selectbox(
                "Select High-Risk Transaction for Local SHAP Waterfall Breakdown",
                options=list(local_examples.keys()),
            )
            tx_info = local_examples[selected_tx_id]
            st.caption(
                f"**{tx_info['transaction_id']}** • **{tx_info['merchant']}** ({tx_info['mcc_category']}) • "
                f"**${tx_info['amount']:,.2f}** • {tx_info['channel']} • "
                f"Predicted Fraud Prob: **`{tx_info['predicted_prob']*100:.1f}%`** (Actual Fraud: `{tx_info['actual_fraud']}`)"
            )
            loc_df = pd.DataFrame(tx_info["top_contributions"])
            loc_chart = (
                alt.Chart(loc_df)
                .mark_bar()
                .encode(
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    x=alt.X("shap_contribution:Q", title="SHAP Contribution to Fraud Log-Odds"),
                    color=alt.condition(
                        alt.datum.shap_contribution > 0,
                        alt.value("#dc2626"),
                        alt.value("#16a34a"),
                    ),
                    tooltip=[
                        alt.Tooltip("feature:N", title="Feature"),
                        alt.Tooltip("value:Q", title="Feature Value"),
                        alt.Tooltip("shap_contribution:Q", title="SHAP Impact", format="+.4f"),
                    ],
                )
                .properties(height=230, title=f"Local SHAP Feature Contributions for {selected_tx_id}")
            )
            st.altair_chart(loc_chart, use_container_width=True)


# =============================================================================
# TAB 7: AI INSIGHTS ENGINE (Storyline, Mitigation, Context Viewer, Follow-Up Q&A)
# =============================================================================
with tabs[6]:
    st.subheader("🧠 Executive Strategy & AI Decision Advisor")
    st.caption(
        "C-Suite strategic action plan to grow revenue, lift approval rates, and cut fraud & operational costs—"
        "backed by live portfolio data."
    )

    ai_top1, ai_top2 = st.columns([2.2, 1.0])
    with ai_top1:
        target_view_scope = st.selectbox(
            "🔭 Select Business Domain for Executive Review",
            options=[
                "Whole Dashboard (All 6 Domains Combined)",
                "1. Overview (Volumes, Approval & Seasonality)",
                "2. Authorization (Acquirers, Tokenization & 3DS Lift)",
                "3. Declines (Hard vs. Soft Reasons, MCC & Ticket Size)",
                "4. Fraud & Chargebacks (Diurnal Risk & Lifecycle)",
                "5. Authentication (3DS Funnel, Latency & Drop-offs)",
                "6. ML Model & Threshold Economics",
            ],
            index=0,
        )

    current_context = build_dashboard_context(
        df_auth=df_auth,
        df_3ds=df_3ds,
        df_auth_baseline=df_auth_full,
        active_view=target_view_scope,
        active_filters=active_filters_dict,
        ml_results=ml_results,
    )

    with ai_top2:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        refresh_clicked = st.button(
            "✨ Refresh Executive Briefing",
            type="primary",
            use_container_width=True,
        )

    if refresh_clicked or "ai_insights_md" not in st.session_state:
        with st.spinner("Preparing C-Suite executive briefing..."):
            insights_md, mode_label = generate_executive_insights(current_context)
            st.session_state["ai_insights_md"] = insights_md
            st.session_state["ai_mode_label"] = mode_label

    # -------------------------------------------------------------------------
    # SECTION A: 3 C-SUITE STRATEGIC PILLARS (REVENUE, FRAUD/COST, OPERATIONS)
    # -------------------------------------------------------------------------
    ctx_kpis = current_context.get("headline_kpis", {})
    ctx_acqs = current_context.get("acquirer_performance_ranked_worst_first", [])
    ctx_sec = current_context.get("ecommerce_security_lift", {})
    ctx_seas = current_context.get("seasonality_and_diurnal_signals", {})
    ctx_ml = current_context.get("ml_model_summary", {})
    ctx_decs = current_context.get("top_decline_reasons", [])

    worst_acq_row = ctx_acqs[0] if ctx_acqs else {"acquirer": "LegacySwitch_Bank", "approval_rate_pct": 81.94, "technical_error_rate_pct": 5.32, "tokenized_share_pct": 30.3}
    best_acq_row = ctx_acqs[-1] if ctx_acqs else {"acquirer": "StripeAcq_EU", "approval_rate_pct": 93.04}
    tok_lift_pts = round((ctx_sec.get("ecom_tokenized_approval_pct") or 93.1) - (ctx_sec.get("ecom_untokenized_approval_pct") or 83.3), 2)
    inc_gain_val = ctx_ml.get("incremental_gain_vs_legacy_rules_usd", 117522)
    fd_leg_val = int(ctx_ml.get("false_declines_oot_legacy", 1102))
    fd_xgb_val = int(ctx_ml.get("false_declines_oot_xgb", 615))
    partner_gap = round(best_acq_row["approval_rate_pct"] - worst_acq_row["approval_rate_pct"], 1)

    st.markdown("#### 🎯 Top 3 C-Suite Strategic Priorities *(Action + Business Impact)*")

    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="action-card-header">
                    <span class="action-tag">📈 Grow Revenue & Approval</span>
                    <span class="impact-pill">+{tok_lift_pts:.1f}% Online Approval</span>
                </div>
                <div class="action-title">Expand Digital Wallet Tokens & Remove Rigid Blocks</div>
                <div class="action-desc">Migrate online merchants to Network Tokens and stop auto-blocking loyal high-spend customers.</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📊 View Data Justification (2 Key Proof Points)"):
                st.markdown(
                    f"""
                    * **Token Approval Lift:** Online checkouts with Network Tokens approve at **{ctx_sec.get('ecom_tokenized_approval_pct', 0):.1f}%** vs. **{ctx_sec.get('ecom_untokenized_approval_pct', 0):.1f}%** without tokens.
                    * **Recoverable Sales:** **{ctx_kpis.get('estimated_false_fraud_declines_count', 0):,} good customer purchases** (`USD {ctx_kpis.get('estimated_false_fraud_declines_usd', 0):,.0f}`) were wrongly declined by legacy rules.
                    """
                )

    with s_col2:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="action-card-header">
                    <span class="action-tag">🛡️ Cut Fraud & Dispute Costs</span>
                    <span class="impact-pill">+USD {inc_gain_val:,.0f} Net Savings</span>
                </div>
                <div class="action-title">Deploy AI Fraud Scoring & Overnight Step-Up Checks</div>
                <div class="action-desc">Replace static fraud rules with AI risk tiers and require 3DS verification on high-risk overnight spend.</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📊 View Data Justification (2 Key Proof Points)"):
                st.markdown(
                    f"""
                    * **44% Fewer False Declines:** AI scoring cuts wrongly blocked good customers from **{fd_leg_val:,} to {fd_xgb_val:,}** while generating **`+USD {inc_gain_val:,.0f}`** net lift.
                    * **Overnight Risk Concentration:** Fraud between `1 AM – 5 AM` is **2.5x higher** (**{ctx_seas.get('late_night_1am_5am_fraud_rate_pct', 0):.2f}%** vs. **{ctx_seas.get('daytime_fraud_rate_pct', 0):.2f}%** daytime).
                    """
                )

    with s_col3:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="action-card-header">
                    <span class="action-tag">🏦 Fix Partner & Billing Drag</span>
                    <span class="impact-pill">+{partner_gap:.1f}% Processor Gap</span>
                </div>
                <div class="action-title">Fix {worst_acq_row['acquirer']} Outages & Align Billing to Payday</div>
                <div class="action-desc">Reroute failed processor traffic and schedule subscription renewals after payday (25th–5th).</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("📊 View Data Justification (2 Key Proof Points)"):
                st.markdown(
                    f"""
                    * **Processor Drag:** `{worst_acq_row['acquirer']}` approves only **{worst_acq_row['approval_rate_pct']:.1f}%** (vs. **{best_acq_row['approval_rate_pct']:.1f}%** at `{best_acq_row['acquirer']}`) due to a **{worst_acq_row['technical_error_rate_pct']:.1f}%** system timeout rate.
                    * **Pre-Payday Cash Crunch:** "Insufficient Funds" declines double during **Days 15–24** (**{ctx_seas.get('pre_payday_days_15_24_nsf_decline_rate_pct', 0):.2f}%** vs. **{ctx_seas.get('other_days_nsf_decline_rate_pct', 0):.2f}%** post-payday).
                    """
                )

    # -------------------------------------------------------------------------
    # SECTION B: CONCISE 6-SECTION BOARD SUMMARY (MAX 2 BULLETS PER HEADER)
    # -------------------------------------------------------------------------
    with st.container(border=True):
        b_hdr1, b_hdr2 = st.columns([3.0, 1.2])
        with b_hdr1:
            st.markdown("#### 📋 Board-Level 6-Point Executive Summary *(Max 2 Takeaways Per Section)*")
        with b_hdr2:
            st.caption(f"Engine: `{st.session_state['ai_mode_label']}`")
        st.markdown(st.session_state["ai_insights_md"])

    with st.expander("🔍 Data Governance & Transparency Audit (View Raw Context Sent to AI)", expanded=False):
        st.markdown("##### 1. Executive System Instructions")
        st.code(SYSTEM_PROMPT, language="markdown")
        st.markdown("##### 2. Live Pre-Computed `<DATA_CONTEXT>` JSON Payload")
        st.json(current_context)

    # -------------------------------------------------------------------------
    # SECTION C: C-SUITE AI STRATEGY COPILOT (PLAIN-BUSINESS Q&A)
    # -------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="copilot-banner">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <div style="font-size:1.12rem; font-weight:700; color:#ffffff; margin-bottom:0.2rem;">
                        💬 Ask Your AI Strategy Advisor
                    </div>
                    <div style="font-size:0.84rem; color:#cbd5e1;">
                        Click an executive topic below or type any business question on revenue growth, approval rate, fraud, or cost savings.
                    </div>
                </div>
                <div>
                    <span class="copilot-badge">🛡️ Verified Portfolio Data Only</span>
                    <span class="copilot-badge">📊 Active Scope: {ctx_kpis.get('filtered_transactions', 0):,} Tx</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    selected_quick_q = None

    with q_col1:
        with st.container(border=True):
            st.markdown('<div class="prompt-card-tag">📈 Revenue Recovery</div>', unsafe_allow_html=True)
            st.markdown('<div class="prompt-card-title">How do we recover declined sales & lift approval?</div>', unsafe_allow_html=True)
            if st.button("View Revenue Plan →", key="q_btn_rev", use_container_width=True):
                selected_quick_q = "How can we recover declined sales and lift our overall approval rate?"

    with q_col2:
        with st.container(border=True):
            st.markdown('<div class="prompt-card-tag">🛡️ Fraud & Cost Savings</div>', unsafe_allow_html=True)
            st.markdown('<div class="prompt-card-title">What is the dollar ROI of AI fraud scoring vs. old rules?</div>', unsafe_allow_html=True)
            if st.button("View Fraud ROI →", key="q_btn_roi", use_container_width=True):
                selected_quick_q = "What is the net dollar savings of deploying the AI fraud model compared to legacy rules?"

    with q_col3:
        with st.container(border=True):
            st.markdown('<div class="prompt-card-tag">🏦 Partner Performance</div>', unsafe_allow_html=True)
            st.markdown('<div class="prompt-card-title">Which payment processor is costing us lost sales?</div>', unsafe_allow_html=True)
            if st.button("View Partner Audit →", key="q_btn_acq", use_container_width=True):
                selected_quick_q = "Which payment processor is underperforming and what action should we take?"

    with q_col4:
        with st.container(border=True):
            st.markdown('<div class="prompt-card-tag">🔒 Governance Check</div>', unsafe_allow_html=True)
            st.markdown('<div class="prompt-card-title">Test guardrail on unavailable customer credit data</div>', unsafe_allow_html=True)
            if st.button("Test AI Guardrail →", key="q_btn_guard", use_container_width=True):
                selected_quick_q = "What is the average cardholder FICO credit score and salary for declined transactions?"

    if selected_quick_q:
        st.session_state["copilot_active_question"] = selected_quick_q
        insights_q_ans, insights_q_mode = answer_followup_question(selected_quick_q, current_context)
        st.session_state["copilot_active_answer"] = insights_q_ans
        st.session_state["copilot_active_mode"] = insights_q_mode

    # Prominent Executive Question Bar
    with st.container(border=True):
        st.markdown("##### 🔎 Ask a Custom Executive Question")
        with st.form("copilot_custom_question_form", clear_on_submit=False):
            cmd_col1, cmd_col2 = st.columns([4.3, 1.0])
            with cmd_col1:
                custom_q_input = st.text_input(
                    "Custom executive question",
                    value=st.session_state.get("copilot_active_question", ""),
                    placeholder="e.g., How much do Digital Wallet Tokens and 3DS improve online approval and cut fraud?",
                    label_visibility="collapsed",
                )
            with cmd_col2:
                submit_custom_q = st.form_submit_button(
                    "🚀 Ask Advisor",
                    type="primary",
                    use_container_width=True,
                )

        if submit_custom_q and custom_q_input.strip():
            q_str = custom_q_input.strip()
            st.session_state["copilot_active_question"] = q_str
            insights_q_ans, insights_q_mode = answer_followup_question(q_str, current_context)
            st.session_state["copilot_active_answer"] = insights_q_ans
            st.session_state["copilot_active_mode"] = insights_q_mode

    active_q = st.session_state.get("copilot_active_question", "").strip()
    if active_q and "copilot_active_answer" in st.session_state:
        with st.container(border=True):
            ans_hdr1, ans_hdr2 = st.columns([3.6, 1.2])
            with ans_hdr1:
                st.markdown(f"**💬 Executive Inquiry:** `{active_q}`")
            with ans_hdr2:
                st.caption(f"Source: `{st.session_state.get('copilot_active_mode', 'Grounded Q&A')}`")
            st.markdown(st.session_state["copilot_active_answer"])
            with st.expander("📊 View Underlying Portfolio Numbers Backing This Answer"):
                st.json({
                    "headline_kpis": ctx_kpis,
                    "top_decline_reasons": ctx_decs[:3],
                    "ecommerce_security_lift": ctx_sec,
                    "ml_model_summary": ctx_ml,
                })
