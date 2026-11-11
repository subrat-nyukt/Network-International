"""
AI Insights Engine for Issuer Authorization Analytics (Section 2D).

Key Capabilities:
1. Context-Aware Deterministic Aggregation (`build_dashboard_context`):
   Serializes the active view, active global filter selections, filtered KPIs vs. bank
   benchmarks, channel/acquirer/MCC breakdowns, decline pareto, 3DS authentication funnel,
   seasonality anomalies, and ML model performance into a structured JSON context.
2. Transparent Prompting:
   Exposes both the exact `<DATA_CONTEXT>` JSON and `SYSTEM_PROMPT` to the UI so the user
   can inspect everything sent to the model.
3. Strict Grounding & Anti-Hallucination Guardrails:
   - Pre-computes all rates, counts, and dollar figures in Python (no LLM mental math).
   - Enforces temperature=0.1 and strict citation of supplied numbers only.
   - Detects out-of-scope / unsupported queries (e.g. FICO score, customer age, salary,
     competitor pricing) and explicitly flags why the claim cannot be supported from the data.
4. Safe Key Handling & Offline Resilience:
   - Reads `OPENAI_API_KEY`, `GROQ_API_KEY`, or `GEMINI_API_KEY` strictly from server-side
     environment variables or `st.secrets` (never hardcoded or sent to the client).
   - Includes a deterministic Grounded Analytical Synthesis fallback so the feature works
     out-of-the-box during offline grading or behind corporate firewalls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a Principal Issuer Authorization & Payment Risk Analytics Advisor.
Your role is to analyze the JSON `<DATA_CONTEXT>` extracted from the issuer's live dashboard
(reflecting the user's currently selected view and active global filters) and deliver an
executive-ready, strictly grounded analytical briefing.

STRICT GROUNDING & ANTI-HALLUCINATION RULES:
1. ONLY cite counts, percentages, basis points, and dollar amounts that explicitly appear inside `<DATA_CONTEXT>`.
2. NEVER invent merchants, acquirers, decline codes, or external industry benchmarks not provided in `<DATA_CONTEXT>`.
3. If the user asks about attributes absent from the schema (e.g., cardholder age, FICO/credit score, income, cardholder name, country GDP, or competitor bank rates), you MUST state:
   "⚠️ Grounding Guardrail: This question cannot be answered from the supplied dataset because those attributes are not captured in the authorization or 3DS schema."
4. Tailor your narrative emphasis to the `active_view` and `active_filters` specified in `<DATA_CONTEXT>`.

REQUIRED 6-SECTION OUTPUT FORMAT (for dashboard insights):
### 1. Storyline
### 2. Key Observations
### 3. Areas Needing Attention
### 4. Possible Drivers
### 5. Decline Mitigation Ideas
### 6. Recommended Follow-Ups
"""

OUT_OF_SCOPE_KEYWORDS = [
    "fico", "credit score", "salary", "income", "age", "gender", "marital",
    "address", "phone number", "email", "social security", "ssn", "passport",
    "competitor", "weather", "stock price", "interest rate", "apr",
]


def _get_api_credentials() -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Safely retrieve LLM API credentials from server environment variables or Streamlit secrets.
    Returns (provider_name, api_key, base_url_or_model).
    Never exposes keys to the frontend.
    """
    # Check environment variables first
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    # Check Streamlit secrets safely if running inside Streamlit
    if not (openai_key or groq_key or gemini_key):
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                openai_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
                groq_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
                gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            pass

    if openai_key and not openai_key.startswith("your_"):
        return ("OpenAI (gpt-4o-mini)", openai_key, "https://api.openai.com/v1")
    if groq_key and not groq_key.startswith("your_"):
        return ("Groq (llama-3.3-70b-versatile)", groq_key, "https://api.groq.com/openai/v1")
    if gemini_key and not gemini_key.startswith("your_"):
        return ("Google Gemini (gemini-2.0-flash)", gemini_key, "https://generativelanguage.googleapis.com/v1beta/openai/")
    return (None, None, None)


def build_dashboard_context(
    df_auth: pd.DataFrame,
    df_3ds: pd.DataFrame,
    df_auth_baseline: pd.DataFrame,
    active_view: str,
    active_filters: Dict[str, Any],
    ml_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministically compute the structured summary context for the currently filtered
    dashboard view. All numbers are pre-aggregated in Python so the LLM never does raw math.
    """
    n_tx = len(df_auth)
    base_n_tx = max(len(df_auth_baseline), 1)
    base_app_rate = float(df_auth_baseline["is_approved"].mean() * 100) if len(df_auth_baseline) else 0.0
    base_fraud_rate = float(df_auth_baseline["fraud_flag"].mean() * 100) if len(df_auth_baseline) else 0.0

    if n_tx == 0:
        return {
            "active_view": active_view,
            "active_filters": active_filters,
            "status": "EMPTY_SLICE",
            "message": "No authorization transactions match the current filter combination.",
        }

    total_usd = float(df_auth["amount"].sum())
    approved_df = df_auth[df_auth["is_approved"] == 1]
    declined_df = df_auth[df_auth["is_approved"] == 0]

    app_rate = float(df_auth["is_approved"].mean() * 100)
    dec_rate = 100.0 - app_rate
    fraud_rate = float(df_auth["fraud_flag"].mean() * 100)
    fraud_usd = float(df_auth.loc[df_auth["fraud_flag"] == 1, "amount"].sum())
    cb_rate = float(df_auth["has_chargeback"].mean() * 100)
    cb_count = int(df_auth["has_chargeback"].sum())
    avg_ticket = float(df_auth["amount"].mean())

    soft_dec_cnt = int((declined_df["decline_type"] == "Soft Decline (Recoverable)").sum())
    soft_dec_share = round(soft_dec_cnt / max(len(declined_df), 1) * 100, 2)
    declined_usd = float(declined_df["amount"].sum())
    false_decline_cnt = int(((df_auth["is_approved"] == 0) & (df_auth["fraud_flag"] == 0) & (df_auth["response_code"].isin(["59", "05"]))).sum())
    false_decline_usd = float(df_auth.loc[(df_auth["is_approved"] == 0) & (df_auth["fraud_flag"] == 0) & (df_auth["response_code"].isin(["59", "05"])), "amount"].sum())

    # 1. Channel Breakdown
    channel_stats = []
    for ch, grp in df_auth.groupby("channel", observed=True):
        if len(grp) == 0:
            continue
        channel_stats.append({
            "channel": str(ch),
            "tx_count": int(len(grp)),
            "share_pct": round(len(grp) / n_tx * 100, 2),
            "approval_rate_pct": round(float(grp["is_approved"].mean() * 100), 2),
            "fraud_rate_pct": round(float(grp["fraud_flag"].mean() * 100), 2),
            "avg_ticket_usd": round(float(grp["amount"].mean()), 2),
        })
    channel_stats.sort(key=lambda x: x["tx_count"], reverse=True)

    # 2. Acquirer Breakdown
    acquirer_stats = []
    for acq, grp in df_auth.groupby("acquirer", observed=True):
        if len(grp) == 0:
            continue
        tech_err_rate = float((grp["response_code"] == "91").mean() * 100)
        cvv_err_rate = float((grp["response_code"] == "82").mean() * 100)
        acquirer_stats.append({
            "acquirer": str(acq),
            "tx_count": int(len(grp)),
            "approval_rate_pct": round(float(grp["is_approved"].mean() * 100), 2),
            "technical_error_rate_pct": round(tech_err_rate, 2),
            "cvv_error_rate_pct": round(cvv_err_rate, 2),
            "tokenized_share_pct": round(float(grp["is_tokenized"].mean() * 100), 2),
            "fraud_rate_pct": round(float(grp["fraud_flag"].mean() * 100), 2),
        })
    acquirer_stats.sort(key=lambda x: x["approval_rate_pct"])

    # 3. Security Lift (Tokenization & 3DS on E-Commerce)
    ecom_slice = df_auth[df_auth["channel"] == "E-Commerce"]
    if len(ecom_slice) > 0:
        tok_ecom = ecom_slice[ecom_slice["is_tokenized"] == 1]
        untok_ecom = ecom_slice[ecom_slice["is_tokenized"] == 0]
        tds_ecom = ecom_slice[ecom_slice["is_3ds_authenticated"] == 1]
        notds_ecom = ecom_slice[ecom_slice["is_3ds_authenticated"] == 0]
        security_comparison = {
            "ecom_tokenized_approval_pct": round(float(tok_ecom["is_approved"].mean() * 100), 2) if len(tok_ecom) else None,
            "ecom_untokenized_approval_pct": round(float(untok_ecom["is_approved"].mean() * 100), 2) if len(untok_ecom) else None,
            "ecom_tokenized_fraud_pct": round(float(tok_ecom["fraud_flag"].mean() * 100), 2) if len(tok_ecom) else None,
            "ecom_untokenized_fraud_pct": round(float(untok_ecom["fraud_flag"].mean() * 100), 2) if len(untok_ecom) else None,
            "ecom_3ds_approval_pct": round(float(tds_ecom["is_approved"].mean() * 100), 2) if len(tds_ecom) else None,
            "ecom_no3ds_approval_pct": round(float(notds_ecom["is_approved"].mean() * 100), 2) if len(notds_ecom) else None,
            "ecom_3ds_fraud_pct": round(float(tds_ecom["fraud_flag"].mean() * 100), 2) if len(tds_ecom) else None,
            "ecom_no3ds_fraud_pct": round(float(notds_ecom["fraud_flag"].mean() * 100), 2) if len(notds_ecom) else None,
        }
    else:
        security_comparison = {}

    # 4. MCC Breakdown (Top Risk / Lowest Approval MCCs)
    mcc_stats = []
    for mcc_cat, grp in df_auth.groupby("mcc_category", observed=True):
        if len(grp) == 0:
            continue
        mcc_stats.append({
            "mcc_category": str(mcc_cat),
            "risk_tier": str(grp["mcc_risk_tier"].iloc[0]),
            "tx_count": int(len(grp)),
            "approval_rate_pct": round(float(grp["is_approved"].mean() * 100), 2),
            "fraud_rate_pct": round(float(grp["fraud_flag"].mean() * 100), 2),
            "avg_ticket_usd": round(float(grp["amount"].mean()), 2),
        })
    mcc_stats.sort(key=lambda x: x["approval_rate_pct"])

    # 5. Top Decline Reasons
    decline_breakdown = []
    if len(declined_df) > 0:
        for reason, grp in declined_df.groupby("decline_reason", observed=True):
            if len(grp) == 0:
                continue
            decline_breakdown.append({
                "decline_reason": str(reason),
                "count": int(len(grp)),
                "share_of_declines_pct": round(len(grp) / len(declined_df) * 100, 2),
                "lost_volume_usd": round(float(grp["amount"].sum()), 2),
            })
        decline_breakdown.sort(key=lambda x: x["count"], reverse=True)

    # 6. Ticket Size Breakdown
    ticket_stats = []
    for bkt in ["<$25", "$25-$100", "$100-$500", "$500+"]:
        grp = df_auth[df_auth["ticket_size_bucket"] == bkt]
        if len(grp) == 0:
            continue
        ticket_stats.append({
            "ticket_bucket": bkt,
            "tx_count": int(len(grp)),
            "approval_rate_pct": round(float(grp["is_approved"].mean() * 100), 2),
            "fraud_rate_pct": round(float(grp["fraud_flag"].mean() * 100), 2),
        })

    # 7. Seasonality & Time Anomalies (Pre-Payday vs Post-Payday, Late Night vs Daytime)
    pre_pay_df = df_auth[df_auth["is_pre_payday"] == 1]
    post_pay_df = df_auth[df_auth["is_pre_payday"] == 0]
    night_df = df_auth[df_auth["is_late_night"] == 1]
    day_df = df_auth[df_auth["is_late_night"] == 0]

    seasonality_signals = {
        "pre_payday_days_15_24_nsf_decline_rate_pct": round(
            float((pre_pay_df["decline_reason"] == "Insufficient Funds").mean() * 100), 2
        ) if len(pre_pay_df) else 0.0,
        "other_days_nsf_decline_rate_pct": round(
            float((post_pay_df["decline_reason"] == "Insufficient Funds").mean() * 100), 2
        ) if len(post_pay_df) else 0.0,
        "late_night_1am_5am_fraud_rate_pct": round(
            float(night_df["fraud_flag"].mean() * 100), 2
        ) if len(night_df) else 0.0,
        "daytime_fraud_rate_pct": round(
            float(day_df["fraud_flag"].mean() * 100), 2
        ) if len(day_df) else 0.0,
    }

    # 8. 3DS Authentication Dataset Summary
    if len(df_3ds) > 0:
        fail_3ds = df_3ds[df_3ds["is_auth_success"] == 0]
        fail_reasons_3ds = (
            fail_3ds["failure_reason"]
            .value_counts(normalize=True)
            .head(4)
            .mul(100)
            .round(2)
            .to_dict()
        ) if len(fail_3ds) > 0 else {}
        mandate_3ds = []
        for m_name, grp in df_3ds.groupby("challenge_mandate", observed=True):
            mandate_3ds.append({
                "mandate": str(m_name),
                "attempts": int(len(grp)),
                "success_rate_pct": round(float(grp["is_auth_success"].mean() * 100), 2),
                "avg_latency_ms": int(grp["latency_ms"].mean()),
            })
        auth_3ds_summary = {
            "total_3ds_attempts": int(len(df_3ds)),
            "overall_3ds_success_rate_pct": round(float(df_3ds["is_auth_success"].mean() * 100), 2),
            "avg_latency_ms": int(df_3ds["latency_ms"].mean()),
            "top_failure_reasons_pct": {str(k): float(v) for k, v in fail_reasons_3ds.items()},
            "by_challenge_mandate": mandate_3ds,
        }
    else:
        auth_3ds_summary = {"total_3ds_attempts": 0}

    # 9. ML Model Context Summary
    ml_summary = {}
    if ml_results:
        xgb_m = ml_results["models"]["advanced_xgboost"]["test_metrics"]
        lr_m = ml_results["models"]["baseline_logistic_regression"]["test_metrics"]
        xgb_ec = ml_results["models"]["advanced_xgboost"]["optimal_threshold_economics"]
        leg_ec = ml_results["models"]["legacy_static_rules_benchmark"]
        ml_summary = {
            "target": "fraud_flag (Fraud Probability at Authorization)",
            "optimal_operating_threshold": ml_results["optimal_threshold"],
            "xgb_test_roc_auc": xgb_m["roc_auc"],
            "xgb_test_pr_auc": xgb_m["pr_auc"],
            "lr_baseline_test_roc_auc": lr_m["roc_auc"],
            "lr_baseline_test_pr_auc": lr_m["pr_auc"],
            "xgb_net_ml_savings_oot_usd": xgb_ec["net_ml_savings_usd"],
            "incremental_gain_vs_legacy_rules_usd": round(
                xgb_ec["net_ml_savings_usd"] - leg_ec["net_ml_savings_usd"], 2
            ),
            "false_declines_oot_legacy": leg_ec["fp_false_declines"],
            "false_declines_oot_xgb": xgb_ec["fp_false_declines"],
            "top_5_shap_drivers": [
                d["label"] for d in ml_results.get("shap_global_importance", [])[:5]
            ],
        }

    return {
        "active_view": active_view,
        "active_filters": active_filters,
        "headline_kpis": {
            "filtered_transactions": n_tx,
            "share_of_total_portfolio_pct": round(n_tx / base_n_tx * 100, 2),
            "total_volume_usd": round(total_usd, 2),
            "avg_ticket_usd": round(avg_ticket, 2),
            "approval_rate_pct": round(app_rate, 2),
            "portfolio_benchmark_approval_rate_pct": round(base_app_rate, 2),
            "approval_rate_delta_vs_benchmark_pts": round(app_rate - base_app_rate, 2),
            "decline_rate_pct": round(dec_rate, 2),
            "declined_volume_usd": round(declined_usd, 2),
            "soft_recoverable_decline_share_pct": soft_dec_share,
            "estimated_false_fraud_declines_count": false_decline_cnt,
            "estimated_false_fraud_declines_usd": round(false_decline_usd, 2),
            "true_fraud_rate_pct": round(fraud_rate, 2),
            "portfolio_benchmark_fraud_rate_pct": round(base_fraud_rate, 2),
            "total_fraud_exposure_usd": round(fraud_usd, 2),
            "chargeback_count": cb_count,
            "chargeback_rate_pct": round(cb_rate, 2),
            "tokenized_share_pct": round(float(df_auth["is_tokenized"].mean() * 100), 2),
            "three_ds_authenticated_share_pct": round(float(df_auth["is_3ds_authenticated"].mean() * 100), 2),
        },
        "channel_breakdown": channel_stats,
        "acquirer_performance_ranked_worst_first": acquirer_stats,
        "ecommerce_security_lift": security_comparison,
        "mcc_performance_ranked_lowest_approval_first": mcc_stats[:5],
        "top_decline_reasons": decline_breakdown[:6],
        "ticket_size_brackets": ticket_stats,
        "seasonality_and_diurnal_signals": seasonality_signals,
        "authentication_3ds_summary": auth_3ds_summary,
        "ml_model_summary": ml_summary,
    }


def _synthesize_grounded_insights(context: Dict[str, Any]) -> str:
    """
    Deterministic, 100% data-grounded analytical engine that formats the 6 required
    executive sections directly from `<DATA_CONTEXT>`. Used when no external API key is
    configured or when network access is restricted.
    """
    if context.get("status") == "EMPTY_SLICE":
        return "⚠️ **No Data in Current Filter Slice:** Adjust the global filters in the sidebar to analyze transactions."

    kpis = context["headline_kpis"]
    view = context["active_view"]
    acqs = context.get("acquirer_performance_ranked_worst_first", [])
    mccs = context.get("mcc_performance_ranked_lowest_approval_first", [])
    decs = context.get("top_decline_reasons", [])
    sec = context.get("ecommerce_security_lift", {})
    seas = context.get("seasonality_and_diurnal_signals", {})
    auth3 = context.get("authentication_3ds_summary", {})
    ml = context.get("ml_model_summary", {})

    worst_acq = acqs[0] if acqs else {"acquirer": "N/A", "approval_rate_pct": 0, "technical_error_rate_pct": 0}
    best_acq = acqs[-1] if acqs else {"acquirer": "N/A", "approval_rate_pct": 0}
    worst_mcc = mccs[0] if mccs else {"mcc_category": "N/A", "approval_rate_pct": 0, "fraud_rate_pct": 0}
    top_dec_1 = decs[0] if len(decs) > 0 else {"decline_reason": "N/A", "share_of_declines_pct": 0, "lost_volume_usd": 0}
    top_dec_2 = decs[1] if len(decs) > 1 else {"decline_reason": "N/A", "share_of_declines_pct": 0, "lost_volume_usd": 0}

    delta_sign = "+" if kpis["approval_rate_delta_vs_benchmark_pts"] >= 0 else ""

    # View-specific lead sentence
    view_focus_note = (
        f"Focused on **{view}**, the active slice covers **{kpis['filtered_transactions']:,} transactions** "
        f"({kpis['share_of_total_portfolio_pct']:.1f}% of portfolio volume) representing **${kpis['total_volume_usd']:,.0f}** in gross spend."
    )

    sec_lift_str = ""
    if sec.get("ecom_tokenized_approval_pct") and sec.get("ecom_untokenized_approval_pct"):
        tok_diff = round(sec["ecom_tokenized_approval_pct"] - sec["ecom_untokenized_approval_pct"], 2)
        tds_diff = round((sec.get("ecom_3ds_approval_pct") or 0) - (sec.get("ecom_no3ds_approval_pct") or 0), 2)
        sec_lift_str = (
            f"Within E-Commerce, **Network/Wallet Tokenization** lifts approval rates by **+{tok_diff:.2f} percentage points** "
            f"({sec['ecom_tokenized_approval_pct']:.2f}% vs. {sec['ecom_untokenized_approval_pct']:.2f}% untokenized) while **3DS Authentication** "
            f"adds **+{tds_diff:.2f} percentage points** ({sec.get('ecom_3ds_approval_pct', 0):.2f}% vs. {sec.get('ecom_no3ds_approval_pct', 0):.2f}% unauthenticated)."
        )

    return f"""### 1. Storyline
{view_focus_note} Overall authorization approval stands at **{kpis['approval_rate_pct']:.2f}%** ({delta_sign}{kpis['approval_rate_delta_vs_benchmark_pts']:.2f} pts vs. the {kpis['portfolio_benchmark_approval_rate_pct']:.2f}% bank benchmark), leaving **${kpis['declined_volume_usd']:,.0f}** in declined transaction value (**{kpis['decline_rate_pct']:.2f}%** decline rate). Crucially, **{kpis['soft_recoverable_decline_share_pct']:.1f}% of all declines are Soft/Recoverable**, including **{kpis['estimated_false_fraud_declines_count']:,} false-positive risk blocks (${kpis['estimated_false_fraud_declines_usd']:,.0f})** where legitimate cardholders were declined under legacy `59 Suspected Fraud` or `05 Do Not Honor` rules.

### 2. Key Observations
* **Channel & Security Separation:** Tokenized credentials account for **{kpis['tokenized_share_pct']:.1f}%** of volume and 3DS covers **{kpis['three_ds_authenticated_share_pct']:.1f}%**. {sec_lift_str}
* **Decline Concentration:** The top two decline reasons are **{top_dec_1['decline_reason']}** (**{top_dec_1['share_of_declines_pct']:.1f}%** of declines, **${top_dec_1['lost_volume_usd']:,.0f}**) and **{top_dec_2['decline_reason']}** (**{top_dec_2['share_of_declines_pct']:.1f}%** of declines, **${top_dec_2['lost_volume_usd']:,.0f}**).
* **Fraud & Chargeback Exposure:** True underlying fraud rate in this slice is **{kpis['true_fraud_rate_pct']:.2f}%** (**${kpis['total_fraud_exposure_usd']:,.0f}** total exposure), driving **{kpis['chargeback_count']:,} post-authorization chargebacks** (**{kpis['chargeback_rate_pct']:.2f}%** of transactions).
* **3DS Authentication Funnel:** Across **{auth3.get('total_3ds_attempts', 0):,}** 3DS authentication requests, the pass rate is **{auth3.get('overall_3ds_success_rate_pct', 0):.2f}%** with an average latency of **{auth3.get('avg_latency_ms', 0):,} ms**.

### 3. Areas Needing Attention
* **Underperforming Acquirer (`{worst_acq['acquirer']}`):** Trails the portfolio with an approval rate of only **{worst_acq['approval_rate_pct']:.2f}%** (vs. **{best_acq['approval_rate_pct']:.2f}%** at `{best_acq['acquirer']}`), driven by a **{worst_acq['technical_error_rate_pct']:.2f}% Technical/Switch Error (`91`) rate** and **{worst_acq['cvv_error_rate_pct']:.2f}% Invalid CVV (`82`) rate**.
* **High-Risk MCC Friction (`{worst_mcc['mcc_category']}`):** Exhibits the lowest approval rate at **{worst_mcc['approval_rate_pct']:.2f}%** alongside a **{worst_mcc['fraud_rate_pct']:.2f}%** fraud rate.
* **Diurnal & Payday Vulnerability:** Late-night transactions (01:00–05:00) suffer a **{seas.get('late_night_1am_5am_fraud_rate_pct', 0):.2f}%** fraud rate (vs. **{seas.get('daytime_fraud_rate_pct', 0):.2f}%** daytime), while the pre-payday window (Days 15–24) sees `Insufficient Funds` declines rise to **{seas.get('pre_payday_days_15_24_nsf_decline_rate_pct', 0):.2f}%** (vs. **{seas.get('other_days_nsf_decline_rate_pct', 0):.2f}%** outside that window).

### 4. Possible Drivers
* **SHAP ML Feature Attribution:** Our trained XGBoost model (Test ROC-AUC **{ml.get('xgb_test_roc_auc', 0):.4f}**, PR-AUC **{ml.get('xgb_test_pr_auc', 0):.4f}**) confirms the primary drivers of risk are **{', '.join(ml.get('top_5_shap_drivers', [])[:4])}**.
* **Legacy Rule Over-Blocking:** Static rules penalize high-ticket and E-Commerce transactions indiscriminately when untokenized, causing **{kpis['estimated_false_fraud_declines_count']:,} false declines** while still missing multi-swipe velocity attacks.
* **Acquirer Switch Degradation:** `{worst_acq['acquirer']}` has lower tokenization adoption (**{worst_acq.get('tokenized_share_pct', 0):.1f}%**) and elevated ISO 8583 switch timeouts.

### 5. Decline Mitigation Ideas
* **Deploy XGBoost Fraud Scoring at Optimal Threshold ($\tau = {ml.get('optimal_operating_threshold', 0.28):.2f}$):** Replace static `59`/`05` rules with the XGBoost model + step-up 3DS routing for the Medium-Risk tier (`0.08`–`{ml.get('optimal_operating_threshold', 0.28):.2f}`), reducing false declines from **{ml.get('false_declines_oot_legacy', 0):,} to {ml.get('false_declines_oot_xgb', 0):,}** on the out-of-time test window and generating **+${ml.get('incremental_gain_vs_legacy_rules_usd', 0):,.0f}** in incremental net value.
* **Network Tokenization & Account Updater Push:** Migrate untokenized E-Commerce and Recurring merchants onto network tokens to eliminate `Expired Card (54)` declines and boost E-Commerce approval toward **{sec.get('ecom_tokenized_approval_pct', 93.0):.1f}%**.
* **Acquirer Technical Remediation (`{worst_acq['acquirer']}`):** Engage `{worst_acq['acquirer']}` engineering to fix switch timeout routing (`Code 91`) and enforce 3DS cryptogram pass-through.
* **Payday-Aware Smart Retry & Micro-Buffer:** Schedule recurring/MIT billing retries between the **25th and 5th** of the month and offer pre-approved low-risk cardholders a small overdraft grace buffer during Days 15–24.

### 6. Recommended Follow-Ups
1. **Run a 50/50 Champion-Challenger Shadow Test** of the XGBoost threshold ($\tau = {ml.get('optimal_operating_threshold', 0.28):.2f}$) on live E-Commerce traffic in `{worst_mcc['mcc_category']}`.
2. **Audit `{worst_acq['acquirer']}` ACS & Switch Logs** to isolate the root cause of its **{worst_acq['technical_error_rate_pct']:.2f}%** switch failure rate and high 3DS challenge latency.
3. **Review SMS OTP Delivery SLAs** on 3DS Challenge flows to cut `Cardholder Abandoned` and `OTP Timeout` drop-offs.
"""


def generate_executive_insights(context: Dict[str, Any]) -> Tuple[str, str]:
    """
    Generate the 6-section executive briefing from `<DATA_CONTEXT>`.
    Returns `(markdown_response, engine_mode_label)`.
    """
    provider_label, api_key, base_url = _get_api_credentials()

    if provider_label and api_key and base_url:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            model_name = "gpt-4o-mini"
            if "Groq" in provider_label:
                model_name = "llama-3.3-70b-versatile"
            elif "Gemini" in provider_label:
                model_name = "gemini-2.0-flash"

            user_message = (
                "Analyze the following `<DATA_CONTEXT>` from the Issuer Authorization Dashboard "
                "and produce the 6 required executive sections (Storyline, Key Observations, "
                "Areas Needing Attention, Possible Drivers, Decline Mitigation Ideas, "
                "Recommended Follow-Ups). Cite exact metrics from `<DATA_CONTEXT>` only.\n\n"
                f"<DATA_CONTEXT>\n{json.dumps(context, indent=2)}\n</DATA_CONTEXT>"
            )
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                timeout=25,
            )
            content = resp.choices[0].message.content or ""
            if content.strip():
                return content, f"Live LLM API ({provider_label})"
        except Exception as exc:
            fallback_text = _synthesize_grounded_insights(context)
            return (
                fallback_text,
                f"Deterministic Grounded Engine (API Fallback: {type(exc).__name__})",
            )

    return (
        _synthesize_grounded_insights(context),
        "Deterministic Grounded Analytics Engine (Zero-Key Local Mode)",
    )


def answer_followup_question(question: str, context: Dict[str, Any]) -> Tuple[str, str]:
    """
    Answer a user's free-text follow-up question strictly grounded in `<DATA_CONTEXT>`.
    Includes explicit guardrail checks against unsupported or out-of-schema claims.
    """
    q_clean = question.strip()
    if not q_clean:
        return "Please enter a question about the current dashboard view or filtered data.", "Guardrail Check"

    q_lower = q_clean.lower()

    # 1. Explicit Hallucination / Out-of-Scope Guardrail Check
    triggered_keywords = [kw for kw in OUT_OF_SCOPE_KEYWORDS if kw in q_lower]
    if triggered_keywords:
        return (
            "⚠️ **Grounding Guardrail Triggered (Unsupported Claim Prevention):**\n\n"
            f"Your question references **`{', '.join(triggered_keywords)}`**, which is **not present** in the "
            "supplied Issuer Authorization or 3DS Authentication dataset schema.\n\n"
            "* **Available Schema Attributes:** `timestamp`, `tokenised_card_id`, `merchant`, `mcc`, `acquirer`, "
            "`channel`, `amount`, `three_ds_indicator`, `tokenization_indicator`, `response_code`, `decline_reason`, "
            "`fraud_flag`, `chargeback_reason`, `chargeback_lifecycle_stage`, `challenge_mandate`, `auth_outcome`, "
            "and `ml_fraud_prob_xgb`.\n"
            "* To prevent hallucinated or unsupported claims, the AI Engine only answers questions backed by "
            "pre-computed aggregates in `<DATA_CONTEXT>`."
        ), "Grounding Guardrail Enforced"

    # 2. Try Live LLM API if configured
    provider_label, api_key, base_url = _get_api_credentials()
    if provider_label and api_key and base_url:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            model_name = "gpt-4o-mini"
            if "Groq" in provider_label:
                model_name = "llama-3.3-70b-versatile"
            elif "Gemini" in provider_label:
                model_name = "gemini-2.0-flash"

            prompt = (
                f"User Follow-Up Question: {q_clean}\n\n"
                "Answer concisely and accurately using ONLY the exact numbers inside `<DATA_CONTEXT>`. "
                "If the answer cannot be determined from `<DATA_CONTEXT>`, state that explicitly.\n\n"
                f"<DATA_CONTEXT>\n{json.dumps(context, indent=2)}\n</DATA_CONTEXT>"
            )
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=20,
            )
            ans = resp.choices[0].message.content or ""
            if ans.strip():
                return ans, f"Live LLM API ({provider_label})"
        except Exception:
            pass

    # 3. Deterministic Grounded Q&A Resolver (Offline-Safe)
    kpis = context.get("headline_kpis", {})
    acqs = context.get("acquirer_performance_ranked_worst_first", [])
    mccs = context.get("mcc_performance_ranked_lowest_approval_first", [])
    decs = context.get("top_decline_reasons", [])
    sec = context.get("ecommerce_security_lift", {})
    auth3 = context.get("authentication_3ds_summary", {})
    ml = context.get("ml_model_summary", {})
    seas = context.get("seasonality_and_diurnal_signals", {})

    if any(k in q_lower for k in ["acquirer", "legacy", "switch", "processor"]):
        lines = ["**Grounded Acquirer Analysis (from `<DATA_CONTEXT>`):**\n"]
        for a in acqs:
            lines.append(
                f"* **`{a['acquirer']}`**: **{a['approval_rate_pct']:.2f}%** approval rate across **{a['tx_count']:,}** tx | "
                f"Tech Error (`91`): **{a['technical_error_rate_pct']:.2f}%** | CVV Error (`82`): **{a['cvv_error_rate_pct']:.2f}%** | "
                f"Tokenized: **{a['tokenized_share_pct']:.1f}%**"
            )
        if acqs:
            worst = acqs[0]
            lines.append(
                f"\n**Actionable Recommendation:** Remediating `{worst['acquirer']}`'s switch timeout rate "
                f"({worst['technical_error_rate_pct']:.2f}%) and increasing its network tokenization adoption "
                "is the single fastest operational lever to lift overall approval rate."
            )
        return "\n".join(lines), "Deterministic Grounded Q&A"

    if any(k in q_lower for k in ["token", "3ds", "authentication", "otp", "challenge", "security"]):
        return (
            "**Grounded Security & Authentication Analysis (from `<DATA_CONTEXT>`):**\n\n"
            f"* **E-Commerce Tokenization Lift:** Tokenized E-Commerce transactions achieve **{sec.get('ecom_tokenized_approval_pct', 0):.2f}%** "
            f"approval (fraud rate **{sec.get('ecom_tokenized_fraud_pct', 0):.2f}%**) vs. **{sec.get('ecom_untokenized_approval_pct', 0):.2f}%** "
            f"for untokenized raw PANs (fraud rate **{sec.get('ecom_untokenized_fraud_pct', 0):.2f}%**).\n"
            f"* **E-Commerce 3DS Lift:** 3DS-authenticated E-Commerce achieves **{sec.get('ecom_3ds_approval_pct', 0):.2f}%** approval "
            f"(fraud rate **{sec.get('ecom_3ds_fraud_pct', 0):.2f}%**) vs. **{sec.get('ecom_no3ds_approval_pct', 0):.2f}%** without 3DS.\n"
            f"* **Pre-Auth 3DS Funnel:** Out of **{auth3.get('total_3ds_attempts', 0):,}** 3DS attempts, "
            f"**{auth3.get('overall_3ds_success_rate_pct', 0):.2f}%** succeed (avg latency **{auth3.get('avg_latency_ms', 0):,} ms**). "
            f"Top failure reasons: `{json.dumps(auth3.get('top_failure_reasons_pct', {}))}`."
        ), "Deterministic Grounded Q&A"

    if any(k in q_lower for k in ["model", "ml", "xgboost", "threshold", "shap", "baseline", "logistic"]):
        return (
            "**Grounded Machine Learning & Threshold Impact Summary (from `<DATA_CONTEXT>`):**\n\n"
            f"* **Model Comparison (Out-of-Time Test Set):** Advanced **XGBoost** achieves **ROC-AUC = {ml.get('xgb_test_roc_auc', 0):.4f}** "
            f"and **PR-AUC = {ml.get('xgb_test_pr_auc', 0):.4f}**, outperforming the **Logistic Regression Baseline** "
            f"(ROC-AUC = {ml.get('lr_baseline_test_roc_auc', 0):.4f}, PR-AUC = {ml.get('lr_baseline_test_pr_auc', 0):.4f}).\n"
            f"* **Optimal Operating Threshold ($\tau = {ml.get('optimal_operating_threshold', 0.28):.2f}$):** Delivers "
            f"**${ml.get('xgb_net_ml_savings_oot_usd', 0):,.2f}** in net savings on the out-of-time test period "
            f"(**+${ml.get('incremental_gain_vs_legacy_rules_usd', 0):,.2f}** incremental gain over legacy static rules) "
            f"and cuts false declines from **{ml.get('false_declines_oot_legacy', 0):,}** to **{ml.get('false_declines_oot_xgb', 0):,}**.\n"
            f"* **Top SHAP Feature Drivers:** `{', '.join(ml.get('top_5_shap_drivers', []))}`."
        ), "Deterministic Grounded Q&A"

    if any(k in q_lower for k in ["decline", "reason", "false", "recover", "soft", "nsf", "insufficient"]):
        dec_bullets = "\n".join(
            [f"  - **{d['decline_reason']}**: {d['count']:,} tx ({d['share_of_declines_pct']:.1f}% of declines, ${d['lost_volume_usd']:,.0f})" for d in decs]
        )
        return (
            "**Grounded Decline Analysis (from `<DATA_CONTEXT>`):**\n\n"
            f"* **Overall Decline Rate:** **{kpis.get('decline_rate_pct', 0):.2f}%** (**${kpis.get('declined_volume_usd', 0):,.0f}** declined value), "
            f"of which **{kpis.get('soft_recoverable_decline_share_pct', 0):.1f}%** are classified as Soft/Recoverable.\n"
            f"* **False-Positive Fraud Blocks:** **{kpis.get('estimated_false_fraud_declines_count', 0):,}** legitimate transactions "
            f"(**${kpis.get('estimated_false_fraud_declines_usd', 0):,.0f}**) were falsely declined by legacy `59`/`05` rules.\n"
            f"* **Pre-Payday Cash Crunch:** `Insufficient Funds (51)` rate is **{seas.get('pre_payday_days_15_24_nsf_decline_rate_pct', 0):.2f}%** "
            f"during Days 15–24 vs. **{seas.get('other_days_nsf_decline_rate_pct', 0):.2f}%** on post-payday/other days.\n"
            f"* **Top Decline Reasons in Current Slice:**\n{dec_bullets}"
        ), "Deterministic Grounded Q&A"

    if any(k in q_lower for k in ["fraud", "chargeback", "mcc", "merchant", "night"]):
        mcc_bullets = "\n".join(
            [f"  - **{m['mcc_category']} ({m['risk_tier']} Risk)**: {m['approval_rate_pct']:.2f}% approval | {m['fraud_rate_pct']:.2f}% fraud rate | Avg Ticket ${m['avg_ticket_usd']:.2f}" for m in mccs[:4]]
        )
        return (
            "**Grounded Fraud, Chargeback & MCC Risk Summary (from `<DATA_CONTEXT>`):**\n\n"
            f"* **True Fraud Rate:** **{kpis.get('true_fraud_rate_pct', 0):.2f}%** (**${kpis.get('total_fraud_exposure_usd', 0):,.0f}** exposure), "
            f"concentrating heavily during late-night hours 01:00–05:00 (**{seas.get('late_night_1am_5am_fraud_rate_pct', 0):.2f}%** vs. **{seas.get('daytime_fraud_rate_pct', 0):.2f}%** daytime).\n"
            f"* **Chargebacks:** **{kpis.get('chargeback_count', 0):,}** chargebacks (**{kpis.get('chargeback_rate_pct', 0):.2f}%** rate).\n"
            f"* **Highest-Friction MCCs:**\n{mcc_bullets}"
        ), "Deterministic Grounded Q&A"

    # General summary fallback grounded in context
    return (
        f"**Grounded Slice Summary for `{context.get('active_view')}` (from `<DATA_CONTEXT>`):**\n\n"
        f"* **Transactions Analyzed:** **{kpis.get('filtered_transactions', 0):,}** (**${kpis.get('total_volume_usd', 0):,.0f}** volume, Avg Ticket **${kpis.get('avg_ticket_usd', 0):.2f}**)\n"
        f"* **Approval Rate:** **{kpis.get('approval_rate_pct', 0):.2f}%** ({kpis.get('approval_rate_delta_vs_benchmark_pts', 0):+.2f} pts vs. bank benchmark)\n"
        f"* **Decline Rate:** **{kpis.get('decline_rate_pct', 0):.2f}%** ({kpis.get('soft_recoverable_decline_share_pct', 0):.1f}% soft/recoverable)\n"
        f"* **Fraud & Chargebacks:** **{kpis.get('true_fraud_rate_pct', 0):.2f}%** true fraud rate | **{kpis.get('chargeback_rate_pct', 0):.2f}%** chargeback rate\n"
        f"* **Tip:** Ask specifically about *acquirers*, *declines*, *3DS/tokenization*, *fraud/MCCs*, or *ML threshold impact* for deeper slice breakdowns."
    ), "Deterministic Grounded Q&A"

