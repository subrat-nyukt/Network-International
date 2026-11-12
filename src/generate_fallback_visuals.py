"""
Generates high-resolution fallback visual snapshots (PNGs) of all 6 dashboard views
in `screenshots/` so reviewers behind corporate firewalls can inspect the visuals
directly in the repository or README.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".tmp_mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generator import generate_datasets
from src.ml_pipeline import train_and_evaluate_models

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "screenshots"


def generate_all_snapshots() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    df_auth, df_3ds = generate_datasets()
    ml_res = train_and_evaluate_models()

    plt.rcParams.update({
        "font.sans-serif": "DejaVu Sans",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 0.8,
        "figure.facecolor": "#f8fafc",
        "axes.facecolor": "#ffffff",
    })

    # -------------------------------------------------------------------------
    # 1. OVERVIEW & SEASONALITY SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.suptitle(
        "View 1 — Portfolio Overview: Daily Volumes, Approval Rate, Channel Mix & Payday Seasonality\n"
        f"(220,000 Auth Tx | Gross Spend: ${df_auth['amount'].sum()/1e6:,.2f}M | Approval Rate: {df_auth['is_approved'].mean()*100:.2f}% | Fraud Rate: {df_auth['fraud_flag'].mean()*100:.2f}%)",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    daily = df_auth.groupby("date").agg(tx=("transaction_id", "count"), app=("is_approved", "mean")).reset_index()
    ax = axes[0, 0]
    ax.bar(daily["date"], daily["tx"], color="#93c5fd", alpha=0.8, label="Daily Tx Volume")
    ax.set_title("Daily Transaction Volume (Jul–Oct 2026, incl. Oct Cyber Sale)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Transactions")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax = axes[0, 1]
    ax.plot(daily["date"], daily["app"] * 100, color="#1d4ed8", linewidth=2)
    ax.set_title("Daily Authorization Approval Rate (%)", fontweight="bold", fontsize=10.5)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_ylim(82, 97)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ch = df_auth.groupby("channel").agg(tx=("transaction_id", "count"), app=("is_approved", "mean")).sort_values("tx")
    ax = axes[1, 0]
    bars = ax.barh(ch.index, ch["tx"] / len(df_auth) * 100, color=["#0ea5e9", "#3b82f6", "#6366f1", "#1d4ed8"])
    ax.set_title("Channel Mix (% of Volume) & Approval Rate", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Share of Portfolio Volume (%)")
    for b, app_v in zip(bars, ch["app"] * 100):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{app_v:.1f}% Appr", va="center", fontsize=9)
    ax.set_xlim(0, 52)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    dom = df_auth.groupby("day_of_month").apply(lambda g: (g["decline_reason"] == "Insufficient Funds").mean() * 100)
    ax = axes[1, 1]
    colors_dom = [
        "#10b981" if (d >= 25 or d <= 5) else ("#f97316" if 15 <= d <= 24 else "#60a5fa")
        for d in dom.index
    ]
    ax.bar(dom.index, dom.values, color=colors_dom)
    ax.set_title("Payday Seasonality: Insufficient Funds Decline Rate by Day of Month", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Day of Month (Green = Post-Payday 25–5 | Orange = Pre-Payday 15–24)")
    ax.set_ylabel("NSF Decline Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(SCREENSHOTS_DIR / "01_overview_and_seasonality.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. AUTHORIZATION PERFORMANCE SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.suptitle(
        "View 2 — Authorization Performance by Acquirer, 3DS & Network Tokenization",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    acq = df_auth.groupby("acquirer")["is_approved"].mean().mul(100).sort_values()
    ax = axes[0]
    cols_acq = ["#ef4444" if idx == "LegacySwitch_Bank" else "#2563eb" for idx in acq.index]
    bars = ax.barh(acq.index, acq.values, color=cols_acq)
    ax.set_title("Approval Rate (%) by Acquirer (LegacySwitch_Bank Anomaly)", fontweight="bold", fontsize=10.5)
    ax.set_xlim(0, 102)
    for b in bars:
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.2f}%", va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    ecom = df_auth[df_auth["channel"] == "E-Commerce"]
    sec_grp = ecom.groupby(["three_ds_indicator", "tokenization_indicator"])["is_approved"].mean().mul(100).unstack()
    ax = axes[1]
    x = np.arange(len(sec_grp.index))
    w = 0.35
    if "Raw PAN (Untokenized)" in sec_grp.columns:
        ax.bar(x - w / 2, sec_grp["Raw PAN (Untokenized)"], width=w, label="Raw PAN (Untokenized)", color="#94a3b8")
    if "Tokenized (Network/Wallet)" in sec_grp.columns:
        ax.bar(x + w / 2, sec_grp["Tokenized (Network/Wallet)"], width=w, label="Tokenized (Network/Wallet)", color="#0284c7")
    ax.set_xticks(x)
    ax.set_xticklabels(sec_grp.index, rotation=12)
    ax.set_ylim(60, 100)
    ax.set_ylabel("E-Commerce Approval Rate (%)")
    ax.set_title("E-Commerce Security Lift: 3DS Status × Tokenization", fontweight="bold", fontsize=10.5)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(SCREENSHOTS_DIR / "02_authorization_performance.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. DECLINES INTELLIGENCE SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.suptitle(
        "View 3 — Decline Intelligence: Top Reasons (Hard vs. Soft) & MCC × Ticket Size Heatmap",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    dec_df = df_auth[df_auth["is_approved"] == 0]
    reasons = dec_df["decline_reason"].value_counts().sort_values()
    ax = axes[0]
    bars = ax.barh(reasons.index, reasons.values, color="#f59e0b")
    ax.set_title("Top Decline Reasons (Transaction Count)", fontweight="bold", fontsize=10.5)
    for b in bars:
        ax.text(b.get_width() + 80, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,}", va="center", fontsize=8.5)
    ax.set_xlim(0, reasons.max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    piv = (
        df_auth.pivot_table(
            index="mcc_category",
            columns="ticket_size_bucket",
            values="is_approved",
            aggfunc=lambda s: (1.0 - s.mean()) * 100,
        )[["<$25", "$25-$100", "$100-$500", "$500+"]]
    )
    ax = axes[1]
    im = ax.imshow(piv.values, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8.5)
    ax.set_title("Decline Rate (%) Heatmap: MCC Category × Ticket Size", fontweight="bold", fontsize=10.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="black" if val < 15 else "white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(SCREENSHOTS_DIR / "03_decline_intelligence.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 4. FRAUD & CHARGEBACKS SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.suptitle(
        "View 4 — Fraud & Chargebacks: Diurnal Late-Night Spike & Chargeback Lifecycle Funnel",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    hr_f = df_auth.groupby("hour")["fraud_flag"].mean().mul(100)
    ax = axes[0]
    cols_hr = ["#dc2626" if 1 <= h <= 5 else "#3b82f6" for h in hr_f.index]
    ax.bar(hr_f.index, hr_f.values, color=cols_hr)
    ax.set_title("Fraud Rate (%) by Hour of Day (Red = 01:00–05:00 Attack Window)", fontweight="bold", fontsize=10.5)
    ax.set_xlabel("Hour of Day (0–23)")
    ax.set_ylabel("True Fraud Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    cb_df = df_auth[df_auth["has_chargeback"] == 1]
    cb_st = cb_df["chargeback_lifecycle_stage"].value_counts().sort_index()
    ax = axes[1]
    bars = ax.barh(cb_st.index, cb_st.values, color="#7c3aed")
    ax.set_title("Chargeback Volume by Lifecycle Stage", fontweight="bold", fontsize=10.5)
    for b in bars:
        ax.text(b.get_width() + 10, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,}", va="center", fontsize=9)
    ax.set_xlim(0, cb_st.max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(SCREENSHOTS_DIR / "04_fraud_and_chargebacks.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. AUTHENTICATION (3DS) FUNNEL SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    fig.suptitle(
        f"View 5 — 3DS Authentication Funnel (134,191 Attempts | Overall Pass Rate: {df_3ds['is_auth_success'].mean()*100:.2f}%)",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    mand = df_3ds.groupby("challenge_mandate")["is_auth_success"].mean().mul(100).sort_values()
    ax = axes[0]
    bars = ax.barh(mand.index, mand.values, color="#0d9488")
    ax.set_title("3DS Authentication Pass Rate (%) by Challenge Mandate", fontweight="bold", fontsize=10.5)
    ax.set_xlim(0, 105)
    for b in bars:
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    fails = df_3ds[df_3ds["is_auth_success"] == 0]["failure_reason"].value_counts().sort_values()
    ax = axes[1]
    bars = ax.barh(fails.index, fails.values, color="#e11d48")
    ax.set_title("3DS Failure Reasons Breakdown", fontweight="bold", fontsize=10.5)
    for b in bars:
        ax.text(b.get_width() + 40, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,}", va="center", fontsize=9)
    ax.set_xlim(0, fails.max() * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(SCREENSHOTS_DIR / "05_authentication_3ds_funnel.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. ML MODEL, SHAP & OPERATING THRESHOLD BUSINESS IMPACT SNAPSHOT
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    xgb_m = ml_res["models"]["advanced_xgboost"]["test_metrics"]
    lr_m = ml_res["models"]["baseline_logistic_regression"]["test_metrics"]
    opt_tau = ml_res["optimal_threshold"]
    fig.suptitle(
        f"View 6 — ML Fraud Model (Out-of-Time Test: XGBoost PR-AUC={xgb_m['pr_auc']:.4f} vs LogReg={lr_m['pr_auc']:.4f} | Optimal τ={opt_tau:.2f})",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )

    # PR Curve
    ax = axes[0]
    ax.plot(
        ml_res["pr_curve"]["xgb"]["recall"],
        ml_res["pr_curve"]["xgb"]["precision"],
        color="#2563eb",
        linewidth=2.5,
        label=f"XGBoost (PR-AUC={xgb_m['pr_auc']:.3f})",
    )
    ax.plot(
        ml_res["pr_curve"]["lr"]["recall"],
        ml_res["pr_curve"]["lr"]["precision"],
        color="#94a3b8",
        linewidth=2,
        linestyle="--",
        label=f"LogReg Baseline (PR-AUC={lr_m['pr_auc']:.3f})",
    )
    ax.set_title("Precision-Recall Curve (OOT Test)", fontweight="bold", fontsize=10)
    ax.set_xlabel("Recall (Fraud Capture Rate)")
    ax.set_ylabel("Precision (Hit Rate)")
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Threshold Curve
    ax = axes[1]
    tc_df = pd.DataFrame(ml_res["threshold_curve_xgb_test"])
    ax.plot(tc_df["threshold"], tc_df["net_ml_savings_usd"] / 1e3, color="#16a34a", linewidth=2.5, label="XGBoost Net Savings ($k)")
    ax.axvline(opt_tau, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Optimal τ = {opt_tau:.2f}")
    ax.set_title("Net Business Benefit Across Thresholds", fontweight="bold", fontsize=10)
    ax.set_xlabel("Operating Probability Threshold (τ)")
    ax.set_ylabel("Net Savings ($ Thousands USD)")
    ax.legend(fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # SHAP Global Drivers
    ax = axes[2]
    shap_top = pd.DataFrame(ml_res["shap_global_importance"][:8]).iloc[::-1]
    ax.barh(shap_top["label"], shap_top["mean_abs_shap"], color="#0284c7")
    ax.set_title("Top 8 Global SHAP Feature Drivers", fontweight="bold", fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(SCREENSHOTS_DIR / "06_ml_model_shap_and_threshold_impact.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print("Generating fallback visual PNG snapshots in screenshots/...")
    generate_all_snapshots()
    print("Saved 6 high-resolution PNG snapshots in screenshots/.")

