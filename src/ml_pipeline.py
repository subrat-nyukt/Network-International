"""
Simplified, Easy-to-Read Machine Learning Pipeline (Section 2B).

Follows 6 clear steps:
  Step 1: Engineer leakage-free features (`engineer_features`)
  Step 2: Split data by time (Train -> Validation -> Out-of-Time Test)
  Step 3: Train Simple Baseline (Logistic Regression) vs. Advanced Model (XGBoost)
  Step 4: Find the best Operating Threshold using Business Unit Economics
  Step 5: Explain predictions with SHAP (Global importance + Local examples)
  Step 6: Save scored data & metrics for the Streamlit Dashboard
"""

import json
import os
from pathlib import Path

# Works in both .py files and Jupyter Notebooks (.ipynb) without NameError!
ROOT_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".tmp_mpl"))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_generator import DATA_DIR, generate_datasets

# -----------------------------------------------------------------------------
# FILE PATHS & BUSINESS ASSUMPTIONS
# -----------------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
ML_RESULTS_JSON = MODELS_DIR / "ml_results.json"
SCORED_PARQUET = DATA_DIR / "scored_predictions.parquet"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
LR_MODEL_PATH = MODELS_DIR / "lr_baseline.joblib"

# Stated Financial Unit Economics (Used to convert probabilities into Dollars)
DEFAULT_ECONOMICS = {
    "interchange_rate": 0.0175,          # +1.75% fee earned by bank on approved good spend
    "false_decline_churn_rate": 0.080,   # -8.0% lost future card spend when a good customer is blocked
    "false_decline_friction_usd": 15.00, # -$15.00 support/friction cost per false decline
    "chargeback_fee_usd": 15.00,         # -$15.00 operational dispute fee per missed fraud
    "fraud_loss_share": 1.00,            # -100% of transaction amount lost when fraud is missed
}

# Clean business names for dashboard charts
FEATURE_LABELS = {
    "unsecured_ecom": "Unsecured E-Com (No 3DS & No Token)",
    "card_tx_count_1h": "Card Velocity (Tx Count Last 1h)",
    "card_tx_count_24h": "Card Velocity (Tx Count Last 24h)",
    "amount_to_card_avg_ratio": "Spend Ratio vs. Cardholder Avg",
    "log_amount": "Transaction Amount (Log USD)",
    "mcc_train_fraud_rate": "MCC Historical Fraud Rate",
    "is_late_night": "Late-Night Window (01:00–05:00)",
    "is_3ds_authenticated": "3DS Authenticated Indicator",
    "is_tokenized": "Network/Wallet Tokenized",
    "amount_to_mcc_avg_ratio": "Amount Ratio vs. MCC Average",
    "acquirer_train_decline_rate": "Acquirer Historical Decline Rate",
    "is_pre_payday": "Pre-Payday Cash Crunch (Days 15–24)",
    "channel_E-Commerce": "Channel: E-Commerce",
    "channel_POS Chip & PIN": "Channel: POS Chip & PIN",
    "channel_POS Contactless": "Channel: POS Contactless",
    "channel_Recurring / MIT": "Channel: Recurring / MIT",
    "mcc_risk_tier_High": "MCC Risk Tier: High",
    "mcc_risk_tier_Medium": "MCC Risk Tier: Medium",
    "mcc_risk_tier_Low": "MCC Risk Tier: Low",
}


# =============================================================================
# STEP 1: FEATURE ENGINEERING (Leakage-Free)
# =============================================================================
def engineer_features(df: pd.DataFrame, train_mask: np.ndarray):
    """
    Create model features from transaction data.
    Historical averages (MCC fraud rate, MCC avg amount, Acquirer decline rate)
    are calculated ONLY on `train_df` so future test data is never leaked.
    """
    train_df = df[train_mask]

    # Learn historical rates from the Training set only
    mcc_fraud_map = train_df.groupby("mcc")["fraud_flag"].mean().to_dict()
    mcc_amt_map = train_df.groupby("mcc")["amount"].mean().to_dict()
    acq_dec_map = (1.0 - train_df.groupby("acquirer")["is_approved"].mean()).to_dict()

    # Build the numeric feature table
    feat = pd.DataFrame(index=df.index)
    feat["log_amount"] = np.log1p(df["amount"])
    feat["amount_to_card_avg_ratio"] = df["amount_to_card_avg_ratio"].astype(float)

    mcc_avg_amt = df["mcc"].map(mcc_amt_map).fillna(train_df["amount"].mean()).astype(float)
    feat["amount_to_mcc_avg_ratio"] = (df["amount"] / mcc_avg_amt.clip(lower=1.0)).clip(0.05, 25.0)

    feat["card_tx_count_1h"] = df["card_tx_count_1h"].astype(float)
    feat["card_tx_count_24h"] = df["card_tx_count_24h"].astype(float)
    feat["is_3ds_authenticated"] = df["is_3ds_authenticated"].astype(float)
    feat["is_tokenized"] = df["is_tokenized"].astype(float)

    # High-risk flag: Online E-Commerce with NO 3DS and NO Token
    feat["unsecured_ecom"] = (
        (df["channel"] == "E-Commerce")
        & (df["is_3ds_authenticated"] == 0)
        & (df["is_tokenized"] == 0)
    ).astype(float)

    feat["is_late_night"] = df["is_late_night"].astype(float)
    feat["is_pre_payday"] = df["is_pre_payday"].astype(float)
    feat["mcc_train_fraud_rate"] = df["mcc"].map(mcc_fraud_map).fillna(train_df["fraud_flag"].mean()).astype(float)
    feat["acquirer_train_decline_rate"] = df["acquirer"].map(acq_dec_map).fillna(1.0 - train_df["is_approved"].mean()).astype(float)

    # One-hot encode Channel and MCC Risk Tier
    dummies = pd.get_dummies(
        df[["channel", "mcc_risk_tier"]].astype(str),
        columns=["channel", "mcc_risk_tier"],
        dtype=float,
    )
    feat = pd.concat([feat, dummies], axis=1)

    encoding_maps = {
        "mcc_fraud_map": {str(k): round(float(v), 5) for k, v in mcc_fraud_map.items()},
        "mcc_amt_map": {str(k): round(float(v), 2) for k, v in mcc_amt_map.items()},
        "acq_dec_map": {str(k): round(float(v), 5) for k, v in acq_dec_map.items()},
    }
    return feat, encoding_maps


# =============================================================================
# STEP 2: BUSINESS IMPACT & THRESHOLD CALCULATOR
# =============================================================================
def compute_threshold_economics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    thresholds: np.ndarray,
    econ: dict = DEFAULT_ECONOMICS,
) -> pd.DataFrame:
    """
    For each probability threshold tau (e.g. 0.06 to 0.90), calculate:
      - How much fraud we catch (True Positives -> Money Saved)
      - How many good customers we falsely block (False Positives -> Money Lost)
      - Net ML Savings ($) = Fraud Prevented ($) - False Decline Cost ($)
    """
    ic_rate = econ.get("interchange_rate", 0.0175)
    churn_rate = econ.get("false_decline_churn_rate", 0.080)
    fd_fee = econ.get("false_decline_friction_usd", 15.00)
    cb_fee = econ.get("chargeback_fee_usd", 15.00)
    f_share = econ.get("fraud_loss_share", 1.00)

    is_fraud = y_true == 1
    is_good = ~is_fraud
    total_fraud_usd = float(amounts[is_fraud].sum())

    rows = []
    for tau in thresholds:
        flagged = y_prob >= tau  # Model says: Block / Decline
        approved = ~flagged      # Model says: Approve

        tp = flagged & is_fraud   # Caught Fraud
        fp = flagged & is_good    # False Decline (Good customer blocked)
        fn = approved & is_fraud  # Missed Fraud
        tn = approved & is_good   # Approved Good Customer

        tp_cnt, fp_cnt, fn_cnt, tn_cnt = int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum())

        # Dollar calculations
        interchange_earned_usd = float(amounts[tn].sum() * ic_rate)
        fraud_prevented_usd = float(amounts[tp].sum() * f_share + tp_cnt * cb_fee)
        missed_fraud_loss_usd = float(amounts[fn].sum() * f_share + fn_cnt * cb_fee)
        false_decline_cost_usd = float(amounts[fp].sum() * (ic_rate + churn_rate) + fp_cnt * fd_fee)

        net_ml_savings_usd = fraud_prevented_usd - false_decline_cost_usd
        net_portfolio_profit_usd = interchange_earned_usd - missed_fraud_loss_usd - false_decline_cost_usd

        precision = tp_cnt / max(tp_cnt + fp_cnt, 1)
        recall = tp_cnt / max(int(is_fraud.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        rows.append({
            "threshold": round(float(tau), 2),
            "approval_rate": round(int(approved.sum()) / max(len(y_true), 1), 4),
            "precision": round(precision, 4),
            "recall_tx": round(recall, 4),
            "recall_usd": round(float(amounts[tp].sum()) / max(total_fraud_usd, 1.0), 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fp_cnt / max(int(is_good.sum()), 1), 4),
            "tp_caught_fraud": tp_cnt,
            "fp_false_declines": fp_cnt,
            "fn_missed_fraud": fn_cnt,
            "tn_approved_good": tn_cnt,
            "interchange_earned_usd": round(interchange_earned_usd, 2),
            "fraud_prevented_usd": round(fraud_prevented_usd, 2),
            "missed_fraud_loss_usd": round(missed_fraud_loss_usd, 2),
            "false_decline_cost_usd": round(false_decline_cost_usd, 2),
            "net_ml_savings_usd": round(net_ml_savings_usd, 2),
            "net_portfolio_profit_usd": round(net_portfolio_profit_usd, 2),
        })

    return pd.DataFrame(rows)


def evaluate_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Calculate standard imbalanced classification metrics at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 5),
        "operating_threshold": round(float(threshold), 2),
        "precision_at_threshold": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_threshold": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_at_threshold": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


# =============================================================================
# STEP 3–6: MAIN TRAINING, SHAP EXPLAINABILITY & SAVING PIPELINE
# =============================================================================
def train_and_evaluate_models(force_retrain: bool = False) -> dict:
    """Train Baseline vs. XGBoost, compute SHAP & threshold economics, and save artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_auth, _ = generate_datasets(force_regenerate=False)

    if not force_retrain and ML_RESULTS_JSON.exists() and SCORED_PARQUET.exists():
        with open(ML_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # 1. Strict Time-Based Split (Chronological: Train -> Validation -> Test)
    # -------------------------------------------------------------------------
    ts = pd.to_datetime(df_auth["timestamp"])
    train_mask = (ts <= "2026-09-15 23:59:59").to_numpy()
    val_mask = ((ts > "2026-09-15 23:59:59") & (ts <= "2026-10-05 23:59:59")).to_numpy()
    test_mask = (ts > "2026-10-05 23:59:59").to_numpy()

    X_all, encoding_maps = engineer_features(df_auth, train_mask)
    y_all = df_auth["fraud_flag"].to_numpy(dtype=int)
    amt_all = df_auth["amount"].to_numpy(dtype=float)

    # Baseline uses static transaction features; XGBoost uses all engineered velocity/ratio features
    baseline_cols = [
        c for c in X_all.columns
        if c in ["log_amount", "is_3ds_authenticated", "is_tokenized", "is_late_night", "is_pre_payday"]
        or c.startswith("channel_")
        or c.startswith("mcc_risk_tier_")
    ]

    # -------------------------------------------------------------------------
    # 2. Train Simple Baseline: Logistic Regression
    # -------------------------------------------------------------------------
    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=42),
    )
    lr_model.fit(X_all.loc[train_mask, baseline_cols], y_all[train_mask])

    lr_prob_val = lr_model.predict_proba(X_all.loc[val_mask, baseline_cols])[:, 1]
    lr_prob_test = lr_model.predict_proba(X_all.loc[test_mask, baseline_cols])[:, 1]
    lr_prob_all = lr_model.predict_proba(X_all[baseline_cols])[:, 1]

    # -------------------------------------------------------------------------
    # 3. Train Advanced Model: XGBoost Classifier
    # -------------------------------------------------------------------------
    xgb_model = XGBClassifier(
        n_estimators=240,
        max_depth=6,
        learning_rate=0.055,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=12,
        reg_lambda=2.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    xgb_model.fit(
        X_all.loc[train_mask],
        y_all[train_mask],
        eval_set=[(X_all.loc[val_mask], y_all[val_mask])],
        verbose=False,
    )

    xgb_prob_val = xgb_model.predict_proba(X_all.loc[val_mask])[:, 1]
    xgb_prob_test = xgb_model.predict_proba(X_all.loc[test_mask])[:, 1]
    xgb_prob_all = xgb_model.predict_proba(X_all)[:, 1]

    joblib.dump(lr_model, LR_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)

    # -------------------------------------------------------------------------
    # 4. Find Optimal Operating Threshold on Validation Set
    # -------------------------------------------------------------------------
    thresholds = np.round(np.arange(0.06, 0.92, 0.02), 2)

    val_econ_xgb = compute_threshold_economics(y_all[val_mask], xgb_prob_val, amt_all[val_mask], thresholds)
    optimal_threshold = float(val_econ_xgb.loc[val_econ_xgb["net_ml_savings_usd"].idxmax(), "threshold"])

    val_econ_lr = compute_threshold_economics(y_all[val_mask], lr_prob_val, amt_all[val_mask], thresholds)
    optimal_lr_threshold = float(val_econ_lr.loc[val_econ_lr["net_ml_savings_usd"].idxmax(), "threshold"])

    # Evaluate on Out-of-Time Test Set
    test_econ_xgb = compute_threshold_economics(y_all[test_mask], xgb_prob_test, amt_all[test_mask], thresholds)
    test_econ_lr = compute_threshold_economics(y_all[test_mask], lr_prob_test, amt_all[test_mask], thresholds)

    xgb_at_opt = test_econ_xgb[test_econ_xgb["threshold"] == optimal_threshold].iloc[0].to_dict()
    lr_at_opt = test_econ_lr[test_econ_lr["threshold"] == optimal_lr_threshold].iloc[0].to_dict()

    # Legacy static rules benchmark ('59 Suspected Fraud' & '05 Do Not Honor')
    legacy_flagged_test = df_auth.loc[test_mask, "response_code"].isin(["59", "05"]).to_numpy(dtype=float)
    legacy_test_summary = compute_threshold_economics(
        y_all[test_mask], legacy_flagged_test, amt_all[test_mask], np.array([0.50])
    ).iloc[0].to_dict()

    # -------------------------------------------------------------------------
    # 5. SHAP Explainability (Global Feature Importance + Top 50 Local Examples)
    # -------------------------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    X_test_df = X_all.loc[test_mask]
    shap_sample = X_test_df.sample(n=min(3000, len(X_test_df)), random_state=42)
    shap_vals_sample = explainer.shap_values(shap_sample)

    mean_abs_shap = np.abs(shap_vals_sample).mean(axis=0)
    shap_global_list = sorted(
        [
            {
                "feature": col,
                "label": FEATURE_LABELS.get(col, col),
                "mean_abs_shap": round(float(val), 4),
            }
            for col, val in zip(X_all.columns, mean_abs_shap)
        ],
        key=lambda x: x["mean_abs_shap"],
        reverse=True,
    )

    # Local SHAP waterfall breakdown for top 50 high-risk test transactions
    top_test_indices = X_test_df.iloc[np.argsort(-xgb_prob_test)[:50]].index
    shap_top50 = explainer.shap_values(X_all.loc[top_test_indices])
    base_value = float(np.atleast_1d(explainer.expected_value)[0])

    local_shap_examples = {}
    for row_pos, idx_val in enumerate(top_test_indices):
        tx_id = str(df_auth.loc[idx_val, "transaction_id"])
        contribs = sorted(
            [
                {
                    "feature": FEATURE_LABELS.get(col, col),
                    "value": round(float(X_all.loc[idx_val, col]), 3),
                    "shap_contribution": round(float(s_val), 4),
                }
                for col, s_val in zip(X_all.columns, shap_top50[row_pos])
            ],
            key=lambda d: abs(d["shap_contribution"]),
            reverse=True,
        )
        local_shap_examples[tx_id] = {
            "transaction_id": tx_id,
            "predicted_prob": round(float(xgb_prob_all[idx_val]), 4),
            "actual_fraud": int(y_all[idx_val]),
            "amount": float(amt_all[idx_val]),
            "merchant": str(df_auth.loc[idx_val, "merchant"]),
            "mcc_category": str(df_auth.loc[idx_val, "mcc_category"]),
            "channel": str(df_auth.loc[idx_val, "channel"]),
            "acquirer": str(df_auth.loc[idx_val, "acquirer"]),
            "decline_reason": str(df_auth.loc[idx_val, "decline_reason"]),
            "base_log_odds": round(base_value, 4),
            "top_contributions": contribs[:8],
        }

    # -------------------------------------------------------------------------
    # 6. Assign Risk Segments & Save Scored Predictions Parquet
    # -------------------------------------------------------------------------
    risk_segment = np.where(
        xgb_prob_all >= optimal_threshold,
        "High Risk (Auto-Block / Step-Up)",
        np.where(xgb_prob_all >= 0.08, "Medium Risk (3DS Challenge)", "Low Risk (Frictionless Approve)"),
    )

    # Simple human-readable tags for each transaction
    reasons_list = []
    for vel1, vel24, ratio, unsec, tier, night, tok, tds in zip(
        X_all["card_tx_count_1h"],
        X_all["card_tx_count_24h"],
        X_all["amount_to_card_avg_ratio"],
        X_all["unsecured_ecom"],
        df_auth["mcc_risk_tier"],
        X_all["is_late_night"],
        X_all["is_tokenized"],
        X_all["is_3ds_authenticated"],
    ):
        tags = []
        if vel1 >= 3:
            tags.append(f"High 1h Velocity ({int(vel1)} tx)")
        elif vel24 >= 5:
            tags.append(f"Elevated 24h Velocity ({int(vel24)} tx)")
        if ratio >= 2.0:
            tags.append(f"Spend Anomaly ({ratio:.1f}x Card Avg)")
        if unsec == 1:
            tags.append("Unsecured E-Com (No 3DS/Token)")
        if tier == "High":
            tags.append("High-Risk MCC")
        if night == 1:
            tags.append("Late-Night Window (1-5 AM)")
        if not tags:
            tags.append("Tokenized + 3DS Verified (Low Risk)" if (tok == 1 and tds == 1) else "Normal Baseline Spend Pattern")
        reasons_list.append(" + ".join(tags[:3]))

    split_name = np.where(train_mask, "1_Train", np.where(val_mask, "2_Validation", "3_Test_OOT"))
    pd.DataFrame({
        "transaction_id": df_auth["transaction_id"],
        "split": pd.Series(split_name, dtype="string"),
        "ml_fraud_prob_xgb": np.round(xgb_prob_all.astype(np.float32), 4),
        "ml_fraud_prob_lr": np.round(lr_prob_all.astype(np.float32), 4),
        "ml_risk_segment": pd.Series(risk_segment, dtype="string"),
        "ml_top_drivers": pd.Series(reasons_list, dtype="string"),
    }).to_parquet(SCORED_PARQUET, index=False)

    # Risk Segment Summary Table on Test Set
    test_df = df_auth.loc[test_mask].copy()
    test_df["ml_fraud_prob_xgb"] = xgb_prob_test
    test_df["ml_risk_segment"] = risk_segment[test_mask]

    seg_summary = []
    for seg_label in [
        "Low Risk (Frictionless Approve)",
        "Medium Risk (3DS Challenge)",
        "High Risk (Auto-Block / Step-Up)",
    ]:
        sub = test_df[test_df["ml_risk_segment"] == seg_label]
        if len(sub) > 0:
            seg_summary.append({
                "risk_segment": seg_label,
                "tx_count": int(len(sub)),
                "volume_share_pct": round(float(len(sub) / len(test_df) * 100), 2),
                "total_amount_usd": round(float(sub["amount"].sum()), 2),
                "avg_predicted_risk_pct": round(float(sub["ml_fraud_prob_xgb"].mean() * 100), 2),
                "actual_fraud_rate_pct": round(float(sub["fraud_flag"].mean() * 100), 2),
                "legacy_approval_rate_pct": round(float(sub["is_approved"].mean() * 100), 2),
                "fraud_captured_count": int(sub["fraud_flag"].sum()),
                "fraud_captured_share_pct": round(float(sub["fraud_flag"].sum() / max(test_df["fraud_flag"].sum(), 1) * 100), 2),
            })

    # ROC and PR curves (downsampled to 80 points for fast dashboard plotting)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_all[test_mask], xgb_prob_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_all[test_mask], lr_prob_test)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_all[test_mask], xgb_prob_test)
    prec_lr, rec_lr, _ = precision_recall_curve(y_all[test_mask], lr_prob_test)

    def _sample80(arr):
        idx = np.linspace(0, len(arr) - 1, 80).astype(int)
        return [round(float(x), 4) for x in arr[idx]]

    results_payload = {
        "problem_choice": {
            "target": "fraud_flag (Transaction Fraud Probability at Authorization)",
            "justification": (
                "Real-time AI risk scoring accurately separates legitimate shoppers from fraudsters.\n\n"
                "• This allows banks to boost card approvals and revenue while blocking real fraud attacks before settlement."
            ),
        },
        "validation_split": {
            "strategy": "Strict Chronological / Out-of-Time Split (Zero Future Leakage)",
            "train": {"period": "2026-07-01 to 2026-09-15 (77 Days)", "rows": int(train_mask.sum()), "fraud_rate": round(float(y_all[train_mask].mean()), 4)},
            "validation": {"period": "2026-09-16 to 2026-10-05 (20 Days)", "rows": int(val_mask.sum()), "fraud_rate": round(float(y_all[val_mask].mean()), 4)},
            "test_oot": {"period": "2026-10-06 to 2026-10-31 (26 Days, incl. Oct Cyber Sale)", "rows": int(test_mask.sum()), "fraud_rate": round(float(y_all[test_mask].mean()), 4)},
        },
        "models": {
            "baseline_logistic_regression": {
                "val_metrics": evaluate_metrics(y_all[val_mask], lr_prob_val, optimal_lr_threshold),
                "test_metrics": evaluate_metrics(y_all[test_mask], lr_prob_test, optimal_lr_threshold),
                "optimal_threshold_economics": lr_at_opt,
            },
            "advanced_xgboost": {
                "val_metrics": evaluate_metrics(y_all[val_mask], xgb_prob_val, optimal_threshold),
                "test_metrics": evaluate_metrics(y_all[test_mask], xgb_prob_test, optimal_threshold),
                "optimal_threshold_economics": xgb_at_opt,
            },
            "legacy_static_rules_benchmark": legacy_test_summary,
        },
        "optimal_threshold": optimal_threshold,
        "unit_economics_assumptions": DEFAULT_ECONOMICS,
        "threshold_curve_xgb_test": test_econ_xgb.to_dict(orient="records"),
        "threshold_curve_lr_test": test_econ_lr.to_dict(orient="records"),
        "roc_curve": {
            "xgb": {"fpr": _sample80(fpr_xgb), "tpr": _sample80(tpr_xgb)},
            "lr": {"fpr": _sample80(fpr_lr), "tpr": _sample80(tpr_lr)},
        },
        "pr_curve": {
            "xgb": {"recall": _sample80(rec_xgb), "precision": _sample80(prec_xgb)},
            "lr": {"recall": _sample80(rec_lr), "precision": _sample80(prec_lr)},
        },
        "shap_global_importance": shap_global_list,
        "risk_segments_test": seg_summary,
        "local_shap_examples": local_shap_examples,
        "encoding_maps": encoding_maps,
    }

    with open(ML_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    return results_payload


if __name__ == "__main__":
    print("Training Baseline (Logistic Regression) vs. Advanced (XGBoost)...")
    res = train_and_evaluate_models(force_retrain=True)
    lr_m = res["models"]["baseline_logistic_regression"]["test_metrics"]
    xgb_m = res["models"]["advanced_xgboost"]["test_metrics"]
    xgb_ec = res["models"]["advanced_xgboost"]["optimal_threshold_economics"]
    leg_ec = res["models"]["legacy_static_rules_benchmark"]

    print(f"Logistic Regression -> ROC-AUC: {lr_m['roc_auc']:.4f} | PR-AUC: {lr_m['pr_auc']:.4f}")
    print(f"XGBoost Advanced    -> ROC-AUC: {xgb_m['roc_auc']:.4f} | PR-AUC: {xgb_m['pr_auc']:.4f}")
    print(f"Optimal Threshold: tau = {res['optimal_threshold']:.2f} | Net Savings: ${xgb_ec['net_ml_savings_usd']:,.2f}")
    print(f"False Declines Reduced: {leg_ec['fp_false_declines']:,} (Legacy) -> {xgb_ec['fp_false_declines']:,} (XGBoost)")
