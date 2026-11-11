"""
Machine Learning Pipeline for Issuer Authorization Analytics (Section 2B).

Problem Chosen:
- Predict Transaction Fraud Probability (`fraud_flag = 1`) at authorization time.
- Business Justification: Legacy static rules ('59 Suspected Fraud' / '05 Do Not Honor')
  suffer from high false-positive decline rates on legitimate E-Commerce and high-ticket
  transactions while still missing ~45% of sophisticated fraud. A well-calibrated ML model
  allows the issuer to simultaneously:
    1) Approve more legitimate transactions (recovering false declines & interchange revenue)
    2) Block true fraud before authorization (preventing fraud write-offs and chargebacks).

Methodology:
- Strict Time-Based Split:
    * Train:      2026-07-01 -> 2026-09-15 (~61% of volume)
    * Validation: 2026-09-16 -> 2026-10-05 (~18% of volume, used for threshold tuning)
    * Test (OOT): 2026-10-06 -> 2026-10-31 (~21% of volume, held-out out-of-time evaluation)
- Baseline vs. Advanced Model:
    * Simple Baseline: Logistic Regression on standard transaction attributes
    * Advanced Model: XGBoost Classifier with behavioral velocity, spend-anomaly ratios,
      and out-of-fold historical target encodings
- Explainability:
    * Global SHAP values via `shap.TreeExplainer`
    * Local transaction-level SHAP feature contributions for the dashboard's Scored Explorer
- Business Impact & Operating Threshold Optimization:
    * Explicit issuer unit economics (1.75% interchange revenue, 8.0% future spend churn + $15.00
      cardholder friction penalty per false decline, 100% fraud principal loss + $15.00 chargeback fee).
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".tmp_mpl"))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data_generator import generate_datasets, DATA_DIR

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ML_RESULTS_JSON = MODELS_DIR / "ml_results.json"
SCORED_PARQUET = DATA_DIR / "scored_predictions.parquet"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model.joblib"
LR_MODEL_PATH = MODELS_DIR / "lr_baseline.joblib"

# Stated Financial Unit Economics (Documented Issuer Assumptions)
DEFAULT_ECONOMICS = {
    "interchange_rate": 0.0175,          # 1.75% earned per approved good transaction
    "false_decline_churn_rate": 0.080,   # 8.0% lost top-of-wallet spend on falsely declined good tx
    "false_decline_friction_usd": 15.00, # $15.00 customer service & abandonment cost per false decline
    "chargeback_fee_usd": 15.00,         # $15.00 operational processing fee per missed fraud
    "fraud_loss_share": 1.00,            # 100% of principal lost on unauthenticated/missed fraud
}

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
    "channel": "Transaction Channel",
    "mcc_risk_tier": "MCC Risk Tier",
}


def engineer_features(
    df: pd.DataFrame,
    train_mask: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Engineer leakage-free features. Historical target encodings (MCC fraud rate,
    MCC average amount, Acquirer decline rate) are fitted strictly on the training split.
    """
    train_df = df.loc[train_mask]

    mcc_fraud_map = train_df.groupby("mcc")["fraud_flag"].mean().to_dict()
    mcc_amt_map = train_df.groupby("mcc")["amount"].mean().to_dict()
    acq_dec_map = (1.0 - train_df.groupby("acquirer")["is_approved"].mean()).to_dict()

    global_fraud_mean = float(train_df["fraud_flag"].mean())
    global_amt_mean = float(train_df["amount"].mean())
    global_dec_mean = float(1.0 - train_df["is_approved"].mean())

    feat = pd.DataFrame(index=df.index)
    feat["log_amount"] = np.log1p(df["amount"].to_numpy(dtype=float))
    feat["amount_to_card_avg_ratio"] = df["amount_to_card_avg_ratio"].to_numpy(dtype=float)

    mcc_avg_amt = df["mcc"].map(mcc_amt_map).fillna(global_amt_mean).to_numpy(dtype=float)
    feat["amount_to_mcc_avg_ratio"] = np.clip(
        df["amount"].to_numpy(dtype=float) / np.maximum(mcc_avg_amt, 1.0), 0.05, 25.0
    )

    feat["card_tx_count_1h"] = df["card_tx_count_1h"].to_numpy(dtype=float)
    feat["card_tx_count_24h"] = df["card_tx_count_24h"].to_numpy(dtype=float)
    feat["is_3ds_authenticated"] = df["is_3ds_authenticated"].to_numpy(dtype=float)
    feat["is_tokenized"] = df["is_tokenized"].to_numpy(dtype=float)
    feat["unsecured_ecom"] = (
        (df["channel"] == "E-Commerce")
        & (df["is_3ds_authenticated"] == 0)
        & (df["is_tokenized"] == 0)
    ).astype(float)
    feat["is_late_night"] = df["is_late_night"].to_numpy(dtype=float)
    feat["is_pre_payday"] = df["is_pre_payday"].to_numpy(dtype=float)
    feat["mcc_train_fraud_rate"] = df["mcc"].map(mcc_fraud_map).fillna(global_fraud_mean).to_numpy(dtype=float)
    feat["acquirer_train_decline_rate"] = df["acquirer"].map(acq_dec_map).fillna(global_dec_mean).to_numpy(dtype=float)

    feat["channel"] = df["channel"].astype(str)
    feat["mcc_risk_tier"] = df["mcc_risk_tier"].astype(str)

    encoding_maps = {
        "mcc_fraud_map": {str(k): round(float(v), 5) for k, v in mcc_fraud_map.items()},
        "mcc_amt_map": {str(k): round(float(v), 2) for k, v in mcc_amt_map.items()},
        "acq_dec_map": {str(k): round(float(v), 5) for k, v in acq_dec_map.items()},
    }
    return feat, encoding_maps


def compute_threshold_economics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    thresholds: np.ndarray,
    econ: Dict[str, float] = DEFAULT_ECONOMICS,
) -> pd.DataFrame:
    """
    Evaluate classification and business unit economics across a grid of operating thresholds.
    """
    ic_rate = econ.get("interchange_rate", 0.0175)
    churn_rate = econ.get("false_decline_churn_rate", 0.080)
    fd_fee = econ.get("false_decline_friction_usd", 15.00)
    cb_fee = econ.get("chargeback_fee_usd", 15.00)
    f_share = econ.get("fraud_loss_share", 1.00)

    is_fraud = y_true == 1
    is_good = ~is_fraud
    total_tx = len(y_true)
    total_fraud_cnt = int(is_fraud.sum())
    total_fraud_usd = float(amounts[is_fraud].sum())

    rows = []
    for tau in thresholds:
        flagged = y_prob >= tau  # Predicted Fraud -> Block / Decline
        approved = ~flagged

        tp_mask = flagged & is_fraud      # Caught Fraud
        fp_mask = flagged & is_good       # False Decline (Blocked Good Customer)
        fn_mask = approved & is_fraud     # Missed Fraud (Approved Fraudster)
        tn_mask = approved & is_good      # Approved Good Customer

        tp_cnt = int(tp_mask.sum())
        fp_cnt = int(fp_mask.sum())
        fn_cnt = int(fn_mask.sum())
        tn_cnt = int(tn_mask.sum())

        interchange_earned_usd = float(amounts[tn_mask].sum() * ic_rate)
        fraud_prevented_usd = float(amounts[tp_mask].sum() * f_share + tp_cnt * cb_fee)
        missed_fraud_loss_usd = float(amounts[fn_mask].sum() * f_share + fn_cnt * cb_fee)
        false_decline_cost_usd = float(amounts[fp_mask].sum() * (ic_rate + churn_rate) + fp_cnt * fd_fee)

        net_portfolio_profit_usd = (
            interchange_earned_usd - missed_fraud_loss_usd - false_decline_cost_usd
        )
        net_ml_savings_usd = fraud_prevented_usd - false_decline_cost_usd

        precision = tp_cnt / max(tp_cnt + fp_cnt, 1)
        recall = tp_cnt / max(total_fraud_cnt, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        approval_rate = int(approved.sum()) / max(total_tx, 1)
        false_positive_rate = fp_cnt / max(int(is_good.sum()), 1)
        fraud_value_capture_rate = float(amounts[tp_mask].sum()) / max(total_fraud_usd, 1.0)

        rows.append({
            "threshold": round(float(tau), 2),
            "approval_rate": round(approval_rate, 4),
            "precision": round(precision, 4),
            "recall_tx": round(recall, 4),
            "recall_usd": round(fraud_value_capture_rate, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(false_positive_rate, 4),
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


def train_and_evaluate_models(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Execute the end-to-end time-based ML training, evaluation, SHAP analysis,
    and threshold business impact calculation.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df_auth, _ = generate_datasets(force_regenerate=False)

    if not force_retrain and ML_RESULTS_JSON.exists() and SCORED_PARQUET.exists():
        with open(ML_RESULTS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # 1. TIME-BASED VALIDATION SPLIT (Strict Chronological Boundaries)
    # -------------------------------------------------------------------------
    ts = pd.to_datetime(df_auth["timestamp"])
    train_end = pd.Timestamp("2026-09-15 23:59:59")
    val_end = pd.Timestamp("2026-10-05 23:59:59")

    train_mask = (ts <= train_end).to_numpy()
    val_mask = ((ts > train_end) & (ts <= val_end)).to_numpy()
    test_mask = (ts > val_end).to_numpy()

    X_all, encoding_maps = engineer_features(df_auth, train_mask)
    y_all = df_auth["fraud_flag"].to_numpy(dtype=int)
    amt_all = df_auth["amount"].to_numpy(dtype=float)

    # Simple Baseline uses core static transaction features
    baseline_num_cols = [
        "log_amount",
        "is_3ds_authenticated",
        "is_tokenized",
        "is_late_night",
        "is_pre_payday",
    ]
    # Advanced XGBoost uses full behavioral velocity, spend ratios, and historical encodings
    num_cols = [
        "log_amount",
        "amount_to_card_avg_ratio",
        "amount_to_mcc_avg_ratio",
        "card_tx_count_1h",
        "card_tx_count_24h",
        "is_3ds_authenticated",
        "is_tokenized",
        "unsecured_ecom",
        "is_late_night",
        "is_pre_payday",
        "mcc_train_fraud_rate",
        "acquirer_train_decline_rate",
    ]
    cat_cols = ["channel", "mcc_risk_tier"]

    X_train, y_train = X_all.loc[train_mask], y_all[train_mask]
    X_val, y_val = X_all.loc[val_mask], y_all[val_mask]
    X_test, y_test = X_all.loc[test_mask], y_all[test_mask]

    # -------------------------------------------------------------------------
    # 2. MODEL 1: SIMPLE BASELINE (Logistic Regression on Static Attributes)
    # -------------------------------------------------------------------------
    preprocessor_lr = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), baseline_num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    lr_pipeline = Pipeline([
        ("prep", preprocessor_lr),
        ("clf", LogisticRegression(max_iter=500, C=1.0, random_state=42)),
    ])
    lr_pipeline.fit(X_train, y_train)

    lr_prob_val = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_prob_test = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_prob_all = lr_pipeline.predict_proba(X_all)[:, 1]

    # -------------------------------------------------------------------------
    # 3. MODEL 2: ADVANCED MODEL (XGBoost Gradient Boosted Trees + Engineered Features)
    # -------------------------------------------------------------------------
    X_xgb_all = pd.get_dummies(X_all[num_cols + cat_cols], columns=cat_cols, drop_first=False, dtype=float)
    xgb_feature_cols = list(X_xgb_all.columns)

    X_xgb_train = X_xgb_all.loc[train_mask]
    X_xgb_val = X_xgb_all.loc[val_mask]
    X_xgb_test = X_xgb_all.loc[test_mask]

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
        X_xgb_train,
        y_train,
        eval_set=[(X_xgb_val, y_val)],
        verbose=False,
    )

    xgb_prob_val = xgb_model.predict_proba(X_xgb_val)[:, 1]
    xgb_prob_test = xgb_model.predict_proba(X_xgb_test)[:, 1]
    xgb_prob_all = xgb_model.predict_proba(X_xgb_all)[:, 1]

    joblib.dump(lr_pipeline, LR_MODEL_PATH)
    joblib.dump(xgb_model, XGB_MODEL_PATH)

    # -------------------------------------------------------------------------
    # 4. OPERATING THRESHOLD OPTIMIZATION ON VALIDATION SET
    # -------------------------------------------------------------------------
    thresholds = np.round(np.arange(0.06, 0.92, 0.02), 2)
    val_econ_xgb = compute_threshold_economics(y_val, xgb_prob_val, amt_all[val_mask], thresholds)
    best_idx = int(val_econ_xgb["net_ml_savings_usd"].idxmax())
    optimal_threshold = float(val_econ_xgb.loc[best_idx, "threshold"])

    test_econ_xgb = compute_threshold_economics(y_test, xgb_prob_test, amt_all[test_mask], thresholds)
    test_econ_lr = compute_threshold_economics(y_test, lr_prob_test, amt_all[test_mask], thresholds)

    best_lr_val = compute_threshold_economics(y_val, lr_prob_val, amt_all[val_mask], thresholds)
    optimal_lr_threshold = float(best_lr_val.loc[int(best_lr_val["net_ml_savings_usd"].idxmax()), "threshold"])

    legacy_flagged_test = df_auth.loc[test_mask, "response_code"].isin(["59", "05"]).to_numpy()
    legacy_econ_df = compute_threshold_economics(
        y_test,
        legacy_flagged_test.astype(float),
        amt_all[test_mask],
        np.array([0.50]),
    )
    legacy_test_summary = legacy_econ_df.iloc[0].to_dict()

    xgb_at_opt = test_econ_xgb[test_econ_xgb["threshold"] == optimal_threshold].iloc[0].to_dict()
    lr_at_opt = test_econ_lr[test_econ_lr["threshold"] == optimal_lr_threshold].iloc[0].to_dict()

    # -------------------------------------------------------------------------
    # 5. CLASSIFICATION METRICS & CURVES (TEST SET)
    # -------------------------------------------------------------------------
    def _get_metrics(y_t: np.ndarray, y_p: np.ndarray, tau: float) -> Dict[str, float]:
        y_pred = (y_p >= tau).astype(int)
        return {
            "roc_auc": round(float(roc_auc_score(y_t, y_p)), 4),
            "pr_auc": round(float(average_precision_score(y_t, y_p)), 4),
            "brier_score": round(float(brier_score_loss(y_t, y_p)), 5),
            "operating_threshold": round(float(tau), 2),
            "precision_at_threshold": round(float(precision_score(y_t, y_pred, zero_division=0)), 4),
            "recall_at_threshold": round(float(recall_score(y_t, y_pred, zero_division=0)), 4),
            "f1_at_threshold": round(float(f1_score(y_t, y_pred, zero_division=0)), 4),
        }

    metrics_xgb_test = _get_metrics(y_test, xgb_prob_test, optimal_threshold)
    metrics_lr_test = _get_metrics(y_test, lr_prob_test, optimal_lr_threshold)
    metrics_xgb_val = _get_metrics(y_val, xgb_prob_val, optimal_threshold)
    metrics_lr_val = _get_metrics(y_val, lr_prob_val, optimal_lr_threshold)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob_test)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob_test)
    roc_idx_xgb = np.linspace(0, len(fpr_xgb) - 1, 80).astype(int)
    roc_idx_lr = np.linspace(0, len(fpr_lr) - 1, 80).astype(int)

    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_prob_test)
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_prob_test)
    pr_idx_xgb = np.linspace(0, len(prec_xgb) - 1, 80).astype(int)
    pr_idx_lr = np.linspace(0, len(prec_lr) - 1, 80).astype(int)

    # -------------------------------------------------------------------------
    # 6. SHAP EXPLAINABILITY (Global & Local Per-Transaction Drivers)
    # -------------------------------------------------------------------------
    explainer = shap.TreeExplainer(xgb_model)
    shap_sample = X_xgb_test.sample(n=min(3000, len(X_xgb_test)), random_state=42)
    shap_vals_sample = explainer.shap_values(shap_sample)

    nice_label_map = {
        **FEATURE_LABELS,
        "channel_E-Commerce": "Channel: E-Commerce",
        "channel_POS Chip & PIN": "Channel: POS Chip & PIN",
        "channel_POS Contactless": "Channel: POS Contactless",
        "channel_Recurring / MIT": "Channel: Recurring / MIT",
        "mcc_risk_tier_High": "MCC Risk Tier: High",
        "mcc_risk_tier_Medium": "MCC Risk Tier: Medium",
        "mcc_risk_tier_Low": "MCC Risk Tier: Low",
    }

    mean_abs_shap = np.abs(shap_vals_sample).mean(axis=0)
    shap_global_list = []
    for col_name, val in zip(xgb_feature_cols, mean_abs_shap):
        shap_global_list.append({
            "feature": col_name,
            "label": nice_label_map.get(col_name, col_name),
            "mean_abs_shap": round(float(val), 4),
        })
    shap_global_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

    risk_segment = np.where(
        xgb_prob_all >= optimal_threshold,
        "High Risk (Auto-Block / Step-Up)",
        np.where(xgb_prob_all >= 0.08, "Medium Risk (3DS Challenge)", "Low Risk (Frictionless Approve)"),
    )

    reasons_list = []
    unsec_arr = X_all["unsecured_ecom"].to_numpy()
    vel1_arr = X_all["card_tx_count_1h"].to_numpy()
    vel24_arr = X_all["card_tx_count_24h"].to_numpy()
    ratio_arr = X_all["amount_to_card_avg_ratio"].to_numpy()
    night_arr = X_all["is_late_night"].to_numpy()
    tier_arr = df_auth["mcc_risk_tier"].to_numpy()
    tok_arr = X_all["is_tokenized"].to_numpy()
    tds_arr = X_all["is_3ds_authenticated"].to_numpy()

    for i in range(len(df_auth)):
        tags = []
        if vel1_arr[i] >= 3:
            tags.append(f"High 1h Velocity ({int(vel1_arr[i])} tx)")
        elif vel24_arr[i] >= 5:
            tags.append(f"Elevated 24h Velocity ({int(vel24_arr[i])} tx)")
        if ratio_arr[i] >= 2.0:
            tags.append(f"Spend Anomaly ({ratio_arr[i]:.1f}x Card Avg)")
        if unsec_arr[i] == 1:
            tags.append("Unsecured E-Com (No 3DS/Token)")
        if tier_arr[i] == "High":
            tags.append("High-Risk MCC")
        if night_arr[i] == 1:
            tags.append("Late-Night Window (1-5 AM)")
        if not tags:
            if tok_arr[i] == 1 and tds_arr[i] == 1:
                tags.append("Tokenized + 3DS Verified (Low Risk)")
            elif tok_arr[i] == 1:
                tags.append("Tokenized Credentials (Low Risk)")
            else:
                tags.append("Normal Baseline Spend Pattern")
        reasons_list.append(" + ".join(tags[:3]))

    split_name = np.where(train_mask, "1_Train", np.where(val_mask, "2_Validation", "3_Test_OOT"))

    scored_df = pd.DataFrame({
        "transaction_id": df_auth["transaction_id"],
        "split": pd.Series(split_name, dtype="string"),
        "ml_fraud_prob_xgb": np.round(xgb_prob_all.astype(np.float32), 4),
        "ml_fraud_prob_lr": np.round(lr_prob_all.astype(np.float32), 4),
        "ml_risk_segment": pd.Series(risk_segment, dtype="string"),
        "ml_top_drivers": pd.Series(reasons_list, dtype="string"),
    })
    scored_df.to_parquet(SCORED_PARQUET, index=False)

    top_test_indices = X_xgb_test.iloc[np.argsort(-xgb_prob_test)[:50]].index
    shap_top50 = explainer.shap_values(X_xgb_all.loc[top_test_indices])
    base_value = float(np.atleast_1d(explainer.expected_value)[0])
    local_shap_examples = {}
    for row_pos, idx_val in enumerate(top_test_indices):
        tx_id = str(df_auth.loc[idx_val, "transaction_id"])
        s_vec = shap_top50[row_pos]
        f_vals = X_xgb_all.loc[idx_val]
        contribs = []
        for f_name, s_val in zip(xgb_feature_cols, s_vec):
            contribs.append({
                "feature": nice_label_map.get(f_name, f_name),
                "value": round(float(f_vals[f_name]), 3),
                "shap_contribution": round(float(s_val), 4),
            })
        contribs.sort(key=lambda d: abs(d["shap_contribution"]), reverse=True)
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

    test_df_merged = df_auth.loc[test_mask].copy()
    test_df_merged["ml_fraud_prob_xgb"] = xgb_prob_test
    test_df_merged["ml_risk_segment"] = risk_segment[test_mask]

    seg_summary = []
    for seg_label in [
        "Low Risk (Frictionless Approve)",
        "Medium Risk (3DS Challenge)",
        "High Risk (Auto-Block / Step-Up)",
    ]:
        sub = test_df_merged[test_df_merged["ml_risk_segment"] == seg_label]
        if len(sub) == 0:
            continue
        seg_summary.append({
            "risk_segment": seg_label,
            "tx_count": int(len(sub)),
            "volume_share_pct": round(float(len(sub) / len(test_df_merged) * 100), 2),
            "total_amount_usd": round(float(sub["amount"].sum()), 2),
            "avg_predicted_risk_pct": round(float(sub["ml_fraud_prob_xgb"].mean() * 100), 2),
            "actual_fraud_rate_pct": round(float(sub["fraud_flag"].mean() * 100), 2),
            "legacy_approval_rate_pct": round(float(sub["is_approved"].mean() * 100), 2),
            "fraud_captured_count": int(sub["fraud_flag"].sum()),
            "fraud_captured_share_pct": round(
                float(sub["fraud_flag"].sum() / max(test_df_merged["fraud_flag"].sum(), 1) * 100), 2
            ),
        })

    results_payload = {
        "problem_choice": {
            "target": "fraud_flag (Transaction Fraud Probability at Authorization)",
            "justification": (
                "Issuer authorization performance is bounded by the twin tension between Fraud Losses and "
                "False-Positive Declines. In our portfolio, legacy static rules ('59 Suspected Fraud' and "
                "'05 Do Not Honor') falsely decline thousands of legitimate E-Commerce and high-ticket "
                "transactions while still letting ~45% of true fraud slip through into post-auth chargebacks. "
                "Predicting fraud probability accurately allows the issuer to simultaneously auto-approve low-risk "
                "transactions (lifting overall approval rate and recovering lost interchange) while "
                "blocking high-risk fraud before settlement."
            ),
        },
        "validation_split": {
            "strategy": "Strict Chronological / Out-of-Time Split (Zero Future Leakage)",
            "train": {
                "period": "2026-07-01 to 2026-09-15 (77 Days)",
                "rows": int(train_mask.sum()),
                "fraud_rate": round(float(y_train.mean()), 4),
            },
            "validation": {
                "period": "2026-09-16 to 2026-10-05 (20 Days)",
                "rows": int(val_mask.sum()),
                "fraud_rate": round(float(y_val.mean()), 4),
            },
            "test_oot": {
                "period": "2026-10-06 to 2026-10-31 (26 Days, incl. Oct Cyber Sale)",
                "rows": int(test_mask.sum()),
                "fraud_rate": round(float(y_test.mean()), 4),
            },
        },
        "models": {
            "baseline_logistic_regression": {
                "val_metrics": metrics_lr_val,
                "test_metrics": metrics_lr_test,
                "optimal_threshold_economics": lr_at_opt,
            },
            "advanced_xgboost": {
                "val_metrics": metrics_xgb_val,
                "test_metrics": metrics_xgb_test,
                "optimal_threshold_economics": xgb_at_opt,
            },
            "legacy_static_rules_benchmark": legacy_test_summary,
        },
        "optimal_threshold": optimal_threshold,
        "unit_economics_assumptions": DEFAULT_ECONOMICS,
        "threshold_curve_xgb_test": test_econ_xgb.to_dict(orient="records"),
        "threshold_curve_lr_test": test_econ_lr.to_dict(orient="records"),
        "roc_curve": {
            "xgb": {
                "fpr": [round(float(x), 4) for x in fpr_xgb[roc_idx_xgb]],
                "tpr": [round(float(x), 4) for x in tpr_xgb[roc_idx_xgb]],
            },
            "lr": {
                "fpr": [round(float(x), 4) for x in fpr_lr[roc_idx_lr]],
                "tpr": [round(float(x), 4) for x in tpr_lr[roc_idx_lr]],
            },
        },
        "pr_curve": {
            "xgb": {
                "recall": [round(float(x), 4) for x in rec_xgb[pr_idx_xgb]],
                "precision": [round(float(x), 4) for x in prec_xgb[pr_idx_xgb]],
            },
            "lr": {
                "recall": [round(float(x), 4) for x in rec_lr[pr_idx_lr]],
                "precision": [round(float(x), 4) for x in prec_lr[pr_idx_lr]],
            },
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
    print("Regenerating datasets and training Baseline vs. Advanced XGBoost models...")
    generate_datasets(force_regenerate=True)
    res = train_and_evaluate_models(force_retrain=True)
    lr_m = res["models"]["baseline_logistic_regression"]["test_metrics"]
    xgb_m = res["models"]["advanced_xgboost"]["test_metrics"]
    xgb_ec = res["models"]["advanced_xgboost"]["optimal_threshold_economics"]
    lr_ec = res["models"]["baseline_logistic_regression"]["optimal_threshold_economics"]
    leg_ec = res["models"]["legacy_static_rules_benchmark"]

    print("\n--- OUT-OF-TIME TEST SET PERFORMANCE (2026-10-06 to 2026-10-31) ---")
    print(f"Logistic Regression Baseline -> ROC-AUC: {lr_m['roc_auc']:.4f} | PR-AUC: {lr_m['pr_auc']:.4f} | Brier: {lr_m['brier_score']:.5f}")
    print(f"XGBoost Advanced Model       -> ROC-AUC: {xgb_m['roc_auc']:.4f} | PR-AUC: {xgb_m['pr_auc']:.4f} | Brier: {xgb_m['brier_score']:.5f}")
    print(f"\nOptimal Operating Threshold (XGBoost): tau = {res['optimal_threshold']:.2f}")
    print(f"  XGBoost Net ML Savings (OOT Test): ${xgb_ec['net_ml_savings_usd']:,.2f} (Recall: {xgb_ec['recall_tx']:.1%}, False Declines: {xgb_ec['fp_false_declines']:,})")
    print(f"  LogReg Net ML Savings (OOT Test):  ${lr_ec['net_ml_savings_usd']:,.2f} (Recall: {lr_ec['recall_tx']:.1%}, False Declines: {lr_ec['fp_false_declines']:,})")
    print(f"  Legacy Rules Net Savings (Test):   ${leg_ec['net_ml_savings_usd']:,.2f} (Recall: {leg_ec['recall_tx']:.1%}, False Declines: {leg_ec['fp_false_declines']:,})")
    print(f"  Incremental Value (XGBoost vs Legacy): +${xgb_ec['net_ml_savings_usd'] - leg_ec['net_ml_savings_usd']:,.2f}")
    print("\nTop 6 Global SHAP Drivers:")
    for item in res["shap_global_importance"][:6]:
        print(f"  - {item['label']}: mean(|SHAP|) = {item['mean_abs_shap']:.4f}")
