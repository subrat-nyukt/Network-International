"""
Synthetic Data Generator for Issuer Authorization & 3DS Authentication Analytics.


Built-in Domain Patterns & Documented Assumptions:
- Channel & Security: POS Contactless / Chip & PIN have ~95-97% approval and ~0.15% fraud;
  E-Commerce has lower approval (~81-84%) and higher fraud (~3.2%). Tokenization and 3DS
  authentication lift E-Commerce approval by +10-15 percentage points and cut fraud by >75%.
- MCC & Ticket Size: High-risk MCCs (Crypto 6051, Digital Gaming 5816, Luxury 5944,
  Electronics 5732) and large ticket sizes ($500+) drive 'Suspected Fraud' and
  'Insufficient Funds' declines.
- Acquirer Differences: 'LegacySwitch_Bank' exhibits degraded switch reliability
  (~5x higher 'Technical / Switch Error' and 'Invalid CVV' declines, lower 3DS/token routing).
- Seasonality & Temporal Cycles:
  * Monthly Payday Cycle: Days 25-5 (post-payday) show higher liquidity and 45% fewer
    'Insufficient Funds' declines; Days 15-24 (pre-payday crunch) show elevated declines.
  * Diurnal Pattern: Late-night hours (01:00-05:00) have 3.5x-4x higher fraud concentration.
  * Promotional Surge: October 15-25 Cyber Retail Week spikes E-Commerce volume by +35%.
- Cardholder Velocity: Compromised cards exhibit rapid multi-transaction bursts within 1-24h.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
AUTH_TX_PATH = DATA_DIR / "authorization_transactions.parquet"
AUTH_3DS_PATH = DATA_DIR / "authentication_3ds.parquet"


MCC_CATALOG = [
    # (mcc, mcc_category, risk_tier, base_fraud_prob, base_amount_mu, base_amount_sigma, ecom_share, merchants)
    ("5411", "Grocery & Supermarkets", "Low", 0.0015, 3.65, 0.65, 0.18,
     ["Carrefour Hyper", "Spinneys Fresh", "Lulu Market", "Waitrose Daily"]),
    ("4900", "Utilities & Telecom", "Low", 0.0012, 4.15, 0.55, 0.72,
     ["Etisalat BillPay", "DEWA SmartGrid", "Du Telecom", "NatGrid Power"]),
    ("5912", "Pharmacies & Healthcare", "Low", 0.0018, 3.50, 0.60, 0.22,
     ["Life Pharmacy", "Aster MedPlus", "Boots Health", "MedCare Direct"]),
    ("5812", "Restaurants & Dining", "Medium", 0.0045, 3.75, 0.70, 0.35,
     ["Talabat Eats", "Deliveroo Plus", "Zuma Bistro", "ShakeShack ME"]),
    ("5651", "Apparel & Department Stores", "Medium", 0.0065, 4.45, 0.75, 0.58,
     ["Zara Online", "Namshi Fashion", "Ounass Style", "H&M Global"]),
    ("4511", "Airlines & Cross-Border Travel", "Medium", 0.0110, 5.85, 0.70, 0.88,
     ["Emirates Air", "Booking.com Intl", "Etihad Holidays", "Expedia Global"]),
    ("5732", "Consumer Electronics", "High", 0.0220, 5.55, 0.85, 0.76,
     ["Apple Store Online", "Sharaf DG Tech", "Amazon Electronics", "Virgin Megastore"]),
    ("5944", "Luxury Goods & Jewelry", "High", 0.0260, 6.10, 0.85, 0.68,
     ["Cartier Boutique", "Rolex Prestige", "Farfetch Luxury", "Bloomingdales Lux"]),
    ("5816", "Digital Goods & Online Gaming", "High", 0.0360, 3.60, 0.90, 0.96,
     ["Steam Games", "PlayStation Net", "Epic Games Store", "Roblox Digital"]),
    ("6051", "Crypto & Quasi-Cash", "High", 0.0480, 5.45, 0.95, 0.98,
     ["Binance Ramp", "CryptoCom Pay", "Coinbase Direct", "MoonPay Global"]),
]

ACQUIRER_CATALOG = [
    # (acquirer_name, share, tech_error_prob, cvv_err_prob, token_adoption_boost, latency_base_ms)
    ("GlobalPay_Net", 0.28, 0.004, 0.006, 0.12, 185),
    ("AdyenSwitch", 0.24, 0.003, 0.005, 0.16, 165),
    ("StripeAcq_EU", 0.21, 0.004, 0.006, 0.14, 175),
    ("RegionalPay_ME", 0.15, 0.011, 0.014, 0.00, 260),
    ("LegacySwitch_Bank", 0.12, 0.052, 0.038, -0.18, 440),  # Underperforming legacy processor
]


def _assign_ticket_bucket(amounts: np.ndarray) -> pd.Series:
    return pd.Series(
        pd.cut(
            amounts,
            bins=[-0.01, 25.0, 100.0, 500.0, 1e9],
            labels=["<$25", "$25-$100", "$100-$500", "$500+"],
        ),
        dtype="string",
    )


def generate_datasets(
    n_auth_rows: int = 220_000,
    n_3ds_rows: Optional[int] = None,
    seed: int = 42,
    force_regenerate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate or load the synthetic Issuer Authorization and 3DS Authentication datasets.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not force_regenerate and AUTH_TX_PATH.exists() and AUTH_3DS_PATH.exists():
        df_auth = pd.read_parquet(AUTH_TX_PATH)
        df_3ds = pd.read_parquet(AUTH_3DS_PATH)
        return df_auth, df_3ds

    rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # 1. TIMELINE GENERATION (4 Months: 2026-07-01 to 2026-10-31 = 123 days)
    # -------------------------------------------------------------------------
    start_ts = pd.Timestamp("2026-07-01 00:00:00")
    total_days = 123
    day_indices = np.arange(total_days)
    dates = pd.date_range(start_ts, periods=total_days, freq="D")

    day_of_month = dates.day.to_numpy()
    day_of_week = dates.dayofweek.to_numpy()
    month_num = dates.month.to_numpy()

    daily_weights = np.ones(total_days, dtype=float)
    is_post_payday_day = (day_of_month >= 25) | (day_of_month <= 5)
    daily_weights[is_post_payday_day] *= 1.18
    daily_weights[day_of_week >= 4] *= 1.12
    daily_weights *= 1.0 + 0.025 * (month_num - 7)
    is_cyber_sale_day = (month_num == 10) & (day_of_month >= 15) & (day_of_month <= 25)
    daily_weights[is_cyber_sale_day] *= 1.35

    daily_probs = daily_weights / daily_weights.sum()
    chosen_day_idx = rng.choice(day_indices, size=n_auth_rows, p=daily_probs)

    hour_weights = np.array([
        0.018, 0.014, 0.012, 0.011, 0.012, 0.016,
        0.025, 0.038, 0.052, 0.062, 0.068, 0.072,
        0.076, 0.074, 0.068, 0.065, 0.066, 0.070,
        0.068, 0.058, 0.046, 0.034, 0.024, 0.019,
    ], dtype=float)
    hour_probs = hour_weights / hour_weights.sum()
    chosen_hours = rng.choice(np.arange(24), size=n_auth_rows, p=hour_probs)
    chosen_mins = rng.integers(0, 60, size=n_auth_rows)
    chosen_secs = rng.integers(0, 60, size=n_auth_rows)

    timestamps = (
        start_ts
        + pd.to_timedelta(chosen_day_idx, unit="D")
        + pd.to_timedelta(chosen_hours, unit="h")
        + pd.to_timedelta(chosen_mins, unit="m")
        + pd.to_timedelta(chosen_secs, unit="s")
    )

    order = np.argsort(timestamps.to_numpy())
    timestamps = timestamps[order]
    chosen_day_idx = chosen_day_idx[order]
    chosen_hours = chosen_hours[order]

    tx_dates = pd.DatetimeIndex(timestamps)
    tx_dom = tx_dates.day.to_numpy()
    tx_dow = tx_dates.dayofweek.to_numpy()
    tx_month = tx_dates.month.to_numpy()
    is_late_night = (chosen_hours >= 1) & (chosen_hours <= 5)
    is_pre_payday = (tx_dom >= 15) & (tx_dom <= 24)
    is_post_payday = (tx_dom >= 25) | (tx_dom <= 5)
    is_promo_period = (tx_month == 10) & (tx_dom >= 15) & (tx_dom <= 25)

    # -------------------------------------------------------------------------
    # 2. CARDHOLDER PROFILES (~18,000 unique tokenised cards)
    # -------------------------------------------------------------------------
    n_cards = 18_000
    card_ids_pool = np.array([f"CRD-{i:06d}" for i in range(1, n_cards + 1)])
    card_activity_w = rng.gamma(shape=1.8, scale=1.0, size=n_cards)
    card_activity_p = card_activity_w / card_activity_w.sum()
    card_affluence = rng.lognormal(mean=0.0, sigma=0.45, size=n_cards)
    card_low_balance_risk = rng.beta(a=1.5, b=8.5, size=n_cards)

    card_idx = rng.choice(np.arange(n_cards), size=n_auth_rows, p=card_activity_p)

    # Inject 1,050 compromised card attack sessions (rapid multi-swipe bursts)
    n_bursts = 1050
    burst_anchor_rows = rng.choice(np.arange(200, n_auth_rows - 25), size=n_bursts, replace=False)
    is_velocity_attack = np.zeros(n_auth_rows, dtype=bool)
    for anchor in burst_anchor_rows:
        burst_len = int(rng.integers(3, 6))
        target_card = card_idx[anchor]
        offsets = rng.integers(1, 14, size=burst_len)
        burst_positions = np.clip(anchor + np.cumsum(offsets), 0, n_auth_rows - 1)
        card_idx[burst_positions] = target_card
        is_velocity_attack[anchor] = True
        is_velocity_attack[burst_positions] = True

    tokenised_card_id = card_ids_pool[card_idx]

    # -------------------------------------------------------------------------
    # 3. MCC, MERCHANT, ACQUIRER, AND CHANNEL ASSIGNMENT
    # -------------------------------------------------------------------------
    mcc_weights = np.array([0.19, 0.08, 0.08, 0.16, 0.13, 0.07, 0.09, 0.05, 0.09, 0.06])
    mcc_idx = rng.choice(len(MCC_CATALOG), size=n_auth_rows, p=mcc_weights)

    high_risk_mcc_choices = rng.choice([6, 7, 8, 9], size=is_velocity_attack.sum(), p=[0.25, 0.15, 0.35, 0.25])
    mcc_idx[is_velocity_attack] = high_risk_mcc_choices

    promo_mask = is_promo_period & (rng.random(n_auth_rows) < 0.22)
    mcc_idx[promo_mask] = rng.choice([4, 6, 7, 8], size=promo_mask.sum(), p=[0.30, 0.40, 0.15, 0.15])

    mcc_codes = np.array([m[0] for m in MCC_CATALOG])[mcc_idx]
    mcc_categories = np.array([m[1] for m in MCC_CATALOG])[mcc_idx]
    mcc_risk_tiers = np.array([m[2] for m in MCC_CATALOG])[mcc_idx]
    mcc_amt_mu = np.array([m[4] for m in MCC_CATALOG])[mcc_idx]
    mcc_amt_sigma = np.array([m[5] for m in MCC_CATALOG])[mcc_idx]
    mcc_ecom_share = np.array([m[6] for m in MCC_CATALOG])[mcc_idx]

    merchant_sub_idx = rng.integers(0, 4, size=n_auth_rows)
    merchant_matrix = np.array([m[7] for m in MCC_CATALOG])
    merchants = merchant_matrix[mcc_idx, merchant_sub_idx]

    acq_names = np.array([a[0] for a in ACQUIRER_CATALOG])
    acq_shares = np.array([a[1] for a in ACQUIRER_CATALOG])
    acq_tech_err = np.array([a[2] for a in ACQUIRER_CATALOG])
    acq_cvv_err = np.array([a[3] for a in ACQUIRER_CATALOG])
    acq_token_boost = np.array([a[4] for a in ACQUIRER_CATALOG])

    acq_idx = rng.choice(len(ACQUIRER_CATALOG), size=n_auth_rows, p=acq_shares)
    acquirers = acq_names[acq_idx]

    ecom_prob = np.clip(
        mcc_ecom_share
        + np.where(is_late_night, 0.22, 0.0)
        + np.where(is_promo_period, 0.08, 0.0)
        + np.where(is_velocity_attack, 0.35, 0.0),
        0.05,
        0.99,
    )
    u_chan = rng.random(n_auth_rows)
    channels = np.empty(n_auth_rows, dtype=object)

    recurring_prob = np.where(mcc_codes == "4900", 0.52, np.where(mcc_codes == "5816", 0.20, 0.04))
    is_recurring = u_chan < recurring_prob
    is_ecom = (~is_recurring) & (u_chan < (recurring_prob + ecom_prob * (1.0 - recurring_prob)))
    is_pos = (~is_recurring) & (~is_ecom)

    pos_contactless_mask = is_pos & (rng.random(n_auth_rows) < 0.74)
    pos_chip_mask = is_pos & (~pos_contactless_mask)

    channels[is_ecom] = "E-Commerce"
    channels[is_recurring] = "Recurring / MIT"
    channels[pos_contactless_mask] = "POS Contactless"
    channels[pos_chip_mask] = "POS Chip & PIN"

    # -------------------------------------------------------------------------
    # 4. AMOUNT (TICKET SIZE) GENERATION
    # -------------------------------------------------------------------------
    raw_amounts = rng.lognormal(mean=mcc_amt_mu, sigma=mcc_amt_sigma, size=n_auth_rows)
    amounts = raw_amounts * (0.65 + 0.35 * card_affluence[card_idx])
    probe_mask = is_velocity_attack & (rng.random(n_auth_rows) < 0.18)
    heavy_attack_mask = is_velocity_attack & (~probe_mask)
    amounts[probe_mask] = rng.uniform(1.00, 4.50, size=probe_mask.sum())
    amounts[heavy_attack_mask] *= rng.uniform(2.2, 4.2, size=heavy_attack_mask.sum())

    amounts = np.round(np.clip(amounts, 1.00, 12500.00), 2)
    ticket_buckets = _assign_ticket_bucket(amounts)

    # -------------------------------------------------------------------------
    # 5. TOKENIZATION & 3D SECURE (3DS) INDICATORS
    # -------------------------------------------------------------------------
    base_token_prob = np.where(
        channels == "POS Contactless", 0.64,
        np.where(
            channels == "POS Chip & PIN", 0.08,
            np.where(channels == "Recurring / MIT", 0.58, 0.44)
        )
    )
    token_prob = np.clip(
        base_token_prob + acq_token_boost[acq_idx] - np.where(is_velocity_attack, 0.36, 0.0),
        0.03,
        0.92,
    )
    is_tokenized = (rng.random(n_auth_rows) < token_prob).astype(int)
    tokenization_indicator = np.where(
        is_tokenized == 1, "Tokenized (Network/Wallet)", "Raw PAN (Untokenized)"
    )

    three_ds_indicator = np.full(n_auth_rows, "Not Applicable (POS)", dtype=object)
    is_cnp = (channels == "E-Commerce") | (channels == "Recurring / MIT")

    cnp_3ds_prob = np.where(channels == "E-Commerce", 0.68, 0.28)
    cnp_3ds_prob = np.clip(
        cnp_3ds_prob
        + np.where(acquirers == "LegacySwitch_Bank", -0.24, 0.04)
        + np.where(amounts > 250.0, 0.14, 0.0)
        - np.where(is_velocity_attack, 0.44, 0.0),
        0.08,
        0.94,
    )
    has_3ds = is_cnp & (rng.random(n_auth_rows) < cnp_3ds_prob)
    challenge_share = np.clip(
        0.35 + np.where(amounts > 200.0, 0.25, 0.0) + np.where(mcc_risk_tiers == "High", 0.20, 0.0),
        0.15,
        0.85,
    )
    is_3ds_challenge = has_3ds & (rng.random(n_auth_rows) < challenge_share)
    is_3ds_frictionless = has_3ds & (~is_3ds_challenge)
    is_no_3ds_cnp = is_cnp & (~has_3ds)

    three_ds_indicator[is_3ds_frictionless] = "3DS Frictionless"
    three_ds_indicator[is_3ds_challenge] = "3DS Challenge Authenticated"
    three_ds_indicator[is_no_3ds_cnp] = "No 3DS / Unauthenticated"
    is_3ds_authenticated = has_3ds.astype(int)

    # -------------------------------------------------------------------------
    # 6. VELOCITY & SPEND RATIO SIGNALS
    # -------------------------------------------------------------------------
    df_temp = pd.DataFrame({
        "timestamp": timestamps,
        "card_idx": card_idx,
        "amount": amounts,
        "mcc_idx": mcc_idx,
    })

    df_temp = df_temp.sort_values(["card_idx", "timestamp"])
    grp = df_temp.groupby("card_idx", sort=False)
    tx_count_1h = grp.rolling("1h", on="timestamp")["amount"].count().to_numpy()
    tx_count_24h = grp.rolling("24h", on="timestamp")["amount"].count().to_numpy()

    cum_sum = grp["amount"].cumsum().to_numpy() - df_temp["amount"].to_numpy()
    cum_cnt = grp.cumcount().to_numpy()
    fallback_card_avg = np.exp(mcc_amt_mu[df_temp["mcc_idx"].to_numpy()])
    hist_card_avg = np.where(cum_cnt > 0, cum_sum / np.maximum(cum_cnt, 1), fallback_card_avg)
    amount_to_card_avg = np.clip(df_temp["amount"].to_numpy() / np.maximum(hist_card_avg, 5.0), 0.1, 25.0)

    inv_order = np.argsort(df_temp.index.to_numpy())
    tx_count_1h = tx_count_1h[inv_order].astype(int)
    tx_count_24h = tx_count_24h[inv_order].astype(int)
    amount_to_card_avg = np.round(amount_to_card_avg[inv_order], 3)

    # -------------------------------------------------------------------------
    # 7. GROUND-TRUTH FRAUD GENERATION (Rich Non-Linear Interactions)
    # -------------------------------------------------------------------------
    unsecured_ecom = ((channels == "E-Commerce") & (is_3ds_authenticated == 0) & (is_tokenized == 0)).astype(float)
    ecom_flag = (channels == "E-Commerce").astype(float)
    high_risk_mcc_flag = (mcc_risk_tiers == "High").astype(float)
    med_risk_mcc_flag = (mcc_risk_tiers == "Medium").astype(float)
    legacy_acq_flag = (acquirers == "LegacySwitch_Bank").astype(float)

    # Non-linear interaction regimes that tree models (XGBoost) excel at discovering
    regime_vel_burst = ((tx_count_1h >= 3) & (amount_to_card_avg >= 1.8) & (is_tokenized == 0)).astype(float)
    regime_night_cnp = (is_late_night & (ecom_flag == 1) & (is_3ds_authenticated == 0) & (amounts > 120.0)).astype(float)
    regime_high_mcc_unsec = ((high_risk_mcc_flag == 1) & (unsecured_ecom == 1) & (amounts > 180.0)).astype(float)
    regime_card_testing = ((amounts <= 5.0) & (ecom_flag == 1) & (high_risk_mcc_flag == 1) & (tx_count_24h >= 3)).astype(float)
    regime_extreme_ratio = ((amount_to_card_avg >= 3.2) & (is_3ds_authenticated == 0) & (med_risk_mcc_flag + high_risk_mcc_flag >= 1)).astype(float)

    fraud_logit = (
        -6.10
        + 0.75 * ecom_flag
        - 0.90 * is_3ds_authenticated
        - 0.85 * is_tokenized
        + 0.70 * high_risk_mcc_flag
        + 0.25 * med_risk_mcc_flag
        + 0.45 * is_late_night.astype(float)
        + 0.22 * np.clip(tx_count_1h - 1, 0, 5)
        + 0.12 * np.clip(np.log1p(amount_to_card_avg) - 0.7, 0, 3.5)
        # Strong non-linear tree-discoverable regimes:
        + 3.15 * regime_vel_burst
        + 2.65 * regime_night_cnp
        + 2.55 * regime_high_mcc_unsec
        + 3.40 * regime_card_testing
        + 2.35 * regime_extreme_ratio
        + 0.85 * legacy_acq_flag * unsecured_ecom
    )
    fraud_prob_true = 1.0 / (1.0 + np.exp(-fraud_logit))
    fraud_flag = (rng.random(n_auth_rows) < fraud_prob_true).astype(int)

    # -------------------------------------------------------------------------
    # 8. AUTHORIZATION OUTCOME, RESPONSE CODES, AND DECLINE REASONS
    # -------------------------------------------------------------------------
    # 1) Insufficient Funds (51): Driven by ticket size, pre-payday cash crunch (Days 15-24)
    nsf_prob = (
        0.010
        + 0.075 * card_low_balance_risk[card_idx]
        + 0.024 * is_pre_payday.astype(float)
        - 0.010 * is_post_payday.astype(float)
        + 0.028 * (amounts > 250.0).astype(float)
        + 0.048 * (amounts > 800.0).astype(float)
        + 0.022 * np.clip((amount_to_card_avg - 2.0) / 5.0, 0.0, 1.0)
    )
    nsf_prob = np.clip(nsf_prob, 0.003, 0.32)

    # 2) Legacy Fraud Rule Blocks ('59 Suspected Fraud' & '05 Do Not Honor'):
    # Crude static rules catch ~55% of actual fraud, but over-block legitimate E-com / High-Ticket / Night spend!
    legacy_rule_suspicion = (
        0.005
        + 0.52 * fraud_flag
        + 0.072 * unsecured_ecom
        + 0.065 * high_risk_mcc_flag * (amounts > 160.0).astype(float)
        + 0.055 * is_late_night.astype(float) * ecom_flag
        + 0.052 * (amounts > 450.0).astype(float) * (1 - is_3ds_authenticated)
        + 0.035 * is_promo_period.astype(float) * (amounts > 250.0).astype(float)
        - 0.025 * is_tokenized
        - 0.028 * is_3ds_authenticated
    )
    legacy_rule_suspicion = np.clip(legacy_rule_suspicion, 0.001, 0.88)

    # 3) Technical / Switch Error ('91'): Heavily concentrated in LegacySwitch_Bank
    tech_err_prob = acq_tech_err[acq_idx] * np.where(channels == "E-Commerce", 1.25, 0.85)

    # 4) Invalid CVV ('82') & 3DS Auth Required ('65')
    cvv_err_prob = np.where(
        (channels == "E-Commerce") & (is_tokenized == 0),
        acq_cvv_err[acq_idx] * 1.8,
        0.001,
    )
    sca_req_prob = np.where(
        (channels == "E-Commerce") & (is_3ds_authenticated == 0) & (amounts > 100.0),
        np.where(acquirers == "LegacySwitch_Bank", 0.045, 0.015),
        0.0005,
    )

    # 5) Expired Card ('54')
    expired_prob = np.where(
        is_tokenized == 1,
        0.0008,
        np.where(channels == "Recurring / MIT", 0.035, np.where(channels == "E-Commerce", 0.013, 0.002)),
    )

    u_dec = rng.random(n_auth_rows)
    response_code = np.full(n_auth_rows, "00", dtype=object)
    decline_reason = np.full(n_auth_rows, "Approved", dtype=object)
    decline_type = np.full(n_auth_rows, "Approved", dtype=object)

    p1 = tech_err_prob
    p2 = p1 + legacy_rule_suspicion
    p3 = p2 + nsf_prob
    p4 = p3 + cvv_err_prob
    p5 = p4 + expired_prob
    p6 = p5 + sca_req_prob

    m_tech = u_dec < p1
    m_risk = (~m_tech) & (u_dec < p2)
    m_nsf = (~m_tech) & (~m_risk) & (u_dec < p3)
    m_cvv = (~m_tech) & (~m_risk) & (~m_nsf) & (u_dec < p4)
    m_exp = (~m_tech) & (~m_risk) & (~m_nsf) & (~m_cvv) & (u_dec < p5)
    m_sca = (~m_tech) & (~m_risk) & (~m_nsf) & (~m_cvv) & (~m_exp) & (u_dec < p6)

    response_code[m_tech] = "91"
    decline_reason[m_tech] = "Technical / Switch Error"
    decline_type[m_tech] = "Soft Decline (Recoverable)"

    m_risk_59 = m_risk & (rng.random(n_auth_rows) < 0.65)
    m_risk_05 = m_risk & (~m_risk_59)
    response_code[m_risk_59] = "59"
    decline_reason[m_risk_59] = "Suspected Fraud"
    decline_type[m_risk_59] = np.where(
        fraud_flag[m_risk_59] == 1, "Hard Decline", "Soft Decline (Recoverable)"
    )

    response_code[m_risk_05] = "05"
    decline_reason[m_risk_05] = "Do Not Honor"
    decline_type[m_risk_05] = np.where(
        fraud_flag[m_risk_05] == 1, "Hard Decline", "Soft Decline (Recoverable)"
    )

    response_code[m_nsf] = "51"
    decline_reason[m_nsf] = "Insufficient Funds"
    decline_type[m_nsf] = "Soft Decline (Recoverable)"

    response_code[m_cvv] = "82"
    decline_reason[m_cvv] = "Invalid CVV"
    decline_type[m_cvv] = "Hard Decline"

    response_code[m_exp] = "54"
    decline_reason[m_exp] = "Expired Card"
    decline_type[m_exp] = "Soft Decline (Recoverable)"

    response_code[m_sca] = "65"
    decline_reason[m_sca] = "3DS Authentication Required"
    decline_type[m_sca] = "Soft Decline (Recoverable)"

    is_approved = (response_code == "00").astype(int)

    # -------------------------------------------------------------------------
    # 9. CHARGEBACKS (Reason & Lifecycle Stage on Approved Transactions)
    # -------------------------------------------------------------------------
    has_chargeback = np.zeros(n_auth_rows, dtype=int)
    chargeback_reason = np.full(n_auth_rows, "None", dtype=object)
    chargeback_stage = np.full(n_auth_rows, "None", dtype=object)

    approved_fraud_mask = (is_approved == 1) & (fraud_flag == 1)
    cb_from_fraud = approved_fraud_mask & (rng.random(n_auth_rows) < 0.88)
    has_chargeback[cb_from_fraud] = 1
    is_cnp_cb = cb_from_fraud & is_cnp
    is_cp_cb = cb_from_fraud & (~is_cnp)
    chargeback_reason[is_cnp_cb] = "Fraud - Card Not Present (10.4)"
    chargeback_reason[is_cp_cb] = "Fraud - Counterfeit / Lost Card (10.1)"

    non_fraud_cb_prob = np.where(
        (is_approved == 1) & (fraud_flag == 0),
        np.where(channels == "Recurring / MIT", 0.0065, np.where(channels == "E-Commerce", 0.0040, 0.0012)),
        0.0,
    )
    cb_dispute = (rng.random(n_auth_rows) < non_fraud_cb_prob)
    has_chargeback[cb_dispute] = 1
    dispute_reasons = rng.choice(
        [
            "Merchandise Not Received (13.1)",
            "Cancelled Recurring Billing (13.2)",
            "Duplicate Processing / Error (12.6)",
        ],
        size=cb_dispute.sum(),
        p=[0.52, 0.30, 0.18],
    )
    chargeback_reason[cb_dispute] = dispute_reasons

    cb_mask = has_chargeback == 1
    days_since_tx = (pd.Timestamp("2026-11-01") - tx_dates).days.to_numpy()
    cb_days = days_since_tx[cb_mask]
    cb_stages_assigned = np.empty(cb_mask.sum(), dtype=object)
    for i, age in enumerate(cb_days):
        if age < 20:
            probs = [0.62, 0.25, 0.08, 0.02, 0.03]
        elif age < 50:
            probs = [0.28, 0.36, 0.16, 0.08, 0.12]
        else:
            probs = [0.08, 0.18, 0.16, 0.22, 0.36]
        cb_stages_assigned[i] = rng.choice(
            [
                "1. First Chargeback",
                "2. Representment",
                "3. Pre-Arbitration",
                "4. Resolved - Issuer Won",
                "4. Write-Off - Issuer Lost",
            ],
            p=probs,
        )
    chargeback_stage[cb_mask] = cb_stages_assigned

    # -------------------------------------------------------------------------
    # 10. ASSEMBLE AUTHORIZATION TRANSACTIONS DATAFRAME
    # -------------------------------------------------------------------------
    df_auth = pd.DataFrame({
        "transaction_id": [f"TXN-{i:07d}" for i in range(1, n_auth_rows + 1)],
        "timestamp": timestamps,
        "date": tx_dates.floor("D"),
        "month": tx_dates.strftime("%Y-%m"),
        "day_of_month": tx_dom.astype(np.int8),
        "day_of_week": tx_dow.astype(np.int8),
        "hour": chosen_hours.astype(np.int8),
        "is_late_night": is_late_night.astype(np.int8),
        "is_pre_payday": is_pre_payday.astype(np.int8),
        "tokenised_card_id": tokenised_card_id,
        "merchant": merchants,
        "mcc": mcc_codes,
        "mcc_category": mcc_categories,
        "mcc_risk_tier": mcc_risk_tiers,
        "acquirer": acquirers,
        "channel": channels,
        "amount": amounts,
        "ticket_size_bucket": ticket_buckets,
        "three_ds_indicator": three_ds_indicator,
        "is_3ds_authenticated": is_3ds_authenticated.astype(np.int8),
        "tokenization_indicator": tokenization_indicator,
        "is_tokenized": is_tokenized.astype(np.int8),
        "card_tx_count_1h": tx_count_1h.astype(np.int16),
        "card_tx_count_24h": tx_count_24h.astype(np.int16),
        "amount_to_card_avg_ratio": amount_to_card_avg.astype(np.float32),
        "is_approved": is_approved.astype(np.int8),
        "response_code": response_code,
        "decline_reason": decline_reason,
        "decline_type": decline_type,
        "fraud_flag": fraud_flag.astype(np.int8),
        "has_chargeback": has_chargeback.astype(np.int8),
        "chargeback_reason": chargeback_reason,
        "chargeback_lifecycle_stage": chargeback_stage,
    })

    for col in [
        "month", "mcc", "mcc_category", "mcc_risk_tier", "acquirer", "channel",
        "ticket_size_bucket", "three_ds_indicator", "tokenization_indicator",
        "response_code", "decline_reason", "decline_type",
        "chargeback_reason", "chargeback_lifecycle_stage",
    ]:
        df_auth[col] = df_auth[col].astype("string")

    # -------------------------------------------------------------------------
    # 11. GENERATE SEPARATE 3DS AUTHENTICATION DATASET (134,191 rows)
    # -------------------------------------------------------------------------
    # Directly link 1-to-1 to all CNP transactions (E-Commerce and Recurring / MIT)
    cnp_pool = df_auth[df_auth["channel"].isin(["E-Commerce", "Recurring / MIT"])].copy().reset_index(drop=True)
    if n_3ds_rows is not None and n_3ds_rows != len(cnp_pool):
        cnp_pool = cnp_pool.sample(
            n=n_3ds_rows, replace=(n_3ds_rows > len(cnp_pool)), random_state=seed
        ).sort_values("timestamp").reset_index(drop=True)
    n_3ds_rows = len(cnp_pool)

    amt_3ds = cnp_pool["amount"].to_numpy()
    risk_3ds = cnp_pool["mcc_risk_tier"].to_numpy()
    acq_3ds = cnp_pool["acquirer"].to_numpy()
    tok_3ds = cnp_pool["is_tokenized"].to_numpy()
    night_3ds = cnp_pool["is_late_night"].to_numpy()
    tds_ind = cnp_pool["three_ds_indicator"].to_numpy()

    # Exact 1-to-1 alignment with three_ds_indicator in df_auth:
    # 3DS Frictionless (40,520) + 3DS Challenge Authenticated (46,299) = 86,819 Success
    # No 3DS / Unauthenticated (47,372) = 47,372 Failure
    is_success = np.isin(tds_ind, ["3DS Frictionless", "3DS Challenge Authenticated"])
    auth_outcome = np.where(is_success, "Success", "Failure")

    mandates = np.empty(n_3ds_rows, dtype=object)
    u_m = rng.random(n_3ds_rows)

    is_low_val = amt_3ds < 35.0
    is_high_risk = (risk_3ds == "High") | (amt_3ds > 200.0) | (night_3ds == 1)

    for i in range(n_3ds_rows):
        if tds_ind[i] == "3DS Frictionless":
            if is_low_val[i] and u_m[i] < 0.65:
                mandates[i] = "TRA / Low-Value Exemption"
            else:
                mandates[i] = "Frictionless (Low Risk)"
        elif tds_ind[i] == "3DS Challenge Authenticated":
            if is_high_risk[i]:
                mandates[i] = "Mandatory SCA Mandate" if u_m[i] < 0.65 else "Issuer Risk-Based Challenge"
            else:
                mandates[i] = "Mandatory SCA Mandate" if u_m[i] < 0.50 else "Issuer Risk-Based Challenge"
        else:  # No 3DS / Unauthenticated
            if is_low_val[i] and u_m[i] < 0.08:
                mandates[i] = "TRA / Low-Value Exemption"
            elif is_high_risk[i]:
                mandates[i] = "Issuer Risk-Based Challenge" if u_m[i] < 0.55 else "Mandatory SCA Mandate"
            else:
                if u_m[i] < 0.58:
                    mandates[i] = "Mandatory SCA Mandate"
                elif u_m[i] < 0.94:
                    mandates[i] = "Issuer Risk-Based Challenge"
                else:
                    mandates[i] = "Frictionless (Low Risk)"

    failure_reason = np.full(n_3ds_rows, "Authenticated Successfully", dtype=object)
    fail_indices = np.where(~is_success)[0]
    for idx in fail_indices:
        m_type = mandates[idx]
        acq_name = acq_3ds[idx]
        if acq_name == "LegacySwitch_Bank" and rng.random() < 0.45:
            failure_reason[idx] = "ACS / Switch Technical Error"
        elif m_type in ("Frictionless (Low Risk)", "TRA / Low-Value Exemption"):
            failure_reason[idx] = rng.choice(
                ["ACS / Switch Technical Error", "Device Fingerprint Mismatch"],
                p=[0.65, 0.35],
            )
        else:
            failure_reason[idx] = rng.choice(
                [
                    "Cardholder Abandoned Challenge",
                    "OTP Timeout / Delayed SMS",
                    "Wrong OTP / Biometric Failed",
                    "ACS / Switch Technical Error",
                ],
                p=[0.42, 0.28, 0.20, 0.10],
            )

    base_lat = np.where(
        mandates == "Frictionless (Low Risk)", 220,
        np.where(mandates == "TRA / Low-Value Exemption", 180, 1550)
    )
    acq_lat_add = np.where(acq_3ds == "LegacySwitch_Bank", 580, np.where(acq_3ds == "RegionalPay_ME", 200, 0))
    latency_ms = np.clip(
        rng.normal(loc=base_lat + acq_lat_add, scale=140, size=n_3ds_rows),
        85,
        8500,
    ).astype(int)

    df_3ds = pd.DataFrame({
        "auth_id": [f"AUTH-{i:07d}" for i in range(1, n_3ds_rows + 1)],
        "transaction_id": cnp_pool["transaction_id"].to_numpy(),
        "timestamp": cnp_pool["timestamp"].to_numpy(),
        "date": cnp_pool["date"].to_numpy(),
        "month": cnp_pool["month"].astype("string"),
        "hour": cnp_pool["hour"].to_numpy(),
        "tokenised_card_id": cnp_pool["tokenised_card_id"].to_numpy(),
        "merchant": cnp_pool["merchant"].to_numpy(),
        "mcc": cnp_pool["mcc"].astype("string"),
        "mcc_category": cnp_pool["mcc_category"].astype("string"),
        "mcc_risk_tier": cnp_pool["mcc_risk_tier"].astype("string"),
        "acquirer": cnp_pool["acquirer"].astype("string"),
        "channel": cnp_pool["channel"].astype("string"),
        "amount": amt_3ds,
        "ticket_size_bucket": cnp_pool["ticket_size_bucket"].astype("string"),
        "tokenization_indicator": cnp_pool["tokenization_indicator"].astype("string"),
        "is_tokenized": tok_3ds.astype(np.int8),
        "three_ds_indicator": cnp_pool["three_ds_indicator"].astype("string"),
        "challenge_mandate": pd.Series(mandates, dtype="string"),
        "auth_outcome": pd.Series(auth_outcome, dtype="string"),
        "is_auth_success": is_success.astype(np.int8),
        "failure_reason": pd.Series(failure_reason, dtype="string"),
        "latency_ms": latency_ms,
    })

    df_auth.to_parquet(AUTH_TX_PATH, index=False)
    df_3ds.to_parquet(AUTH_3DS_PATH, index=False)

    return df_auth, df_3ds


if __name__ == "__main__":
    print("Generating synthetic Issuer Authorization & 3DS Authentication datasets...")
    auth_df, threeds_df = generate_datasets(force_regenerate=True)
    print(f"Authorization dataset shape: {auth_df.shape}")
    print(f"  Overall Approval Rate: {auth_df['is_approved'].mean():.2%}")
    print(f"  Overall True Fraud Rate: {auth_df['fraud_flag'].mean():.2%}")
    print(f"  Chargeback Rate: {auth_df['has_chargeback'].mean():.2%}")
