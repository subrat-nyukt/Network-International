# Issuer Authorization Analytics & AI Insights — Technical Write-Up

**Candidate Technical Report | AI Data Scientist Case Study**

---

## 1. Synthetic Data Design, Built-In Patterns & Assumptions

To simulate a realistic card-issuing bank portfolio without using real cardholder data, `src/data_generator.py` deterministically (`seed=42`) generates two linked datasets across **4 full months (`2026-07-01` to `2026-10-31`, 123 days)**:
1. **Issuer Authorization Dataset (`220,000` transactions, `18,000` repeat tokenized cardholders, `$33.4M` gross volume):** Includes `timestamp`, `tokenised_card_id`, `merchant` (40 merchants across 10 MCCs), `mcc`, `mcc_risk_tier`, `acquirer` (5 processors), `channel`, `amount`, `three_ds_indicator`, `tokenization_indicator`, ISO 8583 `response_code`, `decline_reason`, `decline_type` (Hard vs. Soft/Recoverable), ground-truth `fraud_flag`, `chargeback_reason`, and `chargeback_lifecycle_stage`.
2. **Pre-Authorization 3DS Authentication Dataset (`110,000` attempts):** Tracks card-not-present authentication flows across `challenge_mandate` (*Frictionless*, *Mandatory SCA*, *Issuer Risk-Based Challenge*, *TRA/Low-Value Exemption*), `auth_outcome` (*Success* vs. *Failure*), `failure_reason` (*Cardholder Abandoned*, *OTP Timeout*, *Wrong OTP/Biometric*, *ACS Technical Error*), and `latency_ms`.

### Documented Domain Patterns & Generation Logic
* **Channel & Security Differentials:** In-store Card-Present channels (`POS Contactless` **95.3%**, `POS Chip & PIN` **94.8%** approval) enjoy high trust and minimal fraud (**~0.14%**). Conversely, `E-Commerce` exhibits lower approval (**87.7%**) and higher fraud (**3.65%**). Within E-Commerce, **Network/Wallet Tokenization** and **3DS Authentication** increase approval rates by **+9 to +14 percentage points** and reduce fraud rates by **>75%** compared to untokenized, unauthenticated raw PAN transactions. Additionally, tokenized recurring transactions eliminate `54 Expired Card` declines via automated credential updating.
* **MCC Risk Tiers & Ticket Size Interactions:** High-risk MCCs (`6051 Crypto`, `5816 Digital Gaming`, `5944 Luxury`, `5732 Electronics`) and large ticket brackets (`$500+` or spend >2.2x a cardholder's historical average) drive elevated `59 Suspected Fraud`, `05 Do Not Honor`, and `51 Insufficient Funds` declines.
* **Acquirer Routing & Switch Anomalies:** While modern acquirers (`AdyenSwitch`, `StripeAcq_EU`, `GlobalPay_Net`) achieve **92.6%–93.0%** approval rates, `LegacySwitch_Bank` is modeled with an outdated gateway causing a **5.2% `91 Technical/Switch Error` rate**, elevated `82 Invalid CVV` errors, lower 3DS/token pass-through, and a **620 ms** 3DS latency penalty—depressing its approval rate to **80.8%**.
* **Seasonality, Diurnal & Velocity Cycles:**
  * *Monthly Payday Cycle:* Days **25–5** (post-payday liquidity) show **+18%** volume lift and **45% fewer** `Insufficient Funds` declines compared to the pre-payday cash crunch (**Days 15–24**).
  * *Diurnal Late-Night Concentration:* Overnight hours (**01:00–05:00**) exhibit **3.6x higher fraud density** than daytime hours.
  * *Promotional & Velocity Bursts:* An October 15–25 *Cyber Retail Week* spikes E-Commerce volume by **+35%**, while **1,050 compromised-card velocity sessions** simulate rapid multi-swipe attacks (3–6 swipes within 15–60 minutes, including `$1–$5` card-testing probes).

---

## 2. Machine Learning Approach, Validation & Business Impact

### Problem Selection & Justification
We chose to predict **Transaction Fraud Probability (`fraud_flag = 1`) at the moment of authorization**. From an issuer perspective, fraud rules are the primary controllable lever governing the trade-off between **Approval Rate** and **Fraud Losses**. In our baseline portfolio, legacy static rules (`59 Suspected Fraud` and `05 Do Not Honor`) falsely block thousands of legitimate E-Commerce and high-ticket transactions (**Soft/Recoverable False Declines**) while still missing **~31%** of true fraud (which slips through into costly `10.4` / `10.1` chargebacks). Replacing rigid rules with a calibrated ML probability score enables the bank to simultaneously **auto-approve more good customers** and **block true fraud**.

### Time-Based Validation Design & Leakage Prevention
To prevent future look-ahead leakage and card-memorization bias, transactions are split strictly chronologically:
* **Training Set (`2026-07-01` to `2026-09-15`, 77 days):** `131,836` rows (used for model fitting and out-of-fold historical target encodings such as `mcc_train_fraud_rate` and `acquirer_train_decline_rate`).
* **Validation Set (`2026-09-16` to `2026-10-05`, 20 days):** `34,475` rows (used for hyperparameter tuning and selecting the optimal operating threshold $\tau^*$).
* **Out-of-Time Test Set (`2026-10-06` to `2026-10-31`, 26 days):** `53,689` rows (includes the October Cyber Sale surge; held out completely for final evaluation).

### Baseline vs. Advanced Model Comparison (Out-of-Time Test Set)
We compared a **Simple Baseline (`LogisticRegression` with `StandardScaler` + `OneHotEncoder` on static transaction attributes)** against an **Advanced Model (`XGBClassifier` gradient boosted trees with rolling 1h/24h card velocity, cardholder spend-anomaly ratios, and historical risk encodings)**:

| Model / Benchmark | ROC-AUC | PR-AUC (Avg Precision) | Brier Score (Calibration ↓) | Fraud Recall @ $\tau^*$ | False Declines (OOT Test) | Net ML Value Saved (26-Day Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Static Rules (`59`/`05`)** | — | — | — | 69.2% | 1,102 | \$456,388 |
| **Logistic Regression (Baseline)** | 0.9145 | 0.5138 | 0.01027 | 65.4% | 724 | \$550,048 |
| **XGBoost (Advanced Model)** | **0.9373** | **0.6161** | **0.00834** | **69.5%** | **615** | **\$573,910** |

* **Why XGBoost Wins (+10.23 pts PR-AUC lift):** Because fraud is rare (~1.8% class balance), **PR-AUC** is our primary ranking metric. XGBoost captures non-linear multi-way interactions (e.g., *Unsecured E-Com × High-Risk MCC × 1h Velocity $\ge 3$ × Spend Ratio $\ge 2\times$ Card Average*) that linear Logistic Regression cannot separate.

### Explainability (SHAP) & Operating Threshold Business Economics
Using `shap.TreeExplainer`, the top global drivers of predicted fraud risk are:
1. `Unsecured E-Com (No 3DS & No Token)` ($\text{mean } |\text{SHAP}| = 0.610$)
2. `Spend Ratio vs. Cardholder Historical Average` ($0.479$)
3. `MCC Historical Fraud Rate` ($0.429$)
4. `3DS Authenticated Indicator` ($0.300$) & `Network/Wallet Tokenized` ($0.288$)
5. `Transaction Amount (Log USD)` ($0.193$) & `Card Velocity (1h / 24h)` ($0.145$)

**Operating Threshold Optimization ($\tau^* = 0.12$):** We evaluated thresholds $\tau \in [0.06, 0.90]$ under explicit issuer unit economics:
* **Good Transaction Approved (TN):** Earns **1.75%** interchange revenue.
* **False Decline of Good Customer (FP):** Loses **1.75%** interchange + **8.0%** top-of-wallet spend churn + **\$15.00** customer service/abandonment friction cost.
* **Missed Fraud (FN) vs. Caught Fraud (TP):** loses vs. saves **100%** of transaction principal + **\$15.00** network chargeback processing fee.
At validation-optimal **$\tau^* = 0.12$**, XGBoost generates **\$573,910 in net savings** on the 26-day test set—delivering **+\$117,522 in incremental value** over legacy rules (**~\$1.65M annualized**) while **cutting False Declines by 44.2%** (`1,102` $\rightarrow$ `615`).

---

## 3. How the AI Insights Engine Works (Grounding & Security)

1. **Deterministic Context Serialization (`src/ai_engine.py`):** Rather than passing raw rows or asking an LLM to do arithmetic, whenever the user changes global filters or selects a view, Python deterministically computes a compact `<DATA_CONTEXT>` JSON payload containing exact filtered KPIs, benchmark deltas, channel/acquirer/MCC rankings, hard vs. soft decline Pareto, 3DS failure breakdown, seasonality signals, and ML metrics.
2. **Transparency & Interactive Follow-Up Q&A:** An expandable **"Inspect Exact Context & System Prompt"** drawer in the UI exposes the full `<DATA_CONTEXT>` JSON and system prompt. A free-text input box supports custom follow-up questions.
3. **Limiting Unsupported Claims (Hallucination Guardrails):**
   * *Pre-Computed Numbers Only:* The system prompt (`temperature=0.1`) restricts the model to citing exact values inside `<DATA_CONTEXT>`.
   * *Out-of-Schema Guardrail:* Queries referencing uncaptured attributes (e.g., FICO credit score, cardholder age, salary, or competitor pricing) are intercepted and refused with an explicit schema boundary explanation.
4. **API Key Security & Offline Resilience:** Keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`) are read strictly on the backend via `.env` / `st.secrets` (excluded via `.gitignore`). If no key is provided or corporate firewalls block outbound traffic, the engine automatically switches to its **Deterministic Grounded Analytics Engine**, ensuring 100% reliability during grading.

---

## 4. Honest Limitations & What We Would Do Next in Production

### Limitations
* **Synthetic Simplifications:** Although our generator models multi-way interactions, velocity bursts, and payday seasonality, real issuer data exhibits noisier merchant naming, cross-border FX nuances, and concept drift from evolving fraud rings.
* **Instant Fraud Label Availability:** In production, confirmed fraud and chargebacks arrive with a **30-to-90 day settlement delay** (label maturation lag), whereas our offline training assumes ground-truth labels are known at split boundaries.
* **Cold-Start Cards:** Newly issued cards lack historical spend baselines, requiring fallback to MCC/product-level priors.

### Production Roadmap (What Next)
1. **Delayed-Feedback / Survival Learning:** Account for 60-day chargeback maturation lag using importance-weighted Pu-learning or survival models combined with rapid 24-hour rule/alert labels.
2. **Real-Time Streaming Feature Store:** Deploy sub-15ms Redis/Feast streaming counters for 5-minute, 1-hour, and 24-hour card and merchant velocity at ISO 8583 authorization switch speed.
3. **Three-Tier Dynamic Action Policy:** Instead of binary block/approve, route Medium-Risk scores (`0.08 <= p < 0.12`) to **Step-Up 3DS Biometric Challenge** or **Smart Payday Retry**, and run a **Champion/Challenger A/B shadow test** on 10% of live traffic.

---

## 5. AI Coding Assistant Disclosure (Per Section 4 Ground Rules)
An AI coding assistant was used as a pair-programming accelerator to scaffold modular Python boilerplate, Streamlit/Altair chart layout code, and documentation formatting. All payment domain logic (ISO response codes, 3DS mandate mechanics, acquirer switch degradation, payday liquidity cycles), leakage-free time-based validation design, feature engineering, issuer unit-economics threshold formulation, and verification of every metric and insight were directed, reviewed, and validated by the candidate.

