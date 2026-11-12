# 🎤 20-Minute Interview Walkthrough Script & Q&A Cheat Sheet

Use this structured timeline during your **20-minute live presentation** so you hit every single grading criterion on the rubric.

---

## ⏱️ Minute 0:00 – 2:30 | Business Framing & End-to-End Architecture
* **What to say:**
  > *"Every card-issuing bank faces a core tension: if authorization rules are too strict, we falsely decline good customers—losing 1.75% interchange revenue and top-of-wallet loyalty. If rules are too loose, fraud and chargebacks spike. I built an end-to-end Issuer Authorization & Risk Intelligence Platform covering four pillars: (1) a 220,000-row synthetic authorization dataset + 134,191-row 3DS authentication dataset across 4 months, (2) an out-of-time XGBoost fraud probability model with SHAP explainability and a financial threshold simulator, (3) a 6-view interactive executive dashboard, and (4) a grounded AI Insights Engine."*

---

## ⏱️ Minute 2:30 – 6:00 | Synthetic Data Realism & Domain Patterns (Tab 1 & Tab 2)
* **Click on Tab 1 (`📊 1. Overview`):**
  * Point out the **220,000 transactions ($33.4M spend)** and **91.0% overall approval rate**.
  * Hover over the **Daily Volume & Approval Rate chart** and the **Payday Seasonality chart**:
    * Show how **Post-Payday (Days 25–5)** lifts spend volume by **+18%** and cuts `Insufficient Funds` declines by **45%** compared to the **Pre-Payday Cash Crunch (Days 15–24)**.
    * Point out the **October 15–25 Cyber Retail Week surge** (+35% E-Commerce spike).
* **Click on Tab 2 (`✅ 2. Authorization`):**
  * Point to the **Approval Rate by Acquirer** bar chart:
    * Highlight **`LegacySwitch_Bank`** in red (**80.8% approval** vs. **92.8%** on `AdyenSwitch`/`StripeAcq_EU`) due to its **5.2% Technical/Switch Error (`91`)** rate.
  * Point to the **3DS × Tokenization chart**:
    * Explain how combining **Network Tokenization + 3DS** lifts E-Commerce approval rate by **+12 percentage points** while cutting fraud by **>75%**.

---

## ⏱️ Minute 6:00 – 10:00 | Declines, Fraud, Chargebacks & 3DS Funnel (Tabs 3, 4 & 5)
* **Click on Tab 3 (`🚫 3. Declines`):**
  * Show how we separate **Soft/Recoverable Declines** (`Insufficient Funds`, `False-Positive Fraud Blocks`, `Technical Errors`, `Expired Card`) from **Hard Declines** (`True Fraud`, `Invalid CVV`).
  * Point to the **MCC × Ticket Size Heatmap**: high-risk MCCs (`Crypto 6051`, `Digital Gaming 5816`, `Luxury 5944`) and `$500+` tickets experience the highest decline friction.
* **Click on Tab 4 (`🛡️ 4. Fraud & Chargebacks`):**
  * Point to the **Hour of Day chart**: highlight the red **01:00–05:00 Late-Night Attack Window** where fraud density is **3.6x higher** than daytime.
  * Point to the **Chargeback Reason Codes** (`10.4 CNP Fraud`, `13.1 Merchandise Not Received`, `13.2 Cancelled Recurring`) and **Lifecycle Funnel** (*First Chargeback $\rightarrow$ Representment $\rightarrow$ Pre-Arbitration $\rightarrow$ Write-Off*).
* **Click on Tab 5 (`🔐 5. Authentication (3DS)`):**
  * Explain the key domain insight: *"If a customer fails or abandons 3DS, the transaction dies before ever reaching Authorization! Here across 134,191 3DS attempts (all CNP transactions), we see a 64.7% pass rate (86,819 authentications), with drop-offs driven by `Cardholder Abandoned Challenge`, `OTP Timeout`, and `LegacySwitch_Bank` ACS technical errors."*

---

## ⏱️ Minute 10:00 – 15:30 | Machine Learning Model & Threshold Simulator (Tab 6 — 25% of Grade!)
* **Click on Tab 6 (`🤖 6. ML Model & What-If Simulator`):**
  1. **Why We Chose Fraud Prediction (`fraud_flag = 1`):**
     * Legacy static rules (`59`/`05`) falsely blocked **1,102 legitimate transactions** in the test window while still missing **31% of true fraud**. Predicting fraud probability fixes both sides simultaneously.
  2. **Time-Based Validation Split:**
     * Explain why random `train_test_split` is wrong in banking (future look-ahead leakage and card-burst memorization). Show our chronological split: **Train (Jul 1 – Sep 15)**, **Validation (Sep 16 – Oct 5)**, and **Out-of-Time Test (Oct 6 – Oct 31)**.
  3. **Baseline vs. Advanced Comparison:**
     * Point to the metrics table and PR Curve: **XGBoost achieves PR-AUC = `0.6161` and ROC-AUC = `0.9373`**, beating the **Logistic Regression Baseline (`PR-AUC = 0.5138`, `ROC-AUC = 0.9145`)** by **+10.23 pts PR-AUC** because tree splits capture non-linear combinations (*Unsecured E-Com + 1h Velocity $\ge 3$ + Spend Ratio $\ge 2\times$ Card Average*).
  4. **SHAP Explainability & What-If Threshold Simulator:**
     * Move the **Operating Threshold ($\tau$) slider** live on screen!
     * Explain our stated unit economics (**1.75%** interchange earned, **8% spend churn + \$15 friction cost** per false decline, **100% principal + \$15 chargeback fee** per missed fraud).
     * Show that at validation-optimal **$\tau^* = 0.12$**, XGBoost creates **\$573,910 in net value** on the 26-day test set (**+\$117,522 incremental gain** over legacy rules) and cuts **False Declines by 44.2%** (`1,102` $\rightarrow$ `615`).
     * Select a transaction in the **Local SHAP Waterfall Inspector** to show transaction-level explainability.

---

## ⏱️ Minute 15:30 – 19:00 | Grounded AI Insights Engine & Guardrails (Tab 7 — 20% of Grade!)
* **Click on Tab 7 (`🧠 7. AI Insights Engine`):**
  * Show the **6 structured executive sections** (*Storyline*, *Key Observations*, *Areas Needing Attention*, *Possible Drivers*, *Decline Mitigation Ideas*, *Recommended Follow-Ups*).
  * Change a global filter in the left sidebar (e.g., select preset **`LegacySwitch_Bank Deep-Dive`**) and click **Generate / Refresh AI Executive Insights** to show how the storyline updates dynamically for that exact filter slice.
  * Open the **"🔍 Inspect Exact Context & System Prompt Sent to the Model"** drawer:
    * Explain: *"We never let the LLM do mental math over raw rows. Python deterministically computes this exact JSON `<DATA_CONTEXT>` and instructs the model at `temperature=0.1` to cite only these numbers."*
  * Click the **"🧪 Test Guardrail: Ask about Cardholder FICO & Age"** button:
    * Show how the engine immediately blocks unsupported claims outside the schema!

---

## ⏱️ Minute 19:00 – 20:00 | Honest Limitations & Next Steps
* Conclude with 3 production next steps:
  1. **Delayed-Feedback / Survival Modeling** for the 30–90 day real-world chargeback maturation lag.
  2. **Sub-15ms Streaming Feature Store (Redis/Feast)** for live ISO 8583 velocity features.
  3. **3-Tier Dynamic Routing Policy** (Frictionless Approve vs. Step-Up 3DS Biometric Challenge vs. Auto-Block) deployed via a **90/10 Champion-Challenger shadow test**.

