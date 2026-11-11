"""Generates the formatted 2-page Word write-up (`Case_Study_Writeup_2Pages.docx`)."""

from pathlib import Path
import docx
from docx.shared import Pt, Inches, RGBColor

ROOT = Path(__file__).resolve().parent.parent


def build_docx() -> None:
    doc = docx.Document()
    for s in doc.sections:
        s.top_margin = Inches(0.5)
        s.bottom_margin = Inches(0.5)
        s.left_margin = Inches(0.6)
        s.right_margin = Inches(0.6)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(9.5)
    style.paragraph_format.space_after = Pt(3)
    style.paragraph_format.line_spacing = 1.08

    title = doc.add_paragraph()
    r = title.add_run("Issuer Authorization Analytics & AI Insights — Technical Write-Up (2 Pages)")
    r.bold = True
    r.font.size = Pt(13.5)
    r.font.color.rgb = RGBColor(15, 23, 42)

    sub = doc.add_paragraph(
        "AI Data Scientist Technical Case Study | Synthetic Data, Out-of-Time ML, Interactive Dashboard & Grounded AI Engine"
    )
    sub.runs[0].italic = True
    sub.runs[0].font.size = Pt(9)

    def add_sec(heading_text, bullets):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(heading_text)
        run.bold = True
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(30, 58, 138)
        for b_title, b_body in bullets:
            bp = doc.add_paragraph(style="List Bullet")
            bp.paragraph_format.space_after = Pt(1.5)
            r1 = bp.add_run(b_title + ": ")
            r1.bold = True
            bp.add_run(b_body)

    add_sec("1. Synthetic Data Design, Built-In Patterns & Assumptions (src/data_generator.py)", [
        ("Scale & Schema", "Deterministically (seed=42) generates 220,000 Issuer Authorization transactions ($33.4M gross spend across 18,000 repeat tokenised cards, 40 merchants, 10 MCCs, and 5 acquirers) plus a separate 110,000-row 3DS Authentication dataset spanning 4 full months (Jul 1 to Oct 31, 2026)."),
        ("Channel & Security Differentials", "POS Contactless (95.3% approval) and POS Chip & PIN (94.8% approval) exhibit very low fraud (~0.14%). E-Commerce has lower approval (87.7%) and higher fraud (3.65%). Within E-Commerce, Network/Wallet Tokenization and 3DS Authentication boost approval by +9 to +14 percentage points and slash fraud by >75%. Tokenized Recurring/MIT bills also eliminate Code 54 Expired Card declines via auto-updated credentials."),
        ("MCC Risk Tiers & Ticket Size", "High-risk MCCs (6051 Crypto, 5816 Digital Gaming, 5944 Luxury, 5732 Electronics) and large tickets ($500+ or >2.2x cardholder average) drive Code 59 Suspected Fraud, Code 05 Do Not Honor, and Code 51 Insufficient Funds declines."),
        ("Acquirer Routing Anomaly", "While modern processors (AdyenSwitch, StripeAcq_EU, GlobalPay_Net) achieve 92.6%-93.0% approval, LegacySwitch_Bank is modeled with a gateway defect causing a 5.2% Technical/Switch Error (Code 91) rate, elevated Invalid CVV (Code 82) declines, +620ms 3DS latency, and an 80.8% approval rate."),
        ("Seasonality, Diurnal & Velocity Cycles", "Includes a monthly Payday Cycle (Days 25-5 post-payday liquidity lifts volume +18% and cuts Insufficient Funds declines by 45% vs. Days 15-24 pre-payday crunch), late-night fraud concentration (01:00-05:00 has 3.6x higher fraud density), an Oct 15-25 Cyber Retail Week (+35% E-Com surge), and 1,050 compromised-card rapid velocity bursts (3-6 swipes in <60 mins)."),
    ])

    add_sec("2. Machine Learning Approach, Validation & Threshold Business Impact (src/ml_pipeline.py)", [
        ("Problem Chosen & Justification", "We predict Transaction Fraud Probability (fraud_flag = 1) at authorization time. In our portfolio, legacy static rules (Codes 59/05) falsely decline 1,102 legitimate transactions on the test window while still missing ~31% of true fraud (which becomes post-auth 10.4/10.1 chargebacks). Predicting fraud probability accurately enables the issuer to simultaneously recover false declines and block true fraud."),
        ("Strict Time-Based Validation Split", "Chronological split prevents future data leakage and card-burst memorization: Train (Jul 1 - Sep 15, 77 days, 131,836 rows), Validation (Sep 16 - Oct 5, 20 days, 34,475 rows for threshold tuning), and Out-of-Time Test (Oct 6 - Oct 31, 26 days, 53,689 rows including the October Cyber Sale surge). Historical target encodings (MCC fraud rate, Acquirer decline rate) are fitted strictly on Train."),
        ("Baseline vs. Advanced Model Results (Out-of-Time Test)", "Simple Baseline (Logistic Regression on static attributes) achieves ROC-AUC = 0.9145, PR-AUC = 0.5138, Brier = 0.01027. Advanced Model (XGBoost with 1h/24h velocity, cardholder spend-anomaly ratios, and historical encodings) achieves ROC-AUC = 0.9373, PR-AUC = 0.6161 (+10.23 pts PR-AUC lift), and Brier = 0.00834 (-19% calibration error)."),
        ("Explainability (SHAP Drivers)", "TreeExplainer global SHAP ranks the primary risk drivers as: (1) Unsecured E-Com (No 3DS & No Token, |SHAP|=0.610), (2) Spend Ratio vs. Cardholder Avg (0.479), (3) MCC Historical Fraud Rate (0.429), (4) 3DS Authenticated Indicator (0.300), (5) Network Tokenized (0.288), and (6) Transaction Amount & 1h Velocity."),
        ("Operating Threshold Economics (tau* = 0.12)", "Under explicit issuer unit economics (1.75% interchange earned on approved good spend; 1.75% lost interchange + 8.0% top-of-wallet spend churn + $15.00 friction penalty per false decline; 100% principal loss + $15.00 chargeback fee per missed fraud), the validation-optimal threshold is tau* = 0.12. On the 26-day Out-of-Time Test Set, XGBoost delivers $573,910 in net savings (+$117,522 incremental value over legacy static rules, or ~$1.65M annualized) while reducing False Declines by 44.2% (from 1,102 down to 615)."),
    ])

    add_sec("3. AI Insights Engine Architecture, Security & Grounding (src/ai_engine.py)", [
        ("Deterministic Context Pre-Aggregation", "To prevent numerical hallucinations, the LLM never computes math over raw rows. Python deterministically computes a compact <DATA_CONTEXT> JSON summarizing the active view and global filter selections (headline KPIs vs. benchmark, channel/acquirer/MCC rankings, hard vs. soft decline Pareto, 3DS failure breakdown, seasonality signals, and ML metrics)."),
        ("Transparency & Interactive Follow-Up Q&A", "The dashboard exposes an expandable 'Inspect Exact Context & System Prompt Sent to Model' drawer and a free-text follow-up question box with quick-ask chips."),
        ("Hallucination Guardrails & API Key Security", "Low temperature (0.1) + strict system prompt limits claims to <DATA_CONTEXT>. Out-of-schema queries (e.g., cardholder FICO score, salary, age, or competitor rates) trigger an explicit Grounding Guardrail refusal. API keys (OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY) are loaded strictly server-side via .env / st.secrets (git-ignored), with an automatic Deterministic Grounded Analytics Engine fallback so the app runs 100% reliably offline or behind corporate firewalls."),
    ])

    add_sec("4. Honest Limitations, Production Roadmap & AI Disclosure", [
        ("Limitations", "(1) Synthetic data cannot capture every noisy cross-border FX quirk or adversarial drift of live fraud rings; (2) Real-world chargeback labels take 30-90 days to mature (label delay), whereas offline training assumes labels are known at split boundaries; (3) Cold-start cards lack historical spend ratios."),
        ("What We Would Do Next in Production", "(1) Implement delayed-feedback / survival learning to handle 60-day chargeback maturation lag; (2) Deploy sub-15ms streaming velocity counters in Redis/Feast; (3) Replace binary blocking with a 3-tier policy routing Medium Risk (0.08 <= p < 0.12) to dynamic 3DS Challenge and scheduling pre-payday Insufficient Funds retries for Days 25-5; (4) Run a 90/10 Champion-Challenger shadow test."),
        ("AI Coding Assistant Disclosure", "Per Section 4 Ground Rules, an AI coding assistant was used to accelerate modular Python scaffolding, Streamlit/Altair chart layout, and report formatting. All payment domain design, leakage-free temporal validation, unit-economics formulation, and metric verification were directed and validated by the candidate."),
    ])

    out_path = ROOT / "Case_Study_Writeup_2Pages.docx"
    doc.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    build_docx()

