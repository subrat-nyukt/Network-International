"""
generate_executive_deck.py
Generates a Tier-1 Consultant Executive Presentation (.pptx)
for the 20-minute walkthrough presentation to Network International recruiters.
"""

from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_DIR = Path("presentation")
OUTPUT_DIR.mkdir(exist_ok=True)
PPTX_PATH = OUTPUT_DIR / "Network_International_Executive_Presentation.pptx"

# Corporate Color Palette (Tier-1 Fintech / Network International Theme)
C_NAVY_DARK = RGBColor(15, 23, 42)      # #0F172A - Deep Slate
C_NAVY_CARD = RGBColor(30, 41, 59)      # #1E293B - Card Background for dark slides
C_BLUE_PRIMARY = RGBColor(37, 99, 235)  # #2563EB - Primary Brand Blue
C_BLUE_LIGHT = RGBColor(239, 246, 255)  # #EFF6FF - Light Ice Blue
C_GREEN = RGBColor(16, 185, 129)        # #10B981 - Success Green
C_AMBER = RGBColor(245, 158, 11)        # #F59E0B - Warning Amber
C_RED = RGBColor(220, 38, 38)           # #DC2626 - Alert Red
C_TEXT_MAIN = RGBColor(15, 23, 42)      # Dark text
C_TEXT_MUTED = RGBColor(100, 116, 139)  # Slate-500
C_TEXT_WHITE = RGBColor(255, 255, 255)
C_BG_PAGE = RGBColor(248, 250, 252)     # Slate-50
C_CARD_BG = RGBColor(255, 255, 255)
C_BORDER = RGBColor(226, 232, 240)      # Slate-200


def create_deck():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, tracker: str, title: str, subtitle: str):
        # Tracker category
        tb_track = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        p_track = tb_track.text_frame.paragraphs[0]
        p_track.text = tracker.upper()
        p_track.font.size = Pt(10)
        p_track.font.bold = True
        p_track.font.color.rgb = C_BLUE_PRIMARY

        # Action Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.55))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = C_TEXT_MAIN

        # Subtitle / Key Takeaway lead-in
        tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(11.7), Inches(0.35))
        p_sub = tb_sub.text_frame.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = C_TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1.2)
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Executive Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY_DARK
    bg1.line.fill.background()

    # Title box
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "NETWORK INTERNATIONAL"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = RGBColor(56, 189, 248) # Cyan

    p2 = tf.add_paragraph()
    p2.text = "Next-Generation Issuer Authorization &\nAI Risk Intelligence Platform"
    p2.font.size = Pt(34)
    p2.font.bold = True
    p2.font.color.rgb = C_TEXT_WHITE
    p2.space_before = Pt(12)

    p3 = tf.add_paragraph()
    p3.text = "Balancing Frictionless Conversion, Revenue Recovery & Real-Time Fraud Prevention"
    p3.font.size = Pt(15)
    p3.font.color.rgb = RGBColor(203, 213, 225)
    p3.space_before = Pt(16)

    # Footer Card on Slide 1
    add_card(s1, Inches(1.0), Inches(5.6), Inches(11.333), Inches(1.1), C_NAVY_CARD, RGBColor(51, 65, 85))
    tb_meta = s1.shapes.add_textbox(Inches(1.2), Inches(5.7), Inches(10.9), Inches(0.9))
    tf_meta = tb_meta.text_frame
    p_meta = tf_meta.paragraphs[0]
    p_meta.text = "Case Study & Solution Architecture Walkthrough  |  20-Minute Executive Presentation"
    p_meta.font.size = Pt(11)
    p_meta.font.bold = True
    p_meta.font.color.rgb = C_TEXT_WHITE

    p_meta2 = tf_meta.add_paragraph()
    p_meta2.text = "Candidate: Senior AI Data Scientist  •  Tech Stack: Python, Streamlit, XGBoost, SHAP, Altair, LLM AI Insights Engine"
    p_meta2.font.size = Pt(10)
    p_meta2.font.color.rgb = RGBColor(148, 163, 184)

    # =========================================================================
    # SLIDE 2: EXECUTIVE SUMMARY & THE CORE BANKING DILEMMA
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Summary", "The Issuer's Dilemma: False Declines vs. Missed Fraud Losses",
               "Legacy static rules create a costly double loss: rejecting good customers while still missing 31% of real fraud.")

    cards_s2 = [
        ("THE BUSINESS PROBLEM", C_RED, [
            "• Rigid static rules ('59 Suspected Fraud', '05 Do Not Honor') falsely decline 1,102 good shoppers in test window alone.",
            "• Misses ~31% of true fraud attacks, creating $700K+ in post-auth chargeback losses.",
            "• Results in $7.74M in lost transaction volume and permanent customer top-of-wallet churn."
        ]),
        ("THE AI SOLUTION", C_BLUE_PRIMARY, [
            "• Calibrated XGBoost ML model scoring fraud probability at authorization time.",
            "• Replaces binary Approve/Decline with 3-Tier Risk Routing (Frictionless, 3DS Step-Up, Hard Block).",
            "• Integrates Smart Payday Retries for recurring 'Insufficient Funds' bill declines (Days 25–5)."
        ]),
        ("MEASURABLE IMPACT", C_GREEN, [
            "• +$573,910 Net Savings in 26-day test slice (+$117.5K lift over legacy rules; ~$1.65M annualized).",
            "• 44.2% Reduction in False Declines (1,102 down to 615 false alarms).",
            "• +10% to +14% Approval Lift on E-Commerce via Network Tokenization & 3DS integration."
        ])
    ]

    for i, (title, color, bullets) in enumerate(cards_s2):
        left = Inches(0.8 + i * 3.95)
        add_card(s2, left, Inches(1.7), Inches(3.8), Inches(4.7))
        
        # Color top banner
        bar = s2.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.7), Inches(3.8), Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tb_c = s2.shapes.add_textbox(left + Inches(0.25), Inches(1.95), Inches(3.3), Inches(4.3))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True

        p_t = tf_c.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        for b in bullets:
            p_b = tf_c.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(11)
            p_b.font.color.rgb = C_TEXT_MAIN
            p_b.space_before = Pt(10)

    # Bottom Takeaway Banner
    add_card(s2, Inches(0.8), Inches(6.55), Inches(11.7), Inches(0.65), C_BLUE_LIGHT, C_BLUE_PRIMARY)
    tb_b = s2.shapes.add_textbox(Inches(1.0), Inches(6.6), Inches(11.3), Inches(0.55))
    p_b = tb_b.text_frame.paragraphs[0]
    p_b.text = "★ Key Executive Takeaway: Real-time ML probability scoring simultaneously maximizes interchange revenue and shields the bank from fraud."
    p_b.font.size = Pt(11)
    p_b.font.bold = True
    p_b.font.color.rgb = C_BLUE_PRIMARY

    # =========================================================================
    # SLIDE 3: PORTFOLIO SCALE & 1-TO-1 3DS DATA ARCHITECTURE
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Data Foundation & Architecture", "End-to-End Scale: 220,000 Authorizations Linked to 134,191 3DS Logs",
               "Captures the full 4-month lifecycle from pre-auth checkout authentication to clearing, settlement, and disputes.")

    kpis_s3 = [
        ("220,000", "Total Authorizations", "$41.5M spend across 123 days (Jul–Oct 2026)"),
        ("17,542", "Unique Cardholders", "Repeat multi-swipe card baselines for spend ratios"),
        ("134,191", "3DS Pre-Auth Attempts", "1-to-1 mapped with all Card-Not-Present volume"),
        ("64.70%", "3DS Pass Rate", "86,819 passed; 47,372 abandoned before auth"),
    ]
    for i, (val, title, sub) in enumerate(kpis_s3):
        left = Inches(0.8 + i * 2.95)
        add_card(s3, left, Inches(1.7), Inches(2.85), Inches(1.3))
        tb_k = s3.shapes.add_textbox(left + Inches(0.15), Inches(1.75), Inches(2.55), Inches(1.2))
        tf_k = tb_k.text_frame
        p_v = tf_k.paragraphs[0]
        p_v.text = val
        p_v.font.size = Pt(20)
        p_v.font.bold = True
        p_v.font.color.rgb = C_BLUE_PRIMARY
        
        p_t = tf_k.add_paragraph()
        p_t.text = title.upper()
        p_t.font.size = Pt(9)
        p_t.font.bold = True
        p_t.font.color.rgb = C_TEXT_MUTED
        
        p_s = tf_k.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(8.5)
        p_s.font.color.rgb = C_TEXT_MAIN

    # Architecture 3 Columns below
    arch_cols = [
        ("STAGE 1: PRE-AUTHORIZATION (3DS)", C_BLUE_PRIMARY, [
            "• 134,191 CNP E-Commerce & Recurring Attempts.",
            "• 4 Challenge Mandates: Frictionless, Mandatory SCA, Risk-Based, TRA Exemption.",
            "• Uncovers hidden cart abandonment (OTP timeout, drop-off) before orders reach clearing."
        ]),
        ("STAGE 2: AUTHORIZATION SWITCH (ISO 8583)", RGBColor(16, 185, 129), [
            "• 220,000 Total Transactions (incl. 85,809 in-store POS).",
            "• Real-time decision: 91.33% Approved vs. 8.67% Declined.",
            "• Decline Intelligence: Separates Soft Recoverable (Code 51 NSF) from Hard Declines."
        ]),
        ("STAGE 3: DISPUTE & CHARGEBACK LIFECYCLE", RGBColor(124, 58, 237), [
            "• 1,121 Confirmed Chargebacks tracked across full scheme lifecycle.",
            "• 4 Formal Stages: First Chargeback -> Representment -> Pre-Arbitration -> Write-Off.",
            "• Proves 3DS Network Liability Shift protects merchants against fraud disputes."
        ])
    ]
    for i, (title, color, bullets) in enumerate(arch_cols):
        left = Inches(0.8 + i * 3.95)
        add_card(s3, left, Inches(3.2), Inches(3.8), Inches(3.8))
        bar = s3.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(3.2), Inches(3.8), Inches(0.10))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tb_c = s3.shapes.add_textbox(left + Inches(0.2), Inches(3.4), Inches(3.4), Inches(3.4))
        tf_c = tb_c.text_frame
        tf_c.word_wrap = True
        p_t = tf_c.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        for b in bullets:
            p_b = tf_c.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(10.5)
            p_b.font.color.rgb = C_TEXT_MAIN
            p_b.space_before = Pt(8)

    # =========================================================================
    # SLIDE 4: FOUR STRUCTURAL BEHAVIORAL DOMAIN PATTERNS
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Portfolio Diagnostics & Patterns", "Four Core Behavioral Patterns Driving Issuer Performance",
               "Real-world empirical phenomena modeled deterministically without relying on random noise.")

    quads = [
        ("1. Channel & Tokenization Differentials", C_BLUE_PRIMARY, [
            "• In-Store POS: 95.3% approval and negligible fraud (0.14%) due to physical card trust.",
            "• E-Commerce Lags: 87.7% approval and 3.65% fraud driven by credential theft.",
            "• Tokenization + 3DS Lift: Boosts approval by +9–14% and slashes fraud by >75%."
        ]),
        ("2. Payday Liquidity & Seasonality Cycles", C_GREEN, [
            "• Post-Payday (Days 25–5): High liquidity lifts volume +18% and slashes NSF declines by 45%.",
            "• Cash Crunch (Days 15–24): Balance depletion doubles Insufficient Funds (Code 51) declines.",
            "• Cyber Week (Oct 15–25): +35% volume surge elevates false-alarm pressure."
        ]),
        ("3. Diurnal Late-Night Attack Window", C_RED, [
            "• Overnight Spike (01:00–05:00): Fraud density is 3.6x higher than standard daytime hours.",
            "• Driven by automated fraud botnets and overseas rings testing stolen card lists.",
            "• Justifies strict overnight velocity controls and step-up authentication."
        ]),
        ("4. Acquirer Gateway & Switch Degradation", C_AMBER, [
            "• Modern Processors (Adyen, Stripe, GlobalPay) maintain healthy 92.6%–93.0% approval.",
            "• LegacySwitch_Bank Anomaly: Lags at 80.8% approval due to 5.2% Code 91 technical switch errors.",
            "• Operational Fix: Reconfigure token and 3DS payload parsing with the processor."
        ])
    ]

    for i, (title, color, bullets) in enumerate(quads):
        row = i // 2
        col = i % 2
        left = Inches(0.8 + col * 5.95)
        top = Inches(1.7 + row * 2.65)
        add_card(s4, left, top, Inches(5.75), Inches(2.45))
        
        bar = s4.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.12), Inches(2.45))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tb_q = s4.shapes.add_textbox(left + Inches(0.25), top + Inches(0.15), Inches(5.35), Inches(2.15))
        tf_q = tb_q.text_frame
        tf_q.word_wrap = True

        p_t = tf_q.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(12)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        for b in bullets:
            p_b = tf_q.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(10.5)
            p_b.font.color.rgb = C_TEXT_MAIN
            p_b.space_before = Pt(6)

    # =========================================================================
    # SLIDE 5: MACHINE LEARNING BENCHMARK & SHAP EXPLAINABILITY
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Machine Learning Modeling", "Leakage-Free Validation: Advanced XGBoost Beats Baseline by +10.2 pts",
               "Evaluated on a strictly chronologically held-out Out-of-Time test set during the peak October shopping surge.")

    # Left Column: Validation Split & Benchmark Table
    add_card(s5, Inches(0.8), Inches(1.7), Inches(6.0), Inches(5.3))
    tb_m = s5.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.6), Inches(5.0))
    tf_m = tb_m.text_frame
    tf_m.word_wrap = True

    p = tf_m.paragraphs[0]
    p.text = "Strict Out-of-Time Chronological Split (Zero Leakage)"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_BLUE_PRIMARY

    splits = [
        "• Train Set (Jul 1 – Sep 15, 77 days): 131,836 rows used to fit models and out-of-fold historical risk target encodings.",
        "• Validation Set (Sep 16 – Oct 5, 20 days): 34,475 rows used exclusively to optimize decision threshold tau*.",
        "• Out-of-Time Test (Oct 6 – Oct 31, 26 days): 53,689 rows including the Cyber Retail surge, held out completely."
    ]
    for s in splits:
        p_s = tf_m.add_paragraph()
        p_s.text = s
        p_s.font.size = Pt(9.5)
        p_s.font.color.rgb = C_TEXT_MAIN
        p_s.space_before = Pt(4)

    p_tbl = tf_m.add_paragraph()
    p_tbl.text = "Model Evaluation Benchmark (Out-of-Time Test Set)"
    p_tbl.font.size = Pt(12)
    p_tbl.font.bold = True
    p_tbl.font.color.rgb = C_TEXT_MAIN
    p_tbl.space_before = Pt(12)

    metrics_text = [
        "Metric                 Baseline (LogReg)    Advanced (XGBoost)    Net Lift",
        "PR-AUC (Primary)       0.5138               0.6161                +10.23 pts",
        "ROC-AUC (Rank)         0.9145               0.9373                +2.28 pts",
        "False Declines         724 false blocks     615 false blocks      -15.1% fewer",
        "Net Financial ROI      $550,048             $573,910              +$23,862"
    ]
    for m in metrics_text:
        p_m = tf_m.add_paragraph()
        p_m.text = m
        p_m.font.size = Pt(9)
        p_m.font.name = "Consolas"
        p_m.font.color.rgb = C_NAVY_DARK

    # Right Column: SHAP Feature Importance
    add_card(s5, Inches(7.1), Inches(1.7), Inches(5.4), Inches(5.3))
    tb_sh = s5.shapes.add_textbox(Inches(7.3), Inches(1.85), Inches(5.0), Inches(5.0))
    tf_sh = tb_sh.text_frame
    tf_sh.word_wrap = True

    p_sh = tf_sh.paragraphs[0]
    p_sh.text = "Global Feature Importance (TreeExplainer SHAP)"
    p_sh.font.size = Pt(13)
    p_sh.font.bold = True
    p_sh.font.color.rgb = C_BLUE_PRIMARY

    p_sub = tf_sh.add_paragraph()
    p_sub.text = "What drives the fraud score? (Mean |SHAP Value| on Log-Odds):"
    p_sub.font.size = Pt(10)
    p_sub.font.color.rgb = C_TEXT_MUTED
    p_sub.space_before = Pt(4)

    shap_items = [
        ("1. Unsecured E-Commerce (No Token & No 3DS)", "0.610", C_RED),
        ("2. Spend Ratio vs. Cardholder 30-Day Avg", "0.479", C_RED),
        ("3. MCC Historical Target Encodings", "0.429", C_AMBER),
        ("4. 3DS Authentication Indicator (Verified)", "0.300", C_GREEN),
        ("5. Network Tokenized Indicator (Apple/Google)", "0.288", C_GREEN),
        ("6. Transaction Amount (Log USD)", "0.193", C_BLUE_PRIMARY),
        ("7. Card Swipe Velocity (1h & 24h counters)", "0.145", C_BLUE_PRIMARY),
    ]
    for name, val, col in shap_items:
        p_item = tf_sh.add_paragraph()
        p_item.text = f"• {name}  [|SHAP| = {val}]"
        p_item.font.size = Pt(10)
        p_item.font.color.rgb = C_TEXT_MAIN
        p_item.space_before = Pt(8)

    # =========================================================================
    # SLIDE 6: BUSINESS UNIT ECONOMICS & OPTIMAL THRESHOLD (tau* = 0.12)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Business Value & ROI", "Financial Optimization: Operating Threshold tau* = 0.12 Unlocks $1.65M",
               "Aligning decision boundaries with issuer P&L: Why missing fraud is 10x more costly than a false decline.")

    # Unit Economics Framework Card
    add_card(s6, Inches(0.8), Inches(1.7), Inches(5.75), Inches(4.5))
    tb_u = s6.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.35), Inches(4.2))
    tf_u = tb_u.text_frame
    tf_u.word_wrap = True

    p = tf_u.paragraphs[0]
    p.text = "Issuer P&L Unit Economics Formula"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_BLUE_PRIMARY

    econ_points = [
        "1. True Positive (Fraud Caught): Saves 100% principal + $15 chargeback fee.",
        "2. False Positive (False Decline): Loses 1.75% interchange + 8.0% cardholder spend churn + $15 service friction penalty.",
        "3. True Negative (Good Spend): Earns +1.75% interchange fee.",
        "4. False Negative (Missed Fraud): Loses 100% principal + $15 chargeback fee.",
        "\n★ Net Business Formula:",
        "Net ML Savings ($) = Fraud Prevented ($) - False Decline Cost ($)"
    ]
    for ep in econ_points:
        p_ep = tf_u.add_paragraph()
        p_ep.text = ep
        p_ep.font.size = Pt(10.5)
        p_ep.font.color.rgb = C_TEXT_MAIN
        p_ep.space_before = Pt(6)

    # Optimal Threshold Outcome Card
    add_card(s6, Inches(6.75), Inches(1.7), Inches(5.75), Inches(4.5))
    tb_o = s6.shapes.add_textbox(Inches(6.95), Inches(1.85), Inches(5.35), Inches(4.2))
    tf_o = tb_o.text_frame
    tf_o.word_wrap = True

    p_o = tf_o.paragraphs[0]
    p_o.text = "Optimal Operating Threshold: tau* = 0.12"
    p_o.font.size = Pt(13)
    p_o.font.bold = True
    p_o.font.color.rgb = C_GREEN

    res_points = [
        "• Why 12% instead of 50%? Because fraud is rare (1.5%) and 10x more expensive than false declines, the financial sweet spot sits at 12%.",
        "• Overall Approval Rate: 97.77% of all transaction volume is approved friction-free.",
        "• Fraud Dollar Capture: Catches 87.21% of all fraud dollars ($612K of $701K).",
        "• False Decline Reduction: Cuts false declines by -44.2% (from 1,102 down to 615).",
        "• Net Financial Value: $573,910 in 26 days (+$117,522 incremental value over legacy rules, or ~$1.65M annualized)."
    ]
    for rp in res_points:
        p_rp = tf_o.add_paragraph()
        p_rp.text = rp
        p_rp.font.size = Pt(10.5)
        p_rp.font.color.rgb = C_TEXT_MAIN
        p_rp.space_before = Pt(6)

    # Bottom Banner
    add_card(s6, Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.85), C_BLUE_LIGHT, C_BLUE_PRIMARY)
    tb_b6 = s6.shapes.add_textbox(Inches(1.0), Inches(6.4), Inches(11.3), Inches(0.75))
    p_b6 = tb_b6.text_frame.paragraphs[0]
    p_b6.text = "Interactive What-If Simulator: Built directly into Tab 6 of the platform, enabling executives to slide threshold tau live and observe real-time impacts on false declines, fraud losses, and net P&L."
    p_b6.font.size = Pt(10.5)
    p_b6.font.bold = True
    p_b6.font.color.rgb = C_BLUE_PRIMARY

    # =========================================================================
    # SLIDE 7: INTELLIGENT RISK-BASED ROUTING & SMART PAYDAY RETRIES
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Strategic Recommendations", "Intelligent Risk-Based Routing & Calendar-Aware Payday Retries",
               "Moving from crude binary approve/decline switches to multi-tier dynamic authentication.")

    routes = [
        ("TIER 1: LOW RISK (p < 0.08)", C_GREEN, "Frictionless Instant Approval", [
            "• Applies to 97.1% of transactions ($9.14M volume).",
            "• Actual fraud rate is only 0.38%.",
            "• Zero OTP, zero delay, maximum checkout conversion."
        ]),
        ("TIER 2: MEDIUM RISK (0.08 <= p < 0.12)", C_AMBER, "Step-Up 3DS Challenge", [
            "• Applies to 0.65% of transactions (343 cases; 14.3% fraud).",
            "• Instead of declining, prompt user for quick OTP/Biometric.",
            "• Successful 3DS shifts card scheme liability away from merchant."
        ]),
        ("TIER 3: HIGH RISK (p >= 0.12)", C_RED, "Immediate Auto-Block", [
            "• Isolates 2.23% of volume (1,170 transactions; 47.4% fraud).",
            "• Captures 87.2% of all stolen fraud dollars before settlement.",
            "• Routes complex anomalies to fraud analyst investigation."
        ])
    ]

    for i, (title, color, action, bullets) in enumerate(routes):
        left = Inches(0.8 + i * 3.95)
        add_card(s7, left, Inches(1.7), Inches(3.8), Inches(3.3))
        bar = s7.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.7), Inches(3.8), Inches(0.12))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tb_r = s7.shapes.add_textbox(left + Inches(0.2), Inches(1.9), Inches(3.4), Inches(3.0))
        tf_r = tb_r.text_frame
        tf_r.word_wrap = True

        p_t = tf_r.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(11)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        p_act = tf_r.add_paragraph()
        p_act.text = f"Action: {action}"
        p_act.font.size = Pt(10)
        p_act.font.bold = True
        p_act.font.color.rgb = C_NAVY_DARK
        p_act.space_before = Pt(3)

        for b in bullets:
            p_b = tf_r.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(9.5)
            p_b.font.color.rgb = C_TEXT_MAIN
            p_b.space_before = Pt(6)

    # Bottom Full Width Card: Smart Payday Retry Mechanism
    add_card(s7, Inches(0.8), Inches(5.15), Inches(11.7), Inches(1.95), C_CARD_BG, C_BLUE_PRIMARY)
    tb_pd = s7.shapes.add_textbox(Inches(1.0), Inches(5.25), Inches(11.3), Inches(1.75))
    tf_pd = tb_pd.text_frame
    tf_pd.word_wrap = True

    p_pdt = tf_pd.paragraphs[0]
    p_pdt.text = "CALENDAR-AWARE SMART RETRY ENGINE FOR INSUFFICIENT FUNDS (CODE 51)"
    p_pdt.font.size = Pt(11.5)
    p_pdt.font.bold = True
    p_pdt.font.color.rgb = C_BLUE_PRIMARY

    pd_bullets = [
        "• The Problem: In recurring billing, transactions declining for Code 51 between Days 15–24 fail due to temporary cash crunches, causing passive subscriber churn.",
        "• The Solution: Schedule automated retries during the Post-Payday Window (Days 25–5) when consumer liquidity is 45% higher and balance recovery rates peak.",
        "• Commercial Impact: Recovers ~35–50% of otherwise lost recurring subscription revenue without manual customer intervention."
    ]
    for pdb in pd_bullets:
        p_pdb = tf_pd.add_paragraph()
        p_pdb.text = pdb
        p_pdb.font.size = Pt(10)
        p_pdb.font.color.rgb = C_TEXT_MAIN
        p_pdb.space_before = Pt(4)

    # =========================================================================
    # SLIDE 8: AI INSIGHTS ENGINE, GOVERNANCE & GUARDRAILS
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Enterprise AI & Governance", "Grounded AI Insights Engine: Zero-Hallucination Governance Architecture",
               "LLM synthesis anchored strictly to pre-computed Python analytics payloads with out-of-schema guardrails.")

    ai_cards = [
        ("1. PRE-COMPUTED DATA CONTEXT", C_BLUE_PRIMARY, [
            "• Python deterministically calculates KPIs, deltas, and segment rankings.",
            "• Packs context into a structured <DATA_CONTEXT> JSON payload.",
            "• LLM does ZERO mental math over raw data, eliminating calculation hallucinations."
        ]),
        ("2. STRICT GOVERNANCE PROMPTS", C_GREEN, [
            "• System prompt runs at temperature=0.1.",
            "• Model is strictly instructed to cite only numbers in the context.",
            "• Enforces a 6-section board-ready format with concrete business recommendations."
        ]),
        ("3. OUT-OF-SCHEMA REFUSAL GUARDRAIL", C_RED, [
            "• Intercepts queries regarding missing cardholder attributes (FICO credit score, age, income).",
            "• Gracefully refuses and provides an explicit schema boundary explanation.",
            "• Verified with live test button directly in Tab 7 UI."
        ]),
        ("4. HYBRID OFFLINE RESILIENCE", C_AMBER, [
            "• Compatible with OpenAI, Groq, and Gemini APIs.",
            "• If keys are absent or firewalls block outbound traffic, switches seamlessly to Deterministic Analytics Engine.",
            "• 100% offline grading and demo reliability."
        ])
    ]

    for i, (title, color, bullets) in enumerate(ai_cards):
        left = Inches(0.8 + i * 2.95)
        add_card(s8, left, Inches(1.7), Inches(2.85), Inches(5.3))
        bar = s8.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, Inches(1.7), Inches(2.85), Inches(0.10))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()

        tb_a = s8.shapes.add_textbox(left + Inches(0.15), Inches(1.85), Inches(2.55), Inches(5.0))
        tf_a = tb_a.text_frame
        tf_a.word_wrap = True

        p_t = tf_a.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(10)
        p_t.font.bold = True
        p_t.font.color.rgb = color

        for b in bullets:
            p_b = tf_a.add_paragraph()
            p_b.text = b
            p_b.font.size = Pt(9.5)
            p_b.font.color.rgb = C_TEXT_MAIN
            p_b.space_before = Pt(8)

    # =========================================================================
    # SLIDE 9: HONEST LIMITATIONS & PRODUCTION ROADMAP
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Roadmap & Conclusion", "Production Readiness: Honest Limitations & Next Steps for Scale",
               "Real-world operational enhancements for full-scale issuer deployment at Network International.")

    # Left: Limitations
    add_card(s9, Inches(0.8), Inches(1.7), Inches(5.75), Inches(5.3))
    tb_lim = s9.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(5.35), Inches(5.0))
    tf_lim = tb_lim.text_frame
    tf_lim.word_wrap = True

    p = tf_lim.paragraphs[0]
    p.text = "Honest Technical Limitations"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_AMBER

    lims = [
        ("Label Maturation Lag (30–90 Days):", "In live banking, chargeback labels take 30 to 90 days to settle. Offline models assume immediate ground-truth labels at split boundaries."),
        ("Synthetic Edge Simplifications:", "While our generator captures 1,050 velocity bursts and payday cycles, live production data exhibits noisier merchant strings, multi-currency FX spikes, and evolving cardholder behavior."),
        ("Cold-Start Card Profiles:", "Newly issued card tokens lack 30-day historical spend averages, requiring fallback priors based on MCC and issuing bank BINs.")
    ]
    for h, desc in lims:
        p_h = tf_lim.add_paragraph()
        p_h.text = f"• {h}"
        p_h.font.size = Pt(10.5)
        p_h.font.bold = True
        p_h.font.color.rgb = C_NAVY_DARK
        p_h.space_before = Pt(8)
        p_d = tf_lim.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = C_TEXT_MAIN

    # Right: Production Next Steps
    add_card(s9, Inches(6.75), Inches(1.7), Inches(5.75), Inches(5.3))
    tb_prod = s9.shapes.add_textbox(Inches(6.95), Inches(1.85), Inches(5.35), Inches(5.0))
    tf_prod = tb_prod.text_frame
    tf_prod.word_wrap = True

    p_p = tf_prod.paragraphs[0]
    p_p.text = "Production Implementation Roadmap"
    p_p.font.size = Pt(13)
    p_p.font.bold = True
    p_p.font.color.rgb = C_GREEN

    prods = [
        ("Phase 1: Streaming Feature Store (Redis/Feast)", "Deploy sub-15ms rolling velocity counters (1-minute, 1-hour, 24-hour card swipes) directly at the ISO 8583 authorization switch layer."),
        ("Phase 2: Delayed-Feedback & Survival Learning", "Implement importance-weighted PU-learning to train on mature 60-day historical windows combined with rapid 24-hour anomaly alerts."),
        ("Phase 3: Champion-Challenger A/B Testing", "Deploy the XGBoost model in shadow mode on 10% of live traffic against legacy rules to empirically measure approval lift and verify zero switch latency degradation.")
    ]
    for h, desc in prods:
        p_h = tf_prod.add_paragraph()
        p_h.text = f"• {h}"
        p_h.font.size = Pt(10.5)
        p_h.font.bold = True
        p_h.font.color.rgb = C_NAVY_DARK
        p_h.space_before = Pt(8)
        p_d = tf_prod.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(9.5)
        p_d.font.color.rgb = C_TEXT_MAIN

    # Save
    prs.save(PPTX_PATH)
    print(f"Successfully generated PowerPoint presentation at {PPTX_PATH}")

if __name__ == "__main__":
    create_deck()
