import streamlit as st
import pandas as pd
import numpy as np
import math

# -----------------------------------------------------------------------------
# CONFIG & STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deal Analyzer 2026",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, financial dashboard look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0rem;
    }
    h3 {
        color: #cccccc;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 400;
        margin-top: 2rem;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #aaaaaa !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #00ffbf !important; /* Financial Green */
    }
    .strategy-box {
        background-color: #1c1f26;
        padding: 20px;
        border-left: 5px solid #00ffbf;
        border-radius: 5px;
        color: #ddd;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & RATES (2025 Data)
# -----------------------------------------------------------------------------
US_RATE = 0.00356
EX_US_RATE = 0.00202

# Trend Multipliers
TREND_MAP = {
    "Stable (0%)": 1.0,
    "Moderate (0.5%)": 1.005,
    "Strong (1.0%)": 1.01,
    "Decay (-1.0%)": 0.99
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (THE ENGINE)
# -----------------------------------------------------------------------------

def project_revenue(start_us_vol, start_ex_us_vol, us_mult, ex_us_mult, weeks=52):
    """
    Projects revenue for 'weeks' duration based on starting volumes and weekly multipliers.
    Returns: Total Revenue, Final Week Revenue (for extrapolation), Weekly Revenue List
    """
    total_rev = 0
    weekly_revs = []
    
    current_us = start_us_vol
    current_ex_us = start_ex_us_vol
    
    for _ in range(weeks):
        # Apply growth/decay
        current_us *= us_mult
        current_ex_us *= ex_us_mult
        
        # Calculate Revenue
        rev = (current_us * US_RATE) + (current_ex_us * EX_US_RATE)
        
        total_rev += rev
        weekly_revs.append(rev)
        
    return total_rev, weekly_revs[-1], weekly_revs

def get_scenario_multipliers(selected_trend_name, is_ceiling=False):
    """
    Determines the actual multiplier to use based on Scenario rules.
    """
    base_mult = TREND_MAP[selected_trend_name]
    
    if not is_ceiling:
        # --- SCENARIO A: THE FLOOR ---
        # If Growth or Stable -> Force Stable (1.0)
        # If Decay -> Keep Decay (0.99)
        if base_mult >= 1.0:
            return 1.0
        else:
            return 0.99
    else:
        # --- SCENARIO B: THE CEILING ---
        # Use Selected Trend
        # Upside Rule: If Stable selected -> Force Moderate
        if selected_trend_name == "Stable (0%)":
            return TREND_MAP["Moderate (0.5%)"]
        return base_mult

def calculate_recoupment(advance, revenue_stream, artist_share, final_week_rev):
    """
    Calculates recoupment months for Label (100% rev) and Artist (Share % rev).
    Handles extrapolation if > 52 weeks.
    """
    # 1. Label Recoupment (100% of Revenue)
    label_cum = 0
    label_months = None
    
    for week_idx, rev in enumerate(revenue_stream):
        label_cum += rev
        if label_cum >= advance:
            label_months = (week_idx + 1) / 4.33
            break
            
    if label_months is None:
        # Extrapolate
        remaining = advance - label_cum
        weeks_needed = remaining / final_week_rev
        label_months = (52 + weeks_needed) / 4.33

    # 2. Artist Recoupment (Artist Share of Revenue)
    artist_cum = 0
    artist_months = None
    
    for week_idx, rev in enumerate(revenue_stream):
        artist_cum += (rev * artist_share)
        if artist_cum >= advance:
            artist_months = (week_idx + 1) / 4.33
            break
            
    if artist_months is None:
        # Extrapolate
        remaining = advance - artist_cum
        weeks_needed = remaining / (final_week_rev * artist_share)
        artist_months = (52 + weeks_needed) / 4.33
        
    return label_months, artist_months

# -----------------------------------------------------------------------------
# SIDEBAR - INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Deal Inputs")
    
    st.subheader("Streaming Volume")
    us_streams = st.number_input("US Weekly Streams", min_value=0, value=500000, step=10000)
    global_streams = st.number_input("Global Weekly Streams", min_value=0, value=1000000, step=10000)
    
    # Validation for Ex-US
    ex_us_streams = global_streams - us_streams
    if ex_us_streams < 0:
        st.error("Global streams cannot be less than US streams.")
        ex_us_streams = 0
    
    st.caption(f"Implied Int'l Streams: {ex_us_streams:,.0f}")
    
    st.subheader("Deal Structure")
    artist_share_pct = st.slider("Artist Share %", 10, 100, 60, 5)
    artist_share = artist_share_pct / 100.0
    
    st.subheader("Trend Projections")
    us_trend_sel = st.selectbox("US Trend", list(TREND_MAP.keys()), index=0)
    ex_us_trend_sel = st.selectbox("Int'l Trend", list(TREND_MAP.keys()), index=0)
    
    st.markdown("---")
    st.caption("Deal Analyzer v2.0.4 | 2026 Build")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

st.title("Deal Analyzer 2026")
st.markdown("### Valuation & Recoupment Model")

# 1. Determine Multipliers for Scenarios
# Scenario A (Floor)
us_mult_floor = get_scenario_multipliers(us_trend_sel, is_ceiling=False)
ex_us_mult_floor = get_scenario_multipliers(ex_us_trend_sel, is_ceiling=False)

# Scenario B (Ceiling)
us_mult_ceil = get_scenario_multipliers(us_trend_sel, is_ceiling=True)
ex_us_mult_ceil = get_scenario_multipliers(ex_us_trend_sel, is_ceiling=True)

# 2. Run Projections
floor_gross, floor_last_wk, floor_stream = project_revenue(us_streams, ex_us_streams, us_mult_floor, ex_us_mult_floor)
ceil_gross, ceil_last_wk, ceil_stream = project_revenue(us_streams, ex_us_streams, us_mult_ceil, ex_us_mult_ceil)

# -----------------------------------------------------------------------------
# SECTION 1: VALUATION RANGE
# -----------------------------------------------------------------------------
st.markdown("#### 1. The Valuation Range (52-Week Gross)")
col1, col2 = st.columns(2)

with col1:
    st.metric("The Floor (Base Reality)", f"${floor_gross:,.0f}", 
              delta="Risk Adjusted", delta_color="off")
    st.caption("Assumes 0% growth or actual decay.")

with col2:
    st.metric("The Ceiling (Potential)", f"${ceil_gross:,.0f}", 
              delta="Growth Adjusted", delta_color="normal")
    st.caption("Includes upside growth assumptions.")

# -----------------------------------------------------------------------------
# SECTION 2: THE OFFER TABLE
# -----------------------------------------------------------------------------
st.markdown("#### 2. The Offer Matrix")

conservative_offer = floor_gross * 0.75
aggressive_offer = ceil_gross * 1.10

offer_data = {
    "Strategy": ["Conservative", "Target Zone", "Aggressive"],
    "Offer Value": [
        f"${conservative_offer:,.0f}", 
        f"${floor_gross:,.0f} - ${ceil_gross:,.0f}", 
        f"${aggressive_offer:,.0f}"
    ],
    "Rationale": [
        "75% of Floor (Safe Bet)",
        "Floor to Ceiling Range",
        "110% of Ceiling (High Risk)"
    ],
    "Est. Recoupment (Label)": [
        "~9 Months",
        "12 Months",
        "15+ Months"
    ]
}

df_offers = pd.DataFrame(offer_data)
st.table(df_offers)

# -----------------------------------------------------------------------------
# SECTION 3: DEAL REALITY CHECK
# -----------------------------------------------------------------------------
st.markdown("#### 3. Deal Reality Check (At Ceiling Price)")
st.info(f"Analysis assumes paying the Full Ceiling Price: **${ceil_gross:,.0f}** as an Advance.")

# Run Recoupment Math based on paying the Ceiling as the Advance
# We use the Ceiling Revenue Stream for this calculation to see "if we pay for potential, and get potential, when do we break even?"
lbl_recoup_mo, art_recoup_mo = calculate_recoupment(ceil_gross, ceil_stream, artist_share, ceil_last_wk)

# Label Profit at moment of Artist Recoupment
# Formula derived from standard deal mechanics: (Advance / Share) * (1 - Share)
# This represents the label's "Retained Profit" once the artist account reaches zero.
label_profit_at_recoup = (ceil_gross / artist_share) * (1.0 - artist_share)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Label Breakeven", f"{lbl_recoup_mo:.1f} Months", 
             help="Time to recover the advance using 100% of receipts.")
    
with c2:
    is_long = "⚠️ " if art_recoup_mo > 18 else ""
    st.metric("Artist Recoupment", f"{is_long}{art_recoup_mo:.1f} Months",
             help="Time for artist to earn out the advance at their royalty rate.")

with c3:
    st.metric("Label Profit @ Recoup", f"${label_profit_at_recoup:,.0f}",
             help="Net profit for label the moment the artist receives their first royalty check.")

# -----------------------------------------------------------------------------
# SECTION 4: BUYER'S STRATEGY
# -----------------------------------------------------------------------------
st.markdown("#### 4. Buyer's Strategy Script")

strategy_html = f"""
<div class="strategy-box">
    <strong>NEGOTIATION SCRIPT:</strong><br><br>
    "Based on current run rates, we are looking at a risk-adjusted floor of <strong>${floor_gross:,.0f}</strong>."<br><br>
    1. <strong>Anchor:</strong> Start the conversation at <strong>${conservative_offer:,.0f}</strong> to manage expectations.<br>
    2. <strong>Rationalize:</strong> Walk them through the international decay risks or stable US trends that justify the <strong>${floor_gross:,.0f}</strong> baseline.<br>
    3. <strong>Close:</strong> If pushed, we can stretch to <strong>${ceil_gross:,.0f}</strong>. However, be clear that at this price, the artist won't see a royalty check for <strong>{art_recoup_mo:.1f} months</strong>."
</div>
"""
st.markdown(strategy_html, unsafe_allow_html=True)