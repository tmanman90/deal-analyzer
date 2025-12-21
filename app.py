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
    .insight-box {
        background-color: #262730;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #444;
        color: #ffffff !important; /* Fixed: High contrast text */
    }
    .insight-box strong {
        color: #cccccc; /* Slightly softer white for labels */
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

def parse_input_data(text_input):
    """
    Parses string input into list of integers.
    Handles '223,783' (formatted numbers) and '100, 200' (lists) intelligently.
    """
    try:
        if not text_input.strip():
            return None
        
        # 1. Replace "comma-space" with just space (treats '100, 200' as '100 200')
        text = text_input.replace(", ", " ")
        text = text.replace("\n", " ") # Handle newlines from Excel copy-paste
        
        # 2. Remove remaining commas (treats '223,783' as '223783')
        text = text.replace(",", "")
        
        # 3. Split by whitespace and convert
        cleaned = [int(x.strip()) for x in text.split() if x.strip()]
        
        if len(cleaned) < 2:
            return None
        return cleaned
    except ValueError:
        return None

def analyze_trend(data_list):
    """
    Calculates trend based on 8 weeks of data.
    Logic: Compare Avg of first 4 vs Avg of last 4.
    Returns: Trend Name (Key), Raw % Change
    """
    if len(data_list) < 2:
        return "Stable (0%)", 0.0

    # Split data (Handle cases with less than 8 points gracefully)
    mid_point = len(data_list) // 2
    past_data = data_list[:mid_point]
    recent_data = data_list[mid_point:]

    avg_past = sum(past_data) / len(past_data)
    avg_recent = sum(recent_data) / len(recent_data)

    if avg_past == 0:
        pct_change = 0.0
    else:
        pct_change = (avg_recent - avg_past) / avg_past

    # Logic Rules (The Stabilizer)
    if pct_change < -0.002: # Less than -0.2%
        trend_name = "Decay (-1.0%)"
    elif -0.002 <= pct_change <= 0.002: # Between -0.2% and +0.2%
        trend_name = "Stable (0%)"
    else: # Greater than +0.2%
        trend_name = "Moderate (0.5%)" # HARD CAP applied here

    return trend_name, pct_change

def project_revenue(start_us_vol, start_ex_us_vol, us_mult, ex_us_mult, weeks=52):
    """Projects revenue for 'weeks' duration."""
    total_rev = 0
    weekly_revs = []
    
    current_us = start_us_vol
    current_ex_us = start_ex_us_vol
    
    for _ in range(weeks):
        current_us *= us_mult
        current_ex_us *= ex_us_mult
        rev = (current_us * US_RATE) + (current_ex_us * EX_US_RATE)
        total_rev += rev
        weekly_revs.append(rev)
        
    return total_rev, weekly_revs[-1], weekly_revs

def get_scenario_multipliers(selected_trend_name, is_ceiling=False):
    """Determines multiplier based on Scenario rules."""
    base_mult = TREND_MAP[selected_trend_name]
    
    if not is_ceiling:
        # Scenario A: Floor
        if base_mult >= 1.0:
            return 1.0 # Force Stable if Growth
        else:
            return 0.99 # Keep Decay
    else:
        # Scenario B: Ceiling
        if selected_trend_name == "Stable (0%)":
            return TREND_MAP["Moderate (0.5%)"] # Upside rule
        return base_mult

def calculate_recoupment(advance, revenue_stream, artist_share, final_week_rev):
    """Calculates recoupment months."""
    label_cum = 0
    label_months = None
    for week_idx, rev in enumerate(revenue_stream):
        label_cum += rev
        if label_cum >= advance:
            label_months = (week_idx + 1) / 4.33
            break
            
    if label_months is None:
        remaining = advance - label_cum
        weeks_needed = remaining / final_week_rev
        label_months = (52 + weeks_needed) / 4.33

    artist_cum = 0
    artist_months = None
    for week_idx, rev in enumerate(revenue_stream):
        artist_cum += (rev * artist_share)
        if artist_cum >= advance:
            artist_months = (week_idx + 1) / 4.33
            break
            
    if artist_months is None:
        remaining = advance - artist_cum
        weeks_needed = remaining / (final_week_rev * artist_share)
        artist_months = (52 + weeks_needed) / 4.33
        
    return label_months, artist_months

# -----------------------------------------------------------------------------
# SIDEBAR - INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Deal Inputs")
    
    # NEW: Mode Selection
    input_mode = st.radio("Analysis Mode", ["Simple (Manual)", "Advanced (Auto-Detect)"])
    
    st.markdown("---")
    
    # Initialize variables to ensure scope
    us_streams = 0
    global_streams = 0
    us_trend_sel = "Stable (0%)"
    ex_us_trend_sel = "Stable (0%)"
    us_history = []
    global_history = []
    
    if input_mode == "Simple (Manual)":
        st.subheader("Streaming Volume")
        us_streams = st.number_input("US Weekly Streams", min_value=0, value=500000, step=10000)
        global_streams = st.number_input("Global Weekly Streams", min_value=0, value=1000000, step=10000)
        
        st.subheader("Trend Projections")
        us_trend_sel = st.selectbox("US Trend", list(TREND_MAP.keys()), index=0)
        ex_us_trend_sel = st.selectbox("Int'l Trend", list(TREND_MAP.keys()), index=0)
        
        run_analysis = True # Always run in manual mode

    else: # Advanced Mode
        st.subheader("Historical Data (8 Weeks)")
        st.caption("Paste numbers (separated by Space or Newline). Commas are ignored.")
        
        us_input_raw = st.text_area("US Streams History", "71000 72000 71500 73000 74000 74500 75000 76000")
        global_input_raw = st.text_area("Global Streams History", "150000 152000 151000 153000 155000 156000 158000 160000")
        
        # --- STATE MANAGEMENT FIX ---
        if "analysis_complete" not in st.session_state:
            st.session_state.analysis_complete = False

        # When clicked, set the state to True
        if st.button("Run Analysis", type="primary"):
            st.session_state.analysis_complete = True
            
        # Use session state to determine if we run, not just the button return value
        if st.session_state.analysis_complete:
            us_history = parse_input_data(us_input_raw)
            global_history = parse_input_data(global_input_raw)
            
            if us_history and global_history:
                # 1. Get Current Volume (Last week of data)
                us_streams = us_history[-1]
                global_streams = global_history[-1]
                
                # 2. Auto-Detect Trends
                us_trend_sel, us_pct = analyze_trend(us_history)
                ex_us_history = [g - u for g, u in zip(global_history, us_history)] 
                
                # Safety check for Ex-US negatives
                ex_us_history = [max(0, x) for x in ex_us_history]
                
                ex_us_trend_sel, ex_us_pct = analyze_trend(ex_us_history)
                
                run_analysis = True
            else:
                st.error("Please enter valid numbers separated by spaces or newlines.")
                run_analysis = False
        else:
            run_analysis = False

    st.subheader("Deal Structure")
    artist_share_pct = st.slider("Artist Share %", 10, 100, 60, 5)
    artist_share = artist_share_pct / 100.0

    st.markdown("---")
    st.caption("Deal Analyzer v2.1.3 | Session State Fixed")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------
st.title("Deal Analyzer 2026")

if not run_analysis:
    st.info("👈 Select 'Advanced Mode' and click 'Run Analysis' to begin.")
    st.stop()

# Derive Ex-US current volume
ex_us_streams = global_streams - us_streams
if ex_us_streams < 0: 
    ex_us_streams = 0

# -----------------------------------------------------------------------------
# SECTION 0: AUTO-DETECT VISUALIZATION (Only in Advanced Mode)
# -----------------------------------------------------------------------------
if input_mode == "Advanced (Auto-Detect)":
    st.markdown("### 📊 Historical Analysis")
    
    # Show Charts
    chart_data = pd.DataFrame({
        "US Streams": us_history,
        "Global Streams": global_history
    })
    st.line_chart(chart_data)
    
    # Show Insights
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class="insight-box">
            <strong>US Detected Trend:</strong><br>
            <span style="font-size: 1.2em; color: #00ffbf">{us_trend_sel}</span>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Int'l Detected Trend:</strong><br>
            <span style="font-size: 1.2em; color: #00ffbf">{ex_us_trend_sel}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CORE CALCULATION (Engine)
# -----------------------------------------------------------------------------
st.markdown("### Valuation & Recoupment Model")

# 1. Determine Multipliers
us_mult_floor = get_scenario_multipliers(us_trend_sel, is_ceiling=False)
ex_us_mult_floor = get_scenario_multipliers(ex_us_trend_sel, is_ceiling=False)

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
    st.metric("The Floor (Base Reality)", f"${floor_gross:,.0f}", delta="Risk Adjusted", delta_color="off")
    st.caption("Assumes 0% growth or actual decay.")

with col2:
    st.metric("The Ceiling (Potential)", f"${ceil_gross:,.0f}", delta="Growth Adjusted", delta_color="normal")
    st.caption("Includes upside growth assumptions.")

# -----------------------------------------------------------------------------
# SECTION 2: THE OFFER TABLE
# -----------------------------------------------------------------------------
st.markdown("#### 2. The Offer Matrix")
conservative_offer = floor_gross * 0.75
aggressive_offer = ceil_gross * 1.10

offer_data = {
    "Strategy": ["Conservative", "Target Zone", "Aggressive"],
    "Offer Value": [f"${conservative_offer:,.0f}", f"${floor_gross:,.0f} - ${ceil_gross:,.0f}", f"${aggressive_offer:,.0f}"],
    "Rationale": ["75% of Floor (Safe Bet)", "Floor to Ceiling Range", "110% of Ceiling (High Risk)"],
    "Est. Recoupment": ["~9 Months", "12 Months", "15+ Months"]
}
st.table(pd.DataFrame(offer_data))

# -----------------------------------------------------------------------------
# SECTION 3: DEAL REALITY CHECK
# -----------------------------------------------------------------------------
st.markdown("#### 3. Deal Reality Check")

# Define numerical values for the dropdown
recoup_options = {
    "Conservative (75% of Floor)": conservative_offer,
    "Target Floor (Floor Gross)": floor_gross,
    "Target Ceiling (Ceiling Gross)": ceil_gross,
    "Aggressive (110% of Ceiling)": aggressive_offer
}

# Create columns for the dropdown
rc1, rc2 = st.columns([2, 1])
with rc1:
    selected_option = st.selectbox(
        "Select Proposed Advance for Analysis:", 
        options=list(recoup_options.keys()), 
        index=2  # Default to Target Ceiling
    )
selected_advance = recoup_options[selected_option]

# Extract just the name of the strategy (e.g., "Conservative") for cleaner text
strategy_name = selected_option.split(" (")[0]

st.info(f"Analysis assuming **{strategy_name}** Advance of **${selected_advance:,.0f}** (checked against projected Ceiling Revenue).")

# Run Recoupment Math using the selected advance
lbl_recoup_mo, art_recoup_mo = calculate_recoupment(selected_advance, ceil_stream, artist_share, ceil_last_wk)
label_profit_at_recoup = (selected_advance / artist_share) * (1.0 - artist_share)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Label Breakeven", f"{lbl_recoup_mo:.1f} Months", help="Time to recover advance (100% receipts)")
with c2:
    is_long = "⚠️ " if art_recoup_mo > 18 else ""
    st.metric("Artist Recoupment", f"{is_long}{art_recoup_mo:.1f} Months", help="Time for artist to earn out")
with c3:
    st.metric("Label Profit @ Recoup", f"${label_profit_at_recoup:,.0f}", help="Net profit when artist recoups")

# -----------------------------------------------------------------------------
# SECTION 4: BUYER'S STRATEGY
# -----------------------------------------------------------------------------
st.markdown("#### 4. Buyer's Strategy Script")

strategy_html = f"""
<div class="strategy-box">
    <strong>NEGOTIATION SCRIPT:</strong><br><br>
    "Based on the analysis of the last 8 weeks, we see a <strong>{us_trend_sel}</strong> trend."<br><br>
    1. <strong>Anchor:</strong> Start at <strong>${conservative_offer:,.0f}</strong>.<br>
    2. <strong>Rationalize:</strong> Our risk-adjusted floor is <strong>${floor_gross:,.0f}</strong> given market volatility.<br>
    3. <strong>Close:</strong> If we stretch to <strong>${ceil_gross:,.0f}</strong>, the artist won't see royalties for <strong>{art_recoup_mo:.1f} months</strong>."
</div>
"""
st.markdown(strategy_html, unsafe_allow_html=True)
