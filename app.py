import streamlit as st
import pandas as pd
import numpy as np
import math
import base64
import json
import gspread
from google.oauth2.service_account import Credentials

# Safe Import for OpenAI to prevent app crash if missing/old
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# CONFIG & STYLES
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Deal Analyzer 2026",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# URL STATE MANAGEMENT (SHARE LINK LOGIC)
# -----------------------------------------------------------------------------
# This must run before widgets are instantiated to set their default values
if "defaults" not in st.session_state:
    st.session_state.defaults = {}

# Check query params for shared data
if "initialized" not in st.session_state:
    try:
        # Check if 'data' is in the query parameters
        # Compatible with both st.query_params (new) and potentially older handling
        params = st.query_params
        if "data" in params:
            encoded_data = params["data"]
            
            # Handle list return type (older Streamlit versions sometimes return lists)
            if isinstance(encoded_data, list):
                encoded_data = encoded_data[0]
            
            # 1. Restore Padding if missing (Common copy-paste error)
            missing_padding = len(encoded_data) % 4
            if missing_padding:
                encoded_data += '=' * (4 - missing_padding)
            
            # 2. Decode (Try URL-safe first, then standard)
            try:
                # Try URL-safe decode (replaces -_ with +/)
                json_str = base64.urlsafe_b64decode(encoded_data).decode('utf-8')
            except Exception:
                # Fallback to standard decode
                json_str = base64.b64decode(encoded_data).decode('utf-8')

            loaded_data = json.loads(json_str)
            
            # Store in session state for widgets to use as defaults
            st.session_state.defaults = loaded_data
            st.toast("Configuration loaded from shared link!", icon="✅")
            
            # If advanced mode was shared and analysis was run, trigger the analysis state
            if loaded_data.get("input_mode") == "Advanced (Auto-Detect)":
                st.session_state.analysis_complete = True
    except Exception as e:
        # Silent fail or small warning to avoid crashing app on bad link
        print(f"Share link load error: {e}")
    
    st.session_state.initialized = True

def get_default(key, fallback):
    """Helper to get value from shared state or return fallback"""
    return st.session_state.defaults.get(key, fallback)

# -----------------------------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------------------------
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
    "Decay (-1.0%)": 0.99,
    "Falling Knife (-1.0%)": 0.99,       # New Logic
    "New Artist / Cooling (-1.0%)": 0.99 # New Logic
}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (THE ENGINE)
# -----------------------------------------------------------------------------

def parse_input_data(text_input):
    """
    Parses string input into list of integers.
    Handles '223,783' (formatted numbers) and '100, 200' (lists) intelligently.
    Auto-pads with 0s if fewer than 8 data points are provided.
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
        
        # Auto-Padding Logic
        if len(cleaned) < 8:
            padding_needed = 8 - len(cleaned)
            cleaned = [0] * padding_needed + cleaned
        
        if len(cleaned) < 2:
            return None
        return cleaned
    except ValueError:
        return None

# --- CLASSIC MODEL FUNCTIONS ---

def analyze_trend_classic(data_list):
    """
    Calculates trend based on Hierarchy of Risk.
    Returns: Trend Name (Key), Raw % Change
    """
    # Ensure we have at least 8 points (should be handled by padding, but safe fallback)
    if len(data_list) < 8:
        # If somehow we still have less than 8, pad locally
        data_list = [0] * (8 - len(data_list)) + data_list
    
    # We focus on the last 8 data points for the analysis window
    analysis_window = data_list[-8:]
    
    # Setup
    past_data = analysis_window[:4]    # First 4
    recent_data = analysis_window[4:]  # Last 4
    
    avg_past = sum(past_data) / 4
    avg_recent = sum(recent_data) / 4
    
    current_week = analysis_window[-1]
    previous_week = analysis_window[-2]

    # Check 1: Falling Knife
    # If recent average is positive, but current week crashed >20% below that average
    if avg_recent > 0 and current_week < (avg_recent * 0.8):
        return "Falling Knife (-1.0%)", -0.01

    # Check 2: New Artist Cooling
    # If past volume was low (New Artist context) and momentum is breaking (current < prev)
    if avg_past < 1000 and current_week < previous_week:
        return "New Artist / Cooling (-1.0%)", -0.01

    # Check 3: Standard Trend (Slope Analysis)
    if avg_past == 0:
        slope = 0.0 # Prevent division by zero
    else:
        slope = (avg_recent - avg_past) / avg_past

    if slope < -0.002:
        return "Decay (-1.0%)", slope
    elif -0.002 <= slope <= 0.002:
        return "Stable (0%)", slope
    else:
        # Moderate Growth (Capped)
        return "Moderate (0.5%)", 0.005

def project_revenue_classic(start_us_vol, start_ex_us_vol, us_mult, ex_us_mult, weeks=52):
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

# --- BETA MODEL FUNCTIONS (DYNAMIC MOMENTUM) ---

def analyze_trend_beta(data_list):
    if len(data_list) < 8:
        data_list = [0] * (8 - len(data_list)) + data_list
    analysis_window = data_list[-8:]
    past_data = analysis_window[:4]
    recent_data = analysis_window[4:]
    avg_past = sum(past_data) / 4
    avg_recent = sum(recent_data) / 4
    
    if avg_past == 0:
        slope = 0.0
    else:
        slope = ((avg_recent - avg_past) / avg_past) / 4

    # Safety Clamp: Cap viral growth at +5%
    if slope > 0.05: slope = 0.05
    
    # Status Label
    status = "Stable"
    if slope < -0.005: status = "Decay"
    if slope < -0.05:  status = "High Decay"
    if slope > 0.005:  status = "Growth"
    if slope > 0.025:  status = "High Growth"
    
    return {"status": status, "slope": slope, "avg_vol": avg_recent}

def project_revenue_beta(start_us_vol, start_ex_us_vol, trend_data, weeks=52):
    # Rates
    US_RATE = 0.00356
    EX_US_RATE = 0.00202
    
    total_rev = 0
    weekly_revs = []
    current_us = start_us_vol
    current_ex_us = start_ex_us_vol
    
    initial_rate = trend_data['slope']
    terminal_rate = -0.001 
    
    # Stabilization: Growth slows in 12 weeks, Decay in 26 weeks
    if initial_rate > 0:
        stabilization_weeks = 12.0
    else:
        stabilization_weeks = 26.0
    
    for i in range(weeks):
        if i < stabilization_weeks:
            progress = i / stabilization_weeks
            current_rate = initial_rate * (1 - progress) + terminal_rate * progress
        else:
            current_rate = terminal_rate
            
        current_us *= (1 + current_rate)
        current_ex_us *= (1 + current_rate)
        
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
# AI GENERATOR FUNCTION (UPDATED)
# -----------------------------------------------------------------------------
def generate_ai_strategy(api_key, context_data):
    """
    Generates a negotiation strategy using OpenAI API.
    Refined for "Ammo" (Data Leverage) instead of Concessions.
    """
    if not OPENAI_AVAILABLE:
        return "❌ Error: OpenAI library not installed."

    try:
        client = openai.OpenAI(api_key=api_key)
        
        # 1. Clean the Trend Label to remove misleading hard-caps (e.g. "Cooling (-1.0%)" -> "Cooling")
        # This prevents the AI from quoting the safety cap as the actual decline rate.
        raw_label = context_data['trend']['us_label']
        clean_label = raw_label.split('(')[0].strip() if '(' in raw_label else raw_label

        # 2. Extract Narrative Metrics for Prompt Injection
        us_wow = context_data['narrative_metrics']['us_wow']
        us_drop = context_data['narrative_metrics']['us_drop']
        
        system_prompt = """
You are a Senior Deal Strategist advising the COO. 
Your output is a PRIVATE INTERNAL BRIEFING. 

**CRITICAL DATA RULES (Don't get confused):**
- **The Trend Label is "{clean_label}".** If the raw data shows a different story (e.g. a huge crash), trust the RAW DATA numbers below.
- **Ignore Model Caps:** Do not quote any "1.0%" or "-1.0%" caps associated with the category label. Use the **Real Velocity** stats below.
- **Real Velocity:** US Week-over-Week change is **{us_wow}%**. Drop from peak is **{us_drop}%**. USE THESE NUMBERS.

**NARRATIVE RULES (Tell the Story):**
- **Don't just list numbers.** Give context.
- **Use Time:** "Peak was just {us_weeks_ago} weeks ago."
- **Use Velocity:** "Dropped {us_wow}% week-over-week."
- **Use Shape:** "Exploded to {us_peak} then broke momentum."

**STRATEGY LOGIC:**
- **Aggressive:** "Betting on a rebound."
- **Conservative:** "Pricing to mitigate the active crash."

**OUTPUT FORMAT (Strict HTML):**
1. <b>The Data</b>:
   - US: {us_curr} (Peak: {us_peak})
   - Global: {gl_curr} (Peak: {gl_peak})
2. <b>The Read</b>: 1 blunt sentence on the momentum (e.g. "Viral launch that hit a wall 2 weeks ago").
3. <b>The Playbook</b>:
   - <b>Open</b>: ${conservative}
   - <b>Target</b>: ${target} ({strategy})
   - <b>Rationale</b>: (Why this price? Cite the actual {us_wow}% drop if relevant).
4. <b>The Leverage</b>: Identify the weakness (e.g. "Broken momentum," "Empty catalog").
5. <b>The Ammo (Internal Only)</b>:
   - (Stat 1: US Narrative - e.g. "Down {us_drop}% from peak just {us_weeks_ago} weeks ago").
   - (Stat 2: Global Narrative - e.g. "Momentum broken; dropping {gl_wow}% week-over-week").
   - <b>Strategy</b>: (One phrase tactical instruction).
6. <b>COO Verdict</b>:
   - "Breakeven in {breakeven} mo." (<b>Grade: Safe/Stretch/Risky</b>).
7. <b>Extra</b>:
   - Give a 3 sentence analysis after analyzing all available data.
"""

        # Format System Prompt with Narrative Data
        system_prompt = system_prompt.format(
            clean_label=clean_label, # Inject cleaned label
            strategy=context_data['deal_reality_check']['strategy_name'],
            us_curr=context_data['narrative_metrics']['us_curr'],
            us_peak=context_data['narrative_metrics']['us_peak'],
            us_weeks_ago=context_data['narrative_metrics']['us_weeks_ago'],
            us_drop=context_data['narrative_metrics']['us_drop'],
            us_wow=context_data['narrative_metrics']['us_wow'],
            
            gl_curr=context_data['narrative_metrics']['gl_curr'],
            gl_peak=context_data['narrative_metrics']['gl_peak'],
            gl_weeks_ago=context_data['narrative_metrics']['gl_weeks_ago'],
            gl_drop=context_data['narrative_metrics']['gl_drop'],
            gl_wow=context_data['narrative_metrics']['gl_wow'],

            conservative=f"${context_data['offer_matrix']['conservative']:,.0f}",
            target=f"${context_data['deal_reality_check']['selected_advance']:,.0f}",
            breakeven=f"{context_data['deal_reality_check']['label_breakeven_months']:.1f}"
        )

        user_prompt = f"""
**Case Data:**
- Selected Strategy: {context_data['deal_reality_check']['strategy_name']}
- US Trend Category: {clean_label} (Peak {context_data['narrative_metrics']['us_weeks_ago']} wks ago)
- US WoW Change: {context_data['narrative_metrics']['us_wow']}%
- US Drop from Peak: {context_data['narrative_metrics']['us_drop']}%
"""
        
        response = client.chat.completions.create(
            model="gpt-5.2", 
            temperature=0.6,
            max_completion_tokens=1000, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response.choices[0].message.content.replace("```html", "").replace("```", "").strip()

    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

# -----------------------------------------------------------------------------
# SIDEBAR - INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Deal Inputs")
    
    # NEW: Mode Selection
    # Get default index for radio button
    mode_options = ["Simple (Manual)", "Advanced (Auto-Detect)"]
    default_mode = get_default("input_mode", "Simple (Manual)")
    mode_index = mode_options.index(default_mode) if default_mode in mode_options else 0
    
    input_mode = st.radio("Analysis Mode", mode_options, index=mode_index)
    
    # --- MODEL SELECTION TOGGLE ---
    use_beta = st.toggle("Use Beta Model (Dynamic)", value=False)
    
    st.markdown("---")
    
    # Initialize variables to ensure scope
    us_streams = 0
    global_streams = 0
    us_trend_sel = "Stable (0%)"
    ex_us_trend_sel = "Stable (0%)"
    us_trend_detected = "Stable (0%)" # Ensure variable exists for display scope
    ex_us_trend_detected = "Stable (0%)" # Ensure variable exists for display scope
    us_history = []
    global_history = []
    
    if input_mode == "Simple (Manual)":
        st.subheader("Streaming Volume")
        us_streams = st.number_input("US Weekly Streams", min_value=0, value=get_default("us_streams_manual", 500000), step=10000)
        global_streams = st.number_input("Global Weekly Streams", min_value=0, value=get_default("global_streams_manual", 1000000), step=10000)
        
        st.subheader("Trend Projections")
        # Get indexes for selectboxes
        trend_keys = list(TREND_MAP.keys())
        
        def_us_trend = get_default("us_trend_manual", "Stable (0%)")
        us_trend_idx = trend_keys.index(def_us_trend) if def_us_trend in trend_keys else 0
        
        def_ex_trend = get_default("ex_us_trend_manual", "Stable (0%)")
        ex_trend_idx = trend_keys.index(def_ex_trend) if def_ex_trend in trend_keys else 0
        
        us_trend_sel = st.selectbox("US Trend", trend_keys, index=us_trend_idx)
        ex_us_trend_sel = st.selectbox("Int'l Trend", trend_keys, index=ex_trend_idx)
        
        run_analysis = True # Always run in manual mode

    else: # Advanced Mode
        st.subheader("Historical Data (8 Weeks)")
        st.caption("Paste numbers (separated by Space or Newline). Commas are ignored.")
        
        us_input_raw = st.text_area("US Streams History", get_default("us_history_txt", "71000 72000 71500 73000 74000 74500 75000 76000"))
        global_input_raw = st.text_area("Global Streams History", get_default("global_history_txt", "150000 152000 151000 153000 155000 156000 158000 160000"))
        
        # --- STATE MANAGEMENT FIX ---
        if "analysis_complete" not in st.session_state:
            st.session_state.analysis_complete = False

        # When clicked, set the state to True
        if st.button("Run Analysis", type="primary"):
            st.session_state.analysis_complete = True
            st.session_state.force_reset_trends = True # NEW: Signal to reset overrides to detection
            
        # Use session state to determine if we run, not just the button return value
        if st.session_state.analysis_complete:
            us_history = parse_input_data(us_input_raw)
            global_history = parse_input_data(global_input_raw)
            
            if us_history and global_history:
                # 1. Get Current Volume (Last week of data)
                us_streams = us_history[-1]
                global_streams = global_history[-1]
                
                # 2. Auto-Detect Trends (Using Classic Logic for Dropdowns)
                us_trend_detected, us_pct = analyze_trend_classic(us_history)
                ex_us_history = [g - u for g, u in zip(global_history, us_history)] 
                
                # Safety check for Ex-US negatives
                ex_us_history = [max(0, x) for x in ex_us_history]
                
                ex_us_trend_detected, ex_us_pct = analyze_trend_classic(ex_us_history)
                
                # --- OVERRIDE LOGIC ---
                # If "Run Analysis" was just clicked, force reset the overrides to the new detection
                if st.session_state.get('force_reset_trends', False):
                    st.session_state.us_trend_override = us_trend_detected
                    st.session_state.ex_us_trend_override = ex_us_trend_detected
                    st.session_state.force_reset_trends = False # Reset flag so manual changes stick next time
                
                # If overrides don't exist yet (e.g. first run), set them
                elif 'us_trend_override' not in st.session_state:
                    st.session_state.us_trend_override = us_trend_detected
                    st.session_state.ex_us_trend_override = ex_us_trend_detected
                
                # Set the ACTIVE trend selection to the OVERRIDE value (which might be the same as detection)
                us_trend_sel = st.session_state.us_trend_override
                ex_us_trend_sel = st.session_state.ex_us_trend_override
                
                run_analysis = True
            else:
                st.error("Please enter valid numbers separated by spaces or newlines.")
                run_analysis = False
        else:
            run_analysis = False

    st.subheader("Deal Structure")
    artist_share_pct = st.slider("Artist Share %", 10, 100, get_default("artist_share", 60), 5)
    artist_share = artist_share_pct / 100.0

    st.markdown("---")
    
    # NEW: OpenAI API Input
    st.subheader("🤖 AI Advisor")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter key to unlock AI strategies.")
    
    st.markdown("---")
    
    # -------------------------------------------------------------------------
    # SHARE FUNCTIONALITY
    # -------------------------------------------------------------------------
    st.subheader("🔗 Share Analysis")
    if st.button("Generate Share Link"):
        # Gather all current inputs
        share_payload = {
            "input_mode": input_mode,
            "artist_share": artist_share_pct,
            # Manual Mode Data
            "us_streams_manual": us_streams if input_mode == "Simple (Manual)" else 500000,
            "global_streams_manual": global_streams if input_mode == "Simple (Manual)" else 1000000,
            "us_trend_manual": us_trend_sel if input_mode == "Simple (Manual)" else "Stable (0%)",
            "ex_us_trend_manual": ex_us_trend_sel if input_mode == "Simple (Manual)" else "Stable (0%)",
            # Advanced Mode Data
            "us_history_txt": us_input_raw if input_mode != "Simple (Manual)" else "",
            "global_history_txt": global_input_raw if input_mode != "Simple (Manual)" else "",
            # Section 3: Deal Reality Check
            "selected_reality_check": st.session_state.get("reality_check_picker", get_default("selected_reality_check", "Target Ceiling (Ceiling Gross)"))
        }
        
        # Encode (Using URL-Safe Base64)
        json_str = json.dumps(share_payload)
        b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
        
        # 1. Update the URL in browser (if supported by environment)
        try:
            st.query_params["data"] = b64_str
        except:
            pass
            
        # 2. Show the link manually
        share_url = f"?data={b64_str}"
        
        st.success("Link generated!")
        st.markdown(f"### [🔗 Open Shareable Link]({share_url})")
        st.info("Right-click the link above and select 'Copy Link Address' to share it with others.")
        st.caption("Raw Code (if needed):")
        st.code(share_url, language="text")

    st.caption("Deal Analyzer v2.4 | AI Powered")

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
    st.markdown("### 📈 Historical Analysis")
    
    # Show Charts
    chart_data = pd.DataFrame({
        "US Streams": us_history,
        "Global Streams": global_history
    })
    st.line_chart(chart_data)
    
    # NEW LOGIC: CRASH DETECTION ALERTS (Based on Active/Selected Trend)
    # Alerts logic now uses the active selection (us_trend_sel) to reflect overrides
    if "Falling Knife" in us_trend_sel or "Falling Knife" in ex_us_trend_sel:
        st.error("⚠️ CRASH DETECTED: 'Falling Knife' pattern found. Recent volume has dropped significantly (>20%) below average. Proceed with extreme caution.")

    if "New Artist" in us_trend_sel or "New Artist" in ex_us_trend_sel:
        st.warning("⚠️ VOLATILITY WARNING: New Artist / Cooling pattern detected. Early growth spurts are retracing. Projections are capped.")

    # Show Insights
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class="insight-box">
            <strong>US Active Trend:</strong><br>
            <span style="font-size: 1.2em; color: #00ffbf">{us_trend_sel}</span>
        </div>
        """, unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="insight-box">
            <strong>Int'l Active Trend:</strong><br>
            <span style="font-size: 1.2em; color: #00ffbf">{ex_us_trend_sel}</span>
        </div>
        """, unsafe_allow_html=True)

    # NEW: Audit / Override Section
    with st.expander("⚙️ Audit / Override Trend Settings", expanded=False):
        st.caption("The model auto-detected the trends below. You can manually override them here if you disagree.")
        
        st.markdown(f"**Model Detection:** US=`{us_trend_detected}` | Int'l=`{ex_us_trend_detected}`")
        
        ao1, ao2 = st.columns(2)
        with ao1:
            st.selectbox("Override US Trend", options=list(TREND_MAP.keys()), key="us_trend_override")
        with ao2:
            st.selectbox("Override Int'l Trend", options=list(TREND_MAP.keys()), key="ex_us_trend_override")


# -----------------------------------------------------------------------------
# CORE CALCULATION (Engine)
# -----------------------------------------------------------------------------
st.markdown("### Valuation & Recoupment Model")

if use_beta:
    # --- BETA MODEL (Dynamic Momentum) ---
    st.caption(f"Status: **Beta Model Active** (Slope-based Dynamic Projection)")
    
    # Analyze Trend (Beta)
    # We use the US history as the driver for the slope in this Beta implementation
    # If no history (Manual mode), this defaults to flat/stable
    us_trend_data = analyze_trend_beta(us_history)
    
    # Project Revenue (Beta)
    # Beta model generates a single dynamic trajectory
    beta_gross, beta_last_wk, beta_stream = project_revenue_beta(us_streams, ex_us_streams, us_trend_data)
    
    # Map Beta results to Floor/Ceiling variables to maintain downstream compatibility
    # The Beta model currently provides one "Most Likely" scenario
    floor_gross = beta_gross
    ceil_gross = beta_gross
    
    floor_last_wk = beta_last_wk
    ceil_last_wk = beta_last_wk
    
    floor_stream = beta_stream
    ceil_stream = beta_stream
    
    st.info(f"Beta Logic: Slope {us_trend_data['slope']:.4f} | Status: {us_trend_data['status']}")

else:
    # --- CLASSIC MODEL (Multiplier Map) ---
    # 1. Determine Multipliers
    us_mult_floor = get_scenario_multipliers(us_trend_sel, is_ceiling=False)
    ex_us_mult_floor = get_scenario_multipliers(ex_us_trend_sel, is_ceiling=False)

    us_mult_ceil = get_scenario_multipliers(us_trend_sel, is_ceiling=True)
    ex_us_mult_ceil = get_scenario_multipliers(ex_us_trend_sel, is_ceiling=True)

    # 2. Run Projections
    floor_gross, floor_last_wk, floor_stream = project_revenue_classic(us_streams, ex_us_streams, us_mult_floor, ex_us_mult_floor)
    ceil_gross, ceil_last_wk, ceil_stream = project_revenue_classic(us_streams, ex_us_streams, us_mult_ceil, ex_us_mult_ceil)

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

# Determine index for the selectbox based on loaded defaults
default_option = get_default("selected_reality_check", "Target Ceiling (Ceiling Gross)")
options_list = list(recoup_options.keys())
try:
    default_index = options_list.index(default_option)
except ValueError:
    default_index = 2 # Default to Target Ceiling if match not found

# Create columns for the dropdown
rc1, rc2 = st.columns([2, 1])
with rc1:
    selected_option = st.selectbox(
        "Select Proposed Advance for Analysis:", 
        options=options_list, 
        index=default_index,  # Use calculated index
        key="reality_check_picker" # Key allows access in sidebar logic
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

# Default Static HTML (Preserved for fallback)
strategy_html = f"""
<div class="strategy-box">
    <strong>NEGOTIATION SCRIPT:</strong><br><br>
    "Based on the analysis of the last 8 weeks, we see a <strong>{us_trend_sel}</strong> trend."<br><br>
    1. <strong>Anchor:</strong> Start at <strong>${conservative_offer:,.0f}</strong>.<br>
    2. <strong>Rationalize:</strong> Our risk-adjusted floor is <strong>${floor_gross:,.0f}</strong> given market volatility.<br>
    3. <strong>Close:</strong> If we stretch to <strong>${ceil_gross:,.0f}</strong>, the artist won't see royalties for <strong>{art_recoup_mo:.1f} months</strong>."
</div>
"""

if openai_api_key:
    # -------------------------------------------------------------------------
    # 1. NARRATIVE METRIC HELPERS
    # -------------------------------------------------------------------------
    def get_peak_context(data):
        """Returns (Peak Value, Weeks Ago)"""
        real_data = [x for x in data if x > 0]
        if not real_data: return 0, 0
        peak = max(real_data)
        # Find index of peak in the ORIGINAL list (handling 0s correctly)
        # We search from the end to find the most recent peak if ties exist
        indices = [i for i, x in enumerate(data) if x == peak]
        peak_idx = indices[-1] 
        weeks_ago = len(data) - 1 - peak_idx
        return peak, weeks_ago

    def get_wow_change(data):
        """Returns % change from Last Week to This Week"""
        if len(data) < 2: return 0.0
        curr = data[-1]
        prev = data[-2]
        if prev == 0: return 0.0
        return (curr - prev) / prev

    def get_drop_from_peak(current, peak):
        if peak == 0: return 0.0
        return (current - peak) / peak

    # UPDATED: Smart Growth Calculator (Handles New Artists/Zeros)
    def get_smart_growth(data):
        # Filter out the padded zeros to find true history
        real_data = [x for x in data if x > 0]
        # Need at least 2 points to calculate a trend
        if len(real_data) < 2: 
            return 0.0
        # Compare Current Week vs. The First Week they appeared
        start_val = real_data[0]
        current_val = real_data[-1]
        return (current_val - start_val) / start_val

    try:
        with st.spinner("✨ Analyzing Deal Data with AI Consultant..."):
            # -----------------------------------------------------------------
            # 2. CALCULATE NARRATIVE METRICS
            # -----------------------------------------------------------------
            # US Context
            us_curr = us_history[-1]
            us_peak, us_weeks_ago = get_peak_context(us_history)
            us_drop = get_drop_from_peak(us_curr, us_peak)
            us_wow = get_wow_change(us_history)
            # Use smart growth helper here:
            us_growth = get_smart_growth(us_history)

            # Global Context
            gl_curr = global_history[-1]
            gl_peak, gl_weeks_ago = get_peak_context(global_history)
            gl_drop = get_drop_from_peak(gl_curr, gl_peak)
            gl_wow = get_wow_change(global_history)
            # Use smart growth helper here:
            gl_growth = get_smart_growth(global_history)

            # -----------------------------------------------------------------
            # 3. AI GENERATION
            # -----------------------------------------------------------------
            # Full context data for clarity, though narrative helpers do heavy lifting
            context_data = {
                "narrative_metrics": {
                    "us_curr": f"{us_curr:,.0f}",
                    "us_peak": f"{us_peak:,.0f}",
                    "us_weeks_ago": us_weeks_ago,
                    "us_drop": f"{us_drop*100:.1f}",
                    "us_wow": f"{us_wow*100:.1f}",
                    "gl_curr": f"{gl_curr:,.0f}",
                    "gl_peak": f"{gl_peak:,.0f}",
                    "gl_weeks_ago": gl_weeks_ago,
                    "gl_drop": f"{gl_drop*100:.1f}",
                    "gl_wow": f"{gl_wow*100:.1f}"
                },
                "raw_metrics": {
                    "us_growth_pct": us_growth,
                    "global_growth_pct": gl_growth
                },
                "trend": {
                    "us_label": us_trend_sel,
                    "intl_label": ex_us_trend_sel,
                },
                "valuation": {
                    "floor": float(floor_gross),
                    "ceiling": float(ceil_gross),
                },
                "offer_matrix": {
                    "conservative": float(conservative_offer),
                },
                "deal_reality_check": {
                    "strategy_name": strategy_name,
                    "selected_advance": float(selected_advance),
                    "label_breakeven_months": float(lbl_recoup_mo),
                    "artist_recoup_months": float(art_recoup_mo),
                }
            }
            
            ai_output = generate_ai_strategy(openai_api_key, context_data)
            
            # Clean up potential markdown artifacts
            clean_output = ai_output.replace("```html", "").replace("```", "").strip()
            
            st.markdown(f"""
            <div class="strategy-box" style="border-left: 5px solid #7c4dff;">
                <strong>🤖 AI EXECUTIVE ADVISOR:</strong><br><br>
                {clean_output.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        st.markdown(strategy_html, unsafe_allow_html=True)
else:
    # Fallback to existing static script
    st.markdown(strategy_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PHASE 2: SAVE DEAL SNAPSHOT
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.header("📌 Save Deal Snapshot")
    
    with st.form("save_deal_form"):
        deal_id_input = st.text_input("Deal ID", help="Unique identifier for this deal")
        artist_project_input = st.text_input("Artist / Project")
        date_signed = st.date_input("Date Signed")
        forecast_start = st.date_input("Forecast Start Date")
        executed_advance = st.number_input("Executed Advance ($)", min_value=0.0, step=1000.0)
        
        submitted = st.form_submit_button("Save Snapshot to Tracker")
        
        if submitted:
            # Strip whitespace
            deal_id = deal_id_input.strip()
            artist_name = artist_project_input.strip()

            # 1. Validation
            if not deal_id or not artist_name:
                st.error("Deal ID and Artist/Project are required.")
            else:
                try:
                    # 2. Setup GSheets Auth
                    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
                    # Load secrets
                    if "gcp_service_account" not in st.secrets or "deal_tracker_sheet_id" not in st.secrets:
                        st.error("Missing Google Sheets secrets.")
                    else:
                        creds_dict = dict(st.secrets["gcp_service_account"])
                        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
                        client = gspread.authorize(creds)
                        
                        # 3. Open Sheet
                        sheet = client.open_by_key(st.secrets["deal_tracker_sheet_id"])
                        try:
                            worksheet = sheet.worksheet("DEALS")
                        except gspread.WorksheetNotFound:
                            st.error("Worksheet 'DEALS' not found.")
                            worksheet = None

                        if worksheet:
                            # 4. Check Duplicates
                            existing_ids = worksheet.col_values(1) # Column A
                            if str(deal_id) in existing_ids:
                                st.error(f"Deal ID '{deal_id}' already exists.")
                            else:
                                # 5. Prepare Row
                                # Data: Deal ID, Artist, Signed, Start, Exec Adv, Floor, Ceiling, Strategy, Sel Adv, Breakeven
                                row_data = [
                                    str(deal_id),
                                    str(artist_name),
                                    str(date_signed),
                                    str(forecast_start),
                                    float(executed_advance),
                                    float(floor_gross),
                                    float(ceil_gross),
                                    str(selected_option),
                                    float(selected_advance),
                                    round(float(lbl_recoup_mo), 1)
                                ]
                                
                                worksheet.append_row(row_data)
                                st.success("Snapshot saved to DEALS tracker!")
                except Exception as e:
                    st.error(f"Error saving to tracker: {str(e)}")
