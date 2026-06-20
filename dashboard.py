"""
Interactive Streamlit Dashboard for Parking Congestion Analysis
Run with: streamlit run dashboard.py
"""

import asyncio
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from streamlit_folium import st_folium
from map_component import build_map

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Parking Congestion Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card { background-color:#f0f2f6; padding:20px; border-radius:10px; margin:10px 0; }
    .header-title { color:#1f77b4; font-size:2.5em; font-weight:bold; margin-bottom:10px; }
    .zone-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color:white; padding:20px; border-radius:10px; margin:10px 0;
    }
    .severity-low    { background: linear-gradient(135deg, #56ab2f, #a8e063); color:white; padding:20px; border-radius:10px; margin:10px 0; }
    .severity-medium { background: linear-gradient(135deg, #f7971e, #ffd200); color:white; padding:20px; border-radius:10px; margin:10px 0; }
    .severity-high   { background: linear-gradient(135deg, #f85032, #e73827); color:white; padding:20px; border-radius:10px; margin:10px 0; }
    .severity-critical { background: linear-gradient(135deg, #4b0000, #8b0000); color:white; padding:20px; border-radius:10px; margin:10px 0; }
    .demo-banner {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color:white; padding:15px 20px; border-radius:10px;
        border-left: 4px solid #00d4ff; margin-bottom:15px;
    }
    </style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'api_url'         not in st.session_state: st.session_state.api_url         = 'http://localhost:5000'
if 'prediction_pin'  not in st.session_state: st.session_state.prediction_pin  = None
if 'prediction_hist' not in st.session_state: st.session_state.prediction_hist = []
if 'active_tab'      not in st.session_state: st.session_state.active_tab      = 0
if 'demo_payload'    not in st.session_state: st.session_state.demo_payload     = None

HOTSPOT_ZONES = {
    'Zone 5': {'station':'Upparpet',       'violations':123778, 'high_severity':31.1, 'resources':'8 patrol units', 'color':'#FF6B6B', 'severity_class':'severity-high'},
    'Zone 0': {'station':'Malleshwaram',   'violations':69076,  'high_severity':31.1, 'resources':'5 patrol units', 'color':'#4ECDC4', 'severity_class':'severity-high'},
    'Zone 1': {'station':'HAL Old Airport','violations':48600,  'high_severity':35.4, 'resources':'4 patrol units', 'color':'#45B7D1', 'severity_class':'severity-high'},
    'Zone 2': {'station':'HSR Layout',     'violations':25154,  'high_severity':31.7, 'resources':'2 patrol units', 'color':'#FFA07A', 'severity_class':'severity-medium'},
    'Zone 4': {'station':'Kodigehalli',    'violations':21920,  'high_severity':30.7, 'resources':'1 patrol unit',  'color':'#98D8C8', 'severity_class':'severity-medium'},
    'Zone 3': {'station':'Chikkajala',     'violations':6881,   'high_severity':34.3, 'resources':'1 patrol unit',  'color':'#F7DC6F', 'severity_class':'severity-low'},
}

DAY_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

DEMO_SCENARIOS = {
    "🔴 High Risk — Upparpet Peak Hour": {
        'latitude': 12.9767, 'longitude': 77.5713,
        'cell_violations': 120, 'cell_density': 0.9,
        'hour': 18, 'day_of_week': 4,
        'is_peak_hour': True, 'has_wrong_parking': True,
        'has_no_parking': True, 'has_main_road': True,
        'label': 'Friday 6PM rush, Upparpet main road'
    },
    "🟡 Moderate — HSR Layout Midday": {
        'latitude': 12.9116, 'longitude': 77.6473,
        'cell_violations': 20, 'cell_density': 0.4,
        'hour': 13, 'day_of_week': 2,
        'is_peak_hour': False, 'has_wrong_parking': True,
        'has_no_parking': False, 'has_main_road': False,
        'label': 'Wednesday 1PM, residential area'
    },
    "🟢 Low Risk — Chikkajala Off-Peak": {
        'latitude': 13.2580, 'longitude': 77.5450,
        'cell_violations': 3, 'cell_density': 0.1,
        'hour': 11, 'day_of_week': 6,
        'is_peak_hour': False, 'has_wrong_parking': False,
        'has_no_parking': False, 'has_main_road': False,
        'label': 'Saturday 11AM, low-density area'
    },
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-title">🚗 Parking Congestion Analytics</div>', unsafe_allow_html=True)
st.markdown("*Data-Driven Enforcement Strategy for Bangalore*")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    stab1, stab2 = st.tabs(["API Settings", "About"])

    with stab1:
        api_url = st.text_input("API Base URL", value=st.session_state.api_url)
        st.session_state.api_url = api_url

        if st.button("🔗 Test Connection"):
            try:
                r = requests.get(f"{api_url}/health", timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('models_loaded'):
                        st.success("✓ API Connected — all models loaded")
                        ts = data.get('timestamp','')[:19].replace('T',' ')
                        st.caption(f"Last checked: {ts}")
                    else:
                        st.warning("⚠ API connected but models not fully loaded")
                else:
                    st.error(f"✗ API returned {r.status_code}")
            except Exception:
                st.error(f"✗ Cannot connect to {api_url}")

    with stab2:
        st.markdown("""
        ### About
        Real-time parking violation analysis for Bangalore.

        **Models:**
        - Gradient Boosting — Severity prediction
        - Random Forest — Risk classification
        - K-Means — Hotspot zone detection

        **Stack:** Flask · Streamlit · Folium · scikit-learn
        """)

# ── Tab order: Prediction + Map first ────────────────────────────────────────
tab_dash, tab_pred, tab_map, tab_zone, tab_analytics = st.tabs([
    "📊 Dashboard", "🎯 Prediction", "🗺️ Map", "📍 Zone Analysis", "📈 Analytics"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION  (now first)
# ═════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.header("🎯 Real-Time Prediction")

    # ── Demo scenario launcher ────────────────────────────────────────────────
    st.markdown('<div class="demo-banner">💡 <strong>Quick Demo</strong> — pick a scenario to auto-fill and run instantly</div>',
                unsafe_allow_html=True)

    demo_cols = st.columns(3)
    for i, (label, scenario) in enumerate(DEMO_SCENARIOS.items()):
        with demo_cols[i]:
            if st.button(label, use_container_width=True, key=f"demo_{i}"):
                st.session_state.demo_payload = scenario

    st.divider()

    # Pull demo values if a scenario was selected
    demo = st.session_state.demo_payload or {}

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        if demo:
            st.info(f"📋 Demo loaded: *{demo.get('label','')}*")

        latitude  = st.slider("Latitude",    12.5, 13.5, float(demo.get('latitude',  13.05)), 0.01)
        longitude = st.slider("Longitude",   77.3, 77.9, float(demo.get('longitude', 77.60)), 0.01)
        hour      = st.slider("Hour of Day", 0, 23,      int(demo.get('hour', 18)))

        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_of_week = st.selectbox("Day of Week", days,
                                   index=int(demo.get('day_of_week', 4)))
        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            cell_violations   = st.number_input("Reported Violations (area)", 0, 200,
                                                int(demo.get('cell_violations', 15)))
            is_peak_hour      = st.checkbox("Is Peak Hour",         value=bool(demo.get('is_peak_hour',      True)))
            has_no_parking    = st.checkbox("No Parking Zone",       value=bool(demo.get('has_no_parking',    False)))
        with col_b:
            cell_density      = st.slider("Violation Density", 0.0, 1.0,
                                          float(demo.get('cell_density', 0.5)), 0.05)
            has_wrong_parking = st.checkbox("Wrong Parking Present", value=bool(demo.get('has_wrong_parking', True)))
            has_main_road     = st.checkbox("Main Road",             value=bool(demo.get('has_main_road',     False)))

    with col2:
        st.subheader("Predictions")
        run_now = bool(demo) and st.session_state.get('_demo_just_loaded') != id(demo)

        predict_clicked = st.button("🔮 Get Predictions", use_container_width=True)

        if predict_clicked or run_now:
            if run_now:
                st.session_state['_demo_just_loaded'] = id(demo)

            payload = {
                'latitude':          latitude,
                'longitude':         longitude,
                'cell_violations':   int(cell_violations),
                'cell_density':      cell_density,
                'hour':              int(hour),
                'day_of_week':       DAY_MAP[days[day_of_week]] if isinstance(day_of_week, int) else DAY_MAP[day_of_week],
                'is_peak_hour':      int(is_peak_hour),
                'has_wrong_parking': int(has_wrong_parking),
                'has_no_parking':    int(has_no_parking),
                'has_main_road':     int(has_main_road),
            }

            try:
                with st.spinner("Running models..."):
                    r1 = requests.post(f"{st.session_state.api_url}/predict/severity",
                                       json=payload, timeout=10)
                    r2 = requests.post(f"{st.session_state.api_url}/predict/risk",
                                       json=payload, timeout=10)

                if r1.status_code == 200 and r2.status_code == 200:
                    sev  = r1.json()
                    risk = r2.json()

                    pin = {
                        'lat':              latitude,
                        'lon':              longitude,
                        'severity_score':   sev['severity_score'],
                        'severity_level':   sev['severity_level'],
                        'zone':             sev['zone'],
                        'station':          sev['station'],
                        'is_high_risk':     risk['is_high_risk'],
                        'risk_probability': risk['risk_probability'],
                    }
                    st.session_state.prediction_pin = pin

                    # Add to history (keep last 5)
                    st.session_state.prediction_hist.insert(0, {
                        'Time':      datetime.now().strftime('%H:%M:%S'),
                        'Zone':      sev['zone'],
                        'Station':   sev['station'],
                        'Severity':  f"{sev['severity_score']:.2f}",
                        'Level':     sev['severity_level'],
                        'Risk':      f"{risk['risk_probability']*100:.1f}%",
                        'Action':    risk['recommendation'],
                    })
                    st.session_state.prediction_hist = st.session_state.prediction_hist[:5]

                    st.success("✓ Predictions Generated — pin dropped on Map tab")
                    st.markdown(f"**Location:** {latitude:.4f}, {longitude:.4f}")
                    st.markdown(f"**Zone:** {sev['zone']} ({sev['station']})")
                    st.divider()

                    rc1, rc2 = st.columns(2)

                    with rc1:
                        st.markdown("### Severity")
                        severity_score = sev['severity_score']
                        severity_level = sev['severity_level']
                        color = {'Low':'green','Medium':'orange',
                                 'High':'red','Critical':'darkred'}.get(severity_level,'gray')
                        st.metric("Score", f"{severity_score:.2f} / 5", severity_level)

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=severity_score,
                            domain={'x':[0,1],'y':[0,1]},
                            title={'text':"Severity Level"},
                            gauge={
                                'axis':{'range':[0,5]},
                                'bar':{'color':color},
                                'steps':[
                                    {'range':[0,2],'color':"#90EE90"},
                                    {'range':[2,3],'color':"#FFD700"},
                                    {'range':[3,4],'color':"#FF6347"},
                                    {'range':[4,5],'color':"#8B0000"},
                                ],
                                'threshold':{'line':{'color':'red','width':4},
                                             'thickness':0.75,'value':4}
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                    with rc2:
                        st.markdown("### Risk")
                        risk_prob = risk['risk_probability']
                        if risk['is_high_risk']:
                            st.error("🔴 HIGH RISK ZONE")
                        else:
                            st.info("🟡 MODERATE RISK")
                        st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
                        st.markdown(f"**Action:** {risk['recommendation']}")

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=risk_prob * 100,
                            domain={'x':[0,1],'y':[0,1]},
                            title={'text':"Risk %"},
                            gauge={
                                'axis':{'range':[0,100]},
                                'bar':{'color':'darkblue'},
                                'steps':[
                                    {'range':[0,25],  'color':"#90EE90"},
                                    {'range':[25,50], 'color':"#FFD700"},
                                    {'range':[50,75], 'color':"#FF6347"},
                                    {'range':[75,100],'color':"#8B0000"},
                                ],
                                'threshold':{'line':{'color':'red','width':4},
                                             'thickness':0.75,'value':75}
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error {r1.status_code} / {r2.status_code}: {r1.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API.")
            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

        # ── Prediction history ────────────────────────────────────────────────
        if st.session_state.prediction_hist:
            st.divider()
            st.markdown("### 🕐 Recent Predictions")
            st.dataframe(
                pd.DataFrame(st.session_state.prediction_hist),
                use_container_width=True, hide_index=True
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAP  (now second, directly after Prediction)
# ═════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.header("🗺️ Bangalore Parking Hotspot Map")
    st.markdown(
        "Zone circles are sized by violation count and coloured by severity. "
        "Click any zone for stats. Predictions from the 🎯 tab appear as pins here automatically."
    )

    pin = st.session_state.prediction_pin

    if pin:
        level_emoji = {'Low':'🟢','Medium':'🟡','High':'🔴','Critical':'⛔'}.get(pin['severity_level'],'📍')
        st.success(
            f"{level_emoji} **Active Pin** — {pin['zone']} / {pin['station']} · "
            f"Severity {pin['severity_score']:.2f} ({pin['severity_level']}) · "
            f"Risk {pin['risk_probability']*100:.1f}%"
        )
        if st.button("🗑 Clear Pin"):
            st.session_state.prediction_pin = None
            st.rerun()
    else:
        st.caption("No prediction pin yet — run a prediction in the 🎯 tab to drop one here.")

    # Build map fresh every time using current session state
    # key is stable so st_folium doesn't flicker, but map object always has latest pin
    with st.spinner("Loading zone data from model..."):
        folium_map = build_map(st.session_state.api_url, st.session_state.prediction_pin)

    map_data = st_folium(
        folium_map,
        width="100%",
        height=580,
        returned_objects=["last_clicked"],
        key="bangalore_map_stable",   # stable key — no cache-bust flicker
    )

    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.info(
            f"📌 Clicked: lat **{clicked['lat']:.5f}**, lon **{clicked['lng']:.5f}** "
            "— switch to 🎯 Prediction and paste these coordinates to predict here"
        )

    st.markdown("### Zone Statistics (live from model)")
    try:
        r = requests.get(f"{st.session_state.api_url}/analytics/map", timeout=10)
        if r.status_code == 200:
            zones_api = r.json().get('zones', [])
            ts = r.json().get('timestamp','')[:19].replace('T',' ')
            st.caption(f"Source: KMeans centroids + GBR predictions · Updated: {ts}")
            tbl = pd.DataFrame([{
                'Zone':            z['zone'],
                'Station':         z['station'],
                'Violations':      f"{z['violation_count']:,}",
                'Avg Severity':    f"{z['avg_severity']:.2f} / 5",
                'High Severity %': f"{z['high_severity_pct']:.1f}%",
                'Peak Hour %':     f"{z['peak_hour_pct']:.1f}%",
                'Centroid':        f"{z['centroid_lat']:.4f}, {z['centroid_lon']:.4f}",
            } for z in zones_api])
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not load live zone stats.")
    except Exception:
        st.warning("API not reachable — live zone table unavailable.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.header("Real-Time Dashboard")

    # Fetch live summary from API
    try:
        r = requests.get(f"{st.session_state.api_url}/analytics/map", timeout=8)
        if r.status_code == 200:
            zones_live = r.json().get('zones', [])
            total_violations = sum(z['violation_count'] for z in zones_live)
            avg_sev_all      = np.mean([z['avg_severity'] for z in zones_live])
            high_sev_zones   = sum(1 for z in zones_live if z['avg_severity'] >= 4)
            live_data        = True
        else:
            live_data = False
    except Exception:
        live_data = False

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Violations",    f"{total_violations:,}" if live_data else "295,609",
                "Live" if live_data else "Cached", delta_color="off")
    col2.metric("High-Risk Zones",     str(high_sev_zones) if live_data else "6")
    col3.metric("Model Accuracy (R²)", "0.9416", "+0.5%")
    col4.metric("Patrol Units",        "21",     "Optimised")

    if live_data:
        st.caption("✓ Dashboard metrics loaded live from model")
    else:
        st.caption("⚠ Using cached metrics — API not reachable")

    st.divider()
    st.subheader("🔴 Hotspot Zones Overview")

    cols = st.columns(3)
    for idx, (zone_name, zi) in enumerate(HOTSPOT_ZONES.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="{zi['severity_class']}">
            <h4>{zone_name}</h4>
            <p><strong>Station:</strong> {zi['station']}</p>
            <p><strong>Violations:</strong> {zi['violations']:,}</p>
            <p><strong>High Severity:</strong> {zi['high_severity']}%</p>
            <p><strong>Resources:</strong> {zi['resources']}</p>
            </div>""", unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            x=list(HOTSPOT_ZONES.keys()),
            y=[HOTSPOT_ZONES[z]['violations'] for z in HOTSPOT_ZONES],
            title="Violations by Zone",
            labels={'x':'Zone','y':'Violations'},
            color=[HOTSPOT_ZONES[z]['violations'] for z in HOTSPOT_ZONES],
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            x=list(HOTSPOT_ZONES.keys()),
            y=[HOTSPOT_ZONES[z]['high_severity'] for z in HOTSPOT_ZONES],
            title="High Severity % by Zone",
            labels={'x':'Zone','y':'High Severity %'},
            color=[HOTSPOT_ZONES[z]['high_severity'] for z in HOTSPOT_ZONES],
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ZONE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_zone:
    st.header("📍 Zone-Based Analysis")

    selected_zone = st.selectbox("Select Zone", list(HOTSPOT_ZONES.keys()))
    zi = HOTSPOT_ZONES[selected_zone]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Police Station",   zi['station'])
    col2.metric("Total Violations", f"{zi['violations']:,}")
    col3.metric("High Severity %",  f"{zi['high_severity']}%")
    col4.metric("Resources",        zi['resources'])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zone Locations")
        try:
            r = requests.get(f"{st.session_state.api_url}/analytics/map", timeout=10)
            if r.status_code == 200:
                zones_api = r.json().get('zones', [])
                map_df = pd.DataFrame([{
                    'Zone':       z['zone'],
                    'Violations': z['violation_count'],
                    'Latitude':   z['centroid_lat'],
                    'Longitude':  z['centroid_lon'],
                } for z in zones_api])
                st.caption("✓ Centroids from KMeans model")
            else:
                raise Exception()
        except Exception:
            map_df = pd.DataFrame({
                'Zone':       list(HOTSPOT_ZONES.keys()),
                'Violations': [HOTSPOT_ZONES[z]['violations'] for z in HOTSPOT_ZONES],
                'Latitude':   [12.99, 13.01, 12.93, 12.93, 13.07, 13.25],
                'Longitude':  [77.59, 77.55, 77.69, 77.65, 77.585, 77.54],
            })
            st.caption("⚠ Using static fallback centroids")

        fig = px.scatter_mapbox(
            map_df, lat='Latitude', lon='Longitude',
            hover_name='Zone', hover_data='Violations',
            size='Violations', zoom=11, height=400,
            title="Hotspot Zone Centroids"
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Zone Comparison")
        zones_df = pd.DataFrame([
            {'Zone':z,'Violations':HOTSPOT_ZONES[z]['violations'],
             'High Severity %':HOTSPOT_ZONES[z]['high_severity']}
            for z in HOTSPOT_ZONES
        ])
        fig = px.scatter(
            zones_df, x='Violations', y='High Severity %',
            hover_name='Zone', size='Violations',
            color='High Severity %', color_continuous_scale='RdYlGn_r', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("📈 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Metrics")
        st.dataframe(pd.DataFrame({
            'Metric': ['R² Score','RMSE','Precision','Recall','Data Retention'],
            'Value':  ['0.9416','0.1356','0.99','0.89','98.98%'],
            'Status': ['✓ Excellent','✓ Excellent','✓ Excellent','✓ Good','✓ Excellent'],
        }), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Expected Impact (3 Months)")
        st.dataframe(pd.DataFrame({
            'Metric': ['Violations Prevented','Congestion Reduction',
                       'Main Road Flow','Enforcement Efficiency','Compliance Rate'],
            'Target': ['-25%','-20%','-30%','+40%','>80%'],
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Feature Importance — Severity Prediction")

    feat_df = pd.DataFrame({
        'Feature':    ['Reported Violations','Violation Density','Main Road','No Parking Zone',
                       'Hour of Day','Peak Hour','Wrong Parking','Latitude','Longitude','Day of Week'],
        'Importance': [0.28, 0.22, 0.18, 0.14, 0.08, 0.05, 0.02, 0.01, 0.01, 0.01],
    })
    fig = px.bar(
        feat_df.sort_values('Importance'),
        x='Importance', y='Feature', orientation='h',
        title='Feature Importance (GBR model)',
        color='Importance', color_continuous_scale='Blues', height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("*Parking Congestion Analytics · Bangalore · Model-driven enforcement*")