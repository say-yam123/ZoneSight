"""
Interactive Streamlit Dashboard for Parking Congestion Analysis
Run with: streamlit run dashboard.py
"""

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
    </style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if 'api_url' not in st.session_state:
    st.session_state.api_url = 'http://localhost:5000'
if 'prediction_pin' not in st.session_state:
    st.session_state.prediction_pin = None

# ── Static zone display data (violations are training-data facts) ─────────────
HOTSPOT_ZONES = {
    'Zone 5': {'station':'Upparpet',       'violations':123778, 'high_severity':31.1, 'resources':'8 patrol units', 'color':'#FF6B6B'},
    'Zone 0': {'station':'Malleshwaram',   'violations':69076,  'high_severity':31.1, 'resources':'5 patrol units', 'color':'#4ECDC4'},
    'Zone 1': {'station':'HAL Old Airport','violations':48600,  'high_severity':35.4, 'resources':'4 patrol units', 'color':'#45B7D1'},
    'Zone 2': {'station':'HSR Layout',     'violations':25154,  'high_severity':31.7, 'resources':'2 patrol units', 'color':'#FFA07A'},
    'Zone 4': {'station':'Kodigehalli',    'violations':21920,  'high_severity':30.7, 'resources':'1 patrol unit',  'color':'#98D8C8'},
    'Zone 3': {'station':'Chikkajala',     'violations':6881,   'high_severity':34.3, 'resources':'1 patrol unit',  'color':'#F7DC6F'},
}

DAY_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
           "Friday":4,"Saturday":5,"Sunday":6}

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
        """)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", "🎯 Prediction", "📍 Zone Analysis", "📈 Analytics", "🗺️ Map"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Real-Time Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Violations",    "295,609",  "-5.2%",  delta_color="inverse")
    col2.metric("High-Risk Zones",     "6",        "+1.0%")
    col3.metric("Model Accuracy (R²)", "0.9416",   "+0.5%")
    col4.metric("Patrol Units",        "21",       "Optimised")

    st.divider()
    st.subheader("🔴 Hotspot Zones Overview")

    cols = st.columns(3)
    for idx, (zone_name, zi) in enumerate(HOTSPOT_ZONES.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div class="zone-card">
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
# TAB 2 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🎯 Real-Time Prediction")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        latitude    = st.slider("Latitude",    12.5, 13.5, 13.05, 0.01)
        longitude   = st.slider("Longitude",   77.3, 77.9, 77.60, 0.01)
        hour        = st.slider("Hour of Day", 0, 23, 18)
        day_of_week = st.selectbox("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=4)
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            cell_violations   = st.number_input("Cell Violations", 0, 200, 15)
            is_peak_hour      = st.checkbox("Is Peak Hour",          value=True)
            has_no_parking    = st.checkbox("No Parking Zone",        value=False)
        with col_b:
            cell_density      = st.slider("Cell Density", 0.0, 1.0, 0.5, 0.05)
            has_wrong_parking = st.checkbox("Wrong Parking Present",  value=True)
            has_main_road     = st.checkbox("Main Road",              value=False)

    with col2:
        st.subheader("Predictions")

        if st.button("🔮 Get Predictions", use_container_width=True):
            payload = {
                'latitude':          latitude,
                'longitude':         longitude,
                'cell_violations':   int(cell_violations),
                'cell_density':      cell_density,
                'hour':              int(hour),
                'day_of_week':       DAY_MAP[day_of_week],
                'is_peak_hour':      int(is_peak_hour),
                'has_wrong_parking': int(has_wrong_parking),
                'has_no_parking':    int(has_no_parking),
                'has_main_road':     int(has_main_road),
            }

            try:
                r1 = requests.post(f"{st.session_state.api_url}/predict/severity",
                                   json=payload, timeout=10)
                r2 = requests.post(f"{st.session_state.api_url}/predict/risk",
                                   json=payload, timeout=10)

                if r1.status_code == 200 and r2.status_code == 200:
                    sev  = r1.json()
                    risk = r2.json()

                    # Store pin for map tab
                    st.session_state.prediction_pin = {
                        'lat':              latitude,
                        'lon':              longitude,
                        'severity_score':   sev['severity_score'],
                        'severity_level':   sev['severity_level'],
                        'zone':             sev['zone'],
                        'station':          sev['station'],
                        'is_high_risk':     risk['is_high_risk'],
                        'risk_probability': risk['risk_probability'],
                    }

                    st.success("✓ Predictions Generated")
                    st.markdown(f"**Location:** {latitude:.4f}, {longitude:.4f}")
                    st.markdown(f"**Zone:** {sev['zone']} ({sev['station']})")
                    st.caption("📌 Switch to the 🗺️ Map tab to see this prediction pinned.")
                    st.divider()

                    rc1, rc2 = st.columns(2)

                    with rc1:
                        st.markdown("### Severity Prediction")
                        severity_score = sev['severity_score']
                        severity_level = sev['severity_level']
                        color = {'Low':'green','Medium':'orange',
                                 'High':'red','Critical':'darkred'}.get(severity_level,'gray')
                        st.metric("Severity Score", f"{severity_score:.2f}", severity_level)

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
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

                    with rc2:
                        st.markdown("### Risk Assessment")
                        risk_prob      = risk['risk_probability']
                        recommendation = risk['recommendation']

                        if risk['is_high_risk']:
                            st.error("🔴 HIGH RISK ZONE")
                        else:
                            st.info("🟡 MODERATE RISK")

                        st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
                        st.markdown(f"**Recommendation:** {recommendation}")

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
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"API Error — Severity: {r1.status_code} | Risk: {r2.status_code}\n{r1.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure Flask app is running.")
            except requests.exceptions.Timeout:
                st.error("⏱ Request timed out.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ZONE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📍 Zone-Based Analysis")

    selected_zone = st.selectbox("Select Zone", list(HOTSPOT_ZONES.keys()))
    zi = HOTSPOT_ZONES[selected_zone]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Police Station",  zi['station'])
    col2.metric("Total Violations", f"{zi['violations']:,}")
    col3.metric("High Severity %",  f"{zi['high_severity']}%")
    col4.metric("Resources",        zi['resources'])

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Zone Locations")
        # Fetch live centroids from API if available, fall back to static
        try:
            r = requests.get(f"{st.session_state.api_url}/analytics/map", timeout=10)
            if r.status_code == 200:
                zones_api = r.json().get('zones', [])
                map_df = pd.DataFrame([{
                    'Zone':      z['zone'],
                    'Violations':z['violation_count'],
                    'Latitude':  z['centroid_lat'],
                    'Longitude': z['centroid_lon'],
                } for z in zones_api])
            else:
                raise Exception("API error")
        except Exception:
            map_df = pd.DataFrame({
                'Zone':      list(HOTSPOT_ZONES.keys()),
                'Violations':[HOTSPOT_ZONES[z]['violations'] for z in HOTSPOT_ZONES],
                'Latitude':  [12.99, 13.01, 12.93, 12.93, 13.07, 13.25],
                'Longitude': [77.59, 77.55, 77.69, 77.65, 77.585, 77.54],
            })

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
            {'Zone':z, 'Violations':HOTSPOT_ZONES[z]['violations'],
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
# TAB 4 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("📈 Advanced Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Metrics")
        st.dataframe(pd.DataFrame({
            'Metric':['R² Score','RMSE','Precision','Recall','Data Retention'],
            'Value': ['0.9416','0.1356','0.99','0.89','98.98%'],
            'Status':['✓ Excellent','✓ Excellent','✓ Excellent','✓ Good','✓ Excellent'],
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
        'Feature':   ['cell_violations','cell_density','has_main_road','has_no_parking',
                      'hour','is_peak_hour','has_wrong_parking','latitude','longitude','day_of_week'],
        'Importance':[0.28,0.22,0.18,0.14,0.08,0.05,0.02,0.01,0.01,0.01],
    })
    fig = px.bar(
        feat_df.sort_values('Importance'),
        x='Importance', y='Feature', orientation='h',
        title='Feature Importance', color='Importance',
        color_continuous_scale='Blues', height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — MAP
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🗺️ Bangalore Parking Hotspot Map")
    st.markdown(
        "Zone circles are sized by violation count and coloured by severity. "
        "Click any zone for detailed stats. "
        "Run a prediction in the 🎯 Prediction tab to drop a pin here."
    )

    # Show active pin info if one exists
    pin = st.session_state.prediction_pin
    if pin:
        st.info(
            f"📍 Showing prediction pin — {pin['zone']} / {pin['station']} · "
            f"Severity {pin['severity_score']:.2f} ({pin['severity_level']}) · "
            f"Risk {pin['risk_probability']*100:.1f}%"
        )
        if st.button("🗑 Clear Pin"):
            st.session_state.prediction_pin = None
            st.rerun()
    else:
        st.caption("No prediction pin yet — go to 🎯 Prediction tab and run a prediction.")

    # Build and render map
    with st.spinner("Loading zone data from API..."):
        folium_map = build_map(st.session_state.api_url, st.session_state.prediction_pin)

    map_data = st_folium(
        folium_map,
        width="100%",
        height=580,
        returned_objects=["last_clicked"],
        key="bangalore_map",
    )

    # Show clicked coordinates
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        st.caption(
            f"📌 Last clicked: lat **{clicked['lat']:.5f}**, "
            f"lon **{clicked['lng']:.5f}** — "
            "copy these into the 🎯 Prediction tab to predict here"
        )

    # Live zone table from API
    st.markdown("### Zone Statistics (live from model)")
    try:
        r = requests.get(f"{st.session_state.api_url}/analytics/map", timeout=10)
        if r.status_code == 200:
            zones_api = r.json().get('zones', [])
            tbl = pd.DataFrame([{
                'Zone':             z['zone'],
                'Station':          z['station'],
                'Violations':       f"{z['violation_count']:,}",
                'Avg Severity':     f"{z['avg_severity']:.2f} / 5",
                'High Severity %':  f"{z['high_severity_pct']:.1f}%",
                'Peak Hour %':      f"{z['peak_hour_pct']:.1f}%",
                'Centroid':         f"{z['centroid_lat']:.4f}, {z['centroid_lon']:.4f}",
            } for z in zones_api])
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not load live zone stats from API.")
    except Exception:
        st.warning("API not reachable — live zone table unavailable.")

st.divider()
st.markdown("*Parking Congestion Analytics · Bangalore · Model-driven enforcement*")