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
from datetime import datetime, time
import json

# Page configuration
st.set_page_config(
    page_title="Parking Congestion Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .zone-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_url' not in st.session_state:
    st.session_state.api_url = 'http://localhost:5000'

# Hotspot zones data
HOTSPOT_ZONES = {
    'Zone 5': {'station': 'Upparpet', 'violations': 123778, 'high_severity': 31.1, 'resources': '8 patrol units', 'color': '#FF6B6B'},
    'Zone 0': {'station': 'Malleshwaram', 'violations': 69076, 'high_severity': 31.1, 'resources': '5 patrol units', 'color': '#4ECDC4'},
    'Zone 1': {'station': 'HAL Old Airport', 'violations': 48600, 'high_severity': 35.4, 'resources': '4 patrol units', 'color': '#45B7D1'},
    'Zone 2': {'station': 'HSR Layout', 'violations': 25154, 'high_severity': 31.7, 'resources': '2 patrol units', 'color': '#FFA07A'},
    'Zone 4': {'station': 'Kodigehalli', 'violations': 21920, 'high_severity': 30.7, 'resources': '1 patrol unit', 'color': '#98D8C8'},
    'Zone 3': {'station': 'Chikkajala', 'violations': 6881, 'high_severity': 34.3, 'resources': '1 patrol unit', 'color': '#F7DC6F'},
}

# Main title
st.markdown('<div class="header-title">🚗 Parking Congestion Analytics</div>', unsafe_allow_html=True)
st.markdown("*Data-Driven Enforcement Strategy for Bangalore*", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    tab1, tab2 = st.tabs(["API Settings", "About"])
    
    with tab1:
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_url,
            help="Set the backend API URL"
        )
        st.session_state.api_url = api_url
        
        if st.button("🔗 Test Connection"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✓ API Connected Successfully")
                else:
                    st.error("✗ API Error")
            except:
                st.error("✗ Cannot connect to API. Make sure Flask app is running on " + api_url)
    
    with tab2:
        st.markdown("""
        ### About This Dashboard
        
        This application provides real-time analysis of parking violations 
        and congestion patterns in Bangalore.
        
        **Features:**
        - Real-time severity predictions
        - Risk zone identification
        - Zone-based analytics
        - Enforcement resource allocation
        
        **Models:**
        - Gradient Boosting Severity Predictor
        - Random Forest Risk Classifier
        """)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Prediction", "📍 Zone Analysis", "📈 Analytics"])

# ============= TAB 1: DASHBOARD =============
with tab1:
    st.header("Real-Time Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Violations", "295,609", "-5.2%", delta_color="inverse")
    
    with col2:
        st.metric("High-Risk Zones", "6", "+1.0%")
    
    with col3:
        st.metric("Model Accuracy (R²)", "0.9416", "+0.5%")
    
    with col4:
        st.metric("Patrol Units", "21", "Optimized")
    
    st.divider()
    
    # Hotspot zones overview
    st.subheader("🔴 Hotspot Zones Overview")
    
    cols = st.columns(3)
    for idx, (zone_name, zone_info) in enumerate(HOTSPOT_ZONES.items()):
        col = cols[idx % 3]
        with col:
            st.markdown(f"""
            <div class="zone-card">
            <h4>{zone_name}</h4>
            <p><strong>Station:</strong> {zone_info['station']}</p>
            <p><strong>Violations:</strong> {zone_info['violations']:,}</p>
            <p><strong>High Severity:</strong> {zone_info['high_severity']}%</p>
            <p><strong>Resources:</strong> {zone_info['resources']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Comparison chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Violations by zone
        zone_names = list(HOTSPOT_ZONES.keys())
        violations = [HOTSPOT_ZONES[z]['violations'] for z in zone_names]
        
        fig = px.bar(
            x=zone_names,
            y=violations,
            title="Violations by Zone",
            labels={'x': 'Zone', 'y': 'Number of Violations'},
            color=violations,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # High severity percentage
        zone_names = list(HOTSPOT_ZONES.keys())
        high_severity = [HOTSPOT_ZONES[z]['high_severity'] for z in zone_names]
        
        fig = px.bar(
            x=zone_names,
            y=high_severity,
            title="High Severity % by Zone",
            labels={'x': 'Zone', 'y': 'High Severity %'},
            color=high_severity,
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 2: PREDICTION =============
with tab2:
    st.header("🎯 Real-Time Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Parameters")
        
        latitude = st.slider("Latitude", 12.9, 13.4, 13.05, 0.01)
        longitude = st.slider("Longitude", 77.5, 77.7, 77.60, 0.01)
        
        hour = st.slider("Hour of Day", 0, 23, 18)
        day_of_week = st.selectbox("Day of Week", 
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=4)
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        st.divider()
        
        col_a, col_b = st.columns(2)
        with col_a:
            cell_violations = st.number_input("Cell Violations", 0, 100, 15)
            is_peak_hour = st.checkbox("Is Peak Hour", value=True)
        
        with col_b:
            cell_density = st.slider("Cell Density", 0.0, 1.0, 0.8)
            has_wrong_parking = st.checkbox("Wrong Parking Present", value=True)
        
        col_c, col_d = st.columns(2)
        with col_c:
            has_no_parking = st.checkbox("No Parking Zone", value=False)
        
        with col_d:
            has_main_road = st.checkbox("Main Road", value=True)
    
    with col2:
        st.subheader("Predictions")
        
        if st.button("🔮 Get Predictions", use_container_width=True):
            # Prepare input
            prediction_input = {
                'latitude': latitude,
                'longitude': longitude,
                'cell_violations': int(cell_violations),
                'cell_density': cell_density,
                'hour': int(hour),
                'day_of_week': day_map[day_of_week],
                'is_peak_hour': int(is_peak_hour),
                'has_wrong_parking': int(has_wrong_parking),
                'has_no_parking': int(has_no_parking),
                'has_main_road': int(has_main_road)
            }
            
            try:
                # Call severity prediction
                response1 = requests.post(
                    f"{st.session_state.api_url}/predict/severity",
                    json=prediction_input,
                    timeout=10
                )
                
                # Call risk prediction
                response2 = requests.post(
                    f"{st.session_state.api_url}/predict/risk",
                    json=prediction_input,
                    timeout=10
                )
                
                if response1.status_code == 200 and response2.status_code == 200:
                    severity_data = response1.json()
                    risk_data = response2.json()
                    
                    # Display results
                    st.success("✓ Predictions Generated")
                    
                    st.markdown(f"**Location:** {latitude:.4f}, {longitude:.4f}")
                    st.markdown(f"**Zone:** {severity_data['zone']} ({severity_data['station']})")
                    
                    st.divider()
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("### Severity Prediction")
                        severity_score = severity_data['severity_score']
                        severity_level = severity_data['severity_level']
                        
                        # Determine color
                        if severity_level == 'Low':
                            color = 'green'
                        elif severity_level == 'Medium':
                            color = 'orange'
                        elif severity_level == 'High':
                            color = 'red'
                        else:
                            color = 'darkred'
                        
                        st.metric("Severity Score", f"{severity_score:.2f}", severity_level)
                        
                        # Gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=severity_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Severity Level"},
                            gauge={
                                'axis': {'range': [0, 10]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 2], 'color': "#90EE90"},
                                    {'range': [2, 4], 'color': "#FFD700"},
                                    {'range': [4, 6], 'color': "#FF6347"},
                                    {'range': [6, 10], 'color': "#8B0000"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 6
                                }
                            }
                        ))
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with result_col2:
                        st.markdown("### Risk Assessment")
                        is_high_risk = risk_data['is_high_risk']
                        risk_prob = risk_data['risk_probability']
                        recommendation = risk_data['recommendation']
                        
                        if is_high_risk:
                            st.error(f"🔴 HIGH RISK ZONE")
                        else:
                            st.warning(f"🟡 MODERATE RISK")
                        
                        st.metric("Risk Probability", f"{risk_prob*100:.1f}%")
                        st.markdown(f"**Recommendation:** {recommendation}")
                        
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=risk_prob*100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risk %"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "#90EE90"},
                                    {'range': [25, 50], 'color': "#FFD700"},
                                    {'range': [50, 75], 'color': "#FF6347"},
                                    {'range': [75, 100], 'color': "#8B0000"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 75
                                }
                            }
                        ))
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Error getting predictions from API")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure Flask app is running.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============= TAB 3: ZONE ANALYSIS =============
with tab3:
    st.header("📍 Zone-Based Analysis")
    
    selected_zone = st.selectbox("Select Zone", list(HOTSPOT_ZONES.keys()))
    zone_info = HOTSPOT_ZONES[selected_zone]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Police Station", zone_info['station'])
    
    with col2:
        st.metric("Total Violations", f"{zone_info['violations']:,}")
    
    with col3:
        st.metric("High Severity %", f"{zone_info['high_severity']}%")
    
    with col4:
        st.metric("Resources", zone_info['resources'])
    
    st.divider()
    
    # Zone map visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Zone Location")
        zone_map_data = pd.DataFrame({
            'Zone': list(HOTSPOT_ZONES.keys()),
            'Violations': [HOTSPOT_ZONES[z]['violations'] for z in HOTSPOT_ZONES.keys()],
            'Latitude': [12.99, 13.01, 12.93, 12.93, 13.07, 13.25],
            'Longitude': [77.59, 77.55, 77.69, 77.65, 77.585, 77.54]
        })
        
        fig = px.scatter_mapbox(
            zone_map_data,
            lat='Latitude',
            lon='Longitude',
            hover_name='Zone',
            hover_data='Violations',
            zoom=11,
            height=400,
            title="Hotspot Zones Map"
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Zone Comparison")
        
        zones_df = pd.DataFrame([
            {'Zone': z, 'Violations': HOTSPOT_ZONES[z]['violations'], 'High Severity %': HOTSPOT_ZONES[z]['high_severity']}
            for z in HOTSPOT_ZONES.keys()
        ])
        
        fig = px.scatter(
            zones_df,
            x='Violations',
            y='High Severity %',
            hover_name='Zone',
            size='Violations',
            color='High Severity %',
            color_continuous_scale='RdYlGn_r',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: ANALYTICS =============
with tab4:
    st.header("📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Metrics")
        
        metrics = {
            'Metric': ['R² Score', 'RMSE', 'Precision', 'Recall', 'Data Retention'],
            'Value': ['0.9416', '0.4234', '0.99', '0.89', '98.98%'],
            'Status': ['✓ Excellent', '✓ Good', '✓ Excellent', '✓ Good', '✓ Excellent']
        }
        
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Expected Impact (3 Months)")
        
        impact = {
            'Metric': [
                'Violations Prevented',
                'Congestion Reduction',
                'Main Road Flow',
                'Enforcement Efficiency',
                'Compliance Rate'
            ],
            'Target': ['-25%', '-20%', '-30%', '+40%', '>80%']
        }
        
        impact_df = pd.DataFrame(impact)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Feature importance
    st.subheader("Top Features Affecting Severity")
    
    features_importance = pd.DataFrame({
        'Feature': ['cell_violations', 'cell_density', 'has_main_road', 'has_no_parking', 'hour', 
                   'is_peak_hour', 'has_wrong_parking', 'latitude', 'longitude', 'day_of_week'],
        'Importance': [0.28, 0.22, 0.18, 0.14, 0.08, 0.05, 0.02, 0.01, 0.01, 0.01]
    })
    
    fig = px.bar(
        features_importance.sort_values('Importance', ascending=True),
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Severity Prediction',
        labels={'x': 'Importance Score', 'y': 'Feature'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("*Last Updated: 2024 | Data-Driven Enforcement System for Bangalore Police*")
