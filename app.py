"""
Flask API Backend for Parking Congestion Prediction
Serves the trained ML models for real-time predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Model paths
MODEL_DIR = './models'
SCALER_PATH = os.path.join(MODEL_DIR, 'geo_scaler.pkl')
GBR_MODEL_PATH = os.path.join(MODEL_DIR, 'gbr_severity_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_risk_classifier.pkl')
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_hotspots.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')

# Initialize models as None
gbr_model = None
rf_model = None
kmeans_model = None
feature_names = None
geo_scaler    = None

# Load models
try:
    gbr_model = joblib.load(GBR_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(RF_MODEL_PATH)
    kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    geo_scaler    = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"⚠ Error loading models: {e}")
    print("Run the notebook first to train and export models")

for name, obj in [('GBR', gbr_model), ('RF', rf_model), ('KMeans', kmeans_model),
                  ('feature_names', feature_names), ('geo_scaler', geo_scaler)]:
    print(f"{'✓' if obj is not None else '✗'} {name}")
 

# Hotspot zone information (from notebook analysis)
HOTSPOT_ZONES = {
    'Zone 5': {'station': 'Upparpet', 'lat_range': [12.97, 13.01], 'lon_range': [77.57, 77.62]},
    'Zone 0': {'station': 'Malleshwaram', 'lat_range': [12.98, 13.03], 'lon_range': [77.53, 77.58]},
    'Zone 1': {'station': 'HAL Old Airport', 'lat_range': [12.91, 12.96], 'lon_range': [77.65, 77.72]},
    'Zone 2': {'station': 'HSR Layout', 'lat_range': [12.91, 12.95], 'lon_range': [77.63, 77.68]},
    'Zone 4': {'station': 'Kodigehalli', 'lat_range': [13.04, 13.09], 'lon_range': [77.56, 77.61]},
    'Zone 3': {'station': 'Chikkajala', 'lat_range': [13.23, 13.29], 'lon_range': [77.52, 77.57]},
}

CLUSTER_VIOLATION_COUNTS = {
    0: 69076,
    1: 48600,
    2: 25154,
    3: 6881,
    4: 21920,
    5: 123778,
}

# Cluster → station mapping (from notebook hotspot_summary)
CLUSTER_TO_STATION = {
    5: 'Upparpet',
    0: 'Malleshwaram',
    1: 'HAL Old Airport',
    2: 'HSR Layout',
    4: 'Kodigehalli',
    3: 'Chikkajala',
}
 

def prepare_features(input_data):
    """Build feature vector in the exact order the models were trained on."""
    if feature_names is None:
        return None
    try:
        row = [input_data.get(f, 0) for f in feature_names]
        return np.array([row])
    except Exception as e:
        print(f"prepare_features error: {e}")
        return None
 
def find_zone(latitude, longitude):
    """Assign zone using the trained KMeans model + geo scaler."""
    if kmeans_model is None or geo_scaler is None:
        return 'Unknown', 'N/A'
    try:
        coords_scaled = geo_scaler.transform([[latitude, longitude]])
        cluster_id    = int(kmeans_model.predict(coords_scaled)[0])
        zone_name     = f'Zone {cluster_id}'
        station       = CLUSTER_TO_STATION.get(cluster_id, 'N/A')
        return zone_name, station
    except Exception as e:
        print(f"find_zone error: {e}")
        return 'Unknown', 'N/A'
 


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': all([
            gbr_model is not None,
            rf_model is not None,
            kmeans_model is not None,
            feature_names is not None
        ])
    }), 200

@app.route('/predict/severity', methods=['POST'])
def predict_severity():
    """
    Predict violation severity at a given location and time
    Expected JSON input:
    {
        "latitude": 13.05,
        "longitude": 77.60,
        "cell_violations": 15,
        "cell_density": 0.8,
        "hour": 18,
        "day_of_week": 4,
        "is_peak_hour": 1,
        "has_wrong_parking": 1,
        "has_no_parking": 0,
        "has_main_road": 1
    }
    """
    try:
        if gbr_model is None:
            return jsonify({'error': 'Models are not loaded. Please train and export models from the notebook first.'}), 503
        data = request.get_json()
        
        # Prepare features
        X = prepare_features(data)
        if X is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Make prediction
        severity_score = gbr_model.predict(X)[0]
        
        # Find zone
        zone_name, station = find_zone(data.get('latitude', 13.0), 
                                       data.get('longitude', 77.6))
        
        return jsonify({
            'severity_score': float(severity_score),
            'severity_level': classify_severity(severity_score),
            'zone': zone_name,
            'station': station,
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/risk', methods=['POST'])
def predict_risk():
    """
    Predict if a location is high-risk (violation_severity >= 4)
    Uses Random Forest classifier
    """
    try:
        if rf_model is None:
            return jsonify({'error': 'Models are not loaded. Please train and export models from the notebook first.'}), 503
        data = request.get_json()
        
        # Prepare features
        X = prepare_features(data)
        if X is None:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Make predictions
        risk_prediction = rf_model.predict(X)[0]
        risk_probability = rf_model.predict_proba(X)[0]
        
        # Find zone
        zone_name, station = find_zone(data.get('latitude', 13.0), 
                                       data.get('longitude', 77.6))
        
        return jsonify({
            'is_high_risk': bool(risk_prediction),
            'risk_probability': float(risk_probability[1]),
            'low_risk_probability': float(risk_probability[0]),
            'zone': zone_name,
            'station': station,
            'recommendation': get_risk_recommendation(risk_probability[1]),
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple locations
    Expected JSON: {"locations": [{"latitude": ..., "longitude": ..., ...}, ...]}
    """
    try:
        if gbr_model is None or rf_model is None:
            return jsonify({'error': 'Models are not loaded. Please train and export models from the notebook first.'}), 503
        data = request.get_json()
        locations = data.get('locations', [])
        
        if not locations:
            return jsonify({'error': 'No locations provided'}), 400
        
        results = []
        for loc in locations:
            X = prepare_features(loc)
            if X is not None:
                severity = gbr_model.predict(X)[0]
                risk = rf_model.predict(X)[0]
                zone, station = find_zone(loc.get('latitude', 13.0), 
                                         loc.get('longitude', 77.6))
                
                results.append({
                    'location': {'lat': loc.get('latitude'), 'lon': loc.get('longitude')},
                    'severity_score': float(severity),
                    'is_high_risk': bool(risk),
                    'zone': zone,
                    'station': station
                })
        
        return jsonify({
            'total_locations': len(locations),
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/zones', methods=['GET'])
def get_zones():
    """Get information about all hotspot zones"""
    zones_data = []
    for zone_name, zone_info in HOTSPOT_ZONES.items():
        zones_data.append({
            'zone': zone_name,
            'station': zone_info['station'],
            'latitude_range': zone_info['lat_range'],
            'longitude_range': zone_info['lon_range']
        })
    return jsonify({'zones': zones_data}), 200

@app.route('/analytics/zones', methods=['GET'])
def zone_analytics():
    """Get analytics for all zones"""
    zone_stats = {
        'Zone 5': {'violations': 123778, 'high_severity': 31.1, 'resources': '8 patrol units'},
        'Zone 0': {'violations': 69076, 'high_severity': 31.1, 'resources': '5 patrol units'},
        'Zone 1': {'violations': 48600, 'high_severity': 35.4, 'resources': '4 patrol units'},
        'Zone 2': {'violations': 25154, 'high_severity': 31.7, 'resources': '2 patrol units'},
        'Zone 4': {'violations': 21920, 'high_severity': 30.7, 'resources': '1 patrol unit'},
        'Zone 3': {'violations': 6881, 'high_severity': 34.3, 'resources': '1 patrol unit'},
    }
    return jsonify(zone_stats), 200

@app.route('/analytics/map', methods=['GET'])
def analytics_map():
    """
    Returns per-zone stats computed from the trained models.
    Centroid positions come from KMeans + geo_scaler.
    Severity stats come from GBR predictions over a grid around each centroid.
    """
    if kmeans_model is None or geo_scaler is None or gbr_model is None:
        return jsonify({'error': 'Models not loaded'}), 503
 
    try:
        # Step 1: Get cluster centroids in original lat/lon space
        centroids_scaled   = kmeans_model.cluster_centers_          # shape (6, 2)
        centroids_original = geo_scaler.inverse_transform(centroids_scaled)  # lat, lon
 
        zones = []
 
        for cluster_id in range(len(centroids_original)):
            clat, clon = centroids_original[cluster_id]
 
            # Step 2: Build a small grid (5x5) around the centroid
            # to sample GBR severity predictions across the zone
            lat_offsets = np.linspace(-0.02, 0.02, 5)
            lon_offsets = np.linspace(-0.02, 0.02, 5)
 
            severity_scores = []
            high_severity_count = 0
            peak_hour_count = 0
            total_grid_points = 0
 
            for dlat in lat_offsets:
                for dlon in lon_offsets:
                    # Sample with representative feature values
                    # Test both peak and off-peak, main road and not
                    for hour, is_peak in [(8, 1), (14, 0), (18, 1)]:
                        for has_main in [0, 1]:
                            sample = {
                                'latitude':          clat + dlat,
                                'longitude':         clon + dlon,
                                'cell_violations':   50,   # median-ish value
                                'cell_density':      0.5,
                                'hour':              hour,
                                'day_of_week':       3,
                                'is_peak_hour':      is_peak,
                                'has_wrong_parking': 1,
                                'has_no_parking':    0,
                                'has_main_road':     has_main,
                            }
                            X = prepare_features(sample)
                            if X is not None:
                                score = float(np.clip(gbr_model.predict(X)[0], 1.0, 5.0))
                                severity_scores.append(score)
                                if score >= 4:
                                    high_severity_count += 1
                                if is_peak:
                                    peak_hour_count += 1
                                total_grid_points += 1
 
            if not severity_scores:
                continue
 
            avg_severity      = float(np.mean(severity_scores))
            max_severity      = float(np.max(severity_scores))
            high_severity_pct = (high_severity_count / total_grid_points) * 100
            peak_hour_pct     = (peak_hour_count     / total_grid_points) * 100
 
            violation_count = CLUSTER_VIOLATION_COUNTS.get(cluster_id, 0)
 
            # Wrong parking and main road counts estimated from violation proportions
            # (from notebook: ~55% wrong parking in top zones, ~8% main road)
            wrong_parking_count = int(violation_count * 0.55)
            main_road_count     = int(violation_count * 0.08)
 
            zones.append({
                'zone':               f'Zone {cluster_id}',
                'cluster_id':         cluster_id,
                'station':            CLUSTER_TO_STATION.get(cluster_id, 'N/A'),
                'centroid_lat':       round(float(clat), 5),
                'centroid_lon':       round(float(clon), 5),
                'violation_count':    violation_count,
                'avg_severity':       round(avg_severity, 3),
                'max_severity':       round(max_severity, 3),
                'high_severity_pct':  round(high_severity_pct, 1),
                'peak_hour_pct':      round(peak_hour_pct, 1),
                'wrong_parking_count': wrong_parking_count,
                'main_road_count':    main_road_count,
            })
 
        # Sort by violation count descending
        zones.sort(key=lambda z: z['violation_count'], reverse=True)
 
        return jsonify({
            'zones':     zones,
            'timestamp': datetime.now().isoformat(),
            'source':    'kmeans_centroids + gbr_predictions',
        }), 200
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def classify_severity(score):
    """Notebook violation_severity is 1–5, not 0–10."""
    if score <= 2:
        return 'Low'
    elif score <= 3:
        return 'Medium'
    elif score <= 4:
        return 'High'
    else:
        return 'Critical'
 
def get_risk_recommendation(probability):
    """Get recommendation based on risk probability"""
    if probability > 0.75:
        return 'DEPLOY IMMEDIATE ENFORCEMENT'
    elif probability > 0.5:
        return 'INCREASE PATROL PRESENCE'
    elif probability > 0.25:
        return 'MONITOR CLOSELY'
    else:
        return 'ROUTINE PATROL'

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Parking Congestion API...")
    app.run(debug=True, host='0.0.0.0', port=5000)