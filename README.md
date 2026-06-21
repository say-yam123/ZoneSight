# 🚗 Parking Congestion Analytics - Complete Application

A full-stack application for analyzing and predicting parking violations and congestion patterns in Bangalore using Machine Learning.

## 📋 Quick Summary

**What You Have:**
- ✅ Trained ML Models (Gradient Boosting + Random Forest + K-Means)
- ✅ REST API Backend (Flask)
- ✅ Interactive Dashboard (Streamlit)
- ✅ Real-time Prediction Engine
- ✅ Live Folium Map with OpenStreetMap / Google tiles
- ✅ Model-driven Zone Detection (KMeans centroids)

**What You Can Do:**
- 🎯 Predict violation severity at any location
- 🔴 Identify high-risk parking zones
- 📊 Visualize hotspots and patterns
- 📈 Track enforcement metrics
- 🗺️ Interactive heatmap with live zone boundaries
- 📍 Drop prediction pins on the map in real time

---

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Quick Start
```bash
# Windows
quick_start.bat

# Linux/Mac
python3 quick_start.py
```

### Option 2: Manual Setup

**Step 1: Install Dependencies**
```bash
pip install -r requirements-backend.txt
pip install -r requirements-frontend.txt
```

**Step 3: Start Backend**
```bash
python app.py
```
You should see all 5 models confirmed in the terminal:
```
✓ GBR
✓ RF
✓ KMeans
✓ feature_names
✓ geo_scaler
```

**Step 4: Start Frontend** (in a new terminal)
```bash
streamlit run dashboard.py
```
OR
If you are not running it in venv:
```bash
python -m streamlit run dashboard.py
```
**Step 5: Open Browser**
```
Frontend: http://localhost:8501
API:      http://localhost:5000
```

---

## 📁 Project Structure

```
Traffic-Congestion-Predictor/
│
├── 📓 Parking induced Congestion.ipynb     # Training notebook
├── 🔧 app.py                              # Flask backend API
├── 📊 dashboard.py                        # Streamlit frontend
├── 🗺️  map_component.py                   # Folium map module
│
├── requirements-backend.txt               # Backend dependencies
├── requirements-frontend.txt              # Frontend dependencies
│
├── 🎯 quick_start.py                      # Python quick start
├── 🎯 quick_start.bat                     # Windows quick start
│
├── 📚 SETUP_GUIDE.md                      # Detailed setup guide
├── 📚 NOTEBOOK_EXPORT_CODE.md             # Notebook integration code
├── 📚 README.md                           # This file
│
└── 📁 models/                             # Exported models (auto-created)
    ├── gbr_severity_model.pkl             # Gradient Boosting Regressor
    ├── rf_risk_classifier.pkl             # Random Forest Classifier
    ├── kmeans_hotspots.pkl                # K-Means clustering
    ├── geo_scaler.pkl                     # StandardScaler for lat/lon
    └── feature_names.pkl                  # Feature order for predictions
```

---

## 🎯 Features

### 1. Dashboard (📊)
- Real-time KPI metrics fetched live from the API
- Hotspot zones overview (6 zones) — color-coded by severity
- Violation distribution charts
- High severity percentage comparison
- Cached fallback when API is unavailable

### 2. Real-Time Prediction (🎯)
- 3 one-click demo scenarios (High / Moderate / Low Risk)
- Location-based severity prediction via Gradient Boosting
- Risk assessment classification via Random Forest
- Gauge charts on correct 1–5 severity scale
- Enforcement recommendations
- Prediction history table (last 5 predictions)
- Pin automatically dropped on Map tab after prediction

### 3. Map (🗺️)
- OpenStreetMap base layer (no API key required)
- Optional Google Maps / Satellite tile layers
- Violation heatmap weighted by zone violation count
- Zone circles positioned from real KMeans centroids
- Clickable popups with GBR-derived severity stats
- Prediction pin with severity-colored marker
- Static fallback so map always renders even if API is slow
- Live zone statistics table below the map

### 4. Zone Analysis (📍)
- Zone-specific statistics
- Geographic scatter map using live KMeans centroids
- Zone comparison scatter plot
- Police station assignments
- Resource allocation details

### 5. Advanced Analytics (📈)
- Model performance metrics (corrected RMSE: 0.1356)
- Feature importance visualization
- Expected impact projections
- Renamed features for non-technical audiences

---

## 🔌 API Endpoints

### Health Check
```bash
GET /health
# Returns: {"status": "healthy", "models_loaded": true, "timestamp": "..."}
```

### Severity Prediction
```bash
POST /predict/severity
Body: {
  "latitude": 13.05,
  "longitude": 77.60,
  "cell_violations": 15,
  "cell_density": 0.5,
  "hour": 18,
  "day_of_week": 4,
  "is_peak_hour": 1,
  "has_wrong_parking": 1,
  "has_no_parking": 0,
  "has_main_road": 1
}
# Returns: {"severity_score": 4.99, "severity_level": "High", "zone": "Zone 2", "station": "HSR Layout", ...}
```

### Risk Prediction
```bash
POST /predict/risk
Body: { ...same as above... }
# Returns: {"is_high_risk": true, "risk_probability": 0.91, "recommendation": "DEPLOY IMMEDIATE ENFORCEMENT", ...}
```

### Map Analytics (live from models)
```bash
GET /analytics/map
# Returns zone centroids from KMeans + severity stats from GBR grid sweep
# Source: "kmeans_centroids + gbr_predictions"
```

### Zone Analytics
```bash
GET /analytics/zones
GET /zones
```

### Batch Prediction
```bash
POST /predict/batch
Body: {"locations": [...multiple location objects...]}
```

---

## 📊 Model Information

### Models Included:

**1. Gradient Boosting Regressor** (Severity Prediction)
- Predicts violation impact severity (1–5 scale)
- R² Score: 0.9416 (Excellent)
- RMSE: 0.1356

**2. Random Forest Classifier** (Risk Classification)
- Identifies high-risk locations (Binary: High/Low Risk)
- High-risk threshold: violation_severity ≥ 4
- Precision: 0.99
- Recall: 0.89

**3. K-Means Clustering** (Hotspot Zone Detection)
- 6 identified hotspot zones across Bangalore
- Centroids used live via `geo_scaler.inverse_transform()`
- Zone assignment uses `kmeans.predict()` — no hardcoded bounding boxes

**4. StandardScaler** (Geo Scaler)
- Fitted on latitude/longitude for KMeans zone assignment
- Required for both zone detection and map centroid computation

### Top Features:
1. Reported Violations / area (28%)
2. Violation Density (22%)
3. Main Road Presence (18%)
4. No Parking Zone (14%)
5. Hour of Day (8%)

### Hotspot Zones:
| Zone   | Station         | Violations | Resources      |
|--------|-----------------|------------|----------------|
| Zone 5 | Upparpet        | 123,778    | 8 patrol units |
| Zone 0 | Malleshwaram    | 69,076     | 5 patrol units |
| Zone 1 | HAL Old Airport | 48,600     | 4 patrol units |
| Zone 2 | HSR Layout      | 25,154     | 2 patrol units |
| Zone 4 | Kodigehalli     | 21,920     | 1 patrol unit  |
| Zone 3 | Chikkajala      | 6,881      | 1 patrol unit  |

---

## 🛠️ Technical Stack

### Backend
- **Framework**: Flask
- **ML Models**: scikit-learn 1.7.2
- **Serialization**: Joblib
- **CORS**: Flask-CORS

### Frontend
- **Framework**: Streamlit
- **Map**: Folium + streamlit-folium
- **Visualizations**: Plotly
- **Data Processing**: Pandas, NumPy
- **HTTP Client**: Requests

### Data & Models
- **Model Format**: .pkl (joblib)
- **Data Format**: CSV, DataFrame
- **Platform**: Python 3.11 (recommended)

---

## 📋 System Requirements

- **Python**: 3.11 recommended (3.8+ minimum)
- **scikit-learn**: Must match version used to train models (check with `import sklearn; print(sklearn.__version__)`)
- **RAM**: 4 GB minimum (8 GB recommended)
- **Disk Space**: 500 MB
- **Ports**: 5000 (Backend), 8501 (Frontend)

---

## 🐛 Troubleshooting

### "503 — Models not loaded"
```bash
# Models missing from ./models/ — export from notebook first
# Run all notebook cells, then the export cell at the bottom
# Verify files exist:
ls models/
# Should show: gbr_severity_model.pkl, rf_risk_classifier.pkl,
#              kmeans_hotspots.pkl, geo_scaler.pkl, feature_names.pkl
```

### "STACK_GLOBAL requires str" (geo_scaler fails to load)
```bash
# scikit-learn version mismatch between notebook and venv
# Step 1: Check notebook version
import sklearn; print(sklearn.__version__)

# Step 2: Match it in your venv
pip install scikit-learn==<version from above>

# Step 3: Re-export geo_scaler using joblib (not pickle)
import joblib
joblib.dump(scaler, './models/geo_scaler.pkl')
```

### "Zone: Unknown (N/A)" in predictions
```bash
# geo_scaler.pkl not loaded — check terminal for "✗ geo_scaler"
# Re-export the scaler from the notebook and restart app.py
```

### Map not loading / blank page
```bash
# Install folium dependencies
pip install folium streamlit-folium

# Check API is running before starting dashboard
curl http://localhost:5000/analytics/map
```

### "ConnectionResetError: WinError 10054" on Windows
```python
# Add to top of dashboard.py (already included):
import asyncio, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### "No module named '_loss'" (sklearn version)
```bash
# Train and export models inside the same venv as app.py
pip install jupyter ipykernel
python -m ipykernel install --user --name=parking-venv
jupyter notebook
# Select "parking-venv" kernel, run all cells, export models
```

### "Port already in use"
```bash
python app.py        # default port 5000
streamlit run dashboard.py --server.port 8502
```

---

## ⚠️ Known Behaviour

> **Map Loading**
> The map may take longer to load on first render, and may occasionally appear blank on initial page load.
> If this happens, run any prediction from the **🎯 Prediction tab** — either a demo scenario or a custom input — and the map will load correctly with the prediction pin visible.

---

## 🗺️ Map Configuration

### Default (OpenStreetMap — no key needed)
Works out of the box. No configuration required.

### Optional: Google Maps Tiles (no API key needed)
In `map_component.py`, replace the tile layer:
```python
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
    attr='Google Maps',
    name='Google Maps',
).add_to(m)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google Satellite',
    name='Google Satellite',
).add_to(m)
```

### Optional: Google Maps JavaScript API (full SDK)
Requires a Google Cloud API key with Maps JavaScript API enabled.
Replace `st_folium` in `dashboard.py` with `st.components.v1.html`
and use the `build_google_map(api_key, zones, pin)` function.

---

## 🚀 Deployment

### Local Development
```bash
python app.py                    # Flask development server
streamlit run dashboard.py       # Streamlit development
```

### Production
```bash
# Backend with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend
streamlit run dashboard.py --server.address 0.0.0.0
```

### Docker (Optional)
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements-backend.txt -r requirements-frontend.txt
EXPOSE 5000 8501
CMD ["sh", "-c", "python app.py & streamlit run dashboard.py"]
```

---

## 📊 Model Performance

| Metric              | Value   | Status      |
|---------------------|---------|-------------|
| R² Score (GBR)      | 0.9416  | ✓ Excellent |
| RMSE (GBR)          | 0.1356  | ✓ Excellent |
| Precision (RF)      | 0.99    | ✓ Excellent |
| Recall (RF)         | 0.89    | ✓ Good      |
| Data Retention      | 98.98%  | ✓ Excellent |

## 📊 Expected Impact (3 Months)

| Metric                  | Target |
|-------------------------|--------|
| Violations Prevented    | -25%   |
| Congestion Reduction    | -20%   |
| Main Road Flow          | -30%   |
| Enforcement Efficiency  | +40%   |
| Public Compliance       | >80%   |

---

## ✅ Checklist Before Running

- [ ] Python 3.11 installed
- [ ] scikit-learn version matches notebook environment
- [ ] All 5 model files present in `./models/`
- [ ] Dependencies installed (`pip install -r requirements-*.txt`)
- [ ] `folium` and `streamlit-folium` installed
- [ ] Ports 5000 and 8501 available
- [ ] At least 4 GB RAM available
- [ ] `app.py` running before launching `dashboard.py`

---

## 🎯 Demo Flow (Hackathon Presentation)

1. Open dashboard — **Dashboard tab** shows live metrics from model
2. Go to **Prediction tab** — click "🔴 High Risk — Upparpet Peak Hour"
3. Watch gauges fire red — severity 5/5, risk probability >90%
4. Switch to **Map tab** — pin appears on Bangalore at Upparpet
5. Click **Zone 5 circle** — popup shows KMeans centroid + GBR severity stats
6. Run "🟢 Low Risk — Chikkajala Off-Peak" — pin moves, gauges go green
7. Show **Zone Analysis tab** — centroid positions sourced from live model

---

## 📖 Documentation

- **Full Setup Guide**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **API Reference**: See endpoints section above

---

## 🔐 Security Notes

- API runs on localhost by default
- No authentication required (development mode)
- Coordinate validation rejects non-Bangalore inputs (lat 12.5–13.5, lon 77.3–77.9)
- For production: add authentication, use HTTPS, sanitize inputs

---

**Version**: 2.0
**Last Updated**: 2026
**Status**: ✅ Production Ready
**Python**: 3.11 recommended
**scikit-learn**: 1.7.2