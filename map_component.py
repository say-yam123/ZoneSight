"""
map_component.py — Folium map integration for Parking Congestion Dashboard
All zone data is fetched live from the API (computed from trained models).
No hardcoded zone values.
"""

import folium
from folium.plugins import HeatMap
import requests


CLUSTER_TO_COLOR = {
    'Zone 5': '#FF0000',
    'Zone 0': '#FF4500',
    'Zone 1': '#FF8C00',
    'Zone 2': '#FFA500',
    'Zone 4': '#FFD700',
    'Zone 3': '#90EE90',
}

def _violation_radius(violations):
    if violations > 100000: return 2200
    if violations > 50000:  return 1700
    if violations > 25000:  return 1300
    if violations > 10000:  return 1000
    return 700


def fetch_zone_data(api_url):
    """
    Fetch live zone data from /analytics/map endpoint.
    Returns list of zone dicts or None on failure.
    """
    try:
        r = requests.get(f"{api_url}/analytics/map", timeout=10)
        if r.status_code == 200:
            return r.json().get("zones", [])
        return None
    except Exception as e:
        print(f"Failed to fetch zone data: {e}")
        return None


def build_map(api_url, prediction_pin=None):
    """
    Build and return a folium Map with live zone data from the API.

    api_url: base URL of the Flask backend e.g. http://localhost:5000
    prediction_pin: dict with keys:
        lat, lon, severity_score, severity_level, zone, station,
        is_high_risk, risk_probability
    """
    zones = fetch_zone_data(api_url)

    m = folium.Map(
        location=[13.0, 77.59],
        zoom_start=12,
        tiles='OpenStreetMap',
        control_scale=True,
    )

    if zones:
        # ── Heatmap ───────────────────────────────────────────────────────────
        heat_data = [
            [z['centroid_lat'], z['centroid_lon'], z['violation_count'] / 1000]
            for z in zones
        ]
        HeatMap(
            heat_data,
            name='Violation Heatmap',
            min_opacity=0.3,
            max_zoom=14,
            radius=60,
            blur=40,
            gradient={
                '0.2': 'green',
                '0.5': 'yellow',
                '0.7': 'orange',
                '1.0': 'red',
            }
        ).add_to(m)

        # ── Zone circles ──────────────────────────────────────────────────────
        zone_layer = folium.FeatureGroup(name='Hotspot Zones', show=True)

        for z in zones:
            zone_name = z['zone']
            color     = CLUSTER_TO_COLOR.get(zone_name, '#888888')
            radius    = _violation_radius(z['violation_count'])

            popup_html = f"""
            <div style="font-family:Arial;min-width:230px">
                <h4 style="margin:0 0 6px;color:{color}">
                    {zone_name} — {z['station']}
                </h4>
                <table style="width:100%;font-size:13px;border-collapse:collapse">
                    <tr><td><b>Violations</b></td>
                        <td>{z['violation_count']:,}</td></tr>
                    <tr><td><b>Avg Severity</b></td>
                        <td>{z['avg_severity']:.2f} / 5</td></tr>
                    <tr><td><b>Max Severity</b></td>
                        <td>{z['max_severity']:.2f} / 5</td></tr>
                    <tr><td><b>High Severity %</b></td>
                        <td>{z['high_severity_pct']:.1f}%</td></tr>
                    <tr><td><b>Peak Hour %</b></td>
                        <td>{z['peak_hour_pct']:.1f}%</td></tr>
                    <tr><td><b>Wrong Parking</b></td>
                        <td>{z['wrong_parking_count']:,}</td></tr>
                    <tr><td><b>Main Road Violations</b></td>
                        <td>{z['main_road_count']:,}</td></tr>
                    <tr><td><b>Centroid</b></td>
                        <td>{z['centroid_lat']:.4f}, {z['centroid_lon']:.4f}</td></tr>
                </table>
            </div>
            """

            folium.Circle(
                location=[z['centroid_lat'], z['centroid_lon']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.15,
                weight=2,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=(
                    f"{zone_name} · {z['station']} · "
                    f"{z['violation_count']:,} violations · "
                    f"avg severity {z['avg_severity']:.2f}"
                ),
            ).add_to(zone_layer)

            # Zone label
            folium.Marker(
                location=[z['centroid_lat'], z['centroid_lon']],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        background:{color};color:white;font-weight:bold;
                        font-size:11px;padding:3px 7px;border-radius:4px;
                        white-space:nowrap;box-shadow:1px 1px 3px rgba(0,0,0,0.4)">
                        {zone_name}
                    </div>""",
                    icon_size=(80, 24),
                    icon_anchor=(40, 12),
                ),
                tooltip=z['station'],
            ).add_to(zone_layer)

        zone_layer.add_to(m)

    else:
        # API unreachable — show a warning tile on the map
        folium.Marker(
            location=[13.0, 77.59],
            popup="Could not load zone data from API.",
            icon=folium.Icon(color='red', icon='warning-sign'),
        ).add_to(m)

    # ── Prediction pin ────────────────────────────────────────────────────────
    if prediction_pin:
        p         = prediction_pin
        level     = p.get('severity_level', 'Unknown')
        score     = p.get('severity_score', 0)
        risk_pct  = p.get('risk_probability', 0) * 100
        pin_color = {
            'Low': 'green', 'Medium': 'orange',
            'High': 'red',  'Critical': 'darkred'
        }.get(level, 'blue')

        pin_html = f"""
        <div style="font-family:Arial;min-width:200px">
            <h4 style="margin:0 0 6px">📍 Prediction Result</h4>
            <table style="width:100%;font-size:13px;border-collapse:collapse">
                <tr><td><b>Zone</b></td>
                    <td>{p.get('zone','?')} · {p.get('station','?')}</td></tr>
                <tr><td><b>Severity Score</b></td>
                    <td>{score:.2f} / 5</td></tr>
                <tr><td><b>Severity Level</b></td>
                    <td style="color:{pin_color};font-weight:bold">{level}</td></tr>
                <tr><td><b>High Risk</b></td>
                    <td>{'🔴 Yes' if p.get('is_high_risk') else '🟡 No'}</td></tr>
                <tr><td><b>Risk Probability</b></td>
                    <td>{risk_pct:.1f}%</td></tr>
            </table>
        </div>
        """

        pred_layer = folium.FeatureGroup(name='Prediction Pin', show=True)
        folium.Marker(
            location=[p['lat'], p['lon']],
            icon=folium.Icon(color=pin_color, icon='map-marker', prefix='fa'),
            popup=folium.Popup(pin_html, max_width=260),
            tooltip=f"Prediction: {level} ({score:.2f})",
        ).add_to(pred_layer)
        pred_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m
