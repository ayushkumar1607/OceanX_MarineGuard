"""
MarineGuard — Visualization Utilities
Folium map generation, Plotly radar charts, timeline, and risk color coding.
"""

import folium
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional
import json
import numpy as np
from config import MAP_TILE_STYLE, MAP_ATTR


# ─── Color Palette ───────────────────────────────────────────────────────────
RISK_COLORS = {
    "critical": "#FF3B30",
    "high":     "#FF9500",
    "medium":   "#FFCC00",
    "low":      "#34C759",
    "none":     "#8E8E93",
}

SLICK_COLOR = "#FF6B35"
ORIGIN_ZONE_COLOR = "#4ECDC4"
DRIFT_PATH_COLOR = "#45B7D1"
VESSEL_TRACK_COLORS = [
    "#FF3B30", "#FF9500", "#FFCC00", "#34C759", "#007AFF",
    "#5856D6", "#AF52DE", "#FF2D55", "#A2845E", "#8E8E93",
    "#00C7BE", "#30B0C7", "#32ADE6", "#5AC8FA",
]


def get_risk_color(score: float) -> str:
    """Get color based on attribution score."""
    if score >= 0.75:
        return RISK_COLORS["critical"]
    elif score >= 0.50:
        return RISK_COLORS["high"]
    elif score >= 0.30:
        return RISK_COLORS["medium"]
    elif score >= 0.10:
        return RISK_COLORS["low"]
    else:
        return RISK_COLORS["none"]


def get_risk_label(score: float) -> str:
    """Get risk label based on attribution score."""
    if score >= 0.75:
        return "CRITICAL"
    elif score >= 0.50:
        return "HIGH"
    elif score >= 0.30:
        return "MEDIUM"
    elif score >= 0.10:
        return "LOW"
    else:
        return "NONE"


def create_investigation_map(
    center: List[float],
    zoom: int = 10,
    slick_polygon: Optional[List[List[float]]] = None,
    origin_zone: Optional[List[List[float]]] = None,
    drift_path: Optional[List[List[float]]] = None,
    vessels: Optional[List[Dict[str, Any]]] = None,
    detection_point: Optional[List[float]] = None,
    origin_point: Optional[List[float]] = None,
) -> folium.Map:
    """
    Create the main investigation map with all layers.
    
    Layers:
    - Slick polygon (orange fill)
    - Origin zone (teal dashed border)
    - Drift path (blue animated line)
    - Vessel tracks (color-coded by risk)
    - Detection & origin markers
    """
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
    )
    
    # Add the base map with a clean name for the layer control
    folium.TileLayer(
        tiles=MAP_TILE_STYLE,
        attr=MAP_ATTR if MAP_ATTR else "Map Data",
        name="Esri Dark Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # ── Layer: Slick Polygon ──
    if slick_polygon:
        slick_group = folium.FeatureGroup(name="🛢️ Oil Slick Detection")
        folium.Polygon(
            locations=slick_polygon,
            color=SLICK_COLOR,
            weight=3,
            fill=True,
            fill_color=SLICK_COLOR,
            fill_opacity=0.35,
            popup="<b>Detected Oil Slick</b>",
            tooltip="Oil Slick",
        ).add_to(slick_group)
        slick_group.add_to(m)

    # ── Layer: Origin Zone ──
    if origin_zone:
        origin_group = folium.FeatureGroup(name="🎯 Probable Origin Zone")
        folium.Polygon(
            locations=origin_zone,
            color=ORIGIN_ZONE_COLOR,
            weight=2,
            dash_array="10 6",
            fill=True,
            fill_color=ORIGIN_ZONE_COLOR,
            fill_opacity=0.15,
            popup="<b>Probable Origin Zone</b><br>Estimated release area",
            tooltip="Origin Zone",
        ).add_to(origin_group)
        origin_group.add_to(m)

    # ── Layer: Drift Path ──
    if drift_path:
        drift_group = folium.FeatureGroup(name="🌊 Backward Drift Path")
        folium.PolyLine(
            locations=drift_path,
            color=DRIFT_PATH_COLOR,
            weight=3,
            opacity=0.8,
            dash_array="8 4",
            popup="<b>Backward Drift Path</b><br>Wind + Current reconstruction",
            tooltip="Drift Path",
        ).add_to(drift_group)

        # Add arrow markers along drift path
        if len(drift_path) > 2:
            step = max(1, len(drift_path) // 5)
            for i in range(0, len(drift_path) - 1, step):
                folium.CircleMarker(
                    location=drift_path[i],
                    radius=3,
                    color=DRIFT_PATH_COLOR,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"Drift point {i}",
                ).add_to(drift_group)

        drift_group.add_to(m)

    # ── Layer: Vessel Tracks ──
    if vessels:
        vessel_group = folium.FeatureGroup(name="🚢 Vessel Tracks")
        for idx, vessel in enumerate(vessels):
            color = VESSEL_TRACK_COLORS[idx % len(VESSEL_TRACK_COLORS)]
            score = vessel.get("attribution_score", 0)
            risk_color = get_risk_color(score)

            track = vessel.get("track", [])
            name = vessel.get("name", f"Vessel {idx + 1}")
            mmsi = vessel.get("mmsi", "Unknown")
            vessel_type = vessel.get("vessel_type", "Unknown")

            # Draw vessel track
            if len(track) >= 2:
                folium.PolyLine(
                    locations=[[p["lat"], p["lon"]] for p in track],
                    color=risk_color,
                    weight=2.5,
                    opacity=0.7,
                    popup=f"<b>{name}</b><br>MMSI: {mmsi}<br>Type: {vessel_type}<br>Score: {score:.0%}",
                    tooltip=f"{name} ({score:.0%})",
                ).add_to(vessel_group)

            # Draw AIS gap segments (dashed red)
            gaps = vessel.get("ais_gaps", [])
            for gap in gaps:
                if "start_pos" in gap and "end_pos" in gap:
                    folium.PolyLine(
                        locations=[
                            [gap["start_pos"]["lat"], gap["start_pos"]["lon"]],
                            [gap["end_pos"]["lat"], gap["end_pos"]["lon"]],
                        ],
                        color="#FF3B30",
                        weight=3,
                        opacity=0.9,
                        dash_array="4 8",
                        popup=f"<b>⚠️ AIS Gap</b><br>{name}<br>Duration: {gap.get('duration_min', '?')} min",
                        tooltip=f"AIS Gap ({gap.get('duration_min', '?')} min)",
                    ).add_to(vessel_group)

            # Vessel position marker (last known)
            if track:
                last_pos = track[-1]
                folium.Marker(
                    location=[last_pos["lat"], last_pos["lon"]],
                    popup=f"""
                    <div style="font-family: monospace; min-width: 200px;">
                        <b style="color:{risk_color}">{'🔴' if score >= 0.5 else '🟡' if score >= 0.3 else '🟢'} {name}</b><br>
                        <hr style="margin: 4px 0;">
                        <b>MMSI:</b> {mmsi}<br>
                        <b>Type:</b> {vessel_type}<br>
                        <b>Score:</b> {score:.1%}<br>
                        <b>Risk:</b> {get_risk_label(score)}<br>
                        <b>Speed:</b> {last_pos.get('speed_knots', '?')} kn<br>
                        <b>Heading:</b> {last_pos.get('heading', '?')}°
                    </div>
                    """,
                    icon=folium.Icon(
                        color="red" if score >= 0.5 else "orange" if score >= 0.3 else "green",
                        icon="ship",
                        prefix="fa",
                    ),
                    tooltip=f"{name} ({score:.0%})",
                ).add_to(vessel_group)

        vessel_group.add_to(m)

    # ── Markers: Detection Point ──
    if detection_point:
        folium.Marker(
            location=detection_point,
            popup="<b>🛢️ Slick Detection Centroid</b>",
            icon=folium.Icon(color="orange", icon="exclamation-triangle", prefix="fa"),
            tooltip="Slick Centroid",
        ).add_to(m)

    # ── Markers: Origin Point ──
    if origin_point:
        folium.Marker(
            location=origin_point,
            popup="<b>🎯 Estimated Origin Point</b>",
            icon=folium.Icon(color="blue", icon="bullseye", prefix="fa"),
            tooltip="Origin Point",
        ).add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m


def create_attribution_radar_chart(scores: Dict[str, float], vessel_name: str) -> go.Figure:
    """
    Create a radar (spider) chart showing multi-factor attribution score breakdown.
    """
    categories = list(scores.keys())
    values = list(scores.values())
    # Close the polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=[c.replace("_", " ").title() for c in categories],
        fill="toself",
        fillcolor="rgba(78, 205, 196, 0.25)",
        line=dict(color="#4ECDC4", width=2),
        marker=dict(size=8, color="#4ECDC4"),
        name=vessel_name,
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                gridcolor="rgba(255,255,255,0.1)",
                linecolor="rgba(255,255,255,0.1)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.2)",
                linecolor="rgba(255,255,255,0.2)",
            ),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        margin=dict(l=60, r=60, t=40, b=40),
        height=350,
    )

    return fig


def create_score_comparison_chart(vessels: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a horizontal bar chart comparing overall attribution scores.
    """
    names = [v.get("name", f"Vessel {i+1}") for i, v in enumerate(vessels)]
    scores = [v.get("attribution_score", 0) for v in vessels]
    colors = [get_risk_color(s) for s in scores]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=names,
        x=scores,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255,255,255,0.3)", width=1),
        ),
        text=[f"{s:.0%}" for s in scores],
        textposition="auto",
        textfont=dict(color="white", size=13, family="monospace"),
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%",
            gridcolor="rgba(255,255,255,0.08)",
            title="Attribution Score",
        ),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        margin=dict(l=10, r=20, t=10, b=40),
        height=max(200, len(vessels) * 50),
    )

    return fig


def create_timeline_chart(events: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a forensic timeline showing the sequence of events:
    spill release → drift → detection → investigation.
    """
    fig = go.Figure()

    colors_map = {
        "release":    "#FF3B30",
        "drift":      "#45B7D1",
        "detection":  "#FF9500",
        "ais_gap":    "#FF2D55",
        "vessel":     "#34C759",
        "analysis":   "#5856D6",
    }

    for i, event in enumerate(events):
        event_type = event.get("type", "analysis")
        color = colors_map.get(event_type, "#8E8E93")

        fig.add_trace(go.Scatter(
            x=[event.get("time", "")],
            y=[event.get("label", f"Event {i+1}")],
            mode="markers+text",
            marker=dict(size=16, color=color, symbol="diamond"),
            text=[event.get("description", "")],
            textposition="middle right",
            textfont=dict(size=11, color="white"),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        xaxis=dict(
            title="Time (UTC)",
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=300,
    )

    return fig


def create_confidence_gauge(confidence: float, label: str = "Detection Confidence") -> go.Figure:
    """
    Create a gauge chart for detection confidence.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(size=36, color="white")),
        title=dict(text=label, font=dict(size=14, color="rgba(255,255,255,0.7)")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="white"),
            bar=dict(color="#4ECDC4"),
            bgcolor="rgba(255,255,255,0.05)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color="rgba(255, 59, 48, 0.2)"),
                dict(range=[50, 75], color="rgba(255, 204, 0, 0.2)"),
                dict(range=[75, 100], color="rgba(52, 199, 89, 0.2)"),
            ],
            threshold=dict(
                line=dict(color="white", width=2),
                thickness=0.75,
                value=confidence * 100,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=250,
        margin=dict(l=30, r=30, t=50, b=20),
    )

    return fig


def create_factor_breakdown_chart(factors: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Stacked bar chart showing score contribution from each factor per vessel.
    factors = { vessel_name: { factor_name: factor_score, ... }, ... }
    """
    factor_names = list(list(factors.values())[0].keys()) if factors else []
    vessel_names = list(factors.keys())

    factor_colors = {
        "proximity":  "#FF6B35",
        "temporal":   "#45B7D1",
        "trajectory": "#4ECDC4",
        "ais_gap":    "#FF3B30",
        "behavioral": "#5856D6",
    }

    fig = go.Figure()

    for factor in factor_names:
        fig.add_trace(go.Bar(
            name=factor.replace("_", " ").title(),
            y=vessel_names,
            x=[factors[v].get(factor, 0) for v in vessel_names],
            orientation="h",
            marker_color=factor_colors.get(factor, "#8E8E93"),
        ))

    fig.update_layout(
        barmode="stack",
        xaxis=dict(title="Weighted Score Contribution", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=10, r=20, t=40, b=40),
        height=max(250, len(vessel_names) * 50),
    )

    return fig


def create_synthetic_sar_image_plot(scenario: Dict[str, Any]) -> go.Figure:
    """Generate a realistic pseudo-SAR image representation based on scenario parameters."""
    sar_data = scenario.get("sar_data", {})
    height = sar_data.get("image_height", 512)
    width = sar_data.get("image_width", 512)
    has_slick = sar_data.get("has_slick", False)
    slick_params = sar_data.get("slick_params", {})
    seed = sar_data.get("seed", 42)
    
    rng = np.random.RandomState(seed)
    
    # 1. Base ocean backscatter (speckle noise)
    base_intensity = 0.5
    image = rng.normal(base_intensity, 0.15, (height, width))
    image = np.clip(image, 0.05, 1.0)
    
    # 2. Add slick (lower backscatter / dark patch)
    if has_slick and slick_params:
        y, x = np.ogrid[:height, :width]
        cy, cx = slick_params.get("center_row", 256), slick_params.get("center_col", 256)
        r = slick_params.get("radius", 50)
        elongation = slick_params.get("elongation", 1.0)
        angle_deg = slick_params.get("angle", 0)
        dampening = slick_params.get("dampening", 0.7)
        
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate coordinates
        x_rot = cos_a * (x - cx) - sin_a * (y - cy)
        y_rot = sin_a * (x - cx) + cos_a * (y - cy)
        
        # Ellipse distance equation
        dist_sq = (x_rot / (r * elongation))**2 + (y_rot / r)**2
        
        # Create soft edge mask
        mask = np.exp(-dist_sq * 3.0)
        
        # Apply dampening to backscatter
        image = image * (1.0 - (mask * dampening))
        
    image_uint8 = (image * 255).astype(np.uint8)
    
    fig = px.imshow(image_uint8, color_continuous_scale="gray")
    fig.update_layout(
        title="Sentinel-1 SAR Intensity Map",
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=400,
    )
    return fig
