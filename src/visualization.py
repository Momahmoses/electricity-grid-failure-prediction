"""Maps and charts for grid failure risk."""

import folium
from folium.plugins import HeatMap
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import OUTPUTS_DIR, DISCOS, FAILURE_RISK_THRESHOLDS

RISK_COLORS = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c", "critical": "#8e44ad"}


def classify(p):
    t = FAILURE_RISK_THRESHOLDS
    if p < t["low"]: return "low"
    elif p < t["medium"]: return "medium"
    elif p < t["high"]: return "high"
    return "critical"


def create_grid_risk_map(gdf: gpd.GeoDataFrame, probs: np.ndarray):
    m = folium.Map(location=[8.0, 6.5], zoom_start=6, tiles="CartoDB positron")
    heat = [[row["latitude"], row["longitude"], p] for (_, row), p in zip(gdf.iterrows(), probs)]
    HeatMap(heat, radius=18, blur=22, min_opacity=0.3, name="Failure Risk Heatmap").add_to(m)

    high_layer = folium.FeatureGroup(name="High Risk Lines").add_to(m)
    for (_, row), p in zip(gdf.iterrows(), probs):
        if p > FAILURE_RISK_THRESHOLDS["medium"]:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5, color=RISK_COLORS[classify(p)], fill=True, fill_opacity=0.7,
                popup=f"DisCo: {row['disco']}<br>Risk: {classify(p).upper()}<br>"
                      f"Prob: {p:.2%}<br>Age: {row['line_age_years']:.0f}y<br>"
                      f"Load: {row['load_factor']:.2f}",
            ).add_to(high_layer)

    for disco, info in DISCOS.items():
        folium.Marker(
            [info["lat"], info["lon"]],
            popup=f"<b>{disco} DisCo</b><br>{info['city']}",
            icon=folium.Icon(color="blue", icon="flash"),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, "grid_risk_map.html")
    m.save(out)
    print(f"Grid risk map saved → {out}")


def maintenance_priority_report(gdf: gpd.GeoDataFrame, probs: np.ndarray):
    gdf = gdf.copy()
    gdf["failure_prob"] = probs
    gdf["risk_level"] = gdf["failure_prob"].apply(classify)
    top = gdf.nlargest(20, "failure_prob")[
        ["disco", "latitude", "longitude", "failure_prob", "line_age_years",
         "days_since_last_maintenance", "load_factor", "risk_level"]
    ]
    top.to_csv(os.path.join(OUTPUTS_DIR, "maintenance_priority.csv"), index=False)

    disco_risk = gdf.groupby("disco")["failure_prob"].agg(["mean", "max"]).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(disco_risk))
    ax.bar(disco_risk["disco"], disco_risk["mean"], label="Avg Risk", color="steelblue", alpha=0.8)
    ax.bar(disco_risk["disco"], disco_risk["max"], label="Max Risk", color="coral", alpha=0.5, width=0.4)
    ax.set_ylabel("Failure Probability")
    ax.set_title("Grid Failure Risk by DisCo")
    ax.axhline(FAILURE_RISK_THRESHOLDS["high"], color="red", ls="--", lw=1, label="High threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "disco_risk_summary.png"), dpi=150)
    plt.close()
    print(f"Maintenance priority report saved. Top priority: {top.iloc[0]['disco']} DisCo")
