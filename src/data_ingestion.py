"""Generate synthetic power grid infrastructure & failure data."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SAMPLE_DIR, DISCOS, FEATURE_COLS, TARGET_COL


def generate_grid_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    disco_names = list(DISCOS.keys())
    discos = rng.choice(disco_names, size=n_samples)

    lats, lons = [], []
    for d in discos:
        info = DISCOS[d]
        offset = 0.4
        lats.append(info["lat"] + rng.uniform(-offset, offset))
        lons.append(info["lon"] + rng.uniform(-offset, offset))

    line_age = rng.uniform(1, 45, n_samples)
    load_factor = rng.uniform(0.4, 1.2, n_samples)
    temperature = rng.normal(32, 6, n_samples).clip(18, 48)
    rainfall = rng.exponential(40, n_samples).clip(0, 300)
    wind_speed = rng.exponential(4, n_samples).clip(0, 25)
    humidity = rng.uniform(40, 95, n_samples)
    veg_encroachment = rng.uniform(0, 1, n_samples)
    days_since_maint = rng.exponential(90, n_samples).clip(1, 730)
    past_failures = rng.poisson(2, n_samples)
    transformer_load = rng.uniform(0.5, 1.3, n_samples)
    line_length = rng.exponential(8, n_samples).clip(0.5, 60)
    pop_density = rng.lognormal(7, 1.5, n_samples).clip(100, 50000)

    # Failure probability: physics-based scoring
    risk = (
        (line_age / 45) * 0.20 +
        np.clip(load_factor - 0.8, 0, 0.4) / 0.4 * 0.15 +
        (temperature / 48) * 0.10 +
        (rainfall / 300) * 0.10 +
        (wind_speed / 25) * 0.08 +
        veg_encroachment * 0.12 +
        np.clip(days_since_maint / 730, 0, 1) * 0.15 +
        (past_failures / 10) * 0.10
    )
    failure = (risk + rng.normal(0, 0.05, n_samples) > 0.45).astype(int)

    df = pd.DataFrame({
        "latitude": lats, "longitude": lons, "disco": discos,
        "line_age_years": line_age, "load_factor": load_factor,
        "temperature_c": temperature, "rainfall_mm": rainfall,
        "wind_speed_ms": wind_speed, "humidity_pct": humidity,
        "vegetation_encroachment": veg_encroachment,
        "days_since_last_maintenance": days_since_maint,
        "n_past_failures_12m": past_failures,
        "transformer_load_pct": transformer_load * 100,
        "line_length_km": line_length,
        "population_density": pop_density,
        TARGET_COL: failure,
    })
    return df


def load_or_generate():
    path = os.path.join(SAMPLE_DIR, "grid_data.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = generate_grid_data()
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved {len(df):,} grid records")

    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
