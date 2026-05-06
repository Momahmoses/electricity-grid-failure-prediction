import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Nigerian Distribution Companies (DisCos)
DISCOS = {
    "Eko": {"lat": 6.4698, "lon": 3.5852, "city": "Lagos"},
    "Ikeja": {"lat": 6.6018, "lon": 3.3515, "city": "Lagos"},
    "Abuja": {"lat": 9.0579, "lon": 7.4951, "city": "Abuja"},
    "Kano": {"lat": 12.0022, "lon": 8.5920, "city": "Kano"},
    "Enugu": {"lat": 6.4584, "lon": 7.5464, "city": "Enugu"},
    "Port_Harcourt": {"lat": 4.8156, "lon": 7.0498, "city": "PH"},
}

FEATURE_COLS = [
    "line_age_years", "load_factor", "temperature_c", "rainfall_mm",
    "wind_speed_ms", "humidity_pct", "vegetation_encroachment",
    "days_since_last_maintenance", "n_past_failures_12m",
    "transformer_load_pct", "line_length_km", "population_density",
]
TARGET_COL = "failure_occurred"

RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

FAILURE_RISK_THRESHOLDS = {"low": 0.3, "medium": 0.55, "high": 0.75}
