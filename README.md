# Electricity Grid Failure & Outage Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GIS + ML system predicting power grid failure points across Nigerian Distribution Companies (DisCos) using infrastructure age, weather exposure, load levels, and maintenance history — enabling proactive intervention to reduce outage downtime.

---

## Problem Statement

Nigeria's electricity grid suffers chronic outages averaging 12–20 hours per day. DisCos react to failures instead of preventing them. This system predicts failure probability per grid segment, enabling targeted preventive maintenance.

---

## Features

| Feature | Description |
|---------|-------------|
| Failure Probability Scoring | Random Forest on 12 asset condition features |
| Maintenance Priority Ranking | Top 20 highest-risk segments for immediate action |
| DisCo-Level Risk Breakdown | Coverage for Eko, Ikeja, Abuja, Kano, Enugu, Port Harcourt DisCos |
| Interactive Risk Heatmap | Folium map showing failure risk intensity across service areas |
| Priority Report Generation | Exportable CSV of maintenance work orders |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Machine Learning | Random Forest, scikit-learn |
| Geospatial | GeoPandas, Folium |
| Data | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## Project Structure

```
electricity-grid-failure-prediction/
├── src/
│   ├── data_loader.py     # Grid asset and weather data ingestion
│   ├── model.py           # Random Forest training and prediction pipeline
│   └── visualize.py       # Failure heatmap and risk distribution charts
├── data/raw/              # Grid topology, outage logs, weather data
├── models/                # Saved model artifacts
├── config.py              # Model parameters, risk thresholds
├── main.py                # Pipeline entry point
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/electricity-grid-failure-prediction.git
cd electricity-grid-failure-prediction
pip install -r requirements.txt
python main.py
```

---

## Data Sources

- NERC/TCN grid topology and fault history
- DisCo asset registers (transformer age, capacity, load factor)
- NIMET weather data (temperature, humidity, lightning)
- FERMA road access data for maintenance logistics

---

## Author

**Momah Moses** — Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
