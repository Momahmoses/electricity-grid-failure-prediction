# Electricity Grid Failure & Outage Prediction

A GIS + ML system for predicting power grid failure points across Nigerian Distribution Companies (DisCos) using infrastructure age, weather data, load levels, and maintenance history — enabling proactive intervention to reduce outage downtime.

## Overview

Trains a Random Forest classifier on grid segment data to:
- Predict failure probability per line/transformer segment
- Rank segments by maintenance priority
- Visualize risk distribution per DisCo service territory
- Generate actionable maintenance priority reports

## Features

- **Risk Scoring**: 12 features including line age, load factor, weather, vegetation encroachment
- **DisCo-level Analysis**: Risk breakdown for Eko, Ikeja, Abuja, Kano, Enugu, Port Harcourt DisCos
- **Interactive Heatmap**: Folium map showing failure risk intensity
- **Priority Report**: Top 20 highest-risk segments for immediate maintenance

## Project Structure

```
electricity-grid-failure-prediction/
├── src/
│   ├── data_ingestion.py    # Grid data generation
│   ├── model.py             # RF training & ROC evaluation
│   └── visualization.py     # Risk maps & DisCo summaries
├── data/sample/
├── outputs/
├── config.py
├── main.py
└── requirements.txt
```

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

## Target DisCos

| DisCo | City | Coverage |
|-------|------|---------|
| Eko | Lagos | Island & Mainland South |
| Ikeja | Lagos | Mainland North |
| Abuja | FCT | Federal Capital Territory |
| Kano | Kano | Northwest Zone |
| Enugu | Enugu | Southeast Zone |
| Port Harcourt | Rivers | South-South Zone |

## Data Sources (Production)

- Grid topology: TCN/NERC grid maps
- Weather: NIMET historical data, ERA5
- Maintenance logs: DisCo SCADA systems
- Outage records: NERC outage reports

## Author

**MOMAH MOSES .C.**  
Data Scientist & ML Engineer | [GitHub](https://github.com/Momahmoses)
