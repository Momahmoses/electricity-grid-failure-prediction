"""Main pipeline: Electricity Grid Failure & Outage Prediction."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion import load_or_generate
from src.model import train, evaluate
from src.visualization import create_grid_risk_map, maintenance_priority_report


def main():
    print("=" * 60)
    print("  Electricity Grid Failure & Outage Prediction")
    print("  Nigerian DisCos: Eko, Ikeja, Abuja, Kano, Enugu, PH")
    print("=" * 60)

    print("\n[1/4] Loading grid infrastructure data...")
    gdf = load_or_generate()
    print(f"  {len(gdf):,} grid segments | Failure rate: {gdf['failure_occurred'].mean():.2%}")

    print("\n[2/4] Training failure prediction model...")
    clf, scaler, X_test, y_test, test_gdf = train(gdf)

    print("\n[3/4] Evaluating model...")
    probs = evaluate(clf, X_test, y_test)

    print("\n[4/4] Generating risk maps and maintenance reports...")
    create_grid_risk_map(test_gdf, probs)
    maintenance_priority_report(test_gdf, probs)

    print("\n✓ Pipeline complete. Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
