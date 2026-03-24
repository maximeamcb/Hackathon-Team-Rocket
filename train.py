"""
Team Rocket - Model Training Script
=====================================
Run this script once to train the ML model and save it to disk.
The API will load the saved model on startup.

Usage:
    python train.py
"""

from bot import fetch_data, add_features, train_model, save_model

if __name__ == "__main__":
    print("━━━  Team Rocket — Model Training  ━━━\n")
    print("Downloading MSFT data (3 years)…")
    df = fetch_data(period="3y")
    print(f"  → {len(df)} trading days loaded.\n")

    print("Engineering features…")
    df = add_features(df)
    print(f"  → {len(df)} rows after feature engineering.\n")

    print("Training Gradient Boosting Classifier…")
    model, scaler = train_model(df)

    print("\nSaving model to disk…")
    save_model(model, scaler)

    print("\n✓ Training complete! You can now start the API with:")
    print("  uvicorn api:app --host 0.0.0.0 --port 8000\n")
