"""
Team Rocket - Trading Bot Core
================================
ML-powered MSFT trading bot for the EFREI SIMS Hackathon 2026.

Strategy:
  - Collect 2 years of MSFT historical data via yfinance
  - Engineer technical indicators (RSI, MACD, Bollinger Bands, momentum...)
  - Train a Gradient Boosting Classifier to predict next-day direction
  - Generate BUY / SELL orders with smart position sizing
"""

import os
import joblib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# ── Configuration ────────────────────────────────────────────────────────────
TEAM_ID  = "Team Rocket"
TOKEN    = "1c3c5675-242e-46b1-ad5a-24cfde0261ce"
SYMBOL   = "MSFT"
STATUS_URL = f"https://sims.efrei.educentre.fr/api/v1/startups/{TOKEN}/status"

MODEL_PATH  = "model.joblib"
SCALER_PATH = "scaler.joblib"

FEATURE_COLS = [
    "Return_1d", "Return_5d", "Return_10d",
    "MACD", "MACD_hist", "RSI",
    "BB_position", "Volume_ratio",
    "High_Low_ratio", "Close_Open_ratio",
    "Momentum_5", "Momentum_10",
    "Price_SMA5_ratio", "Price_SMA20_ratio", "SMA5_SMA20_ratio",
    "DayOfWeek",
]

# ── Data & Feature Engineering ────────────────────────────────────────────────

def fetch_data(symbol: str = SYMBOL, period: str = "3y") -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators used as ML features."""
    df = df.copy()

    # ── Returns ──────────────────────────────────────────────────────────────
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)

    # ── Moving averages ───────────────────────────────────────────────────────
    df["SMA_5"]  = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["BB_position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # ── Candle ratios ─────────────────────────────────────────────────────────
    df["High_Low_ratio"]    = df["High"] / df["Low"]
    df["Close_Open_ratio"]  = df["Close"] / df["Open"]

    # ── Momentum ──────────────────────────────────────────────────────────────
    df["Momentum_5"]  = df["Close"] / df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10)

    # ── Price / MA ratios ─────────────────────────────────────────────────────
    df["Price_SMA5_ratio"]  = df["Close"] / df["SMA_5"]
    df["Price_SMA20_ratio"] = df["Close"] / df["SMA_20"]
    df["SMA5_SMA20_ratio"]  = df["SMA_5"] / df["SMA_20"]

    # ── Temporal ──────────────────────────────────────────────────────────────
    df["DayOfWeek"] = df.index.dayofweek

    # ── Target: 1 = next close higher than today's close ─────────────────────
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df


# ── Model Training ────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Train a Gradient Boosting Classifier on all available history.
    Uses TimeSeriesSplit to validate without data leakage.
    Returns (model, scaler).
    """
    X = df[FEATURE_COLS].values
    y = df["Target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        subsample=0.8,
        random_state=42,
    )

    # Validate with time-series split (no look-ahead)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        preds = model.predict(X_scaled[val_idx])
        scores.append(accuracy_score(y[val_idx], preds))

    print(f"[Team Rocket] Cross-val accuracy: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Final fit on ALL data
    model.fit(X_scaled, y)
    return model, scaler


def save_model(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Team Rocket] Model saved → {MODEL_PATH}, {SCALER_PATH}")


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[Team Rocket] No saved model found — training now…")
        df = fetch_data()
        df = add_features(df)
        model, scaler = train_model(df)
        save_model(model, scaler)
    else:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    return model, scaler


# ── Prediction ────────────────────────────────────────────────────────────────

def get_prediction(model, scaler, df: pd.DataFrame):
    """
    Predict whether tomorrow's close > today's close.
    Returns (prediction: int, probabilities: array).
    """
    latest = df[FEATURE_COLS].iloc[[-1]].values
    X_scaled = scaler.transform(latest)
    pred  = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]
    return pred, proba


# ── Order Generation ──────────────────────────────────────────────────────────

def generate_orders(
    prediction: int,
    proba: np.ndarray,
    current_price: float,
    portfolio_status: dict,
) -> list[dict]:
    """
    Convert ML prediction into concrete BUY / SELL orders.

    Rules:
      • BUY price  = close × 1.005  (slight premium → maximises execution)
      • SELL price = close × 0.995  (slight discount → maximises execution)
      • High-confidence (> 55 %): trade normally (up to 15 % of cash, ≤ 20 shares)
      • Low-confidence  (≤ 55 %): submit a minimal 1-share order to avoid the
        daily-no-submission penalty (-0.5 pt/day)
      • Never spend more than available cash; never sell more than held shares
    """
    orders = []
    cash         = float(portfolio_status.get("cash") or 100_000)
    shares_held  = int(portfolio_status.get("msftShares") or 0)
    confidence   = float(proba[prediction])

    # ── High-confidence signal ────────────────────────────────────────────────
    if confidence > 0.55:
        if prediction == 1:  # BUY
            budget   = min(cash * 0.15, cash - 200)
            quantity = max(1, int(budget // current_price))
            quantity = min(quantity, 20)

            if quantity >= 1 and budget >= current_price:
                orders.append({
                    "symbol":   SYMBOL,
                    "side":     "BUY",
                    "quantity": quantity,
                    "price":    round(current_price * 1.005, 2),
                })
                print(f"[Team Rocket] BUY  {quantity} × {SYMBOL} @ {current_price * 1.005:.2f}  (confidence {confidence:.2%})")

        else:  # SELL
            if shares_held > 0:
                quantity = max(1, shares_held // 2)
                orders.append({
                    "symbol":   SYMBOL,
                    "side":     "SELL",
                    "quantity": quantity,
                    "price":    round(current_price * 0.995, 2),
                })
                print(f"[Team Rocket] SELL {quantity} × {SYMBOL} @ {current_price * 0.995:.2f}  (confidence {confidence:.2%})")

    # ── Fallback — always submit something to avoid daily penalty ────────────
    if not orders:
        print(f"[Team Rocket] No order generated (conf={confidence:.2%}, pred={'UP' if prediction==1 else 'DOWN'}, shares={shares_held}) — submitting fallback 1-share BUY.")
        if cash >= current_price * 1.005:
            orders.append({
                "symbol":   SYMBOL,
                "side":     "BUY",
                "quantity": 1,
                "price":    round(current_price * 1.005, 2),
            })

    return orders


# ── Portfolio Status ──────────────────────────────────────────────────────────

def get_portfolio_status() -> dict:
    """Query the EFREI SIMS simulation API for our portfolio state."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(STATUS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[Team Rocket] Warning: could not fetch portfolio status — {e}")
        return {"cash": 100_000, "msftShares": 0}


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """
    End-to-end pipeline: fetch data → predict → generate orders.
    Returns the full TradingBotResponse dict.
    """
    model, scaler       = load_model()
    df                  = fetch_data()
    df                  = add_features(df)
    current_price       = float(df["Close"].iloc[-1])
    prediction, proba   = get_prediction(model, scaler, df)
    portfolio_status    = get_portfolio_status()
    orders              = generate_orders(prediction, proba, current_price, portfolio_status)

    response = {
        "teamId":    TEAM_ID,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "orders":    orders,
    }
    return response


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    print("\n━━━  Team Rocket Trading Bot  ━━━\n")
    result = run_pipeline()
    print("\n── Response ──")
    print(json.dumps(result, indent=2))
    print("\n── Portfolio Status ──")
    print(json.dumps(get_portfolio_status(), indent=2))
