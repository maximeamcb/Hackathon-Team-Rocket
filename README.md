# 🚀 Team Rocket — MSFT Trading Bot
**EFREI SIMS Hackathon 2026** • Berké / Zago / Boquet

---

## Overview

Team Rocket's trading bot is an **ML-powered MSFT day-trading system** that:

1. Collects 3 years of MSFT daily OHLCV data (via Yahoo Finance)
2. Engineers 16 technical indicators as features
3. Trains a **Gradient Boosting Classifier** to predict next-day price direction
4. Generates BUY / SELL orders sized conservatively to maximize execution
5. Exposes a **public REST API** called daily by the EFREI SIMS evaluation system

---

## Machine Learning Approach

### Features (16 total)
| Category | Features |
|---|---|
| Returns | 1-day, 5-day, 10-day returns |
| Trend | MACD, MACD histogram |
| Momentum | 5-day & 10-day momentum ratios |
| Oscillator | RSI (14) |
| Volatility | Bollinger Band position |
| Volume | Volume / 20-day avg |
| Candle | High/Low ratio, Close/Open ratio |
| MA ratios | Price/SMA5, Price/SMA20, SMA5/SMA20 |
| Temporal | Day of week |

### Model
- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
- **Target**: Binary — will tomorrow's close > today's close?
- **Validation**: TimeSeriesSplit (5 folds, no look-ahead bias)
- **Typical accuracy**: ~55–58%

### Trading Logic
- **BUY** when model predicts UP with confidence > 55%
  - Quantity: 15% of available cash, capped at 20 shares
  - Price: `close × 1.005` (slight premium → ensures execution within day's range)
- **SELL** when model predicts DOWN with confidence > 55% AND we hold shares
  - Quantity: 50% of current holdings
  - Price: `close × 0.995` (slight discount → ensures execution within day's range)

---

## Project Structure

```
├── bot.py          # Core ML logic (data, features, model, orders)
├── api.py          # FastAPI server — public endpoint
├── train.py        # Standalone training script
├── requirements.txt
├── Dockerfile      # Cloud deployment
├── Procfile        # Railway / Heroku
└── README.md
```

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (saves model.joblib + scaler.joblib)
python train.py

# 3. Start the API server
uvicorn api:app --host 0.0.0.0 --port 8000

# 4. Test the orders endpoint
curl -X POST http://localhost:8000/orders

# 5. Check portfolio status
curl http://localhost:8000/status
```

---

## API Reference

### `POST /orders`
Called daily by the EFREI SIMS evaluation system.

**Response:**
```json
{
  "teamId": "Team Rocket",
  "timestamp": "2026-03-24T18:00:00Z",
  "orders": [
    {
      "symbol": "MSFT",
      "side": "BUY",
      "quantity": 5,
      "price": 382.50
    }
  ]
}
```

### `GET /status`
Returns current portfolio state from the EFREI SIMS simulation API.

```json
{
  "teamId": "...",
  "status": "ACTIVE",
  "cash": 95000.0,
  "realizedPnL": 1200.0,
  "executedOrdersCount": 12,
  "ignoredOrdersCount": 2,
  "msftShares": 10
}
```

### `POST /retrain`
Force a model re-train with the latest market data.

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

---

## Cloud Deployment (Railway — recommended)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select `maximeamcb/Hackathon-Team-Rocket`
4. Railway auto-detects the `Dockerfile` and deploys
5. Copy the generated URL (e.g. `https://hackathon-team-rocket.up.railway.app`)
6. Your public endpoint: `POST https://hackathon-team-rocket.up.railway.app/orders`

> **Alternative:** [render.com](https://render.com) → New Web Service → same steps.

---

## Portfolio Monitoring

```bash
# Check Team Rocket portfolio directly
curl https://sims.efrei.educentre.fr/api/v1/startups/1c3c5675-242e-46b1-ad5a-24cfde0261ce/status
```

---

## Token
`1c3c5675-242e-46b1-ad5a-24cfde0261ce`
