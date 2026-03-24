"""
Team Rocket - Trading Bot API
==============================
FastAPI server that exposes the public endpoint called by the EFREI SIMS
evaluation system every trading day.

Endpoints:
  POST /orders   → returns today's BUY/SELL orders (TradingBotResponse)
  GET  /status   → returns Team Rocket's portfolio status
  GET  /health   → liveness probe
  POST /retrain  → force a model re-train with fresh data
"""

import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from bot import (
    TEAM_ID,
    fetch_data,
    add_features,
    train_model,
    save_model,
    load_model,
    get_prediction,
    generate_orders,
    get_portfolio_status,
    run_pipeline,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Team Rocket — MSFT Trading Bot",
    description="EFREI SIMS Hackathon 2026 • Berké / Zago / Boquet",
    version="1.0.0",
)

# Pre-load model at startup so the first request is fast
@app.on_event("startup")
async def startup_event():
    print("[Team Rocket] Loading model on startup…")
    load_model()
    print("[Team Rocket] Ready 🚀")


# ── Main endpoint (called daily by the evaluation system) ─────────────────────
@app.post("/orders", summary="Generate next-day trading orders")
async def post_orders():
    """
    Called by the EFREI SIMS evaluation system every trading day.
    Returns BUY / SELL orders for the next trading session.
    """
    try:
        response = run_pipeline()
        return JSONResponse(content=response)
    except Exception as exc:
        # Even on error, return a valid (empty orders) response so we don't
        # get a daily submission penalty.
        fallback = {
            "teamId":    TEAM_ID,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "orders":    [],
        }
        print(f"[Team Rocket] ERROR in pipeline: {exc}")
        return JSONResponse(content=fallback, status_code=200)


# ── Portfolio status ──────────────────────────────────────────────────────────
@app.get("/status", summary="Get Team Rocket portfolio status")
async def get_status():
    """Query the EFREI SIMS simulation API and return our current portfolio."""
    data = get_portfolio_status()
    return JSONResponse(content=data)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", summary="Liveness probe")
async def health():
    return {"status": "ok", "team": TEAM_ID, "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Manual retrain ────────────────────────────────────────────────────────────
@app.post("/retrain", summary="Force model re-train with latest data")
async def retrain():
    """Re-download 3 years of MSFT data and re-train the model from scratch."""
    try:
        df = fetch_data()
        df = add_features(df)
        model, scaler = train_model(df)
        save_model(model, scaler)
        return {"status": "success", "rows_used": len(df), "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
