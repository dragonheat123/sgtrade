# SG GenCo Trading & Portfolio Optimization Platform

A demo trading platform for a Singapore genco: stochastic portfolio
optimization (MILP over a scenario tree), NEMS-style bid band generation,
risk metrics, and a market-elasticity model in which the genco's own offers
move the clearing price — with the elasticity forecast reconstructed from
the delayed offer stack via factor-graph fusion.

## Layout

```
backend/    FastAPI app (Python)
  app/services/optimizer.py    two-stage scenario MILP, price-maker market model
  app/services/elasticity.py   elasticity forecast from the delayed offer stack
  app/services/forecasting.py  synthetic P50/P10/P90 forecasts + actuals
  app/services/bidbands.py     NEMS-style bid band recommendations
  app/services/risk.py         CVaR/VaR, exposure metrics, case advice
  demo_elasticity.py           demo: price-taker vs price-maker optimization
  demo_stack_elasticity.py     demo: offer-stack elasticity forecast + ramps
frontend/   React + Vite + ECharts UI
```

## Prerequisites

- Python 3.11+
- Node 18+ (only needed for the frontend)

## Run the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

On first start the app creates and seeds a SQLite database (`sgtrade.db` in
the working directory) with assets, a retail contract, a hedge, forecasts
and demo users. Set `DATABASE_URL=postgresql://...` to use PostgreSQL
instead.

API docs: http://127.0.0.1:8000/docs

All API calls need an `X-Token` header. Demo tokens equal the usernames
(e.g. `trader1`, `optimizer1`, `risk1`, `operator1`, `commercial1`,
`admin`); fetch the list from `GET /api/auth/users` or log in via
`POST /api/auth/login` with `{"username": "trader1"}`.

Quick smoke test:

```bash
curl -s -X POST http://127.0.0.1:8000/api/optimize \
  -H 'X-Token: trader1' -H 'Content-Type: application/json' \
  -d '{"mode": "balanced", "method": "stochastic"}' | head -c 400

curl -s 'http://127.0.0.1:8000/api/elasticity/forecast' -H 'X-Token: trader1'
```

## Run the frontend

Dev mode (proxies `/api` to the backend on port 8000):

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173
```

Or build it and let FastAPI serve the static bundle directly:

```bash
cd frontend && npm install && npm run build
# restart the backend; it mounts frontend/dist at http://127.0.0.1:8000/
```

Log in with any demo user (no password; the token is the username).

## Run the demos

Both are standalone scripts (no server or database needed):

```bash
cd backend
pip install -r requirements.txt

python demo_elasticity.py        # price-taker vs price-maker: the optimizer
                                 # withholds volume at the peak, and the
                                 # "reality check" settles each plan at the
                                 # prices its own volume actually causes

python demo_stack_elasticity.py  # elasticity forecast from delayed offer
                                 # stacks: factor-graph fusion of stack
                                 # history, outage notices, unit telemetry
                                 # and weather; ramp-constrained re-clearing
                                 # couples elasticity across intervals; the
                                 # resulting curve re-shapes the optimal plan
```

## Market elasticity in the optimizer

The optimizer treats the genco as a price maker: cleared volume moves the
price along a linear residual-demand curve, linearized with offer blocks
priced at marginal revenue (stays MILP, no extra binaries). Key knobs:

- `optimizer.DEFAULT_MARKET` — impact defaults (% price move per 100 MW,
  scalar or length-48 curve) and block count.
- `PortfolioInputs.market` — per-run overrides. The API's `build_inputs`
  feeds in the per-interval curve from `elasticity.forecast_for_date`.
- shock `{"market_impact_factor": 0.0}` — price-taker run (legacy
  behaviour); `2.0` — thin market. Both exist as scenario presets.
- `GET /api/elasticity/forecast` — the forecast curve with posteriors,
  clearing prices, supply cushion and ramp diagnostics.

The synthetic publication feed in `elasticity.synthesize_published_stacks`
stands in for real EMC offer-stack ingestion; replace that one function to
go live.
