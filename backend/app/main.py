"""SG GenCo Portfolio Optimization & Trading Platform — API."""
import csv
import datetime as dt
import io
import json

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from . import models, seed
from .database import Base, engine, get_db
from .services import bidbands, elasticity, explain, forecasting, optimizer, risk

Base.metadata.create_all(bind=engine)
app = FastAPI(title="SG GenCo Trading & Portfolio Optimization Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

TODAY = dt.date.today()
_db = next(get_db())
seed.seed(_db, TODAY)
_db.close()

# ----------------------------------------------------------------- auth ----
PERMISSIONS = {
    "run_optimization": {"trader", "portfolio_optimizer", "admin"},
    "approve_bids": {"trader", "commercial_manager", "admin"},
    "override": {"trader", "admin"},
    "configure_assets": {"plant_operator", "portfolio_optimizer", "admin"},
    "configure_contracts": {"commercial_manager", "admin"},
    "upload_forecasts": {"trader", "portfolio_optimizer", "risk_manager", "admin"},
    "export": {"trader", "commercial_manager", "risk_manager", "admin"},
}


def current_user(x_token: str = Header(default=""), db: Session = Depends(get_db)) -> models.User:
    user = db.query(models.User).filter_by(api_token=x_token, active=True).first()
    if not user:
        raise HTTPException(401, "Invalid or missing X-Token header")
    return user


def require(action: str):
    def dep(user: models.User = Depends(current_user)):
        if user.role not in PERMISSIONS.get(action, set()):
            raise HTTPException(403, f"Role '{user.role}' cannot perform '{action}'")
        return user
    return dep


def audit(db, user, action, entity=None, entity_id=None, detail=None):
    db.add(models.AuditLog(username=user.username if user else None,
                           role=user.role if user else None, action=action,
                           entity=entity, entity_id=str(entity_id) if entity_id else None,
                           detail=detail or {}))
    db.commit()


class LoginReq(BaseModel):
    username: str


@app.post("/api/auth/login")
def login(req: LoginReq, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(username=req.username, active=True).first()
    if not user:
        raise HTTPException(401, "Unknown user")
    audit(db, user, "login")
    return {"token": user.api_token, "username": user.username,
            "display_name": user.display_name, "role": user.role,
            "permissions": sorted(a for a, roles in PERMISSIONS.items() if user.role in roles)}


@app.get("/api/auth/users")
def demo_users(db: Session = Depends(get_db)):
    return [{"username": u.username, "display_name": u.display_name, "role": u.role}
            for u in db.query(models.User).filter_by(active=True)]


# ------------------------------------------------------------- helpers ----
def _date(d: str | None) -> dt.date:
    return dt.date.fromisoformat(d) if d else TODAY


def _asset(db, code) -> models.Asset:
    a = db.query(models.Asset).filter_by(code=code).first()
    if not a:
        raise HTTPException(404, f"asset {code} not found")
    return a


def build_inputs(db: Session, date: dt.date, shocks: dict | None = None) -> optimizer.PortfolioInputs:
    forecasting.generate_default_forecasts(db, date)
    fc = {k: forecasting.load_forecast(db, k, date)
          for k in ["solar_import", "solar_local", "contract_demand", "usep",
                    "reserve_price", "regulation_price"]}
    p1 = _asset(db, "PP1").params
    p2 = _asset(db, "PP2").params
    bt = _asset(db, "BESS1").params
    imp = _asset(db, "SOLAR-MY").params
    contract = db.query(models.Contract).filter_by(active=True).first()
    hedges = [{"direction": h.direction, "volume_mw": h.volume_mw, "price": h.price,
               "start_interval": h.start_interval, "end_interval": h.end_interval}
              for h in db.query(models.ForwardPosition).filter_by(active=True)]
    return optimizer.PortfolioInputs(
        fc=fc, p1=p1, p2=p2, batt=bt,
        import_limit_mw=imp.get("import_limit_mw", 100),
        import_cost=imp.get("ppa_cost_per_mwh", 62),
        contract_price=contract.contract_price if contract else 145.0,
        under_penalty=contract.under_delivery_penalty if contract else 180.0,
        hedges=hedges, shocks=shocks or {},
        # per-interval price impact forecast from the delayed offer stack
        market={"energy_impact_pct_per_100mw": elasticity.impact_curve(date)})


# ------------------------------------------------------------ portfolio ----
@app.get("/api/portfolio/overview")
def portfolio_overview(date: str | None = None, db: Session = Depends(get_db),
                       user=Depends(current_user)):
    d = _date(date)
    inp = build_inputs(db, d)
    fc = inp.fc
    dem = np.array(fc["contract_demand"]["p50"])
    solar = (np.minimum(np.array(fc["solar_import"]["p50"]), inp.import_limit_mw)
             + np.array(fc["solar_local"]["p50"]))
    run = (db.query(models.OptimizationRun).filter_by(trade_date=d, status="solved")
             .order_by(models.OptimizationRun.id.desc()).first())
    net_open = solar - dem  # before plants/battery
    hedge_mw = sum(h["volume_mw"] * (1 if h["direction"] == "sell" else -1)
                   for h in inp.hedges)
    return {
        "trade_date": d.isoformat(),
        "forecasts": fc,
        "assets": [{"code": a.code, "name": a.name, "type": a.asset_type,
                    "params": a.params} for a in db.query(models.Asset).filter_by(active=True)],
        "contract_demand_p50": fc["contract_demand"]["p50"],
        "net_solar_minus_demand": [round(float(v), 1) for v in net_open],
        "hedge_position_mw": hedge_mw,
        "battery_soc_init": inp.batt.get("soc_init_mwh"),
        "latest_run_id": run.id if run else None,
        "latest_run_summary": run.summary if run else None,
        "kpis": {
            "peak_demand_mw": float(dem.max()),
            "total_demand_mwh": round(float(dem.sum() * 0.5), 0),
            "total_solar_mwh": round(float(solar.sum() * 0.5), 0),
            "max_deficit_mw": round(float((-net_open).max()), 1),
            "avg_usep_forecast": round(float(np.mean(fc["usep"]["p50"])), 1),
            "peak_usep_forecast": round(float(np.max(fc["usep"]["p90"])), 1),
        },
    }


# ------------------------------------------------------------ forecasts ----
@app.get("/api/forecasts/{kind}")
def get_forecast(kind: str, date: str | None = None, paths: int = 0,
                 db: Session = Depends(get_db), user=Depends(current_user)):
    d = _date(date)
    forecasting.generate_default_forecasts(db, d)
    fc = forecasting.load_forecast(db, kind, d)
    if not fc:
        raise HTTPException(404, f"no forecast '{kind}' for {d}")
    out = dict(fc)
    out.update(forecasting.uncertainty_stats(fc))
    if paths:
        out["scenario_paths"] = forecasting.scenario_paths(fc, n_paths=min(paths, 50))
    return out


@app.post("/api/forecasts/{kind}/upload")
async def upload_forecast(kind: str, file: UploadFile, date: str | None = None,
                          db: Session = Depends(get_db),
                          user=Depends(require("upload_forecasts"))):
    d = _date(date)
    text = (await file.read()).decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "interval" not in rows[0] or "p50" not in rows[0]:
        raise HTTPException(400, "CSV must have columns: interval,p50[,p10,p90,sigma]")
    sid = forecasting.ingest_uploaded_forecast(db, kind, d, rows)
    audit(db, user, "upload_forecast", "forecast_series", sid,
          {"kind": kind, "rows": len(rows)})
    return {"series_id": sid, "rows": len(rows)}


class ActualsReq(BaseModel):
    actuals: list[float | None]


@app.post("/api/forecasts/{kind}/actuals")
def post_actuals(kind: str, req: ActualsReq, date: str | None = None,
                 db: Session = Depends(get_db),
                 user=Depends(require("upload_forecasts"))):
    n = forecasting.record_actuals(db, kind, _date(date), req.actuals)
    audit(db, user, "record_actuals", "forecast_series", kind, {"points": n})
    return {"points_recorded": n}


@app.get("/api/forecasts/errors/history")
def forecast_errors(kind: str | None = None, db: Session = Depends(get_db),
                    user=Depends(current_user)):
    q = db.query(models.ForecastError)
    if kind:
        q = q.filter_by(kind=kind)
    errs = q.limit(2000).all()
    return [{"kind": e.kind, "interval": e.interval, "forecast": e.forecast,
             "actual": e.actual, "error": e.error} for e in errs]


# ----------------------------------------------------------- elasticity ----
@app.get("/api/elasticity/forecast")
def elasticity_forecast(date: str | None = None, user=Depends(current_user)):
    """Per-interval price-impact forecast reconstructed from the delayed
    offer stack (factor-graph fusion of stack history, outage notices,
    same-day unit telemetry and weather)."""
    return {"trade_date": _date(date).isoformat(),
            **elasticity.forecast_for_date(_date(date))}


# --------------------------------------------------------------- assets ----
@app.get("/api/assets")
def list_assets(db: Session = Depends(get_db), user=Depends(current_user)):
    return [{"id": a.id, "code": a.code, "name": a.name, "type": a.asset_type,
             "params": a.params, "active": a.active}
            for a in db.query(models.Asset)]


class AssetUpdate(BaseModel):
    params: dict


@app.put("/api/assets/{code}")
def update_asset(code: str, req: AssetUpdate, db: Session = Depends(get_db),
                 user=Depends(require("configure_assets"))):
    a = _asset(db, code)
    old = dict(a.params)
    merged = dict(a.params)
    merged.update(req.params)
    a.params = merged
    db.commit()
    audit(db, user, "update_asset", "asset", code, {"old": old, "new": merged})
    return {"code": code, "params": merged}


# ------------------------------------------------------------ contracts ----
@app.get("/api/contracts")
def list_contracts(db: Session = Depends(get_db), user=Depends(current_user)):
    out = []
    for c in db.query(models.Contract).filter_by(active=True):
        vols = sorted((v for v in c.volumes), key=lambda v: v.interval)
        out.append({"id": c.id, "name": c.name, "counterparty": c.counterparty,
                    "contract_price": c.contract_price,
                    "tolerance_band_pct": c.tolerance_band_pct,
                    "under_delivery_penalty": c.under_delivery_penalty,
                    "over_delivery_price": c.over_delivery_price,
                    "firm": c.firm, "settlement_rule": c.settlement_rule,
                    "priority": c.priority,
                    "demand_uncertainty_pct": c.demand_uncertainty_pct,
                    "volumes_mw": [v.volume_mw for v in vols]})
    return out


class ContractUpdate(BaseModel):
    contract_price: float | None = None
    tolerance_band_pct: float | None = None
    under_delivery_penalty: float | None = None
    over_delivery_price: float | None = None
    demand_uncertainty_pct: float | None = None


@app.put("/api/contracts/{cid}")
def update_contract(cid: int, req: ContractUpdate, db: Session = Depends(get_db),
                    user=Depends(require("configure_contracts"))):
    c = db.query(models.Contract).get(cid)
    if not c:
        raise HTTPException(404, "contract not found")
    changes = {k: v for k, v in req.model_dump().items() if v is not None}
    for k, v in changes.items():
        setattr(c, k, v)
    db.commit()
    audit(db, user, "update_contract", "contract", cid, changes)
    return {"id": cid, "updated": changes}


@app.get("/api/contracts/coverage")
def contract_coverage(date: str | None = None, db: Session = Depends(get_db),
                      user=Depends(current_user)):
    """Required energy, expected shortfall/surplus, penalty exposure, and the
    merit-order cost of serving the contract with each asset."""
    d = _date(date)
    inp = build_inputs(db, d)
    fc = inp.fc
    dem = np.array(fc["contract_demand"]["p50"])
    dem_sig = np.array(fc["contract_demand"]["sigma"])
    solar = (np.minimum(np.array(fc["solar_import"]["p50"]), inp.import_limit_mw)
             + np.array(fc["solar_local"]["p50"]))
    solar_sig = np.hypot(np.array(fc["solar_import"]["sigma"]),
                         np.array(fc["solar_local"]["sigma"]))
    firm = inp.p1["p_max"] + inp.p2["p_max"] + inp.batt["max_discharge_mw"]
    gap = dem - solar
    from scipy.stats import norm
    sig = np.sqrt(dem_sig ** 2 + solar_sig ** 2)
    p_short_no_firm = norm.sf((solar + firm - dem) / np.maximum(sig, 1e-6))
    exp_surplus = np.maximum(-gap, 0)
    contract = db.query(models.Contract).filter_by(active=True).first()
    usep_avg = float(np.mean(fc["usep"]["p50"]))
    return {
        "trade_date": d.isoformat(),
        "required_mw": [round(float(v), 1) for v in dem],
        "solar_available_mw": [round(float(v), 1) for v in solar],
        "gap_mw": [round(float(v), 1) for v in gap],
        "expected_surplus_mw": [round(float(v), 1) for v in exp_surplus],
        "shortfall_prob_with_full_firm": [round(float(v), 4) for v in p_short_no_firm],
        "required_mwh": round(float(dem.sum() * 0.5), 1),
        "max_gap_mw": round(float(gap.max()), 1),
        "penalty_exposure_per_mwh": (contract.under_delivery_penalty + contract.contract_price)
        if contract else None,
        "cost_to_serve": [
            {"source": "Local solar", "marginal_cost": 0.0, "limit_mw": "forecast-dependent"},
            {"source": "Malaysia import", "marginal_cost": inp.import_cost,
             "limit_mw": inp.import_limit_mw},
            {"source": "Plant 1 (CCGT)", "marginal_cost": inp.p1["marginal_cost"],
             "limit_mw": inp.p1["p_max"]},
            {"source": "Battery discharge", "marginal_cost": round(
                usep_avg / inp.batt.get("round_trip_eff", .88) * 0.55
                + inp.batt.get("degradation_cost_per_mwh", 6), 1),
             "limit_mw": inp.batt["max_discharge_mw"]},
            {"source": "Plant 2 (OCGT)", "marginal_cost": inp.p2["marginal_cost"],
             "limit_mw": inp.p2["p_max"]},
            {"source": "Market purchase", "marginal_cost": round(usep_avg * 1.1, 1),
             "limit_mw": "unbounded (price risk)"},
        ],
    }


# ------------------------------------------------------------- hedges ----
@app.get("/api/hedges")
def list_hedges(db: Session = Depends(get_db), user=Depends(current_user)):
    return [{"id": h.id, "name": h.name, "direction": h.direction,
             "volume_mw": h.volume_mw, "price": h.price,
             "start_interval": h.start_interval, "end_interval": h.end_interval,
             "instrument": h.instrument}
            for h in db.query(models.ForwardPosition).filter_by(active=True)]


# ----------------------------------------------------------- optimize ----
class OptimizeReq(BaseModel):
    date: str | None = None
    mode: str = "balanced"            # conservative | balanced | aggressive
    method: str = "stochastic"        # deterministic | stochastic
    shocks: dict = {}


def _run_and_store(db, user, req: OptimizeReq, scenario_name: str | None = None):
    d = _date(req.date)
    inp = build_inputs(db, d, req.shocks)
    result = optimizer.solve(inp, mode=req.mode, method=req.method)
    if result.get("status") != "solved":
        raise HTTPException(500, f"optimization failed: {result.get('message')}")

    for iv in result["intervals"]:
        iv["explanation"] = explain.interval_explanation(iv, inp, req.mode)
    bands = bidbands.generate(result, inp, req.mode)
    rm = risk.metrics(result, inp)
    cases = risk.case_recommendations(result)
    batt = explain.battery_strategy(result, inp)

    run = models.OptimizationRun(
        trade_date=d, mode=req.mode, method=req.method, status="solved",
        objective_value=result["objective_value"],
        expected_profit=result["expected_profit"], cvar_profit=result["cvar_profit"],
        scenario_overrides=req.shocks, created_by=user.username,
        summary={"expected_profit": result["expected_profit"],
                 "cvar_profit": result["cvar_profit"],
                 "shortfall_prob_day": result["shortfall_prob_day"],
                 "worst_scenario": result["worst_scenario"],
                 "scenario_name": scenario_name})
    db.add(run)
    db.flush()
    for iv in result["intervals"]:
        db.add(models.DispatchInterval(
            run_id=run.id, interval=iv["interval"],
            solar_import_mw=iv["solar_import_mw"], solar_local_mw=iv["solar_local_mw"],
            solar_curtailed_mw=iv["solar_curtailed_mw"],
            plant1_mw=iv["plant1_mw"], plant2_mw=iv["plant2_mw"],
            batt_charge_mw=iv["batt_charge_mw"], batt_discharge_mw=iv["batt_discharge_mw"],
            batt_soc_mwh=iv["batt_soc_mwh"], contract_mw=iv["contract_mw"],
            contract_shortfall_mw=iv["contract_shortfall_mw"],
            energy_sell_mw=iv["energy_sell_mw"], energy_buy_mw=iv["energy_buy_mw"],
            reserve_mw=iv["reserve_mw"], regulation_mw=iv["regulation_mw"],
            risk_buffer_mw=iv["risk_buffer_mw"], shortfall_prob=iv["shortfall_prob"],
            imbalance_prob=iv["imbalance_prob"],
            binding_constraint=iv["binding_constraint"], explanation=iv["explanation"]))
    for b in bands:
        db.add(models.BidBand(run_id=run.id, **{k: b[k] for k in
               ["interval", "asset_code", "market", "band_no", "price",
                "quantity_mw", "dispatch_prob", "rationale", "sensitivity"]}))
    for metric, val in rm.items():
        if isinstance(val, (int, float)) and val is not None:
            db.add(models.RiskMetric(run_id=run.id, metric=metric, value=float(val)))
    db.commit()
    audit(db, user, "run_optimization", "optimization_run", run.id,
          {"mode": req.mode, "method": req.method, "shocks": req.shocks})
    return run, result, bands, rm, cases, batt


@app.post("/api/optimize")
def run_optimization(req: OptimizeReq, db: Session = Depends(get_db),
                     user=Depends(require("run_optimization"))):
    run, result, bands, rm, cases, batt = _run_and_store(db, user, req)
    return {"run_id": run.id, "result": result, "bid_bands": bands,
            "risk": rm, "cases": cases, "battery_strategy": batt}


@app.get("/api/runs")
def list_runs(db: Session = Depends(get_db), user=Depends(current_user)):
    runs = (db.query(models.OptimizationRun)
              .order_by(models.OptimizationRun.id.desc()).limit(30).all())
    return [{"id": r.id, "trade_date": r.trade_date.isoformat(), "mode": r.mode,
             "method": r.method, "status": r.status,
             "expected_profit": r.expected_profit, "cvar_profit": r.cvar_profit,
             "shocks": r.scenario_overrides, "created_by": r.created_by,
             "created_at": r.created_at.isoformat(), "summary": r.summary}
            for r in runs]


@app.get("/api/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db), user=Depends(current_user)):
    r = db.query(models.OptimizationRun).get(run_id)
    if not r:
        raise HTTPException(404, "run not found")
    ivs = [{c.name: getattr(iv, c.name) for c in models.DispatchInterval.__table__.columns
            if c.name not in ("id", "run_id")} for iv in r.intervals]
    bands = [{c.name: getattr(b, c.name) for c in models.BidBand.__table__.columns
              if c.name != "run_id"} for b in r.bid_bands]
    rm = {m.metric: m.value for m in
          db.query(models.RiskMetric).filter_by(run_id=run_id)}
    return {"id": r.id, "trade_date": r.trade_date.isoformat(), "mode": r.mode,
            "method": r.method, "expected_profit": r.expected_profit,
            "cvar_profit": r.cvar_profit, "summary": r.summary,
            "shocks": r.scenario_overrides, "intervals": ivs, "bid_bands": bands,
            "risk_metrics": rm}


# ----------------------------------------------------------- scenarios ----
SCENARIO_PRESETS = [
    {"name": "Solar import −10%", "shocks": {"solar_import_factor": 0.9}},
    {"name": "Solar import −20%", "shocks": {"solar_import_factor": 0.8}},
    {"name": "Solar import −50%", "shocks": {"solar_import_factor": 0.5}},
    {"name": "Local solar cloud event", "shocks": {"solar_local_factor": 0.45}},
    {"name": "Contract demand +15%", "shocks": {"demand_factor": 1.15}},
    {"name": "Plant 1 trips", "shocks": {"plant1_trip": True}},
    {"name": "Plant 2 trips", "shocks": {"plant2_trip": True}},
    {"name": "Battery unavailable", "shocks": {"battery_avail": 0.0}},
    {"name": "USEP spike +$250", "shocks": {"usep_spike_adder": 250}},
    {"name": "Reserve price ×3", "shocks": {"reserve_price_factor": 3.0}},
    {"name": "Regulation price ×3", "shocks": {"regulation_price_factor": 3.0}},
    {"name": "Import constraint 50%", "shocks": {"import_limit_factor": 0.5}},
    {"name": "Fuel cost +30%", "shocks": {"fuel_factor": 1.3}},
    {"name": "Thin market (2× price impact)", "shocks": {"market_impact_factor": 2.0}},
    {"name": "Price-taker (no market impact)", "shocks": {"market_impact_factor": 0.0}},
]


@app.get("/api/scenarios/presets")
def scenario_presets(user=Depends(current_user)):
    return SCENARIO_PRESETS


class SimulateReq(OptimizeReq):
    name: str = "custom scenario"
    baseline_run_id: int | None = None


@app.post("/api/scenarios/simulate")
def simulate(req: SimulateReq, db: Session = Depends(get_db),
             user=Depends(require("run_optimization"))):
    run, result, bands, rm, cases, batt = _run_and_store(db, user, req, req.name)
    db.add(models.ScenarioResult(
        run_id=run.id, name=req.name, shocks=req.shocks,
        expected_profit=result["expected_profit"], cvar_profit=result["cvar_profit"],
        shortfall_mwh=result["expected_shortfall_mwh"],
        shortfall_prob=result["shortfall_prob_day"],
        detail={"worst_scenario": result["worst_scenario"]}))
    db.commit()
    baseline = None
    if req.baseline_run_id:
        b = db.query(models.OptimizationRun).get(req.baseline_run_id)
        if b:
            baseline = {"run_id": b.id, "expected_profit": b.expected_profit,
                        "cvar_profit": b.cvar_profit,
                        "delta_expected": round((result["expected_profit"] or 0)
                                                - (b.expected_profit or 0), 2)}
    return {"run_id": run.id, "name": req.name, "result": result,
            "bid_bands": bands, "risk": rm, "cases": cases,
            "battery_strategy": batt, "baseline_comparison": baseline}


@app.get("/api/scenarios/results")
def scenario_results(db: Session = Depends(get_db), user=Depends(current_user)):
    rows = (db.query(models.ScenarioResult)
              .order_by(models.ScenarioResult.id.desc()).limit(30).all())
    return [{"id": s.id, "run_id": s.run_id, "name": s.name, "shocks": s.shocks,
             "expected_profit": s.expected_profit, "cvar_profit": s.cvar_profit,
             "shortfall_mwh": s.shortfall_mwh, "shortfall_prob": s.shortfall_prob,
             "created_at": s.created_at.isoformat()} for s in rows]


# ----------------------------------------------------- bid bands / export ----
class OverrideReq(BaseModel):
    field: str
    new_value: float
    justification: str


@app.post("/api/bidbands/{band_id}/approve")
def approve_band(band_id: int, db: Session = Depends(get_db),
                 user=Depends(require("approve_bids"))):
    b = db.query(models.BidBand).get(band_id)
    if not b:
        raise HTTPException(404, "bid band not found")
    b.status = "approved"
    db.commit()
    audit(db, user, "approve_bid_band", "bid_band", band_id)
    return {"id": band_id, "status": "approved"}


@app.post("/api/runs/{run_id}/approve_all")
def approve_all(run_id: int, db: Session = Depends(get_db),
                user=Depends(require("approve_bids"))):
    n = (db.query(models.BidBand).filter_by(run_id=run_id, status="recommended")
           .update({"status": "approved"}))
    db.commit()
    audit(db, user, "approve_all_bid_bands", "optimization_run", run_id, {"count": n})
    return {"approved": n}


@app.post("/api/bidbands/{band_id}/override")
def override_band(band_id: int, req: OverrideReq, db: Session = Depends(get_db),
                  user=Depends(require("override"))):
    b = db.query(models.BidBand).get(band_id)
    if not b:
        raise HTTPException(404, "bid band not found")
    if req.field not in ("price", "quantity_mw"):
        raise HTTPException(400, "can only override price or quantity_mw")
    if not req.justification.strip():
        raise HTTPException(400, "justification is required for overrides")
    old = getattr(b, req.field)
    setattr(b, req.field, req.new_value)
    b.status = "overridden"
    db.add(models.UserOverride(run_id=b.run_id, bid_band_id=band_id,
                               username=user.username, field=req.field,
                               old_value=str(old), new_value=str(req.new_value),
                               justification=req.justification))
    db.commit()
    audit(db, user, "override_bid_band", "bid_band", band_id,
          {"field": req.field, "old": old, "new": req.new_value,
           "justification": req.justification})
    return {"id": band_id, "field": req.field, "old": old, "new": req.new_value}


@app.get("/api/runs/{run_id}/export/bids", response_class=PlainTextResponse)
def export_bids(run_id: int, db: Session = Depends(get_db),
                user=Depends(require("export"))):
    bands = (db.query(models.BidBand).filter_by(run_id=run_id)
               .order_by(models.BidBand.interval, models.BidBand.asset_code,
                         models.BidBand.band_no).all())
    if not bands:
        raise HTTPException(404, "no bid bands for run")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["interval", "period_starting", "asset", "market", "band",
                "price_sgd_mwh", "quantity_mw", "status"])
    for b in bands:
        h, m = divmod(b.interval * 30, 60)
        w.writerow([b.interval, f"{h:02d}:{m:02d}", b.asset_code, b.market,
                    b.band_no, b.price, b.quantity_mw, b.status])
    audit(db, user, "export_bids", "optimization_run", run_id)
    return out.getvalue()


@app.get("/api/runs/{run_id}/export/risk")
def export_risk(run_id: int, db: Session = Depends(get_db),
                user=Depends(require("export"))):
    rm = {m.metric: m.value for m in
          db.query(models.RiskMetric).filter_by(run_id=run_id)}
    r = db.query(models.OptimizationRun).get(run_id)
    audit(db, user, "export_risk_report", "optimization_run", run_id)
    return {"run_id": run_id, "trade_date": r.trade_date.isoformat() if r else None,
            "mode": r.mode if r else None, "summary": r.summary if r else None,
            "metrics": rm}


# ---------------------------------------------------------------- audit ----
@app.get("/api/audit")
def audit_trail(limit: int = 100, db: Session = Depends(get_db),
                user=Depends(current_user)):
    rows = (db.query(models.AuditLog).order_by(models.AuditLog.id.desc())
              .limit(min(limit, 500)).all())
    return [{"id": a.id, "username": a.username, "role": a.role, "action": a.action,
             "entity": a.entity, "entity_id": a.entity_id, "detail": a.detail,
             "at": a.created_at.isoformat()} for a in rows]


@app.get("/api/overrides")
def list_overrides(db: Session = Depends(get_db), user=Depends(current_user)):
    rows = (db.query(models.UserOverride).order_by(models.UserOverride.id.desc())
              .limit(100).all())
    return [{"id": o.id, "run_id": o.run_id, "bid_band_id": o.bid_band_id,
             "username": o.username, "field": o.field, "old": o.old_value,
             "new": o.new_value, "justification": o.justification,
             "at": o.created_at.isoformat()} for o in rows]


# serve built frontend if present
import os
_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
