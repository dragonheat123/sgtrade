"""SQLAlchemy data model for the SG trading & portfolio optimization platform.

Covers: assets, forecasts (+errors), contracts, plant/battery constraints,
market prices, optimization runs, dispatch schedules, market allocations,
bid bands, scenario results, risk metrics, settlement, user overrides,
users/roles, audit log.
"""
import datetime as dt

from sqlalchemy import (JSON, Boolean, Column, Date, DateTime, Float,
                        ForeignKey, Integer, String, Text)
from sqlalchemy.orm import relationship

from .database import Base


def now():
    return dt.datetime.utcnow()


# ---------------------------------------------------------------- assets ----
class Asset(Base):
    __tablename__ = "assets"
    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)   # e.g. PP1, BESS1
    name = Column(String, nullable=False)
    asset_type = Column(String, nullable=False)  # solar_import | solar_local | thermal | battery
    # generic technical parameters (interpreted per asset_type)
    params = Column(JSON, default=dict)
    # thermal: p_min, p_max, ramp_mw_per_interval, marginal_cost, startup_cost,
    #          shutdown_cost, min_up_intervals, min_down_intervals, forced_outage_prob,
    #          emissions_t_per_mwh
    # battery: capacity_mwh, max_charge_mw, max_discharge_mw, round_trip_eff,
    #          degradation_cost_per_mwh, soc_init_mwh, soc_end_target_mwh,
    #          contract_buffer_mwh
    # solar_import: capacity_mw, import_limit_mw
    # solar_local: capacity_mw
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=now)


# ------------------------------------------------------------- forecasts ----
class ForecastSeries(Base):
    """One forecast run for one quantity on one trading date (48 half-hour intervals)."""
    __tablename__ = "forecast_series"
    id = Column(Integer, primary_key=True)
    kind = Column(String, nullable=False)
    # solar_import | solar_local | contract_demand | usep | reserve_price |
    # regulation_price | plant_outage | battery_availability
    trade_date = Column(Date, nullable=False)
    source = Column(String, default="internal")  # internal | uploaded
    created_at = Column(DateTime, default=now)
    points = relationship("ForecastPoint", back_populates="series",
                          cascade="all, delete-orphan", order_by="ForecastPoint.interval")


class ForecastPoint(Base):
    __tablename__ = "forecast_points"
    id = Column(Integer, primary_key=True)
    series_id = Column(Integer, ForeignKey("forecast_series.id"), nullable=False)
    interval = Column(Integer, nullable=False)        # 0..47 half-hourly
    p50 = Column(Float, nullable=False)
    p10 = Column(Float)
    p90 = Column(Float)
    sigma = Column(Float)                             # forecast error std-dev
    actual = Column(Float)                            # filled after the fact
    series = relationship("ForecastSeries", back_populates="points")


class ForecastError(Base):
    """Realised forecast errors, used for error-distribution learning."""
    __tablename__ = "forecast_errors"
    id = Column(Integer, primary_key=True)
    kind = Column(String, nullable=False)
    trade_date = Column(Date, nullable=False)
    interval = Column(Integer, nullable=False)
    forecast = Column(Float)
    actual = Column(Float)
    error = Column(Float)
    pct_error = Column(Float)


# ------------------------------------------------------------- contracts ----
class Contract(Base):
    __tablename__ = "contracts"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    counterparty = Column(String)
    contract_price = Column(Float, nullable=False)        # $/MWh
    tolerance_band_pct = Column(Float, default=0.0)       # +/- band before penalties
    under_delivery_penalty = Column(Float, default=0.0)   # $/MWh beyond tolerance
    over_delivery_price = Column(Float, default=0.0)      # $/MWh paid for excess (opportunity cost)
    firm = Column(Boolean, default=True)                  # take-or-pay / firm delivery
    settlement_rule = Column(String, default="interval")  # interval | daily
    priority = Column(Integer, default=1)                 # 1 = serve before market trading
    demand_uncertainty_pct = Column(Float, default=5.0)
    active = Column(Boolean, default=True)
    volumes = relationship("ContractVolume", back_populates="contract",
                           cascade="all, delete-orphan")


class ContractVolume(Base):
    __tablename__ = "contract_volumes"
    id = Column(Integer, primary_key=True)
    contract_id = Column(Integer, ForeignKey("contracts.id"), nullable=False)
    trade_date = Column(Date, nullable=False)
    interval = Column(Integer, nullable=False)
    volume_mw = Column(Float, nullable=False)
    contract = relationship("Contract", back_populates="volumes")


class ForwardPosition(Base):
    """Forward / hedge book."""
    __tablename__ = "forward_positions"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    direction = Column(String, default="sell")     # sell | buy
    volume_mw = Column(Float, nullable=False)      # flat MW across hedge window
    price = Column(Float, nullable=False)          # $/MWh strike
    start_interval = Column(Integer, default=0)
    end_interval = Column(Integer, default=47)
    trade_date = Column(Date, nullable=False)
    instrument = Column(String, default="cfd")     # cfd | physical
    active = Column(Boolean, default=True)


# ---------------------------------------------------------- optimization ----
class OptimizationRun(Base):
    __tablename__ = "optimization_runs"
    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    mode = Column(String, default="balanced")      # conservative | balanced | aggressive
    method = Column(String, default="stochastic")  # deterministic | stochastic
    status = Column(String, default="pending")     # pending | solved | failed
    objective_value = Column(Float)                # expected risk-adjusted profit
    expected_profit = Column(Float)
    cvar_profit = Column(Float)                    # CVaR(alpha) of profit
    scenario_overrides = Column(JSON, default=dict)
    summary = Column(JSON, default=dict)
    created_by = Column(String)
    created_at = Column(DateTime, default=now)
    intervals = relationship("DispatchInterval", back_populates="run",
                             cascade="all, delete-orphan", order_by="DispatchInterval.interval")
    bid_bands = relationship("BidBand", back_populates="run", cascade="all, delete-orphan")


class DispatchInterval(Base):
    """Recommended schedule + market allocation per interval (expected / base scenario)."""
    __tablename__ = "dispatch_intervals"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    interval = Column(Integer, nullable=False)
    # dispatch (MW)
    solar_import_mw = Column(Float, default=0)
    solar_local_mw = Column(Float, default=0)
    solar_curtailed_mw = Column(Float, default=0)
    plant1_mw = Column(Float, default=0)
    plant2_mw = Column(Float, default=0)
    batt_charge_mw = Column(Float, default=0)
    batt_discharge_mw = Column(Float, default=0)
    batt_soc_mwh = Column(Float, default=0)
    # market allocation (MW)
    contract_mw = Column(Float, default=0)
    contract_shortfall_mw = Column(Float, default=0)
    energy_sell_mw = Column(Float, default=0)
    energy_buy_mw = Column(Float, default=0)
    reserve_mw = Column(Float, default=0)
    regulation_mw = Column(Float, default=0)
    risk_buffer_mw = Column(Float, default=0)
    # economics & risk
    expected_profit = Column(Float, default=0)
    shortfall_prob = Column(Float, default=0)
    imbalance_prob = Column(Float, default=0)
    binding_constraint = Column(String)
    explanation = Column(Text)
    run = relationship("OptimizationRun", back_populates="intervals")


class BidBand(Base):
    __tablename__ = "bid_bands"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    interval = Column(Integer, nullable=False)
    asset_code = Column(String, nullable=False)
    market = Column(String, default="energy")      # energy | reserve | regulation
    band_no = Column(Integer, nullable=False)      # 1..10 (NEMS allows up to 10 bands)
    price = Column(Float, nullable=False)          # $/MWh offer price
    quantity_mw = Column(Float, nullable=False)
    dispatch_prob = Column(Float)                  # P(band clears)
    rationale = Column(Text)
    sensitivity = Column(JSON, default=dict)       # {solar_error: ..., demand_error: ...}
    status = Column(String, default="recommended") # recommended | approved | overridden | exported
    run = relationship("OptimizationRun", back_populates="bid_bands")


# --------------------------------------------------------- scenarios/risk ----
class ScenarioResult(Base):
    __tablename__ = "scenario_results"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"))
    name = Column(String, nullable=False)
    shocks = Column(JSON, default=dict)
    expected_profit = Column(Float)
    cvar_profit = Column(Float)
    shortfall_mwh = Column(Float)
    shortfall_prob = Column(Float)
    detail = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)


class RiskMetric(Base):
    __tablename__ = "risk_metrics"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"))
    metric = Column(String, nullable=False)   # var_95, cvar_95, shortfall_prob, ...
    value = Column(Float)
    unit = Column(String)
    detail = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)


class SettlementResult(Base):
    __tablename__ = "settlement_results"
    id = Column(Integer, primary_key=True)
    trade_date = Column(Date, nullable=False)
    interval = Column(Integer, nullable=False)
    scheduled_mw = Column(Float)
    actual_mw = Column(Float)
    usep_actual = Column(Float)
    imbalance_mwh = Column(Float)
    imbalance_cost = Column(Float)
    contract_revenue = Column(Float)
    market_revenue = Column(Float)
    total_pnl = Column(Float)


# --------------------------------------------------------- users / audit ----
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    role = Column(String, nullable=False)
    # trader | portfolio_optimizer | risk_manager | plant_operator | commercial_manager | admin
    api_token = Column(String, unique=True, nullable=False)
    active = Column(Boolean, default=True)


class UserOverride(Base):
    __tablename__ = "user_overrides"
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"))
    bid_band_id = Column(Integer, ForeignKey("bid_bands.id"))
    username = Column(String, nullable=False)
    field = Column(String, nullable=False)
    old_value = Column(String)
    new_value = Column(String)
    justification = Column(Text, nullable=False)
    created_at = Column(DateTime, default=now)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    username = Column(String)
    role = Column(String)
    action = Column(String, nullable=False)
    entity = Column(String)
    entity_id = Column(String)
    detail = Column(JSON, default=dict)
    created_at = Column(DateTime, default=now)
