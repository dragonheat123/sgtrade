"""Demo seed: assets, contract, hedge, users (one per role)."""
import datetime as dt

from sqlalchemy.orm import Session

from . import models
from .services import forecasting


def seed(db: Session, trade_date: dt.date):
    if db.query(models.Asset).count() == 0:
        db.add_all([
            models.Asset(code="SOLAR-MY", name="Malaysia Solar Import", asset_type="solar_import",
                         params={"capacity_mw": 120, "import_limit_mw": 100,
                                 "ppa_cost_per_mwh": 62}),
            models.Asset(code="SOLAR-SG", name="Local Solar Portfolio", asset_type="solar_local",
                         params={"capacity_mw": 60, "marginal_cost": 0}),
            models.Asset(code="PP1", name="Power Plant 1 (CCGT)", asset_type="thermal",
                         params={"p_min": 80, "p_max": 220, "ramp_mw_per_interval": 60,
                                 "marginal_cost": 95, "startup_cost": 8000,
                                 "shutdown_cost": 2000, "min_up_intervals": 4,
                                 "min_down_intervals": 4, "forced_outage_prob": 0.02,
                                 "reserve_cap_mw": 40, "regulation_cap_mw": 15,
                                 "init_on": True, "init_mw": 120,
                                 "emissions_t_per_mwh": 0.37}),
            models.Asset(code="PP2", name="Power Plant 2 (OCGT peaker)", asset_type="thermal",
                         params={"p_min": 40, "p_max": 150, "ramp_mw_per_interval": 120,
                                 "marginal_cost": 162, "startup_cost": 3000,
                                 "shutdown_cost": 800, "min_up_intervals": 2,
                                 "min_down_intervals": 2, "forced_outage_prob": 0.03,
                                 "reserve_cap_mw": 60, "regulation_cap_mw": 10,
                                 "init_on": False, "init_mw": 0,
                                 "emissions_t_per_mwh": 0.55}),
            models.Asset(code="BESS1", name="Battery Energy Storage", asset_type="battery",
                         params={"capacity_mwh": 200, "max_charge_mw": 50,
                                 "max_discharge_mw": 50, "round_trip_eff": 0.88,
                                 "degradation_cost_per_mwh": 6,
                                 "soc_init_mwh": 100, "soc_end_target_mwh": 80,
                                 "soc_min_mwh": 10, "contract_buffer_mwh": 40}),
        ])

    if db.query(models.User).count() == 0:
        for uname, disp, role in [
            ("trader1", "Tan Wei (Trader)", "trader"),
            ("optimizer1", "Priya N (Portfolio Optimizer)", "portfolio_optimizer"),
            ("risk1", "Daniel L (Risk Manager)", "risk_manager"),
            ("operator1", "Hafiz R (Plant Operator)", "plant_operator"),
            ("commercial1", "Sarah K (Commercial Manager)", "commercial_manager"),
            ("admin", "System Admin", "admin"),
        ]:
            db.add(models.User(username=uname, display_name=disp, role=role,
                               api_token=uname))  # demo: token == username

    db.commit()
    forecasting.generate_default_forecasts(db, trade_date)

    if db.query(models.Contract).count() == 0:
        c = models.Contract(
            name="Retail Block — JTC Industrial", counterparty="JTC Estates Pte Ltd",
            contract_price=145.0, tolerance_band_pct=5.0,
            under_delivery_penalty=180.0, over_delivery_price=40.0,
            firm=True, settlement_rule="interval", priority=1,
            demand_uncertainty_pct=5.0)
        db.add(c)
        db.flush()
        dem = forecasting.load_forecast(db, "contract_demand", trade_date)
        for t in range(forecasting.N_INTERVALS):
            db.add(models.ContractVolume(contract_id=c.id, trade_date=trade_date,
                                         interval=t, volume_mw=dem["p50"][t]))

    if db.query(models.ForwardPosition).count() == 0:
        db.add(models.ForwardPosition(
            name="Q2 baseload CFD", direction="sell", volume_mw=50.0, price=118.0,
            start_interval=0, end_interval=47, trade_date=trade_date, instrument="cfd"))
    db.commit()
