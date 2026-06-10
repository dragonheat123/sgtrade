"""Demo: market elasticity (price-maker) in the portfolio optimizer.

The genco is large relative to the SG market, so its cleared volume moves the
clearing price:  USEP_eff(t) = USEP(t) * (1 - impact * net_sales(t)).

This script runs the same trading day three ways:

  A. price-taker      — old behaviour, prices assumed fixed
  B. elastic market   — default impact (3% per 100 MW on energy)
  C. thin market      — 2x impact via the `market_impact_factor` shock

and then "settles" each plan at the prices its own volume would actually
cause, showing how much profit the price-taker plan overstates.

Run from backend/:  python demo_elasticity.py
"""
import time

import numpy as np

from app.services import optimizer
from app.services.forecasting import (_demand_shape, _solar_shape, _usep_shape,
                                      DEFAULT_SIGMA_PCT, Z10, Z90)

DT = optimizer.DT


def fc_entry(p50, sigma_pct):
    p50 = np.asarray(p50, dtype=float)
    sig = np.abs(p50) * sigma_pct
    return {"p50": p50.tolist(), "sigma": sig.tolist(),
            "p10": np.maximum(0.0, p50 + Z10 * sig).tolist(),
            "p90": (p50 + Z90 * sig).tolist()}


def build_inputs(**overrides) -> optimizer.PortfolioInputs:
    """Same portfolio as app.seed, built without the database."""
    usep = _usep_shape()
    fc = {
        "solar_import": fc_entry(_solar_shape(120.0, 0.92), DEFAULT_SIGMA_PCT["solar_import"]),
        "solar_local": fc_entry(_solar_shape(60.0, 0.88), DEFAULT_SIGMA_PCT["solar_local"]),
        "contract_demand": fc_entry(_demand_shape(180.0, 320.0), DEFAULT_SIGMA_PCT["contract_demand"]),
        "usep": fc_entry(usep, DEFAULT_SIGMA_PCT["usep"]),
        "reserve_price": fc_entry(np.maximum(8.0, usep * 0.16 - 4.0), DEFAULT_SIGMA_PCT["reserve_price"]),
        "regulation_price": fc_entry(np.maximum(12.0, usep * 0.20 - 2.0), DEFAULT_SIGMA_PCT["regulation_price"]),
    }
    return optimizer.PortfolioInputs(
        fc=fc,
        p1={"p_min": 80, "p_max": 220, "ramp_mw_per_interval": 60, "marginal_cost": 95,
            "startup_cost": 8000, "shutdown_cost": 2000, "reserve_cap_mw": 40,
            "regulation_cap_mw": 15, "init_on": True, "init_mw": 120},
        p2={"p_min": 40, "p_max": 150, "ramp_mw_per_interval": 120, "marginal_cost": 162,
            "startup_cost": 3000, "shutdown_cost": 800, "reserve_cap_mw": 60,
            "regulation_cap_mw": 10, "init_on": False, "init_mw": 0},
        batt={"capacity_mwh": 200, "max_charge_mw": 50, "max_discharge_mw": 50,
              "round_trip_eff": 0.88, "degradation_cost_per_mwh": 6,
              "soc_init_mwh": 100, "soc_end_target_mwh": 80, "soc_min_mwh": 10,
              "contract_buffer_mwh": 40},
        import_limit_mw=100, import_cost=62, contract_price=145.0, under_penalty=180.0,
        hedges=[{"direction": "sell", "volume_mw": 50.0, "price": 118.0,
                 "start_interval": 0, "end_interval": 47}],
        **overrides)


def settle_under_elasticity(result: dict, imp_e: float, hedge_mw: float) -> dict:
    """Re-settle a plan's base-scenario energy + hedge position at the prices
    its own net volume would actually cause (uniform-price settlement)."""
    planned, realized = 0.0, 0.0
    for iv in result["intervals"]:
        usep, sell, buy = iv["usep"], iv["energy_sell_mw"], iv["energy_buy_mw"]
        usep_eff = usep * (1 - imp_e * (sell - buy))
        planned += (usep * sell - usep * optimizer.BUY_PREMIUM * buy) * DT
        realized += (usep_eff * sell - usep_eff * optimizer.BUY_PREMIUM * buy) * DT
        # sell-side CFD gains when the genco's own sales depress the spot price
        realized += hedge_mw * (usep - usep_eff) * DT
    return {"planned": planned, "realized": realized,
            "slippage": planned - realized}


def run(label, inp):
    t0 = time.time()
    res = optimizer.solve(inp, mode="balanced", method="stochastic")
    assert res["status"] == "solved", res
    print(f"  [{label}] solved in {time.time() - t0:.1f}s")
    return res


def main():
    print("=" * 78)
    print("MARKET ELASTICITY DEMO — genco offers move the clearing price")
    print("=" * 78)

    print("\nSolving the same trading day under three market assumptions...")
    res_a = run("A  price-taker         ", build_inputs(market={"enabled": False}))
    res_b = run("B  elastic (default)   ", build_inputs())
    res_c = run("C  thin market (2x)    ", build_inputs(shocks={"market_impact_factor": 2.0}))

    runs = [("A price-taker", res_a), ("B elastic", res_b), ("C thin market", res_c)]

    print("\n--- Plan comparison " + "-" * 58)
    hdr = f"{'':24s}{'A price-taker':>16s}{'B elastic':>16s}{'C thin (2x)':>16s}"
    print(hdr)
    rows = [
        ("expected profit ($)", lambda r: f"{r['expected_profit']:>16,.0f}"),
        ("CVaR profit ($)", lambda r: f"{r['cvar_profit']:>16,.0f}"),
        ("energy sold (MWh)", lambda r: f"{r['market_impact']['energy_sold_mwh']:>16,.0f}"),
        ("energy bought (MWh)", lambda r: f"{r['market_impact']['energy_bought_mwh']:>16,.0f}"),
        ("avg USEP forecast", lambda r: f"{r['market_impact']['avg_usep_forecast']:>16,.1f}"),
        ("avg sell price recvd", lambda r: f"{r['market_impact']['avg_sell_price_received']:>16,.1f}"),
        ("max price impact ($)", lambda r: f"{r['market_impact']['max_price_impact']:>16,.1f}"),
        ("reserve offered (MWh)", lambda r: f"{sum(iv['reserve_mw'] for iv in r['intervals']) * DT:>16,.0f}"),
    ]
    for name, fmt in rows:
        print(f"{name:24s}" + "".join(fmt(r) for _, r in runs))

    print("\n--- Evening peak detail (price-taker vs elastic) " + "-" * 29)
    print(f"{'period':>8s}{'USEP fc':>9s} | {'A sell MW':>10s}{'A eff $':>9s} | "
          f"{'B sell MW':>10s}{'B eff $':>9s}{'withheld':>10s}")
    imp_e = res_b["market_impact"]["energy_impact_pct_per_100mw"] / 10000.0
    for t in range(34, 44):
        a, b = res_a["intervals"][t], res_b["intervals"][t]
        h, m = divmod(t * 30, 60)
        a_eff = a["usep"] * (1 - imp_e * (a["energy_sell_mw"] - a["energy_buy_mw"]))
        print(f"{h:02d}:{m:02d}   {a['usep']:>9,.0f} | {a['energy_sell_mw']:>10,.0f}"
              f"{a_eff:>9,.0f} | {b['energy_sell_mw']:>10,.0f}"
              f"{b['usep_effective']:>9,.0f}"
              f"{a['energy_sell_mw'] - b['energy_sell_mw']:>10,.0f}")

    print("\n--- Reality check: settle each plan at the prices it actually causes ---")
    print(f"(elastic truth: {res_b['market_impact']['energy_impact_pct_per_100mw']:.1f}% "
          f"price move per 100 MW net sales; 50 MW sell CFD settles on moved price)\n")
    hedge_mw = 50.0
    st_a = settle_under_elasticity(res_a, imp_e, hedge_mw)
    st_b = settle_under_elasticity(res_b, imp_e, hedge_mw)
    print(f"{'':26s}{'planned $':>14s}{'realized $':>14s}{'slippage $':>14s}")
    print(f"{'A price-taker plan':26s}{st_a['planned']:>14,.0f}{st_a['realized']:>14,.0f}"
          f"{st_a['slippage']:>14,.0f}")
    print(f"{'B elasticity-aware plan':26s}{st_b['planned']:>14,.0f}{st_b['realized']:>14,.0f}"
          f"{st_b['slippage']:>14,.0f}")
    imp_c = res_c["market_impact"]["energy_impact_pct_per_100mw"] / 10000.0
    st_a2 = settle_under_elasticity(res_a, imp_c, hedge_mw)
    st_c2 = settle_under_elasticity(res_c, imp_c, hedge_mw)
    print(f"{'A plan in thin market':26s}{st_a2['planned']:>14,.0f}{st_a2['realized']:>14,.0f}"
          f"{st_a2['slippage']:>14,.0f}")
    print(f"{'C thin-market plan':26s}{st_c2['planned']:>14,.0f}{st_c2['realized']:>14,.0f}"
          f"{st_c2['slippage']:>14,.0f}")
    print(f"\n  -> The price-taker plan books ${st_a['slippage']:,.0f} of energy revenue "
          f"that evaporates\n     once its own volume moves the price "
          f"(${st_a2['slippage']:,.0f} in the thin market).\n     "
          f"The elastic plan withholds volume at the peak, keeps the price up, and\n     "
          f"realizes ${st_b['realized'] - st_a['realized']:,.0f} more spot margin "
          f"(${st_c2['realized'] - st_a2['realized']:,.0f} in the thin market).")
    print(f"\n  B's expected-profit figure (${res_b['expected_profit']:,.0f}) already "
          f"prices the move in;\n  A's (${res_a['expected_profit']:,.0f}) is only "
          f"achievable if the genco had no price impact.")


if __name__ == "__main__":
    main()
