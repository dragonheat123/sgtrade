"""Risk engine: portfolio risk metrics from a solved optimization result,
plus base / conservative / stress / worst-credible case recommendations.
"""
import numpy as np

DT = 0.5


def metrics(result: dict, inp) -> dict:
    ivs = result["intervals"]
    fc = inp.fc
    sig_si = np.array(fc["solar_import"]["sigma"], dtype=float)
    sig_sl = np.array(fc["solar_local"]["sigma"], dtype=float)
    sig_dem = np.array(fc["contract_demand"]["sigma"], dtype=float)
    usep = np.array([iv["usep"] for iv in ivs])

    open_mw = np.array([iv["energy_sell_mw"] - iv["energy_buy_mw"] for iv in ivs])
    hedged_mw = sum(h["volume_mw"] for h in inp.hedges if h.get("direction", "sell") == "sell")
    price_exposure = float(np.sum(np.abs(open_mw) * DT))           # MWh exposed to spot
    hedge_cover = hedged_mw * len(ivs) * DT
    hedge_eff = float(min(1.0, hedge_cover / price_exposure)) if price_exposure > 0 else 1.0

    sp = result["scenario_profits"]
    profits = np.array([s["profit"] for s in sp])
    probs = np.array([s["prob"] for s in sp])

    imb = np.array([iv["imbalance_prob"] for iv in ivs])
    exp_imb_mwh = float(np.sum(np.sqrt(sig_si ** 2 + sig_sl ** 2 + sig_dem ** 2)
                               * imb * DT * 0.8))

    return {
        "expected_profit": result["expected_profit"],
        "var95_profit": result["var95_profit"],
        "cvar90_profit": result["cvar_profit"],
        "worst_case_profit": float(profits.min()) if len(profits) else None,
        "worst_case_name": sp[int(np.argmin(profits))]["name"] if len(sp) else None,
        "shortfall_prob_day": result["shortfall_prob_day"],
        "expected_shortfall_mwh": result["expected_shortfall_mwh"],
        "max_interval_shortfall_prob": float(max(iv["shortfall_prob"] for iv in ivs)),
        "expected_imbalance_mwh": round(exp_imb_mwh, 1),
        "solar_error_exposure_mwh": round(float(np.sum(np.sqrt(sig_si**2 + sig_sl**2)) * DT), 1),
        "demand_error_exposure_mwh": round(float(np.sum(sig_dem) * DT), 1),
        "plant_outage_exposure": next((s["profit"] - result["expected_profit"]
                                       for s in sp if "plant" in s["name"]), None),
        "battery_outage_exposure": next((s["profit"] - result["expected_profit"]
                                         for s in sp if "battery" in s["name"]), None),
        "market_price_exposure_mwh": round(price_exposure, 1),
        "hedge_effectiveness": round(hedge_eff, 3),
        "avg_risk_buffer_mw": round(float(np.mean([iv["risk_buffer_mw"] for iv in ivs])), 1),
        "peak_usep": float(usep.max()),
        "scenario_profits": sp,
    }


def case_recommendations(result: dict) -> list[dict]:
    """Recommended posture under base / conservative / stress / worst-credible cases."""
    ivs = result["intervals"]
    peak = [iv for iv in ivs if 34 <= iv["interval"] <= 42]
    avg_buf = float(np.mean([iv["risk_buffer_mw"] for iv in ivs]))
    peak_res = float(np.mean([iv["reserve_mw"] for iv in peak])) if peak else 0.0
    sp = result["scenario_profits"]
    worst = min(sp, key=lambda s: s["profit"])
    return [
        {"case": "base",
         "action": "Run the recommended schedule. Offer the flexibility tranche (band 3) "
                   "into energy; keep the planned reserve/regulation allocation.",
         "expected_profit": result["expected_profit"]},
        {"case": "conservative",
         "action": f"Pull band-3 energy offers in the evening peak and raise the risk buffer "
                   f"above {avg_buf:.0f} MW. Pre-buy expected contract gap before 17:00 when "
                   f"USEP is lower; hold battery SoC near full into the peak.",
         "expected_profit": None},
        {"case": "stress",
         "action": f"If solar tracks P10 by midday: start/raise Plant 2 early (ramp limits "
                   f"bind later), cancel battery arbitrage discharge and reassign "
                   f"{peak_res:.0f} MW of reserve back to energy for contract cover.",
         "expected_profit": None},
        {"case": "worst_credible",
         "action": f"Worst credible scenario is '{worst['name']}' "
                   f"(P&L {worst['profit']:,.0f}). Cap exposure: buy the full expected "
                   f"shortfall forward, hold both plants at the level where reserve "
                   f"commitments stay deliverable, accept over-coverage cost.",
         "expected_profit": worst["profit"]},
    ]
