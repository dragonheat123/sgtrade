"""Bid band recommendation engine for Singapore NEMS-style offers.

For each dispatchable asset, builds up to 3 price/quantity bands per interval
from the optimized schedule:
  band 1: must-run / contract-cover tranche, priced to clear
  band 2: economic tranche at marginal cost + mode-dependent margin
  band 3: flexibility tranche priced at a risk premium (capacity the optimizer
          wants kept available for shortfall / peak risk — only sells if paid)

Each band carries dispatch probability (P(USEP >= price) from the price
forecast error distribution), sensitivities to solar/demand forecast error,
and a plain-English rationale.
"""
import numpy as np
from scipy.stats import norm

MODE_MARGIN = {"conservative": 1.20, "balanced": 1.10, "aggressive": 1.03}
MODE_RISK_PREMIUM = {"conservative": 2.2, "balanced": 1.6, "aggressive": 1.2}


def _dispatch_prob(price, usep_p50, usep_sigma):
    sig = max(usep_sigma, 1e-6)
    return float(norm.sf((price - usep_p50) / sig))


def generate(result: dict, inp, mode: str) -> list[dict]:
    fc = inp.fc
    usep50 = np.array(fc["usep"]["p50"], dtype=float)
    usepsig = np.array(fc["usep"]["sigma"], dtype=float)
    si_sig = np.array(fc["solar_import"]["sigma"], dtype=float)
    sl_sig = np.array(fc["solar_local"]["sigma"], dtype=float)
    dem_sig = np.array(fc["contract_demand"]["sigma"], dtype=float)
    margin = MODE_MARGIN.get(mode, 1.10)
    riskp = MODE_RISK_PREMIUM.get(mode, 1.6)

    bands = []
    for iv in result["intervals"]:
        t = iv["interval"]
        u50, usig = float(usep50[t]), float(usepsig[t])
        solar_err = float(np.hypot(si_sig[t], sl_sig[t]))
        dem_err = float(dem_sig[t])

        def add(asset, market, band_no, price, qty, rationale):
            if qty < 0.5:
                return
            bands.append({
                "interval": t, "asset_code": asset, "market": market,
                "band_no": band_no, "price": round(price, 2),
                "quantity_mw": round(qty, 1),
                "dispatch_prob": round(_dispatch_prob(price, u50, usig), 3)
                if market == "energy" else None,
                "rationale": rationale,
                "sensitivity": {
                    "solar_error_mw": round(solar_err, 1),
                    "demand_error_mw": round(dem_err, 1),
                    "qty_if_solar_p10": round(max(0.0, qty - 0.8 * solar_err), 1),
                    "qty_if_demand_p90": round(max(0.0, qty - 0.6 * dem_err), 1),
                },
            })

        for code, pkey, disp, res, reg, on in [
            ("PP1", inp.p1, iv["plant1_mw"], iv["reserve_split"]["plant1"],
             iv["regulation_split"]["plant1"], iv["u1"]),
            ("PP2", inp.p2, iv["plant2_mw"], iv["reserve_split"]["plant2"],
             iv["regulation_split"]["plant2"], iv["u2"]),
        ]:
            if not on:
                continue
            mc = pkey["marginal_cost"]
            pmin, pmax = pkey["p_min"], pkey["p_max"]
            committed = max(disp, pmin)
            # band 1: min-gen tranche priced to stay dispatched
            add(code, "energy", 1, max(0.5 * mc, 1.0), min(pmin, committed),
                f"{code} minimum-generation tranche priced below cost to avoid "
                f"a shut-down/start-up cycle; keeps the unit on for contract cover.")
            # band 2: scheduled economic tranche
            q2 = max(0.0, disp - pmin)
            add(code, "energy", 2, mc * margin, q2,
                f"{code} economic tranche at marginal cost "
                f"(${mc:.0f}/MWh) plus {int((margin-1)*100)}% {mode} margin; "
                f"covers contract demand and clears when USEP is at/above cost.")
            # band 3: flexibility tranche — headroom not promised to reserve/regulation
            head = max(0.0, pmax - disp - res - reg)
            add(code, "energy", 3, max(mc * riskp, u50 + riskp * usig * 0.5), head,
                f"{code} flexibility tranche: {head:.0f} MW headroom retained for "
                f"solar shortfall (sigma {solar_err:.0f} MW) and demand surprise; "
                f"only sells at a risk premium that compensates losing the buffer.")
            if res > 0.5:
                add(code, "reserve", 1, 0.8 * float(np.array(fc['reserve_price']['p50'])[t]), res,
                    f"{code} reserve offer — headroom the optimizer values more in "
                    f"reserve than energy at this interval.")
            if reg > 0.5:
                add(code, "regulation", 1, 0.8 * float(np.array(fc['regulation_price']['p50'])[t]), reg,
                    f"{code} regulation capacity offer.")

        # battery energy offers
        dis, rb, gb = iv["batt_discharge_mw"], iv["reserve_split"]["battery"], iv["regulation_split"]["battery"]
        cycle_cost = inp.batt.get("degradation_cost_per_mwh", 6.0)
        opp = u50 * 0.8 / max(inp.batt.get("round_trip_eff", 0.88), 0.5) + cycle_cost
        if dis > 0.5:
            add("BESS1", "energy", 1, max(opp, u50 * 0.9), dis,
                f"Battery discharge tranche: opportunity cost = charge energy / "
                f"round-trip efficiency + degradation (${cycle_cost:.0f}/MWh). "
                f"Discharges into the price peak.")
        if rb > 0.5:
            add("BESS1", "reserve", 1, 0.8 * float(np.array(fc['reserve_price']['p50'])[t]), rb,
                "Battery reserve offer: fast response, energy-backed for 30 minutes.")
        if gb > 0.5:
            add("BESS1", "regulation", 1, 0.8 * float(np.array(fc['regulation_price']['p50'])[t]), gb,
                "Battery regulation offer: symmetric headroom held in both directions.")
    return bands
