"""Explainability layer: plain-English narrative for every interval and for the
battery strategy, covering what was decided, why, the binding constraint,
the dominant uncertainty, and the financial/operational trade-off.
"""
import numpy as np

PEAK_START, PEAK_END = 34, 42  # 17:00–21:00

CONSTRAINT_TEXT = {
    "import_limit": "the Malaysia import interconnector limit is binding",
    "plant1_max": "Plant 1 is at maximum output (incl. reserve headroom)",
    "plant2_max": "Plant 2 is at maximum output (incl. reserve headroom)",
    "plant1_min_gen": "Plant 1 is pinned at minimum stable generation",
    "soc_max": "the battery is full",
    "soc_min": "the battery is at its minimum state of charge",
    "battery_power": "the battery power rating is fully allocated",
    "none": "no physical constraint is binding",
}


def label(t):
    h, m = divmod(t * 30, 60)
    return f"{h:02d}:{m:02d}"


def interval_explanation(iv: dict, inp, mode: str) -> str:
    t = iv["interval"]
    parts = []
    dem = iv["demand_mw"]
    solar = iv["solar_import_mw"] + iv["solar_local_mw"]
    sig_solar = float(np.hypot(inp.fc["solar_import"]["sigma"][t],
                               inp.fc["solar_local"]["sigma"][t]))
    sig_dem = float(inp.fc["contract_demand"]["sigma"][t])

    # dispatch story
    if iv["plant1_mw"] > 0.5 or iv["plant2_mw"] > 0.5:
        plant_bits = []
        if iv["plant1_mw"] > 0.5:
            plant_bits.append(f"Plant 1 at {iv['plant1_mw']:.0f} MW")
        if iv["plant2_mw"] > 0.5:
            plant_bits.append(f"Plant 2 at {iv['plant2_mw']:.0f} MW")
        why = ("contract demand exceeds expected solar"
               if solar < dem else "it is cheaper than buying from the market and backs the reserve offer")
        parts.append(f"{' and '.join(plant_bits)} because {why}.")
    elif solar >= dem:
        parts.append("Thermal plants are backed off: solar covers the contract this interval.")

    if iv["batt_discharge_mw"] > 0.5:
        reason = ("USEP is at its peak — discharging captures the price spread"
                  if PEAK_START <= t <= PEAK_END else
                  "discharge covers the contract more cheaply than market purchase")
        parts.append(f"Battery discharges {iv['batt_discharge_mw']:.0f} MW: {reason}.")
    elif iv["batt_charge_mw"] > 0.5:
        src = "surplus solar" if solar > dem else "cheap off-peak energy"
        parts.append(f"Battery charges {iv['batt_charge_mw']:.0f} MW from {src} "
                     f"to position for the evening peak (SoC {iv['batt_soc_mwh']:.0f} MWh).")
    elif iv["reserve_split"]["battery"] + iv["regulation_split"]["battery"] > 0.5:
        parts.append(f"Battery holds SoC at {iv['batt_soc_mwh']:.0f} MWh — its capacity earns "
                     f"more in reserve/regulation than the energy spread this interval.")

    # market allocation story
    alloc = []
    if iv["energy_sell_mw"] > 0.5:
        alloc.append(f"{iv['energy_sell_mw']:.0f} MW offered to the energy market")
    if iv["energy_buy_mw"] > 0.5:
        alloc.append(f"{iv['energy_buy_mw']:.0f} MW bought from the market to cover the contract")
    if iv["reserve_mw"] > 0.5:
        alloc.append(f"{iv['reserve_mw']:.0f} MW held for reserve")
    if iv["regulation_mw"] > 0.5:
        alloc.append(f"{iv['regulation_mw']:.0f} MW allocated to regulation")
    if iv["risk_buffer_mw"] > 0.5:
        alloc.append(f"{iv['risk_buffer_mw']:.0f} MW retained as risk buffer against forecast error")
    if alloc:
        parts.append("Allocation: " + "; ".join(alloc) + ".")

    if iv["contract_shortfall_mw"] > 0.1:
        parts.append(f"WARNING: planned shortfall of {iv['contract_shortfall_mw']:.1f} MW — "
                     f"penalty cheaper than the marginal cover cost in this hour.")

    # uncertainty + constraint
    dom = "solar forecast error" if sig_solar > sig_dem else "contract demand uncertainty"
    parts.append(f"Dominant uncertainty: {dom} "
                 f"(solar sigma {sig_solar:.0f} MW, demand sigma {sig_dem:.0f} MW); "
                 f"P(shortfall) {iv['shortfall_prob']:.0%}, P(imbalance) {iv['imbalance_prob']:.0%}.")
    binding = iv["binding_constraint"].split(",")[0]
    parts.append(f"Binding constraint: {CONSTRAINT_TEXT.get(binding, binding)}.")
    return f"[{label(t)}] " + " ".join(parts)


def battery_strategy(result: dict, inp) -> dict:
    """Day-level battery narrative: why it charges/discharges/holds, value split."""
    ivs = result["intervals"]
    usep = [iv["usep"] for iv in ivs]
    charge_ivs = [iv for iv in ivs if iv["batt_charge_mw"] > 0.5]
    dis_ivs = [iv for iv in ivs if iv["batt_discharge_mw"] > 0.5]
    res_mw = sum(iv["reserve_split"]["battery"] for iv in ivs) / len(ivs)
    reg_mw = sum(iv["regulation_split"]["battery"] for iv in ivs) / len(ivs)
    chg_mwh = sum(iv["batt_charge_mw"] for iv in ivs) * 0.5
    dis_mwh = sum(iv["batt_discharge_mw"] for iv in ivs) * 0.5
    avg_chg_price = (np.mean([iv["usep"] for iv in charge_ivs]) if charge_ivs else 0)
    avg_dis_price = (np.mean([iv["usep"] for iv in dis_ivs]) if dis_ivs else 0)
    eff = inp.batt.get("round_trip_eff", 0.88)
    deg = inp.batt.get("degradation_cost_per_mwh", 6.0)
    arb_value = dis_mwh * avg_dis_price - chg_mwh * avg_chg_price - (chg_mwh + dis_mwh) * deg / 2

    roles = []
    if chg_mwh > 1:
        roles.append({
            "role": "energy_arbitrage",
            "text": f"Charges {chg_mwh:.0f} MWh at avg ${avg_chg_price:.0f}/MWh "
                    f"({', '.join(label(iv['interval']) for iv in charge_ivs[:4])}…) and "
                    f"discharges {dis_mwh:.0f} MWh at avg ${avg_dis_price:.0f}/MWh into the "
                    f"evening peak. Net arbitrage value ≈ ${arb_value:,.0f} after "
                    f"{eff:.0%} round-trip efficiency and ${deg}/MWh degradation."})
    if res_mw > 0.5 or reg_mw > 0.5:
        roles.append({
            "role": "ancillary_services",
            "text": f"Holds on average {res_mw:.0f} MW for reserve and {reg_mw:.0f} MW for "
                    f"regulation — paid for capacity without cycling the cells; SoC is kept "
                    f"high enough to back these offers for 30 minutes."})
    buf = inp.batt.get("contract_buffer_mwh", 0)
    if buf > 0:
        roles.append({
            "role": "contract_protection",
            "text": f"Keeps a {buf:.0f} MWh buffer through the evening peak (17:00–21:00) so a "
                    f"solar under-delivery or demand spike can be covered without buying at "
                    f"peak USEP. This is the risk-reduction value of the battery: it caps the "
                    f"worst-case imbalance cost."})
    hold_ivs = [iv for iv in ivs if iv["batt_charge_mw"] < 0.5 and iv["batt_discharge_mw"] < 0.5]
    if hold_ivs:
        roles.append({
            "role": "hold",
            "text": f"Holds SoC for {len(hold_ivs)} intervals: cycling there earns less than "
                    f"degradation cost, or the energy is worth more saved for the peak / "
                    f"reserved for ancillary commitments."})
    return {"roles": roles,
            "throughput_mwh": round(chg_mwh + dis_mwh, 1),
            "arbitrage_value": round(float(arb_value), 0),
            "avg_reserve_mw": round(res_mw, 1), "avg_regulation_mw": round(reg_mw, 1)}
