"""Demo: forecasting elasticity from the delayed offer stack.

EMC publishes the offer stack with a delay (D-2..D-7 here). A factor graph
fuses those delayed stacks with today's evidence — outage notices, unit
telemetry ("running machines"), fuel drift, weather — into a posterior over
today's stack, which is re-cleared at every half-hour to produce a per-
interval price-impact (elasticity) curve with uncertainty. The curve then
replaces the flat default in the portfolio optimizer.

Run from backend/:  python demo_stack_elasticity.py
"""
import time

import numpy as np

from app.services import elasticity, optimizer
from demo_elasticity import build_inputs

DT = optimizer.DT


def main():
    print("=" * 78)
    print("ELASTICITY FORECAST FROM THE DELAYED OFFER STACK")
    print("=" * 78)

    rng = np.random.default_rng(42)
    stacks = elasticity.synthesize_published_stacks(rng, maint_unit="CCGT-9")
    telemetry = {"CCGT-4": False}        # seen offline at gate closure
    notices = {"ST-E2": 0.0}             # planned maintenance notice for today
    print(f"\nDelayed stacks ingested: D-{', D-'.join(str(s['age_days']) for s in stacks)}")
    print(f"Same-day evidence: telemetry {telemetry} | outage notices {notices}")

    beliefs = elasticity.fuse(stacks, outage_notices=notices, telemetry=telemetry)
    fc = elasticity.forecast(beliefs, seed=42)

    print("\n--- Factor-graph posteriors " + "-" * 50)
    fi = fc["fuel_index"]
    print(f"fuel/offer-price index: {fi['mean']:.3f} +/- {fi['sigma']:.3f} "
          f"(stacks drifted 0.98 -> 1.045 over the week)")
    print(f"{'unit':12s}{'P(available)':>14s}   story")
    av = fc["unit_availability"]
    for name, story in [("CCGT-9", "on maintenance D-7..D-3, back in the D-2 stack"),
                        ("CCGT-4", "in every stack, but telemetry says tripped today"),
                        ("ST-E2", "outage notice for today (clamped)"),
                        ("CCGT-1", "no adverse evidence")]:
        print(f"{name:12s}{av[name]:>14.2f}   {story}")

    print("\n--- Re-cleared stack: today's elasticity curve " + "-" * 31)
    print(f"{'period':>8s}{'clearing $':>12s}{'cushion MW':>12s}"
          f"{'impact %/100MW':>16s}{'sigma':>8s}")
    for t in [6, 16, 22, 27, 34, 36, 38, 40, 42, 46]:
        h, m = divmod(t * 30, 60)
        print(f"{h:02d}:{m:02d}   {fc['clearing_price_p50'][t]:>12,.0f}"
              f"{fc['supply_cushion_p50'][t]:>12,.0f}"
              f"{fc['impact_pct_per_100mw'][t]:>16.2f}"
              f"{fc['impact_sigma'][t]:>8.2f}")
    curve = fc["impact_pct_per_100mw"]
    print(f"\n  -> midday is flat (solar glut, fat CCGT bands), the evening peak is "
          f"steep\n     (clearing on OCGT/peaker bands with a thin cushion). "
          f"Flat default was 3.0;\n     forecast ranges "
          f"{min(curve):.1f} .. {max(curve):.1f} %/100MW.")

    print("\n--- Optimizer: flat 3% assumption vs stack forecast " + "-" * 25)
    t0 = time.time()
    res_flat = optimizer.solve(build_inputs(), mode="balanced", method="stochastic")
    res_curve = optimizer.solve(
        build_inputs(market={"energy_impact_pct_per_100mw": curve}),
        mode="balanced", method="stochastic")
    assert res_flat["status"] == res_curve["status"] == "solved"
    print(f"  both solved in {time.time() - t0:.1f}s")

    def bucket(res, lo, hi):
        return sum(iv["energy_sell_mw"] for iv in res["intervals"][lo:hi]) * DT

    print(f"\n{'':30s}{'flat 3%':>12s}{'stack fc':>12s}")
    for name, lo, hi in [("night sells MWh (00-07)", 0, 14),
                         ("midday sells MWh (10-16)", 20, 32),
                         ("evening sells MWh (17-22)", 34, 44)]:
        print(f"{name:30s}{bucket(res_flat, lo, hi):>12,.0f}{bucket(res_curve, lo, hi):>12,.0f}")
    print(f"{'expected profit ($)':30s}{res_flat['expected_profit']:>12,.0f}"
          f"{res_curve['expected_profit']:>12,.0f}")
    print(f"{'max price impact ($)':30s}"
          f"{res_flat['market_impact']['max_price_impact']:>12,.1f}"
          f"{res_curve['market_impact']['max_price_impact']:>12,.1f}")
    print("\n  -> With the forecast curve the optimizer withholds where the stack is"
          "\n     actually steep (evening) and sells more freely where it is flat,"
          "\n     instead of spreading a uniform 3% caution across the whole day.")


if __name__ == "__main__":
    main()
