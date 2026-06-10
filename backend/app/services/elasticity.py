"""Forecasting market elasticity from the delayed offer stack.

EMC publishes the full energy offer stack with a delay (modelled here as
D-2..D-7). Today's price-impact curve is obtained by reconstructing today's
stack from those delayed observations and re-clearing it around the forecast
operating point of every half-hour period.

Reconstruction is posed as a factor graph:

  variables                       factors / evidence
  -----------------------------   ------------------------------------------
  A_u  availability of unit u     Beta prior; recency-weighted presence in
                                  each delayed stack; outage notices (clamp);
                                  same-day telemetry ("running machines")
  F    fuel / offer-price index   Gaussian prior; implied index of each
                                  delayed stack's CCGT first offer bands
  CF_t solar capacity factor      weather forecast (cloud) evidence
  D_t  system demand              demand forecast + temperature evidence
  b_t  price impact (elasticity)  deterministic re-clearing of the sampled
                                  stack at D_t and D_t - 100 MW

The graph is a tree (every latent has independent evidence; they only meet
at the clearing node), so exact inference is a single pass of conjugate
Beta/Gaussian updates — no loopy belief propagation needed. The clearing
factor is the one non-linearity, so the marginal of b_t is obtained by Monte
Carlo over the closed-form posteriors of its parents. The resulting per-
interval p50/sigma feed the optimizer's market-elasticity model
(`market["energy_impact_pct_per_100mw"]` accepts a length-48 curve).
"""
import datetime as dt

import numpy as np

from .forecasting import _demand_shape, _solar_shape

T = 48
PRICE_CAP = 4500.0            # NEMS offer price cap, clears here on shortage
RECENCY = 0.85                # weight decay per day of publication delay
OBS_STRENGTH = 4.0            # pseudo-counts per stack observation
TELEMETRY_STRENGTH = 60.0     # same-day "running machines" evidence
FUEL_PRIOR = (1.0, 0.06)      # Gaussian prior on the fuel/offer index
FUEL_OBS_SIGMA = 0.03

# Competitor fleet (the rest of the market; our own genco's units are
# excluded — elasticity is the slope of the residual curve we face).
# bands: fraction of available capacity offered at mc * multiplier.
_CCGT_BANDS = [(0.55, 1.00), (0.30, 1.18), (0.15, 1.50)]
FLEET = [
    {"name": f"CCGT-{n}", "tech": "ccgt", "cap": cap, "mc": mc, "bands": _CCGT_BANDS}
    for n, (cap, mc) in enumerate([
        (430, 86), (420, 89), (410, 92), (400, 94), (390, 96), (380, 98),
        (370, 100), (360, 103), (350, 106), (340, 108), (335, 110), (330, 112)], 1)
] + [
    {"name": "ST-E1", "tech": "st", "cap": 250, "mc": 152, "bands": [(0.65, 1.0), (0.35, 1.22)]},
    {"name": "ST-E2", "tech": "st", "cap": 220, "mc": 158, "bands": [(0.65, 1.0), (0.35, 1.22)]},
    {"name": "ST-E3", "tech": "st", "cap": 240, "mc": 163, "bands": [(0.65, 1.0), (0.35, 1.22)]},
    {"name": "OCGT-F1", "tech": "ocgt", "cap": 170, "mc": 168, "bands": [(0.6, 1.0), (0.4, 1.3)]},
    {"name": "OCGT-F2", "tech": "ocgt", "cap": 150, "mc": 174, "bands": [(0.6, 1.0), (0.4, 1.3)]},
    {"name": "OCGT-F3", "tech": "ocgt", "cap": 180, "mc": 179, "bands": [(0.6, 1.0), (0.4, 1.3)]},
    {"name": "PEAK-G1", "tech": "peaker", "cap": 300, "mc": 300, "bands": [(0.5, 1.0), (0.5, 1.7)]},
    {"name": "PEAK-G2", "tech": "peaker", "cap": 260, "mc": 420, "bands": [(0.5, 1.0), (0.5, 1.6)]},
    {"name": "IMPORT-MY", "tech": "import", "cap": 600, "mc": 66, "bands": [(1.0, 1.0)]},
    {"name": "COGEN-WTE", "tech": "mustrun", "cap": 800, "mc": 0, "bands": [(1.0, 1.0)]},
    {"name": "SOLAR-AGG", "tech": "solar", "cap": 1100, "mc": 0, "bands": [(1.0, 1.0)]},
]
FUEL_TECHS = {"ccgt", "st", "ocgt", "peaker"}

DEFAULT_WEATHER = {"cloud_factor_p50": 0.90, "cloud_factor_sigma": 0.12,
                   "temp_delta_p50": 0.0, "temp_delta_sigma": 1.2}
DEMAND_TEMP_SENS = 0.018      # fractional demand change per degC (aircon load)


# ------------------------------------------------- synthetic publications ----
def synthesize_published_stacks(rng: np.random.Generator,
                                maint_unit: str = "CCGT-9",
                                ages: range = range(7, 1, -1)) -> list[dict]:
    """Stand-in for the EMC data feed: delayed offer stacks for D-7..D-2.

    Embedded story: `maint_unit` on maintenance until D-3, fuel index
    drifting up ~1.3%/day, occasional random derates."""
    stacks = []
    for age in ages:
        fuel = 0.98 + 0.013 * (7 - age)
        offers = {}
        for u in FLEET:
            avail = 1.0
            if u["name"] == maint_unit and age >= 3:
                avail = 0.0
            elif u["tech"] in FUEL_TECHS and rng.random() < 0.05:
                avail = rng.uniform(0.4, 0.8)   # forced partial derate
            f = fuel if u["tech"] in FUEL_TECHS else 1.0
            offers[u["name"]] = {
                "avail": round(avail, 3),
                "bands": [[round(u["mc"] * mult * f * rng.normal(1.0, 0.015), 2),
                           round(u["cap"] * avail * frac, 1)]
                          for frac, mult in u["bands"]],
            }
        stacks.append({"age_days": age, "offers": offers})
    return stacks


# ----------------------------------------------------- factor-graph fusion ----
def fuse(stacks: list[dict], outage_notices: dict | None = None,
         telemetry: dict | None = None) -> dict:
    """Exact single-pass message passing on the (tree) factor graph:
    Beta updates for unit availability, Gaussian update for the fuel index."""
    units = {u["name"]: {"a": 18.0, "b": 2.0, "forced": None} for u in FLEET}
    mc_by_name = {u["name"]: u for u in FLEET}
    f_mu, f_sig = FUEL_PRIOR
    f_prec, f_mean = 1.0 / f_sig ** 2, f_mu / f_sig ** 2

    for st in stacks:
        w = RECENCY ** st["age_days"]
        implied = []
        for u in FLEET:
            off = st["offers"].get(u["name"])
            f = float(off["avail"]) if off else 0.0   # absent => offline
            units[u["name"]]["a"] += OBS_STRENGTH * w * f
            units[u["name"]]["b"] += OBS_STRENGTH * w * (1.0 - f)
            if off and u["tech"] == "ccgt" and f > 0.1:
                implied.append(off["bands"][0][0] / u["mc"])
        if implied:
            prec = w / FUEL_OBS_SIGMA ** 2
            f_prec += prec
            f_mean += prec * float(np.mean(implied))

    for name, online in (telemetry or {}).items():
        if name in units:
            units[name]["a" if online else "b"] += TELEMETRY_STRENGTH
    for name, derate in (outage_notices or {}).items():
        if name in units:
            units[name]["forced"] = float(derate)

    for name, b in units.items():
        b["mean"] = (b["forced"] if b["forced"] is not None
                     else b["a"] / (b["a"] + b["b"]))
        b["tech"] = mc_by_name[name]["tech"]
    return {"units": units,
            "fuel": {"mean": f_mean / f_prec, "sigma": (1.0 / f_prec) ** 0.5}}


# -------------------------------------------------------- stack re-clearing ----
def _clear(cum_mw: np.ndarray, prices: np.ndarray, residual: float) -> float:
    if residual <= 0:
        return 0.0
    i = int(np.searchsorted(cum_mw, residual))
    return float(prices[i]) if i < len(prices) else PRICE_CAP


def forecast(beliefs: dict, weather: dict | None = None,
             demand_p50: np.ndarray | None = None,
             n_samples: int = 400, seed: int = 11) -> dict:
    """Marginalize the clearing factor by Monte Carlo over the posteriors:
    sample fuel/availability/weather/demand, rebuild the stack, and take the
    price move from clearing 100 MW less residual demand at every interval."""
    rng = np.random.default_rng(seed)
    w = dict(DEFAULT_WEATHER, **(weather or {}))
    dem50 = (np.asarray(demand_p50, dtype=float) if demand_p50 is not None
             else _demand_shape(4400.0, 6300.0))
    solar_prof = _solar_shape(1.0, 1.0)
    units = beliefs["units"]
    fuel = beliefs["fuel"]

    beta = np.zeros((n_samples, T))
    price = np.zeros((n_samples, T))
    cushion = np.zeros((n_samples, T))
    for n in range(n_samples):
        f = rng.normal(fuel["mean"], fuel["sigma"])
        cf = float(np.clip(rng.normal(w["cloud_factor_p50"], w["cloud_factor_sigma"]), 0.05, 1.15))
        dem_lvl = (1.0 + DEMAND_TEMP_SENS * rng.normal(w["temp_delta_p50"], w["temp_delta_sigma"])) \
            * rng.normal(1.0, 0.015)
        avail = {nm: (b["forced"] if b["forced"] is not None
                      else rng.beta(b["a"], b["b"])) for nm, b in units.items()}
        bands, mustrun, solar_cap = [], 0.0, 0.0
        for u in FLEET:
            mw_avail = u["cap"] * avail[u["name"]]
            if u["tech"] == "mustrun":
                mustrun += mw_avail
            elif u["tech"] == "solar":
                solar_cap += mw_avail
            else:
                fm = f if u["tech"] in FUEL_TECHS else 1.0
                bands += [(u["mc"] * mult * fm, mw_avail * frac) for frac, mult in u["bands"]]
        bands.sort()
        prices = np.array([p for p, _ in bands])
        cum = np.cumsum([q for _, q in bands])
        for t in range(T):
            solar = solar_cap * cf * solar_prof[t]
            dem = dem50[t] * dem_lvl + rng.normal(0.0, 80.0)
            residual = dem - mustrun - solar
            p0 = _clear(cum, prices, residual)
            pdn = _clear(cum, prices, residual - 100.0)
            beta[n, t] = np.clip(100.0 * (p0 - pdn) / max(p0, 1.0), 0.0, 40.0)
            price[n, t] = p0
            cushion[n, t] = cum[-1] + mustrun + solar - dem

    r2 = lambda a: [round(float(v), 2) for v in a]
    return {
        "impact_pct_per_100mw": r2(beta.mean(axis=0)),
        "impact_sigma": r2(beta.std(axis=0)),
        "clearing_price_p50": r2(price.mean(axis=0)),
        "supply_cushion_p50": [round(float(v), 0) for v in cushion.mean(axis=0)],
        "fuel_index": {"mean": round(fuel["mean"], 4), "sigma": round(fuel["sigma"], 4)},
        "unit_availability": {nm: round(b["mean"], 3) for nm, b in units.items()},
    }


# ------------------------------------------------------------- orchestration ----
_CACHE: dict = {}


def forecast_for_date(d: dt.date, weather: dict | None = None) -> dict:
    """Full pipeline for a trade date: pull the delayed stacks (synthetic feed
    here, real EMC ingestion would replace it), fuse, and re-clear."""
    key = (d.isoformat(), tuple(sorted((weather or {}).items())))
    if key not in _CACHE:
        seed = d.toordinal()
        rng = np.random.default_rng(seed)
        stacks = synthesize_published_stacks(rng)
        # demo evidence: one unit reported tripped at gate closure, one on
        # planned maintenance today (real feeds: SCADA/market notices)
        fleet_th = [u["name"] for u in FLEET if u["tech"] in FUEL_TECHS]
        telemetry = {fleet_th[int(rng.integers(len(fleet_th)))]: False}
        notices = {fleet_th[int(rng.integers(len(fleet_th)))]: 0.0}
        beliefs = fuse(stacks, outage_notices=notices, telemetry=telemetry)
        out = forecast(beliefs, weather=weather, seed=seed)
        out["evidence"] = {"telemetry": telemetry, "outage_notices": notices,
                           "stacks_used": [s["age_days"] for s in stacks]}
        _CACHE[key] = out
    return _CACHE[key]


def impact_curve(d: dt.date) -> list[float]:
    return forecast_for_date(d)["impact_pct_per_100mw"]
