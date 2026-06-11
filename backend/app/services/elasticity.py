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
  O_ut dispatch state of unit u   deterministic ramp-constrained merit-order
                                  transition from O_u,t-1
  b_t  price impact (elasticity)  re-clearing of the *ramp-feasible* stack
                                  at D_t and D_t - 100 MW

The latents form a polytree: availability/fuel/weather have independent
evidence, so exact inference is a single pass of conjugate Beta/Gaussian
updates — no loopy belief propagation needed. The dispatch state O_t chains
the intervals together: only capacity reachable within one interval from the
previous dispatch point (|out_t - out_t-1| <= ramp_u) can respond to a
perturbation, so b_t is conditioned on b_t-1 — when competitor units are
ramp-pinned on the evening pickup, the short-run curve is much steeper than
the static merit order, and the steepness persists for several intervals.
Because the O_t transition is deterministic given its parents, it is handled
exactly by forward simulation inside the Monte Carlo that marginalizes the
clearing factor. The resulting per-interval p50/sigma feed the optimizer's
market-elasticity model (`market["energy_impact_pct_per_100mw"]` accepts a
length-48 curve).
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
# ramp: MW per 30-min interval — couples the elasticity across intervals.
_CCGT_BANDS = [(0.55, 1.00), (0.30, 1.18), (0.15, 1.50)]
FLEET = [
    {"name": f"CCGT-{n}", "tech": "ccgt", "cap": cap, "mc": mc, "bands": _CCGT_BANDS,
     "ramp": round(0.25 * cap)}
    for n, (cap, mc) in enumerate([
        (430, 86), (420, 89), (410, 92), (400, 94), (390, 96), (380, 98),
        (370, 100), (360, 103), (350, 106), (340, 108), (335, 110), (330, 112)], 1)
] + [
    {"name": "ST-E1", "tech": "st", "cap": 250, "mc": 152, "bands": [(0.65, 1.0), (0.35, 1.22)], "ramp": 38},
    {"name": "ST-E2", "tech": "st", "cap": 220, "mc": 158, "bands": [(0.65, 1.0), (0.35, 1.22)], "ramp": 33},
    {"name": "ST-E3", "tech": "st", "cap": 240, "mc": 163, "bands": [(0.65, 1.0), (0.35, 1.22)], "ramp": 36},
    {"name": "OCGT-F1", "tech": "ocgt", "cap": 170, "mc": 168, "bands": [(0.6, 1.0), (0.4, 1.3)], "ramp": 170},
    {"name": "OCGT-F2", "tech": "ocgt", "cap": 150, "mc": 174, "bands": [(0.6, 1.0), (0.4, 1.3)], "ramp": 150},
    {"name": "OCGT-F3", "tech": "ocgt", "cap": 180, "mc": 179, "bands": [(0.6, 1.0), (0.4, 1.3)], "ramp": 180},
    {"name": "PEAK-G1", "tech": "peaker", "cap": 300, "mc": 300, "bands": [(0.5, 1.0), (0.5, 1.7)], "ramp": 300},
    {"name": "PEAK-G2", "tech": "peaker", "cap": 260, "mc": 420, "bands": [(0.5, 1.0), (0.5, 1.6)], "ramp": 260},
    {"name": "IMPORT-MY", "tech": "import", "cap": 600, "mc": 66, "bands": [(1.0, 1.0)], "ramp": 200},
    {"name": "COGEN-WTE", "tech": "mustrun", "cap": 800, "mc": 0, "bands": [(1.0, 1.0)], "ramp": 0},
    {"name": "SOLAR-AGG", "tech": "solar", "cap": 1100, "mc": 0, "bands": [(1.0, 1.0)], "ramp": 0},
]
FUEL_TECHS = {"ccgt", "st", "ocgt", "peaker"}

DEFAULT_WEATHER = {"cloud_factor_p50": 0.90, "cloud_factor_sigma": 0.12,
                   "temp_delta_p50": 0.0, "temp_delta_sigma": 1.2}
DEMAND_TEMP_SENS = 0.018      # fractional demand change per degC (aircon load)
DEMAND_NOISE_MW = 80.0        # intraday demand forecast error (marginal sigma)
DEMAND_NOISE_AR1 = 0.8        # errors persist intraday (matches forecasting.py)


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
def _price_at(cum_mw: np.ndarray, prices: np.ndarray, q: float) -> float:
    if q <= 0:
        return 0.0
    i = int(np.searchsorted(cum_mw, q))
    return float(prices[i]) if i < len(prices) else PRICE_CAP


def forecast(beliefs: dict, weather: dict | None = None,
             demand_p50: np.ndarray | None = None,
             n_samples: int = 400, seed: int = 11) -> dict:
    """Marginalize the clearing factor by Monte Carlo over the posteriors.

    Each sample simulates the day sequentially: units carry a dispatch state
    forward, and at every interval only the band capacity reachable within
    one ramp (|out_t - out_t-1| <= ramp_u) can respond. The elasticity is the
    price move from clearing 100 MW less residual demand on that ramp-
    feasible stack, so b_t is conditioned on the previous interval's dispatch
    (and hence on b_t-1). A static (ramp-free) clear is kept as a diagnostic."""
    rng = np.random.default_rng(seed)
    w = dict(DEFAULT_WEATHER, **(weather or {}))
    dem50 = (np.asarray(demand_p50, dtype=float) if demand_p50 is not None
             else _demand_shape(4400.0, 6300.0))
    solar_prof = _solar_shape(1.0, 1.0)
    units = beliefs["units"]
    fuel = beliefs["fuel"]

    priced = [u for u in FLEET if u["tech"] not in ("mustrun", "solar")]
    nu = len(priced)

    beta = np.zeros((n_samples, T))
    beta_st = np.zeros((n_samples, T))
    price = np.zeros((n_samples, T))
    prem = np.zeros((n_samples, T))
    cushion = np.zeros((n_samples, T))
    dres = np.zeros((n_samples, T))   # residual-demand ramp rate (MW/interval)
    for n in range(n_samples):
        f = rng.normal(fuel["mean"], fuel["sigma"])
        cf = float(np.clip(rng.normal(w["cloud_factor_p50"], w["cloud_factor_sigma"]), 0.05, 1.15))
        dem_lvl = (1.0 + DEMAND_TEMP_SENS * rng.normal(w["temp_delta_p50"], w["temp_delta_sigma"])) \
            * rng.normal(1.0, 0.015)
        avail = {nm: (b["forced"] if b["forced"] is not None
                      else rng.beta(b["a"], b["b"])) for nm, b in units.items()}
        cap = np.array([u["cap"] * avail[u["name"]] for u in priced])
        ramp = np.array([u["ramp"] * avail[u["name"]] for u in priced])
        mustrun = sum(u["cap"] * avail[u["name"]] for u in FLEET if u["tech"] == "mustrun")
        solar_cap = sum(u["cap"] * avail[u["name"]] for u in FLEET if u["tech"] == "solar")

        # offer segments in price order (fixed per sample); each segment is a
        # band slice [s0, s1] of its unit's available capacity
        sp, su, f0, f1 = [], [], [], []
        for i, u in enumerate(priced):
            fm = f if u["tech"] in FUEL_TECHS else 1.0
            c = 0.0
            for frac, mult in u["bands"]:
                sp.append(u["mc"] * mult * fm); su.append(i)
                f0.append(c); c += frac; f1.append(c)
        order = np.argsort(sp, kind="stable")
        sp = np.asarray(sp)[order]
        su = np.asarray(su)[order]
        s0 = np.asarray(f0)[order] * cap[su]
        s1 = np.asarray(f1)[order] * cap[su]
        st_width = s1 - s0
        st_cum = np.cumsum(st_width)

        out = None  # per-unit dispatch state carried across intervals
        err, res_prev = 0.0, None   # AR(1) intraday demand forecast error
        for t in range(T):
            solar = solar_cap * cf * solar_prof[t]
            err = (DEMAND_NOISE_AR1 * err
                   + (1 - DEMAND_NOISE_AR1 ** 2) ** 0.5 * rng.normal(0.0, DEMAND_NOISE_MW))
            dem = dem50[t] * dem_lvl + err
            residual = dem - mustrun - solar
            dres[n, t] = residual - (res_prev if res_prev is not None else residual)
            res_prev = residual
            if out is None:  # steady overnight start: static dispatch
                fill = np.clip(residual - np.concatenate(([0.0], st_cum[:-1])), 0.0, st_width)
                out = np.bincount(su, weights=fill, minlength=nu)
            # ramp-feasible window around the previous dispatch point
            lo = np.maximum(0.0, out - ramp)
            hi = np.minimum(cap, out + ramp)
            slo = np.clip(s0, lo[su], hi[su])
            shi = np.clip(s1, lo[su], hi[su])
            inc = shi - slo
            cum = np.cumsum(inc)
            need = residual - float(lo.sum())
            p0 = _price_at(cum, sp, need)
            pdn = _price_at(cum, sp, need - 100.0)
            pst = _price_at(st_cum, sp, residual)
            pst_dn = _price_at(st_cum, sp, residual - 100.0)
            fill = np.clip(need - np.concatenate(([0.0], cum[:-1])), 0.0, inc)
            out = lo + np.bincount(su, weights=fill, minlength=nu)
            beta[n, t] = np.clip(100.0 * (p0 - pdn) / max(p0, 1.0), 0.0, 40.0)
            beta_st[n, t] = np.clip(100.0 * (pst - pst_dn) / max(pst, 1.0), 0.0, 40.0)
            price[n, t] = p0
            prem[n, t] = p0 - pst
            cushion[n, t] = float(hi.sum()) + mustrun + solar - dem

    # lag-1 autocorrelation of the elasticity anomalies — the temporal
    # coupling introduced by competitor ramp constraints (the static value
    # isolates what shared daily drivers alone explain)
    def lag1(m):
        a = m - m.mean(axis=0)
        return float((a[:, 1:] * a[:, :-1]).sum()
                     / max(np.sqrt((a[:, 1:] ** 2).sum() * (a[:, :-1] ** 2).sum()), 1e-9))

    # the ramp mechanism, measured directly: the excess of ramp-aware over
    # static elasticity tracks how hard the rest of the market is being asked
    # to ramp away from its previous-interval dispatch point (interval-level;
    # the excess also outlasts the ramp while pinned units catch up, which is
    # the persistence and caps this correlation)
    excess = (beta - beta_st).mean(axis=0)[1:]
    ramping = dres.mean(axis=0)[1:]
    ramp_corr = float(np.corrcoef(excess, ramping)[0, 1])

    r2 = lambda arr: [round(float(v), 2) for v in arr]
    return {
        "impact_pct_per_100mw": r2(beta.mean(axis=0)),
        "impact_sigma": r2(beta.std(axis=0)),
        "impact_static_pct_per_100mw": r2(beta_st.mean(axis=0)),
        "impact_lag1_autocorr": round(lag1(beta), 3),
        "impact_lag1_autocorr_static": round(lag1(beta_st), 3),
        "ramp_excess_vs_residual_ramp_corr": round(ramp_corr, 3),
        "ramp_premium_p50": r2(prem.mean(axis=0)),
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
