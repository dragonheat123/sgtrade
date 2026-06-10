"""Stochastic portfolio optimization engine.

Two-stage scenario MILP solved with HiGHS (via scipy.optimize.milp):

  First stage (here-and-now, identical across scenarios):
    plant commitment / start-up / shut-down, reserve offers, regulation offers
  Second stage (recourse, per scenario):
    plant dispatch, solar usage/curtailment, battery charge/discharge/SoC,
    energy market sales/purchases, contract shortfall

  Objective:  maximize (1-lambda) * E[profit] + lambda * CVaR_alpha[profit]
  where lambda comes from the trader's risk mode
  (conservative / balanced / aggressive).

Deterministic mode is the same model with a single (base) scenario and
lambda = 0 — satisfying the "deterministic first, then stochastic" rollout.

Note on battery mutual exclusivity: simultaneous charge+discharge is not
excluded with binaries; with positive degradation cost on both directions and
non-negative prices it is never optimal, which holds in this market model.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.stats import norm

T = 48
DT = 0.5  # hours per interval

RISK_LAMBDA = {"conservative": 0.65, "balanced": 0.30, "aggressive": 0.05}
CVAR_ALPHA = 0.90
BUY_PREMIUM = 1.10        # buying energy back priced at USEP * premium (imbalance-style cost)
SELL_CAP_MW = 800.0
BUY_CAP_MW = 400.0


@dataclass
class Scenario:
    name: str
    prob: float
    solar_import: np.ndarray
    solar_local: np.ndarray
    demand: np.ndarray
    usep: np.ndarray
    rprice: np.ndarray
    gprice: np.ndarray
    p1max_factor: float = 1.0
    p2max_factor: float = 1.0
    batt_avail: float = 1.0
    fuel_factor: float = 1.0
    import_limit_factor: float = 1.0


@dataclass
class PortfolioInputs:
    # forecasts (P50 / sigma arrays length 48)
    fc: dict                      # {kind: {"p50": [...], "sigma": [...], "p10": [...], "p90": [...]}}
    # plant 1 / 2 params
    p1: dict
    p2: dict
    batt: dict
    import_limit_mw: float = 100.0
    import_cost: float = 62.0     # $/MWh PPA cost of Malaysian import
    contract_price: float = 145.0
    under_penalty: float = 180.0  # $/MWh penalty on shortfall (on top of lost revenue)
    hedges: list = field(default_factory=list)  # [{direction, volume_mw, price, start, end}]
    shocks: dict = field(default_factory=dict)


def build_scenarios(inp: PortfolioInputs, stochastic: bool = True) -> list[Scenario]:
    """Scenario tree from forecast bands + discrete event scenarios + manual shocks."""
    fc = inp.fc
    sh = inp.shocks

    def arr(kind, key="p50"):
        return np.array(fc[kind][key], dtype=float)

    def shocked(name, base):
        f = float(sh.get(name + "_factor", 1.0))
        a = base * f
        spike = sh.get(name + "_spike_adder")
        if spike:
            a = a + float(spike)
        return a

    si, sl = shocked("solar_import", arr("solar_import")), shocked("solar_local", arr("solar_local"))
    dem = shocked("demand", arr("contract_demand"))
    usep = shocked("usep", arr("usep"))
    rp = shocked("reserve_price", arr("reserve_price"))
    gp = shocked("regulation_price", arr("regulation_price"))
    ff = float(sh.get("fuel_factor", 1.0))
    p1f = 0.0 if sh.get("plant1_trip") else 1.0
    p2f = 0.0 if sh.get("plant2_trip") else 1.0
    bav = float(sh.get("battery_avail", 1.0))
    impf = float(sh.get("import_limit_factor", 1.0))

    base = dict(solar_import=si, solar_local=sl, demand=dem, usep=usep, rprice=rp,
                gprice=gp, p1max_factor=p1f, p2max_factor=p2f, batt_avail=bav,
                fuel_factor=ff, import_limit_factor=impf)

    if not stochastic:
        return [Scenario(name="base", prob=1.0, **base)]

    def variant(name, prob, **mods):
        d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        for k, v in mods.items():
            if isinstance(d[k], np.ndarray) and np.isscalar(v):
                d[k] = d[k] * v
            else:
                d[k] = v
        return Scenario(name=name, prob=prob, **d)

    si10, si90 = shocked("solar_import", arr("solar_import", "p10")), shocked("solar_import", arr("solar_import", "p90"))
    sl10, sl90 = shocked("solar_local", arr("solar_local", "p10")), shocked("solar_local", arr("solar_local", "p90"))
    dem90 = shocked("demand", arr("contract_demand", "p90"))
    dem10 = shocked("demand", arr("contract_demand", "p10"))
    usep90 = shocked("usep", arr("usep", "p90"))
    usep10 = shocked("usep", arr("usep", "p10"))

    scens = [
        Scenario(name="normal", prob=0.30, **base),
        variant("low_solar_high_demand", 0.14, solar_import=si10, solar_local=sl10, demand=dem90),
        variant("high_solar_low_demand", 0.12, solar_import=si90, solar_local=sl90, demand=dem10),
        variant("price_spike", 0.10, usep=usep90 * 1.6, rprice=base["rprice"] * 2.2,
                gprice=base["gprice"] * 1.8),
        variant("low_price", 0.10, usep=usep10),
        variant("plant1_derate", 0.08, p1max_factor=min(p1f, 0.55)),
        variant("import_curtailment", 0.08, solar_import=si * 0.35,
                import_limit_factor=min(impf, 0.5)),
        variant("battery_derate", 0.08, batt_avail=min(bav, 0.4)),
    ]
    total = sum(s.prob for s in scens)
    for s in scens:
        s.prob /= total
    return scens


class _Idx:
    """Flat variable indexer: blocks of length T per name (+ scalars)."""
    def __init__(self):
        self.n = 0
        self.blocks = {}

    def add_block(self, name, length=T):
        self.blocks[name] = self.n
        self.n += length
        return self.blocks[name]

    def __call__(self, name, t=0):
        return self.blocks[name] + t


def solve(inp: PortfolioInputs, mode: str = "balanced",
          method: str = "stochastic") -> dict:
    scens = build_scenarios(inp, stochastic=(method == "stochastic"))
    lam = RISK_LAMBDA.get(mode, 0.30) if method == "stochastic" else 0.0
    S = len(scens)
    p1, p2, bt = inp.p1, inp.p2, inp.batt

    idx = _Idx()
    # first stage
    for nm in ["u1", "u2", "su1", "su2", "sd1", "sd2",
               "r1", "r2", "rb", "g1", "g2", "gb"]:
        idx.add_block(nm)
    # second stage per scenario
    ss_names = ["p1", "p2", "si", "sl", "ch", "dis", "soc", "sell", "buy", "short"]
    for s in range(S):
        for nm in ss_names:
            idx.add_block(f"{nm}_{s}")
    i_eta = idx.add_block("eta", 1)
    i_d = idx.add_block("d", S)
    N = idx.n

    lb = np.zeros(N)
    ub = np.full(N, np.inf)
    integrality = np.zeros(N)
    cexp = np.zeros(N)            # expected-profit coefficients (to maximize)

    # ---- first-stage bounds
    for nm in ["u1", "u2", "su1", "su2", "sd1", "sd2"]:
        ub[idx(nm):idx(nm) + T] = 1.0
    integrality[idx("u1"):idx("u1") + T] = 1
    integrality[idx("u2"):idx("u2") + T] = 1
    ub[idx("r1"):idx("r1") + T] = p1.get("reserve_cap_mw", p1["ramp_mw_per_interval"])
    ub[idx("r2"):idx("r2") + T] = p2.get("reserve_cap_mw", p2["ramp_mw_per_interval"])
    ub[idx("g1"):idx("g1") + T] = p1.get("regulation_cap_mw", 0.0)
    ub[idx("g2"):idx("g2") + T] = p2.get("regulation_cap_mw", 0.0)
    ub[idx("rb"):idx("rb") + T] = bt["max_discharge_mw"]
    ub[idx("gb"):idx("gb") + T] = bt["max_discharge_mw"]

    soc_min = bt.get("soc_min_mwh", 0.05 * bt["capacity_mwh"])
    buffer_scale = {"conservative": 1.0, "balanced": 0.6, "aggressive": 0.2}.get(mode, 0.6)
    peak_buffer = bt.get("contract_buffer_mwh", 0.0) * buffer_scale

    # ---- second-stage bounds
    for s, sc in enumerate(scens):
        imp_lim = inp.import_limit_mw * sc.import_limit_factor
        for t in range(T):
            ub[idx(f"p1_{s}", t)] = p1["p_max"] * sc.p1max_factor
            ub[idx(f"p2_{s}", t)] = p2["p_max"] * sc.p2max_factor
            ub[idx(f"si_{s}", t)] = min(max(sc.solar_import[t], 0.0), imp_lim)
            ub[idx(f"sl_{s}", t)] = max(sc.solar_local[t], 0.0)
            ub[idx(f"ch_{s}", t)] = bt["max_charge_mw"] * sc.batt_avail
            ub[idx(f"dis_{s}", t)] = bt["max_discharge_mw"] * sc.batt_avail
            lo = soc_min + (peak_buffer if 34 <= t <= 42 else 0.0)
            if t == T - 1:
                lo = max(lo, bt.get("soc_end_target_mwh", soc_min))
            lb[idx(f"soc_{s}", t)] = min(lo, bt["capacity_mwh"])
            ub[idx(f"soc_{s}", t)] = bt["capacity_mwh"]
            ub[idx(f"sell_{s}", t)] = SELL_CAP_MW
            ub[idx(f"buy_{s}", t)] = BUY_CAP_MW
            ub[idx(f"short_{s}", t)] = max(sc.demand[t], 0.0)
    lb[i_eta] = -1e8

    # ---- per-scenario profit coefficient vectors (and constants)
    sc_coef = [dict() for _ in range(S)]
    sc_const = np.zeros(S)
    deg = bt.get("degradation_cost_per_mwh", 6.0)
    for s, sc in enumerate(scens):
        cf = sc_coef[s]
        mc1 = p1["marginal_cost"] * sc.fuel_factor
        mc2 = p2["marginal_cost"] * sc.fuel_factor
        for t in range(T):
            cf[idx(f"sell_{s}", t)] = sc.usep[t] * DT
            cf[idx(f"buy_{s}", t)] = -sc.usep[t] * BUY_PREMIUM * DT
            cf[idx(f"short_{s}", t)] = -(inp.contract_price + inp.under_penalty) * DT
            cf[idx(f"p1_{s}", t)] = -mc1 * DT
            cf[idx(f"p2_{s}", t)] = -mc2 * DT
            cf[idx(f"si_{s}", t)] = -inp.import_cost * DT
            cf[idx(f"ch_{s}", t)] = -deg * 0.5 * DT
            cf[idx(f"dis_{s}", t)] = -deg * 0.5 * DT
            # first-stage vars valued at this scenario's prices
            cf[idx("r1", t)] = cf.get(idx("r1", t), 0) + sc.rprice[t] * DT
            cf[idx("r2", t)] = cf.get(idx("r2", t), 0) + sc.rprice[t] * DT
            cf[idx("rb", t)] = cf.get(idx("rb", t), 0) + sc.rprice[t] * DT
            cf[idx("g1", t)] = cf.get(idx("g1", t), 0) + sc.gprice[t] * DT
            cf[idx("g2", t)] = cf.get(idx("g2", t), 0) + sc.gprice[t] * DT
            cf[idx("gb", t)] = cf.get(idx("gb", t), 0) + sc.gprice[t] * DT
            cf[idx("su1", t)] = -p1.get("startup_cost", 0.0)
            cf[idx("su2", t)] = -p2.get("startup_cost", 0.0)
            cf[idx("sd1", t)] = -p1.get("shutdown_cost", 0.0)
            cf[idx("sd2", t)] = -p2.get("shutdown_cost", 0.0)
        sc_const[s] = float(np.sum(sc.demand) * inp.contract_price * DT)
        for h in inp.hedges:
            sgn = 1.0 if h.get("direction", "sell") == "sell" else -1.0
            for t in range(int(h.get("start_interval", 0)), int(h.get("end_interval", T - 1)) + 1):
                sc_const[s] += sgn * (h["price"] - sc.usep[t]) * h["volume_mw"] * DT
        for col, v in cf.items():
            cexp[col] += scens[s].prob * v

    # ---- constraints
    rows, cols, vals, clo, chi = [], [], [], [], []
    r = 0

    def add(coefs: dict, lo, hi):
        nonlocal r
        for c, v in coefs.items():
            rows.append(r); cols.append(c); vals.append(v)
        clo.append(lo); chi.append(hi)
        r += 1

    init_on = {"u1": 1.0 if p1.get("init_on", True) else 0.0,
               "u2": 1.0 if p2.get("init_on", False) else 0.0}
    init_p = {"p1": p1.get("init_mw", p1["p_min"] if p1.get("init_on", True) else 0.0),
              "p2": p2.get("init_mw", 0.0)}

    for t in range(T):  # start-up / shut-down logic
        for u, su, sd in [("u1", "su1", "sd1"), ("u2", "su2", "sd2")]:
            if t == 0:
                add({idx(su, 0): 1, idx(u, 0): -1}, -init_on[u], np.inf)
                add({idx(sd, 0): 1, idx(u, 0): 1}, init_on[u], np.inf)
            else:
                add({idx(su, t): 1, idx(u, t): -1, idx(u, t - 1): 1}, 0, np.inf)
                add({idx(sd, t): 1, idx(u, t): 1, idx(u, t - 1): -1}, 0, np.inf)

    eff_c = math.sqrt(bt.get("round_trip_eff", 0.88))
    eff_d = math.sqrt(bt.get("round_trip_eff", 0.88))

    for s, sc in enumerate(scens):
        for t in range(T):
            # energy balance
            add({idx(f"si_{s}", t): 1, idx(f"sl_{s}", t): 1, idx(f"p1_{s}", t): 1,
                 idx(f"p2_{s}", t): 1, idx(f"dis_{s}", t): 1, idx(f"ch_{s}", t): -1,
                 idx(f"buy_{s}", t): 1, idx(f"sell_{s}", t): -1, idx(f"short_{s}", t): 1},
                float(sc.demand[t]), float(sc.demand[t]))
            # plant capacity with reserve+regulation headroom; floor with regulation-down room
            for pp, u, rr, gg, prm in [("p1", "u1", "r1", "g1", p1), ("p2", "u2", "r2", "g2", p2)]:
                pmax = prm["p_max"] * (sc.p1max_factor if pp == "p1" else sc.p2max_factor)
                add({idx(f"{pp}_{s}", t): 1, idx(rr, t): 1, idx(gg, t): 1,
                     idx(u, t): -pmax}, -np.inf, 0)
                add({idx(f"{pp}_{s}", t): 1, idx(gg, t): -1,
                     idx(u, t): -prm["p_min"]}, 0, np.inf)
                # ramp (relaxed across start-up/shut-down via su/sd big-M;
                # big-M is nominal capacity so a trip/derate stays feasible)
                rmp = prm["ramp_mw_per_interval"]
                big_m = prm["p_max"]
                if t == 0:
                    add({idx(f"{pp}_{s}", 0): 1, idx(f"su{pp[-1]}", 0): -big_m},
                        -np.inf, rmp + init_p[pp])
                    add({idx(f"{pp}_{s}", 0): -1, idx(f"sd{pp[-1]}", 0): -big_m},
                        -np.inf, rmp - init_p[pp])
                else:
                    add({idx(f"{pp}_{s}", t): 1, idx(f"{pp}_{s}", t - 1): -1,
                         idx(f"su{pp[-1]}", t): -big_m}, -np.inf, rmp)
                    add({idx(f"{pp}_{s}", t): -1, idx(f"{pp}_{s}", t - 1): 1,
                         idx(f"sd{pp[-1]}", t): -big_m}, -np.inf, rmp)
            # battery power envelope incl. reserve/regulation headroom
            add({idx(f"dis_{s}", t): 1, idx(f"ch_{s}", t): -1, idx("rb", t): 1,
                 idx("gb", t): 1}, -np.inf, bt["max_discharge_mw"] * sc.batt_avail)
            add({idx(f"ch_{s}", t): 1, idx(f"dis_{s}", t): -1, idx("gb", t): 1},
                -np.inf, bt["max_charge_mw"] * sc.batt_avail)
            # SoC dynamics
            coefs = {idx(f"soc_{s}", t): 1, idx(f"ch_{s}", t): -eff_c * DT,
                     idx(f"dis_{s}", t): DT / eff_d}
            if t == 0:
                add(coefs, bt["soc_init_mwh"], bt["soc_init_mwh"])
            else:
                coefs[idx(f"soc_{s}", t - 1)] = -1
                add(coefs, 0, 0)
            # energy backing for reserve/regulation offers (30 min delivery)
            add({idx(f"soc_{s}", t): 1, idx("rb", t): -0.5, idx("gb", t): -0.5},
                soc_min, np.inf)

    # CVaR rows: eta - d_s - profit_s(x) <= const_s
    for s in range(S):
        coefs = {i_eta: 1.0, i_d + s: -1.0}
        for col, v in sc_coef[s].items():
            coefs[col] = coefs.get(col, 0.0) - v
        add(coefs, -np.inf, float(sc_const[s]))

    A = csr_matrix((vals, (rows, cols)), shape=(r, N))
    probs = np.array([sc.prob for sc in scens])
    c = -( (1 - lam) * cexp )
    c[i_eta] -= lam
    c[i_d:i_d + S] += lam * probs / (1 - CVAR_ALPHA)

    res = milp(c=c, constraints=LinearConstraint(A, np.array(clo), np.array(chi)),
               bounds=Bounds(lb, ub), integrality=integrality,
               options={"time_limit": 90, "mip_rel_gap": 0.002})
    if res.x is None:
        return {"status": "failed", "message": str(res.message)}

    x = res.x
    exp_const = float(np.dot(probs, sc_const))
    profits = np.array([sc_const[s] + sum(v * x[col] for col, v in sc_coef[s].items())
                        for s in range(S)])
    expected_profit = float(np.dot(probs, profits))
    order = np.argsort(profits)
    cum, cvar_num, cvar_den = 0.0, 0.0, 0.0
    tail = 1 - CVAR_ALPHA
    for i in order:
        take = min(probs[i], tail - cum)
        if take <= 1e-12:
            break
        cvar_num += take * profits[i]; cvar_den += take; cum += take
    cvar = float(cvar_num / cvar_den) if cvar_den > 0 else float(profits[order[0]])
    objective = (1 - lam) * expected_profit + lam * cvar

    return _extract(inp, scens, x, idx, mode, method, lam,
                    expected_profit, cvar, objective, profits, probs)


def _extract(inp, scens, x, idx, mode, method, lam,
             expected_profit, cvar, objective, profits, probs):
    """Pull the base-scenario schedule, market allocation, risk stats and
    binding-constraint diagnostics out of the solution vector."""
    p1, p2, bt = inp.p1, inp.p2, inp.batt
    base = 0  # scenarios[0] is the normal/base scenario
    sc0 = scens[base]
    g = lambda nm, t: float(x[idx(nm, t)])
    S = len(scens)

    fc = inp.fc
    sig_combined = np.sqrt(np.array(fc["solar_import"]["sigma"], dtype=float) ** 2
                           + np.array(fc["solar_local"]["sigma"], dtype=float) ** 2
                           + np.array(fc["contract_demand"]["sigma"], dtype=float) ** 2)

    intervals = []
    tol = 1e-4
    for t in range(T):
        si, sl = g(f"si_{base}", t), g(f"sl_{base}", t)
        pp1, pp2 = g(f"p1_{base}", t), g(f"p2_{base}", t)
        ch, dis, soc = g(f"ch_{base}", t), g(f"dis_{base}", t), g(f"soc_{base}", t)
        sell, buy, short = g(f"sell_{base}", t), g(f"buy_{base}", t), g(f"short_{base}", t)
        r_tot = g("r1", t) + g("r2", t) + g("rb", t)
        g_tot = g("g1", t) + g("g2", t) + g("gb", t)
        avail_solar = min(max(sc0.solar_import[t], 0), inp.import_limit_mw * sc0.import_limit_factor) \
            + max(sc0.solar_local[t], 0)
        curtailed = max(0.0, avail_solar - si - sl)
        dem = float(sc0.demand[t])

        # binding constraint detection (base scenario)
        binding = []
        imp_lim = inp.import_limit_mw * sc0.import_limit_factor
        if sc0.solar_import[t] > imp_lim - tol and si > imp_lim - 0.1:
            binding.append("import_limit")
        if g("u1", t) > 0.5 and pp1 + g("r1", t) + g("g1", t) > p1["p_max"] * sc0.p1max_factor - 0.1:
            binding.append("plant1_max")
        if g("u2", t) > 0.5 and pp2 + g("r2", t) + g("g2", t) > p2["p_max"] * sc0.p2max_factor - 0.1:
            binding.append("plant2_max")
        if g("u1", t) > 0.5 and pp1 < p1["p_min"] + 0.1:
            binding.append("plant1_min_gen")
        if soc > bt["capacity_mwh"] - 0.1:
            binding.append("soc_max")
        if soc < float(np.maximum(0, x[idx(f"soc_{base}", t)] * 0 + 0.05 * bt["capacity_mwh"])) + 0.6:
            binding.append("soc_min")
        if dis + g("rb", t) + g("gb", t) > bt["max_discharge_mw"] * sc0.batt_avail - 0.1:
            binding.append("battery_power")

        headroom = (p1["p_max"] * sc0.p1max_factor * g("u1", t) - pp1 - g("r1", t) - g("g1", t)
                    + p2["p_max"] * sc0.p2max_factor * g("u2", t) - pp2 - g("r2", t) - g("g2", t)
                    + bt["max_discharge_mw"] * sc0.batt_avail - dis + ch - g("rb", t) - g("gb", t))
        headroom = max(headroom, 0.0)
        sig = max(float(sig_combined[t]), 1e-6)
        p_imbal = float(norm.sf(headroom / sig))
        short_freq = float(sum(probs[s] for s in range(S)
                               if x[idx(f"short_{s}", t)] > 0.05))
        p_short = max(short_freq, float(norm.sf((headroom + buy_room(buy)) / sig)) * 0.5)

        intervals.append({
            "interval": t,
            "solar_import_mw": rnd(si), "solar_local_mw": rnd(sl),
            "solar_curtailed_mw": rnd(curtailed),
            "plant1_mw": rnd(pp1), "plant2_mw": rnd(pp2),
            "batt_charge_mw": rnd(ch), "batt_discharge_mw": rnd(dis),
            "batt_soc_mwh": rnd(soc),
            "contract_mw": rnd(dem - short), "contract_shortfall_mw": rnd(short),
            "energy_sell_mw": rnd(sell), "energy_buy_mw": rnd(buy),
            "reserve_mw": rnd(r_tot), "regulation_mw": rnd(g_tot),
            "reserve_split": {"plant1": rnd(g("r1", t)), "plant2": rnd(g("r2", t)),
                              "battery": rnd(g("rb", t))},
            "regulation_split": {"plant1": rnd(g("g1", t)), "plant2": rnd(g("g2", t)),
                                 "battery": rnd(g("gb", t))},
            "risk_buffer_mw": rnd(headroom),
            "usep": rnd(float(sc0.usep[t])),
            "demand_mw": rnd(dem),
            "shortfall_prob": round(p_short, 4),
            "imbalance_prob": round(p_imbal, 4),
            "binding_constraint": ",".join(binding) if binding else "none",
            "u1": int(g("u1", t) > 0.5), "u2": int(g("u2", t) > 0.5),
        })

    scen_profits = [{"name": scens[s].name, "prob": round(float(probs[s]), 4),
                     "profit": rnd(float(profits[s])),
                     "shortfall_mwh": rnd(sum(x[idx(f"short_{s}", t)] for t in range(T)) * DT)}
                    for s in range(S)]
    worst = min(scen_profits, key=lambda d: d["profit"])
    exp_short = float(sum(probs[s] * sum(x[idx(f"short_{s}", t)] for t in range(T)) * DT
                          for s in range(S)))
    var95 = float(np.percentile(np.repeat(profits, (probs * 1000).astype(int) + 1), 5))

    return {
        "status": "solved", "mode": mode, "method": method, "risk_lambda": lam,
        "objective_value": rnd(objective), "expected_profit": rnd(expected_profit),
        "cvar_profit": rnd(cvar), "var95_profit": rnd(var95),
        "worst_scenario": worst, "scenario_profits": scen_profits,
        "expected_shortfall_mwh": rnd(exp_short),
        "shortfall_prob_day": round(float(sum(probs[s] for s in range(S)
            if any(x[idx(f"short_{s}", t)] > 0.05 for t in range(T)))), 4),
        "intervals": intervals,
    }


def rnd(v, d=2):
    return round(float(v), d)


def buy_room(buy):
    return max(0.0, BUY_CAP_MW - buy)
