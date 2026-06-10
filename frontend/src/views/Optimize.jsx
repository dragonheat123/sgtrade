import { useState } from 'react'
import { api, fmt, money, tlabel, TLABELS } from '../api.js'
import Chart, { axisX, axisY } from '../components/Chart.jsx'

export default function Optimize({ can, run, setRun, go }) {
  const [mode, setMode] = useState('balanced')
  const [method, setMethod] = useState('stochastic')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [sel, setSel] = useState(38)

  const optimize = async () => {
    setBusy(true); setErr('')
    try { setRun(await api.post('/api/optimize', { mode, method })) }
    catch (e) { setErr(e.message) }
    setBusy(false)
  }

  const ivs = run?.result?.intervals
  const r = run?.result

  const dispatchOpt = ivs && {
    title: { text: 'Recommended dispatch (base scenario)', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS), yAxis: axisY('MW'),
    series: [
      { name: 'MY solar', type: 'bar', stack: 'g', color: '#f5b955', data: ivs.map(i => i.solar_import_mw) },
      { name: 'Local solar', type: 'bar', stack: 'g', color: '#3ecf8e', data: ivs.map(i => i.solar_local_mw) },
      { name: 'Plant 1', type: 'bar', stack: 'g', color: '#4da3ff', data: ivs.map(i => i.plant1_mw) },
      { name: 'Plant 2', type: 'bar', stack: 'g', color: '#a78bfa', data: ivs.map(i => i.plant2_mw) },
      { name: 'Battery discharge', type: 'bar', stack: 'g', color: '#3ad6d6', data: ivs.map(i => i.batt_discharge_mw) },
      { name: 'Market buy', type: 'bar', stack: 'g', color: '#8b97b5', data: ivs.map(i => i.energy_buy_mw) },
      { name: 'Battery charge', type: 'bar', stack: 'g', color: '#1f6f6f', data: ivs.map(i => -i.batt_charge_mw) },
      { name: 'Contract demand', type: 'line', symbol: 'none', color: '#f06a6a', lineStyle: { width: 2.5 }, data: ivs.map(i => i.demand_mw) },
    ],
  }
  const marketOpt = ivs && {
    title: { text: 'Four-market allocation', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS), yAxis: axisY('MW'),
    series: [
      { name: 'Contract-covered', type: 'bar', stack: 'm', color: '#f06a6a', data: ivs.map(i => i.contract_mw) },
      { name: 'Energy market offer', type: 'bar', stack: 'm', color: '#4da3ff', data: ivs.map(i => i.energy_sell_mw) },
      { name: 'Reserve', type: 'bar', stack: 'm', color: '#a78bfa', data: ivs.map(i => i.reserve_mw) },
      { name: 'Regulation', type: 'bar', stack: 'm', color: '#f5b955', data: ivs.map(i => i.regulation_mw) },
      { name: 'Risk buffer', type: 'bar', stack: 'm', color: '#3a4a72', data: ivs.map(i => i.risk_buffer_mw) },
    ],
  }
  const socOpt = ivs && {
    title: { text: 'Battery state of charge & shortfall probability', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS),
    yAxis: [axisY('MWh'), { ...axisY('prob'), max: 1, splitLine: { show: false } }],
    series: [
      { name: 'SoC', type: 'line', symbol: 'none', color: '#3ad6d6', areaStyle: { opacity: .2 }, data: ivs.map(i => i.batt_soc_mwh) },
      { name: 'P(shortfall)', type: 'line', yAxisIndex: 1, symbol: 'none', color: '#f06a6a', lineStyle: { type: 'dashed' }, data: ivs.map(i => i.shortfall_prob) },
      { name: 'P(imbalance)', type: 'line', yAxisIndex: 1, symbol: 'none', color: '#f5b955', lineStyle: { type: 'dashed' }, data: ivs.map(i => i.imbalance_prob) },
    ],
  }
  const onClick = (p) => setSel(p.dataIndex ?? sel)
  const si = ivs?.[sel]

  return (
    <>
      <h1>Dispatch & Optimization</h1>
      <p className="sub">Stochastic co-optimization of contract cover, energy, reserve, regulation, battery and risk buffer.</p>
      <div className="controls">
        <label className="fld">Risk mode
          <select value={mode} onChange={e => setMode(e.target.value)}>
            <option value="conservative">Conservative (λ=0.65 on CVaR)</option>
            <option value="balanced">Balanced (λ=0.30)</option>
            <option value="aggressive">Aggressive (λ=0.05)</option>
          </select>
        </label>
        <label className="fld">Method
          <select value={method} onChange={e => setMethod(e.target.value)}>
            <option value="stochastic">Stochastic (8 scenarios + CVaR)</option>
            <option value="deterministic">Deterministic (P50 only)</option>
          </select>
        </label>
        <button className="btn" disabled={busy || !can('run_optimization')} onClick={optimize}>
          {busy ? 'Solving MILP…' : 'Run optimization'}
        </button>
        {!can('run_optimization') && <span className="badge gray">your role cannot run optimizations</span>}
        {err && <span className="err">{err}</span>}
      </div>

      {r && <>
        <div className="row">
          <div className="card kpi"><div className="l">Expected profit</div><div className="v green">{money(r.expected_profit)}</div></div>
          <div className="card kpi"><div className="l">Risk-adjusted objective</div><div className="v blue">{money(r.objective_value)}</div></div>
          <div className="card kpi"><div className="l">CVaR 90% profit</div><div className="v amber">{money(r.cvar_profit)}</div></div>
          <div className="card kpi"><div className="l">P(any shortfall)</div><div className="v">{(r.shortfall_prob_day * 100).toFixed(1)}%</div></div>
          <div className="card kpi"><div className="l">Expected shortfall</div><div className="v">{fmt(r.expected_shortfall_mwh, 1)} MWh</div></div>
          <div className="card kpi"><div className="l">Worst scenario</div><div className="v red">{r.worst_scenario.name}<br /><small>{money(r.worst_scenario.profit)}</small></div></div>
        </div>
        <div className="row"><div className="card full"><Chart option={dispatchOpt} height={320} onClick={onClick} /></div></div>
        <div className="row">
          <div className="card"><Chart option={marketOpt} height={280} onClick={onClick} /></div>
          <div className="card"><Chart option={socOpt} height={280} onClick={onClick} /></div>
        </div>
        <div className="row">
          <div className="card" style={{ flex: 1.4 }}>
            <h2>Recommendation explanation — interval {tlabel(sel)} <span className="badge gray">click a chart bar to change</span></h2>
            {si && <>
              <div className="explain">{si.explanation}</div>
              <table>
                <tbody>
                  <tr><td>Plant 1 / Plant 2</td><td className="mono">{fmt(si.plant1_mw)} / {fmt(si.plant2_mw)} MW</td>
                    <td>Battery</td><td className="mono">{si.batt_discharge_mw > 0 ? `discharge ${fmt(si.batt_discharge_mw)}` : si.batt_charge_mw > 0 ? `charge ${fmt(si.batt_charge_mw)}` : 'hold'} MW · SoC {fmt(si.batt_soc_mwh)} MWh</td></tr>
                  <tr><td>Contract / shortfall</td><td className="mono">{fmt(si.contract_mw)} / {fmt(si.contract_shortfall_mw, 1)} MW</td>
                    <td>Energy sell / buy</td><td className="mono">{fmt(si.energy_sell_mw)} / {fmt(si.energy_buy_mw)} MW</td></tr>
                  <tr><td>Reserve / regulation</td><td className="mono">{fmt(si.reserve_mw)} / {fmt(si.regulation_mw)} MW</td>
                    <td>Risk buffer</td><td className="mono">{fmt(si.risk_buffer_mw)} MW</td></tr>
                  <tr><td>Binding constraint</td><td colSpan="3"><span className="badge amber">{si.binding_constraint}</span></td></tr>
                </tbody>
              </table>
            </>}
          </div>
          <div className="card">
            <h2>Battery strategy</h2>
            {run.battery_strategy.roles.map(x => (
              <div className="explain" key={x.role} style={{ borderLeftColor: 'var(--green)' }}>
                <b style={{ textTransform: 'capitalize' }}>{x.role.replaceAll('_', ' ')}</b><br />{x.text}
              </div>
            ))}
          </div>
        </div>
        <div className="row">
          <div className="card full">
            <h2>Scenario profits (stochastic tree)</h2>
            <div className="scroll">
              <table>
                <thead><tr><th>Scenario</th><th>Probability</th><th>Profit</th><th>Shortfall MWh</th></tr></thead>
                <tbody>{r.scenario_profits.map(s => (
                  <tr key={s.name}>
                    <td>{s.name}</td><td className="mono">{(s.prob * 100).toFixed(0)}%</td>
                    <td className="mono" style={{ color: s.profit < 0 ? 'var(--red)' : 'var(--green)' }}>{money(s.profit)}</td>
                    <td className="mono">{fmt(s.shortfall_mwh, 1)}</td>
                  </tr>))}</tbody>
              </table>
            </div>
            <p className="sub" style={{ margin: '10px 0 0' }}>
              Bid bands for this run are in <a style={{ color: 'var(--accent)', cursor: 'pointer' }} onClick={() => go('bidbands')}>Bid Bands</a>; risk detail in <a style={{ color: 'var(--accent)', cursor: 'pointer' }} onClick={() => go('risk')}>Risk Dashboard</a>.
            </p>
          </div>
        </div>
      </>}
      {!r && <div className="card"><p className="sub" style={{ margin: 0 }}>No run yet — choose a risk mode and click <b>Run optimization</b>.</p></div>}
    </>
  )
}
