import { useEffect, useState } from 'react'
import { api, fmt, money } from '../api.js'

export default function Scenarios({ can, run }) {
  const [presets, setPresets] = useState([])
  const [selected, setSelected] = useState({})
  const [mode, setMode] = useState('balanced')
  const [busy, setBusy] = useState(false)
  const [results, setResults] = useState([])
  const [latest, setLatest] = useState(null)
  const [err, setErr] = useState('')

  useEffect(() => {
    api.get('/api/scenarios/presets').then(setPresets).catch(e => setErr(e.message))
    api.get('/api/scenarios/results').then(setResults).catch(() => {})
  }, [])

  const toggle = (name, shocks) => {
    const next = { ...selected }
    if (next[name]) delete next[name]
    else next[name] = shocks
    setSelected(next)
  }
  const combinedShocks = Object.values(selected).reduce((a, s) => ({ ...a, ...s }), {})
  const names = Object.keys(selected)

  const simulate = async () => {
    setBusy(true); setErr('')
    try {
      const r = await api.post('/api/scenarios/simulate', {
        mode, shocks: combinedShocks,
        name: names.join(' + ') || 'custom scenario',
        baseline_run_id: run?.run_id || null,
      })
      setLatest(r)
      setResults(await api.get('/api/scenarios/results'))
    } catch (e) { setErr(e.message) }
    setBusy(false)
  }

  return (
    <>
      <h1>Scenario Simulator</h1>
      <p className="sub">Stack shocks, re-optimize the whole portfolio, compare dispatch / bid bands / risk against the baseline run.</p>
      <div className="card full" style={{ marginBottom: 14 }}>
        <h2>Shocks</h2>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {presets.map(p => (
            <button key={p.name} className={'scenario-chip' + (selected[p.name] ? ' active' : '')}
              onClick={() => toggle(p.name, p.shocks)}>{p.name}</button>
          ))}
        </div>
        <div className="controls" style={{ marginTop: 14, marginBottom: 0 }}>
          <label className="fld">Risk mode
            <select value={mode} onChange={e => setMode(e.target.value)}>
              <option value="conservative">Conservative</option><option value="balanced">Balanced</option><option value="aggressive">Aggressive</option>
            </select>
          </label>
          <button className="btn" disabled={busy || !can('run_optimization') || names.length === 0} onClick={simulate}>
            {busy ? 'Re-optimizing…' : `Simulate${names.length ? ` (${names.length} shock${names.length > 1 ? 's' : ''})` : ''}`}
          </button>
          {err && <span className="err">{err}</span>}
        </div>
      </div>

      {latest && <div className="row">
        <div className="card kpi"><div className="l">Scenario</div><div className="v" style={{ fontSize: 15 }}>{latest.name}</div></div>
        <div className="card kpi"><div className="l">Expected profit</div><div className="v green">{money(latest.result.expected_profit)}</div></div>
        <div className="card kpi"><div className="l">CVaR 90%</div><div className="v amber">{money(latest.result.cvar_profit)}</div></div>
        <div className="card kpi"><div className="l">Expected shortfall</div><div className="v">{fmt(latest.result.expected_shortfall_mwh, 1)} MWh</div></div>
        <div className="card kpi"><div className="l">P(shortfall)</div><div className="v red">{(latest.result.shortfall_prob_day * 100).toFixed(0)}%</div></div>
        {latest.baseline_comparison && <div className="card kpi"><div className="l">Δ vs baseline #{latest.baseline_comparison.run_id}</div>
          <div className={'v ' + (latest.baseline_comparison.delta_expected >= 0 ? 'green' : 'red')}>{money(latest.baseline_comparison.delta_expected)}</div></div>}
      </div>}

      {latest && <div className="row">
        <div className="card full">
          <h2>Recommended posture under this scenario</h2>
          {latest.cases.map(c => (
            <div className="explain" key={c.case} style={{ borderLeftColor: { base: 'var(--accent)', conservative: 'var(--green)', stress: 'var(--amber)', worst_credible: 'var(--red)' }[c.case] }}>
              <b style={{ textTransform: 'capitalize' }}>{c.case.replaceAll('_', ' ')}</b>
              {c.expected_profit != null && <span className="badge gray" style={{ marginLeft: 8 }}>{money(c.expected_profit)}</span>}
              <br />{c.action}
            </div>
          ))}
        </div>
      </div>}

      <div className="card full">
        <h2>Scenario history</h2>
        <div className="scroll">
          <table>
            <thead><tr><th>When</th><th>Name</th><th>Shocks</th><th>Expected profit</th><th>CVaR</th><th>Shortfall MWh</th><th>P(shortfall)</th></tr></thead>
            <tbody>{results.map(s => (
              <tr key={s.id}>
                <td className="mono">{s.created_at.slice(11, 19)}</td>
                <td>{s.name}</td>
                <td style={{ color: 'var(--muted)', fontSize: 11.5 }}>{JSON.stringify(s.shocks)}</td>
                <td className="mono" style={{ color: s.expected_profit < 0 ? 'var(--red)' : 'var(--green)' }}>{money(s.expected_profit)}</td>
                <td className="mono">{money(s.cvar_profit)}</td>
                <td className="mono">{fmt(s.shortfall_mwh, 1)}</td>
                <td className="mono">{(s.shortfall_prob * 100).toFixed(0)}%</td>
              </tr>))}</tbody>
          </table>
        </div>
      </div>
    </>
  )
}
