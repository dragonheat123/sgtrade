import { useEffect, useState } from 'react'
import { api, fmt, TLABELS } from '../api.js'
import Chart, { axisX, axisY } from '../components/Chart.jsx'

export default function Contracts({ can }) {
  const [cov, setCov] = useState(null)
  const [contracts, setContracts] = useState([])
  const [err, setErr] = useState('')
  const [edit, setEdit] = useState(null)

  const load = () => Promise.all([
    api.get('/api/contracts/coverage').then(setCov),
    api.get('/api/contracts').then(setContracts),
  ]).catch(e => setErr(e.message))
  useEffect(() => { load() }, [])

  const save = async () => {
    try {
      await api.put(`/api/contracts/${edit.id}`, {
        contract_price: +edit.contract_price,
        tolerance_band_pct: +edit.tolerance_band_pct,
        under_delivery_penalty: +edit.under_delivery_penalty,
        demand_uncertainty_pct: +edit.demand_uncertainty_pct,
      })
      setEdit(null); load()
    } catch (e) { setErr(e.message) }
  }

  if (err) return <div className="err">{err}</div>
  if (!cov) return <p className="sub">Loading…</p>
  const c = contracts[0]

  const covOpt = {
    title: { text: 'Contract obligation vs solar availability (P50)', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS), yAxis: axisY('MW'),
    series: [
      { name: 'Required (contract)', type: 'line', symbol: 'none', color: '#f06a6a', lineStyle: { width: 2.5 }, data: cov.required_mw },
      { name: 'Solar available', type: 'line', symbol: 'none', color: '#3ecf8e', areaStyle: { opacity: .25 }, data: cov.solar_available_mw },
      { name: 'Gap to cover (firm assets / market)', type: 'bar', color: '#f5b955', data: cov.gap_mw.map(v => Math.max(v, 0)) },
    ],
  }

  return (
    <>
      <h1>Contract Obligations</h1>
      <p className="sub">Coverage plan, penalty exposure and merit-order cost of serving the contract.</p>
      <div className="row">
        <div className="card kpi"><div className="l">Required energy</div><div className="v">{fmt(cov.required_mwh)} MWh</div></div>
        <div className="card kpi"><div className="l">Max gap after solar</div><div className="v red">{fmt(cov.max_gap_mw)} MW</div></div>
        <div className="card kpi"><div className="l">Penalty exposure</div><div className="v amber">${fmt(cov.penalty_exposure_per_mwh)}/MWh short</div></div>
        <div className="card kpi"><div className="l">Peak shortfall prob (all firm assets)</div>
          <div className="v">{(Math.max(...cov.shortfall_prob_with_full_firm) * 100).toFixed(2)}%</div></div>
      </div>
      <div className="row"><div className="card full"><Chart option={covOpt} height={300} /></div></div>
      <div className="row">
        <div className="card">
          <h2>Merit order — cost of serving the contract</h2>
          <table>
            <thead><tr><th>Source</th><th>Marginal cost $/MWh</th><th>Limit</th></tr></thead>
            <tbody>{cov.cost_to_serve.map(r => (
              <tr key={r.source}><td>{r.source}</td><td className="mono">${fmt(r.marginal_cost, 1)}</td><td className="mono">{typeof r.limit_mw === 'number' ? `${r.limit_mw} MW` : r.limit_mw}</td></tr>
            ))}</tbody>
          </table>
          <p className="sub" style={{ margin: '10px 0 0' }}>The optimizer fills the gap in this order unless reserve/regulation value or ramp constraints change the merit order.</p>
        </div>
        {c && <div className="card">
          <h2>{c.name} <span className="badge blue">{c.firm ? 'firm' : 'non-firm'}</span></h2>
          <table>
            <tbody>
              <tr><td>Counterparty</td><td>{c.counterparty}</td></tr>
              <tr><td>Contract price</td><td className="mono">${fmt(c.contract_price)}/MWh</td></tr>
              <tr><td>Tolerance band</td><td className="mono">±{c.tolerance_band_pct}%</td></tr>
              <tr><td>Under-delivery penalty</td><td className="mono">${fmt(c.under_delivery_penalty)}/MWh</td></tr>
              <tr><td>Over-delivery price</td><td className="mono">${fmt(c.over_delivery_price)}/MWh</td></tr>
              <tr><td>Settlement</td><td>{c.settlement_rule}</td></tr>
              <tr><td>Demand uncertainty</td><td className="mono">±{c.demand_uncertainty_pct}%</td></tr>
              <tr><td>Priority vs market trading</td><td>{c.priority === 1 ? 'Contract first' : c.priority}</td></tr>
            </tbody>
          </table>
          {can('configure_contracts') &&
            <div style={{ marginTop: 10 }}><button className="btn secondary" onClick={() => setEdit({ ...c })}>Edit terms</button></div>}
        </div>}
      </div>
      {edit && <div className="modal-bg" onClick={() => setEdit(null)}>
        <div className="modal" onClick={e => e.stopPropagation()}>
          <h2>Edit contract terms</h2>
          {['contract_price', 'tolerance_band_pct', 'under_delivery_penalty', 'demand_uncertainty_pct'].map(f => (
            <label className="fld" key={f} style={{ marginBottom: 10 }}>{f.replaceAll('_', ' ')}
              <input type="number" value={edit[f]} onChange={e => setEdit({ ...edit, [f]: e.target.value })} />
            </label>
          ))}
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" onClick={save}>Save</button>
            <button className="btn secondary" onClick={() => setEdit(null)}>Cancel</button>
          </div>
        </div>
      </div>}
    </>
  )
}
