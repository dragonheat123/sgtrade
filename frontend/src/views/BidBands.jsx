import { useMemo, useState } from 'react'
import { api, fmt, tlabel } from '../api.js'

export default function BidBands({ can, run, setRun }) {
  const [market, setMarket] = useState('energy')
  const [hour, setHour] = useState(19)
  const [ovr, setOvr] = useState(null)
  const [err, setErr] = useState('')
  const [msg, setMsg] = useState('')

  const bands = run?.bid_bands || []
  const runId = run?.run_id
  const shown = useMemo(() =>
    bands.filter(b => b.market === market && Math.floor(b.interval / 2) === hour),
    [bands, market, hour])

  const approveAll = async () => {
    try {
      const r = await api.post(`/api/runs/${runId}/approve_all`)
      setMsg(`${r.approved} bands approved`)
      setRun({ ...run, bid_bands: bands.map(b => b.status === 'recommended' ? { ...b, status: 'approved' } : b) })
    } catch (e) { setErr(e.message) }
  }
  const exportBids = async () => {
    const r = await fetch(`/api/runs/${runId}/export/bids`, { headers: { 'X-Token': sessionStorage.getItem('token') } })
    const blob = await r.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `bids_run${runId}.csv`
    a.click()
  }
  const saveOverride = async () => {
    try {
      const idOnServer = ovr.id
      await api.post(`/api/bidbands/${idOnServer}/override`, {
        field: ovr.field, new_value: +ovr.value, justification: ovr.justification,
      })
      setRun({
        ...run,
        bid_bands: bands.map(b => bandKey(b) === bandKey(ovr.band) ? { ...b, [ovr.field]: +ovr.value, status: 'overridden' } : b),
      })
      setOvr(null); setMsg('Override recorded with justification')
    } catch (e) { setErr(e.message) }
  }

  // bands returned by /api/optimize have no DB id; fetch them from the stored run on demand
  const openOverride = async (band) => {
    setErr('')
    try {
      const stored = await api.get(`/api/runs/${runId}`)
      const match = stored.bid_bands.find(b =>
        b.interval === band.interval && b.asset_code === band.asset_code &&
        b.market === band.market && b.band_no === band.band_no)
      if (!match) throw new Error('band not found in stored run')
      setOvr({ id: match.id, band, field: 'price', value: band.price, justification: '' })
    } catch (e) { setErr(e.message) }
  }

  if (!run) return <><h1>Bid Bands</h1><div className="card"><p className="sub" style={{ margin: 0 }}>Run an optimization first (Dispatch & Optimization).</p></div></>

  const statusBadge = (s) => <span className={'badge ' + ({ recommended: 'blue', approved: 'green', overridden: 'amber', exported: 'gray' }[s] || 'gray')}>{s}</span>

  return (
    <>
      <h1>Bid Band Recommendations <span className="badge gray">run #{runId} · {run.result.mode}</span></h1>
      <p className="sub">NEMS-style price/quantity bands per asset and interval, with dispatch probability and sensitivities.</p>
      <div className="controls">
        <label className="fld">Market
          <select value={market} onChange={e => setMarket(e.target.value)}>
            <option value="energy">Energy</option><option value="reserve">Reserve</option><option value="regulation">Regulation</option>
          </select>
        </label>
        <label className="fld">Hour
          <select value={hour} onChange={e => setHour(+e.target.value)}>
            {Array.from({ length: 24 }, (_, h) => <option key={h} value={h}>{String(h).padStart(2, '0')}:00–{String(h).padStart(2, '0')}:59</option>)}
          </select>
        </label>
        {can('approve_bids') && <button className="btn" onClick={approveAll}>Approve all bands</button>}
        {can('export') && <button className="btn secondary" onClick={exportBids}>Export bid file (CSV)</button>}
        {msg && <span className="badge green">{msg}</span>}
        {err && <span className="err">{err}</span>}
      </div>
      <div className="card full">
        <div className="scroll">
          <table>
            <thead><tr>
              <th>Interval</th><th>Asset</th><th>Band</th><th>Price $/MWh</th><th>Qty MW</th>
              <th>P(dispatch)</th><th>Qty @ solar P10</th><th>Qty @ demand P90</th><th>Status</th><th>Rationale</th><th></th>
            </tr></thead>
            <tbody>
              {shown.map(b => (
                <tr key={bandKey(b)}>
                  <td className="mono">{tlabel(b.interval)}</td>
                  <td><b>{b.asset_code}</b></td>
                  <td className="mono">{b.band_no}</td>
                  <td className="mono">${fmt(b.price, 2)}</td>
                  <td className="mono">{fmt(b.quantity_mw, 1)}</td>
                  <td className="mono">{b.dispatch_prob == null ? '—' : (b.dispatch_prob * 100).toFixed(0) + '%'}</td>
                  <td className="mono">{b.sensitivity?.qty_if_solar_p10 ?? '—'}</td>
                  <td className="mono">{b.sensitivity?.qty_if_demand_p90 ?? '—'}</td>
                  <td>{statusBadge(b.status || 'recommended')}</td>
                  <td style={{ maxWidth: 380, color: 'var(--muted)', fontSize: 12 }}>{b.rationale}</td>
                  <td>{can('override') && <button className="btn secondary" style={{ padding: '3px 8px', fontSize: 11.5 }} onClick={() => openOverride(b)}>Override</button>}</td>
                </tr>
              ))}
              {shown.length === 0 && <tr><td colSpan="11" style={{ color: 'var(--muted)' }}>No {market} bands in this hour.</td></tr>}
            </tbody>
          </table>
        </div>
      </div>
      {ovr && <div className="modal-bg" onClick={() => setOvr(null)}>
        <div className="modal" onClick={e => e.stopPropagation()}>
          <h2>Override band — {ovr.band.asset_code} {tlabel(ovr.band.interval)} band {ovr.band.band_no}</h2>
          <label className="fld" style={{ marginBottom: 10 }}>Field
            <select value={ovr.field} onChange={e => setOvr({ ...ovr, field: e.target.value, value: ovr.band[e.target.value === 'price' ? 'price' : 'quantity_mw'] })}>
              <option value="price">price</option><option value="quantity_mw">quantity_mw</option>
            </select>
          </label>
          <label className="fld" style={{ marginBottom: 10 }}>New value
            <input type="number" value={ovr.value} onChange={e => setOvr({ ...ovr, value: e.target.value })} />
          </label>
          <label className="fld" style={{ marginBottom: 12 }}>Justification (required, audited)
            <textarea rows="3" value={ovr.justification} onChange={e => setOvr({ ...ovr, justification: e.target.value })} />
          </label>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn" disabled={!ovr.justification.trim()} onClick={saveOverride}>Save override</button>
            <button className="btn secondary" onClick={() => setOvr(null)}>Cancel</button>
          </div>
        </div>
      </div>}
    </>
  )
}

const bandKey = (b) => `${b.interval}-${b.asset_code}-${b.market}-${b.band_no}`
