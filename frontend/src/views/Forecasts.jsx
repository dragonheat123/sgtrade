import { useEffect, useRef, useState } from 'react'
import { api, fmt, TLABELS } from '../api.js'
import Chart, { axisX, axisY } from '../components/Chart.jsx'

const KINDS = [
  ['solar_import', 'Malaysia solar import (MW)'],
  ['solar_local', 'Local solar (MW)'],
  ['contract_demand', 'Contract demand (MW)'],
  ['usep', 'USEP ($/MWh)'],
  ['reserve_price', 'Reserve price ($/MWh)'],
  ['regulation_price', 'Regulation price ($/MWh)'],
]

export default function Forecasts({ can }) {
  const [kind, setKind] = useState('solar_import')
  const [fc, setFc] = useState(null)
  const [err, setErr] = useState('')
  const [msg, setMsg] = useState('')
  const fileRef = useRef(null)

  const load = (k) => api.get(`/api/forecasts/${k}?paths=12`).then(setFc).catch(e => setErr(e.message))
  useEffect(() => { load(kind) }, [kind])

  const upload = async (e) => {
    const f = e.target.files[0]
    if (!f) return
    const fd = new FormData()
    fd.append('file', f)
    try {
      const r = await api.upload(`/api/forecasts/${kind}/upload`, fd)
      setMsg(`Uploaded ${r.rows} rows (series #${r.series_id})`)
      load(kind)
    } catch (e2) { setErr(e2.message) }
    e.target.value = ''
  }

  if (err) return <div className="err">{err}</div>
  if (!fc) return <p className="sub">Loading…</p>

  const fanOpt = {
    title: { text: 'P50 with P10–P90 confidence band', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS), yAxis: axisY(''),
    series: [
      { name: 'P10', type: 'line', symbol: 'none', stack: 'band', lineStyle: { opacity: 0 }, data: fc.p10, color: '#4da3ff' },
      { name: 'P10–P90', type: 'line', symbol: 'none', stack: 'band', lineStyle: { opacity: 0 }, areaStyle: { color: '#4da3ff', opacity: .18 }, data: fc.p90.map((v, i) => v - fc.p10[i]) },
      { name: 'P50', type: 'line', symbol: 'none', color: '#4da3ff', lineStyle: { width: 2.5 }, data: fc.p50 },
      ...(fc.actual?.some(a => a != null) ? [{ name: 'Actual', type: 'line', symbol: 'none', color: '#3ecf8e', data: fc.actual }] : []),
    ],
  }
  const pathsOpt = {
    title: { text: 'Monte-Carlo scenario paths (AR(1)-correlated errors)', textStyle: { fontSize: 13, color: '#dce4f5' } },
    legend: { show: false },
    xAxis: axisX(TLABELS), yAxis: axisY(''),
    series: (fc.scenario_paths || []).map((p, i) => ({
      name: `path ${i}`, type: 'line', symbol: 'none',
      lineStyle: { width: 1, opacity: .35 }, color: '#a78bfa', data: p,
    })).concat([{ name: 'P50', type: 'line', symbol: 'none', color: '#4da3ff', lineStyle: { width: 2.5 }, data: fc.p50 }]),
  }
  const probOpt = {
    title: { text: 'Probability of shortfall / excess vs P50', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS), yAxis: { ...axisY('prob'), max: 1 },
    series: [
      { name: 'P(below P50)', type: 'line', symbol: 'none', color: '#f06a6a', areaStyle: { opacity: .15 }, data: fc.prob_shortfall },
      { name: 'P(above P50)', type: 'line', symbol: 'none', color: '#3ecf8e', data: fc.prob_excess },
    ],
  }

  return (
    <>
      <h1>Forecasting</h1>
      <p className="sub">Every forecast carries uncertainty: P50, P10/P90, error sigma, scenario paths. Source: <span className="badge gray">{fc.source}</span></p>
      <div className="controls">
        <label className="fld">Forecast
          <select value={kind} onChange={e => setKind(e.target.value)}>
            {KINDS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
          </select>
        </label>
        {can('upload_forecasts') && <>
          <button className="btn secondary" onClick={() => fileRef.current.click()}>Upload CSV (interval,p50[,p10,p90,sigma])</button>
          <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }} onChange={upload} />
        </>}
        {msg && <span className="badge green">{msg}</span>}
      </div>
      <div className="row">
        <div className="card kpi"><div className="l">Mean error sigma</div><div className="v">{fmt(fc.error_distribution.mean_sigma, 1)}</div></div>
        <div className="card kpi"><div className="l">Max error sigma</div><div className="v amber">{fmt(fc.error_distribution.max_sigma, 1)}</div></div>
        <div className="card kpi"><div className="l">Avg P10 band width</div><div className="v">{fmt(fc.error_distribution.p10_band_width, 1)}</div></div>
        <div className="card kpi"><div className="l">Day P50 total</div><div className="v blue">{fmt(fc.p50.reduce((a, b) => a + b, 0) / 2)} {kind.includes('price') || kind === 'usep' ? '' : 'MWh'}</div></div>
      </div>
      <div className="row"><div className="card full"><Chart option={fanOpt} height={300} /></div></div>
      <div className="row">
        <div className="card"><Chart option={pathsOpt} height={270} /></div>
        <div className="card"><Chart option={probOpt} height={270} /></div>
      </div>
    </>
  )
}
