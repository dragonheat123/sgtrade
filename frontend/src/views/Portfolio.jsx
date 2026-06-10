import { useEffect, useState } from 'react'
import { api, fmt, money, TLABELS } from '../api.js'
import Chart, { axisX, axisY } from '../components/Chart.jsx'

export default function Portfolio({ go }) {
  const [ov, setOv] = useState(null)
  const [err, setErr] = useState('')
  useEffect(() => { api.get('/api/portfolio/overview').then(setOv).catch(e => setErr(e.message)) }, [])
  if (err) return <div className="err">{err}</div>
  if (!ov) return <p className="sub">Loading portfolio…</p>

  const fc = ov.forecasts
  const solarTotal = fc.solar_import.p50.map((v, t) => Math.min(v, 100) + fc.solar_local.p50[t])
  const net = ov.net_solar_minus_demand
  const k = ov.kpis

  const positionOpt = {
    title: { text: 'Forecast position by trading interval (P50)', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS),
    yAxis: [axisY('MW'), { ...axisY('$/MWh'), splitLine: { show: false } }],
    series: [
      { name: 'Malaysia solar import', type: 'line', stack: 's', areaStyle: { opacity: .5 }, symbol: 'none', color: '#f5b955', data: fc.solar_import.p50.map(v => Math.min(v, 100)) },
      { name: 'Local solar', type: 'line', stack: 's', areaStyle: { opacity: .5 }, symbol: 'none', color: '#3ecf8e', data: fc.solar_local.p50 },
      { name: 'Contract demand', type: 'line', symbol: 'none', color: '#f06a6a', lineStyle: { width: 2.5 }, data: fc.contract_demand.p50 },
      { name: 'USEP forecast', type: 'line', yAxisIndex: 1, symbol: 'none', color: '#4da3ff', lineStyle: { type: 'dashed' }, data: fc.usep.p50 },
    ],
  }
  const netOpt = {
    title: { text: 'Net solar surplus / deficit vs contract (before plants & battery)', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS),
    yAxis: axisY('MW'),
    series: [{
      name: 'Net position', type: 'bar', data: net,
      itemStyle: { color: p => p.value >= 0 ? '#3ecf8e' : '#f06a6a' },
    }],
  }
  const priceOpt = {
    title: { text: 'Market price forecasts', textStyle: { fontSize: 13, color: '#dce4f5' } },
    xAxis: axisX(TLABELS),
    yAxis: axisY('$/MWh'),
    series: [
      { name: 'USEP', type: 'line', symbol: 'none', color: '#4da3ff', data: fc.usep.p50 },
      { name: 'Reserve', type: 'line', symbol: 'none', color: '#a78bfa', data: fc.reserve_price.p50 },
      { name: 'Regulation', type: 'line', symbol: 'none', color: '#f5b955', data: fc.regulation_price.p50 },
    ],
  }

  return (
    <>
      <h1>Portfolio Dashboard</h1>
      <p className="sub">Trade date {ov.trade_date} · 48 half-hourly intervals · {ov.latest_run_id ? `latest optimization run #${ov.latest_run_id}` : 'no optimization run yet'}</p>
      <div className="row">
        <div className="card kpi"><div className="l">Peak contract demand</div><div className="v">{fmt(k.peak_demand_mw)} MW</div></div>
        <div className="card kpi"><div className="l">Contract energy</div><div className="v">{fmt(k.total_demand_mwh)} MWh</div></div>
        <div className="card kpi"><div className="l">Solar energy (P50)</div><div className="v green">{fmt(k.total_solar_mwh)} MWh</div></div>
        <div className="card kpi"><div className="l">Max deficit</div><div className="v red">{fmt(k.max_deficit_mw)} MW</div></div>
        <div className="card kpi"><div className="l">Avg / peak USEP fcst</div><div className="v blue">${fmt(k.avg_usep_forecast)} / ${fmt(k.peak_usep_forecast)}</div></div>
        <div className="card kpi"><div className="l">Hedge position</div><div className="v amber">{fmt(ov.hedge_position_mw)} MW sold</div></div>
        <div className="card kpi"><div className="l">Battery SoC (start)</div><div className="v">{fmt(ov.battery_soc_init)} MWh</div></div>
        {ov.latest_run_summary && <div className="card kpi"><div className="l">Expected profit (last run)</div><div className="v green">{money(ov.latest_run_summary.expected_profit)}</div></div>}
      </div>
      <div className="row">
        <div className="card full"><Chart option={positionOpt} height={320} /></div>
      </div>
      <div className="row">
        <div className="card"><Chart option={netOpt} height={260} /></div>
        <div className="card"><Chart option={priceOpt} height={260} /></div>
      </div>
      <div className="row">
        <div className="card full">
          <h2>Assets</h2>
          <div className="scroll">
            <table>
              <thead><tr><th>Code</th><th>Name</th><th>Type</th><th>Key parameters</th></tr></thead>
              <tbody>
                {ov.assets.map(a => (
                  <tr key={a.code}>
                    <td><b>{a.code}</b></td><td>{a.name}</td>
                    <td><span className="badge blue">{a.type}</span></td>
                    <td className="mono" style={{ color: 'var(--muted)' }}>
                      {a.type === 'thermal' && `Pmin ${a.params.p_min} / Pmax ${a.params.p_max} MW · MC $${a.params.marginal_cost}/MWh · ramp ${a.params.ramp_mw_per_interval} MW/30min · start $${fmt(a.params.startup_cost)} · FOR ${(a.params.forced_outage_prob * 100).toFixed(0)}%`}
                      {a.type === 'battery' && `${a.params.capacity_mwh} MWh · ±${a.params.max_discharge_mw} MW · RTE ${(a.params.round_trip_eff * 100).toFixed(0)}% · degradation $${a.params.degradation_cost_per_mwh}/MWh · EoD target ${a.params.soc_end_target_mwh} MWh · contract buffer ${a.params.contract_buffer_mwh} MWh`}
                      {a.type === 'solar_import' && `${a.params.capacity_mw} MW capacity · import limit ${a.params.import_limit_mw} MW · PPA $${a.params.ppa_cost_per_mwh}/MWh`}
                      {a.type === 'solar_local' && `${a.params.capacity_mw} MW capacity · zero marginal cost`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ marginTop: 10, marginBottom: 0 }}>
            Run the optimizer in <a style={{ color: 'var(--accent)', cursor: 'pointer' }} onClick={() => go('optimize')}>Dispatch & Optimization</a> to get the recommended schedule, market split and bid bands.
          </p>
        </div>
      </div>
    </>
  )
}
