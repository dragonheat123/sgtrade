import { api, fmt, money } from '../api.js'
import Chart, { axisY } from '../components/Chart.jsx'

export default function Risk({ run, can }) {
  if (!run) return <><h1>Risk Dashboard</h1><div className="card"><p className="sub" style={{ margin: 0 }}>Run an optimization first (Dispatch & Optimization).</p></div></>
  const rk = run.risk
  const sp = rk.scenario_profits

  const profileOpt = {
    title: { text: 'Profit by scenario', textStyle: { fontSize: 13, color: '#dce4f5' } },
    legend: { show: false },
    grid: { left: 170, right: 30, top: 34, bottom: 28 },
    xAxis: { ...axisY('$'), type: 'value' },
    yAxis: { type: 'category', data: sp.map(s => `${s.name} (${(s.prob * 100).toFixed(0)}%)`), axisLabel: { color: '#8b97b5' }, axisLine: { lineStyle: { color: '#283353' } } },
    series: [{
      type: 'bar', data: sp.map(s => s.profit),
      itemStyle: { color: p => p.value < rk.cvar90_profit ? '#f06a6a' : '#3ecf8e' },
      markLine: {
        symbol: 'none', lineStyle: { color: '#f5b955' },
        label: { color: '#f5b955', formatter: 'CVaR90' },
        data: [{ xAxis: rk.cvar90_profit }],
      },
    }],
  }

  const exportRisk = async () => {
    const data = await api.get(`/api/runs/${run.run_id}/export/risk`)
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `risk_report_run${run.run_id}.json`
    a.click()
  }

  const M = ({ l, v, cls }) => (
    <div className="card kpi"><div className="l">{l}</div><div className={'v ' + (cls || '')}>{v}</div></div>
  )

  return (
    <>
      <h1>Risk & Uncertainty Dashboard <span className="badge gray">run #{run.run_id} · {run.result.mode} · {run.result.method}</span></h1>
      <p className="sub">Downside risk, exposure decomposition and recommended actions per case.</p>
      <div className="controls">
        {can('export') && <button className="btn secondary" onClick={exportRisk}>Export risk report</button>}
      </div>
      <div className="row">
        <M l="Expected profit" v={money(rk.expected_profit)} cls="green" />
        <M l="VaR 95%" v={money(rk.var95_profit)} cls="amber" />
        <M l="CVaR 90%" v={money(rk.cvar90_profit)} cls="amber" />
        <M l={`Worst case (${rk.worst_case_name})`} v={money(rk.worst_case_profit)} cls="red" />
      </div>
      <div className="row">
        <M l="P(contract shortfall, day)" v={(rk.shortfall_prob_day * 100).toFixed(1) + '%'} />
        <M l="Expected shortfall" v={fmt(rk.expected_shortfall_mwh, 1) + ' MWh'} />
        <M l="Max interval shortfall prob" v={(rk.max_interval_shortfall_prob * 100).toFixed(1) + '%'} />
        <M l="Expected imbalance" v={fmt(rk.expected_imbalance_mwh, 1) + ' MWh'} />
      </div>
      <div className="row">
        <M l="Solar error exposure" v={fmt(rk.solar_error_exposure_mwh) + ' MWh'} cls="amber" />
        <M l="Demand error exposure" v={fmt(rk.demand_error_exposure_mwh) + ' MWh'} />
        <M l="Plant outage Δprofit" v={money(rk.plant_outage_exposure)} cls="red" />
        <M l="Battery outage Δprofit" v={money(rk.battery_outage_exposure)} cls="red" />
        <M l="Spot price exposure" v={fmt(rk.market_price_exposure_mwh) + ' MWh'} />
        <M l="Hedge effectiveness" v={(rk.hedge_effectiveness * 100).toFixed(0) + '%'} cls="blue" />
        <M l="Avg risk buffer" v={fmt(rk.avg_risk_buffer_mw, 1) + ' MW'} />
      </div>
      <div className="row">
        <div className="card"><Chart option={profileOpt} height={300} /></div>
        <div className="card">
          <h2>Recommended actions by case</h2>
          {run.cases.map(c => (
            <div className="explain" key={c.case} style={{ borderLeftColor: { base: 'var(--accent)', conservative: 'var(--green)', stress: 'var(--amber)', worst_credible: 'var(--red)' }[c.case] }}>
              <b style={{ textTransform: 'capitalize' }}>{c.case.replaceAll('_', ' ')}</b>
              {c.expected_profit != null && <span className="badge gray" style={{ marginLeft: 8 }}>{money(c.expected_profit)}</span>}
              <br />{c.action}
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
