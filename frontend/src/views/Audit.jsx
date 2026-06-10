import { useEffect, useState } from 'react'
import { api } from '../api.js'

export default function Audit() {
  const [logs, setLogs] = useState([])
  const [overrides, setOverrides] = useState([])
  const [err, setErr] = useState('')
  useEffect(() => {
    api.get('/api/audit?limit=120').then(setLogs).catch(e => setErr(e.message))
    api.get('/api/overrides').then(setOverrides).catch(() => {})
  }, [])
  if (err) return <div className="err">{err}</div>
  return (
    <>
      <h1>Audit Trail</h1>
      <p className="sub">Every login, optimization run, approval, override and export is recorded.</p>
      {overrides.length > 0 && <div className="card full" style={{ marginBottom: 14 }}>
        <h2>Trader overrides</h2>
        <div className="scroll"><table>
          <thead><tr><th>When</th><th>User</th><th>Run</th><th>Band</th><th>Field</th><th>Old → New</th><th>Justification</th></tr></thead>
          <tbody>{overrides.map(o => (
            <tr key={o.id}><td className="mono">{o.at.slice(0, 19).replace('T', ' ')}</td><td>{o.username}</td>
              <td className="mono">#{o.run_id}</td><td className="mono">#{o.bid_band_id}</td><td>{o.field}</td>
              <td className="mono">{o.old} → {o.new}</td><td style={{ color: 'var(--muted)' }}>{o.justification}</td></tr>
          ))}</tbody>
        </table></div>
      </div>}
      <div className="card full">
        <div className="scroll"><table>
          <thead><tr><th>When (UTC)</th><th>User</th><th>Role</th><th>Action</th><th>Entity</th><th>Detail</th></tr></thead>
          <tbody>{logs.map(l => (
            <tr key={l.id}>
              <td className="mono">{l.at.slice(0, 19).replace('T', ' ')}</td>
              <td>{l.username}</td><td><span className="badge gray">{l.role}</span></td>
              <td><b>{l.action}</b></td>
              <td className="mono">{l.entity}{l.entity_id ? ` #${l.entity_id}` : ''}</td>
              <td style={{ color: 'var(--muted)', fontSize: 11.5, maxWidth: 420, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{JSON.stringify(l.detail)}</td>
            </tr>
          ))}</tbody>
        </table></div>
      </div>
    </>
  )
}
