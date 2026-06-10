import { useEffect, useState } from 'react'
import { api, clearToken, setToken } from './api.js'
import Audit from './views/Audit.jsx'
import BidBands from './views/BidBands.jsx'
import Contracts from './views/Contracts.jsx'
import Forecasts from './views/Forecasts.jsx'
import Optimize from './views/Optimize.jsx'
import Portfolio from './views/Portfolio.jsx'
import Risk from './views/Risk.jsx'
import Scenarios from './views/Scenarios.jsx'

const VIEWS = [
  ['portfolio', 'Portfolio Dashboard'],
  ['forecasts', 'Forecasts'],
  ['contracts', 'Contract Obligations'],
  ['optimize', 'Dispatch & Optimization'],
  ['bidbands', 'Bid Bands'],
  ['scenarios', 'Scenario Simulator'],
  ['risk', 'Risk Dashboard'],
  ['audit', 'Audit Trail'],
]

function Login({ onLogin }) {
  const [users, setUsers] = useState([])
  const [sel, setSel] = useState('trader1')
  const [err, setErr] = useState('')
  useEffect(() => { api.get('/api/auth/users').then(setUsers).catch(e => setErr(e.message)) }, [])
  const login = async () => {
    try {
      const u = await api.post('/api/auth/login', { username: sel })
      setToken(u.token)
      onLogin(u)
    } catch (e) { setErr(e.message) }
  }
  return (
    <div className="login-wrap">
      <div className="card">
        <h1>SG <span style={{ color: 'var(--accent)' }}>GenCo</span> Trading Platform</h1>
        <p className="sub">Portfolio optimization · four-market trading · risk</p>
        <label className="fld">Sign in as (demo)
          <select value={sel} onChange={e => setSel(e.target.value)}>
            {users.map(u => <option key={u.username} value={u.username}>{u.display_name} — {u.role}</option>)}
          </select>
        </label>
        <div style={{ marginTop: 14 }}>
          <button className="btn" onClick={login}>Sign in</button>
        </div>
        {err && <div className="err">{err}</div>}
      </div>
    </div>
  )
}

export default function App() {
  const [user, setUser] = useState(null)
  const [view, setView] = useState('portfolio')
  // shared state: latest optimization run payload, so views stay in sync
  const [run, setRun] = useState(null)

  if (!user) return <Login onLogin={setUser} />
  const can = (a) => user.permissions.includes(a)
  const props = { user, can, run, setRun, go: setView }

  return (
    <>
      <div className="sidebar">
        <div className="logo">SG <span>GenCo</span><br /><small style={{ color: 'var(--muted)', fontWeight: 400 }}>Trading & Optimization</small></div>
        {VIEWS.map(([k, label]) => (
          <button key={k} className={'nav-item' + (view === k ? ' active' : '')} onClick={() => setView(k)}>{label}</button>
        ))}
        <div className="userbox">
          <b>{user.display_name}</b>
          {user.role}
          <div style={{ marginTop: 6 }}>
            <button className="btn secondary" style={{ padding: '4px 10px', fontSize: 11.5 }}
              onClick={() => { clearToken(); setUser(null) }}>Sign out</button>
          </div>
        </div>
      </div>
      <div className="main">
        {view === 'portfolio' && <Portfolio {...props} />}
        {view === 'forecasts' && <Forecasts {...props} />}
        {view === 'contracts' && <Contracts {...props} />}
        {view === 'optimize' && <Optimize {...props} />}
        {view === 'bidbands' && <BidBands {...props} />}
        {view === 'scenarios' && <Scenarios {...props} />}
        {view === 'risk' && <Risk {...props} />}
        {view === 'audit' && <Audit {...props} />}
      </div>
    </>
  )
}
