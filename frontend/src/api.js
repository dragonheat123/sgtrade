let token = sessionStorage.getItem('token') || ''

export function setToken(t) { token = t; sessionStorage.setItem('token', t) }
export function clearToken() { token = ''; sessionStorage.removeItem('token') }

async function req(method, path, body, isForm) {
  const opts = { method, headers: { 'X-Token': token } }
  if (body && !isForm) {
    opts.headers['Content-Type'] = 'application/json'
    opts.body = JSON.stringify(body)
  } else if (body) {
    opts.body = body
  }
  const r = await fetch(path, opts)
  const text = await r.text()
  let data
  try { data = JSON.parse(text) } catch { data = text }
  if (!r.ok) throw new Error(typeof data === 'object' ? (data.detail || JSON.stringify(data)) : data)
  return data
}

export const api = {
  get: (p) => req('GET', p),
  post: (p, b) => req('POST', p, b),
  put: (p, b) => req('PUT', p, b),
  upload: (p, formData) => req('POST', p, formData, true),
}

export const fmt = (v, d = 0) =>
  v == null ? '—' : Number(v).toLocaleString('en-SG', { maximumFractionDigits: d, minimumFractionDigits: d })
export const money = (v) => v == null ? '—' : (v < 0 ? '-' : '') + '$' + fmt(Math.abs(v))
export const tlabel = (t) => `${String(Math.floor(t / 2)).padStart(2, '0')}:${t % 2 ? '30' : '00'}`
export const TLABELS = Array.from({ length: 48 }, (_, t) => tlabel(t))
