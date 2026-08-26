import { useEffect, useRef, useState } from 'react'

const get = async (path) => {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

/**
 * Fetch once on mount, and again when `deps` change.
 * Pass a null path to skip the request entirely - callers with a nullable
 * selection should not have to invent a placeholder URL to fetch.
 * Returns [data, error, loading].
 */
export function useFetch(path, deps = []) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (!path) { setData(null); setError(null); return }
    let live = true
    setError(null)
    get(path).then(
      (d) => live && setData(d),
      (e) => live && setError(e.message),
    )
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return [data, error, path !== null && data === null && !error]
}

/**
 * Subscribe to the scored event stream.
 *
 * Keeps a bounded ring of the most recent events: an operations console that grows
 * without limit stops being usable after a few minutes, and the analyst only ever
 * looks at the top of the list anyway.
 */
export function useStream({ limit = 90, paused = false, rate = 12 } = {}) {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const socket = useRef(null)

  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws/stream`)
    socket.current = ws
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onerror = () => setConnected(false)
    ws.onmessage = (msg) => {
      const payload = JSON.parse(msg.data)
      if (payload.type !== 'event') return
      setEvents((prev) => [payload.data, ...prev].slice(0, limit))
    }
    return () => ws.close()
  }, [limit])

  useEffect(() => {
    const ws = socket.current
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ rate: paused ? 0.001 : rate }))
    }
  }, [paused, rate])

  return { events, connected }
}

export const inr = (n) => {
  if (n == null) return '—'
  if (n >= 1e7) return `₹${(n / 1e7).toFixed(2)}Cr`
  if (n >= 1e5) return `₹${(n / 1e5).toFixed(1)}L`
  if (n >= 1e3) return `₹${(n / 1e3).toFixed(1)}k`
  return `₹${Math.round(n)}`
}

export const pct = (n, digits = 1) => (n == null ? '—' : `${(n * 100).toFixed(digits)}%`)
