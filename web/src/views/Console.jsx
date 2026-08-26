import { useEffect, useState } from 'react'
import { useFetch, useStream, inr } from '../api'
import Score from '../components/Score'

/**
 * The analyst's working surface: everything streams, but only the escalations
 * demand attention. Allowed payments stay deliberately quiet - no colour, no
 * emphasis - so that the eye is drawn only to what needs a decision.
 */
export default function Console() {
  const [paused, setPaused] = useState(false)
  const [rate, setRate] = useState(12)
  const { events, connected } = useStream({ limit: 120, paused, rate })

  // Seed the queue from scored history so it is never empty on arrival. At a 0.6% fraud rate
  // you would otherwise watch a few hundred allowed payments before the first escalation.
  const [seed] = useFetch('/api/stream/snapshot?limit=25&alerts_only=true')
  const [seeded, setSeeded] = useState([])
  useEffect(() => { if (seed) setSeeded(seed.slice().reverse()) }, [seed])

  const liveAlerts = events.filter((e) => e.action !== 'allow')
  const seen = new Set(liveAlerts.map((e) => e.event_id))
  const alerts = [...liveAlerts, ...seeded.filter((e) => !seen.has(e.event_id))].slice(0, 40)
  const [focus, setFocus] = useState(null)
  const shown = focus ?? alerts[0] ?? null

  return (
    <div className="grid" style={{ gap: 16, gridTemplateColumns: 'minmax(0, 1.6fr) minmax(0, 1fr)' }}>
      <div className="panel" style={{ minWidth: 0 }}>
        <div className="panel-head">
          <h2>Authorisation stream</h2>
          <div className="spacer" />
          <button className="ctl" aria-pressed={paused} onClick={() => setPaused((p) => !p)}>
            {paused ? 'resume' : 'pause'}
          </button>
          {[6, 12, 30].map((r) => (
            <button key={r} className="ctl" aria-pressed={rate === r} onClick={() => setRate(r)}>
              {r}/s
            </button>
          ))}
          <span className="eyebrow">{connected ? 'live' : 'offline'}</span>
        </div>
        <div className="panel-body flush scroll" style={{ maxHeight: '68vh' }}>
          {events.length === 0 ? (
            <div className="empty">Waiting for the stream…</div>
          ) : (
            events.map((e) => (
              <div
                key={e.event_id}
                className={`stream-row${e.action === 'block' ? ' flag' : ''}`}
                onClick={() => e.action !== 'allow' && setFocus(e)}
                style={{ cursor: e.action === 'allow' ? 'default' : 'pointer' }}
              >
                <span className="mono" style={{ fontSize: 11, color: 'var(--text-faint)' }}>{e.rail}</span>
                <span className="mono">{inr(e.amount)}</span>
                <span className="payee hide-sm">{e.payee}</span>
                <span className="hide-sm" style={{ gridColumn: 'span 2' }}>
                  <Score value={e.fraud_score} />
                </span>
                <span className={`chip ${e.action}`}>{e.action.replace('_', ' ')}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="grid" style={{ gap: 14, alignContent: 'start', minWidth: 0 }}>
        <div className="panel">
          <div className="panel-head">
            <h2>Case detail</h2>
            <div className="spacer" />
            {focus && <button className="ctl" onClick={() => setFocus(null)}>follow latest</button>}
          </div>
          <div className="panel-body">
            {!shown ? (
              <div className="empty">No escalations yet. Allowed payments need no review.</div>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
                  <span className="stat">
                    <span className="n money" style={{ fontSize: 26 }}>{inr(shown.amount)}</span>
                  </span>
                  <span className={`chip ${shown.action}`}>{shown.action.replace('_', ' ')}</span>
                </div>
                <div className="note" style={{ marginTop: 6 }}>
                  <span className="mono">{shown.rail}</span> · {shown.channel} · {shown.city}
                </div>

                <div className="eyebrow" style={{ marginTop: 16 }}>why it was flagged</div>
                {shown.reasons?.length ? (
                  <ul className="note" style={{ paddingLeft: 18, marginBottom: 0 }}>
                    {shown.reasons.map((r) => <li key={r}>{r}</li>)}
                  </ul>
                ) : (
                  <p className="note">No single dominant driver — escalated on combined risk.</p>
                )}

                {shown.drivers?.length > 0 && (
                  <>
                    <div className="eyebrow" style={{ marginTop: 16 }}>contributing features</div>
                    <table style={{ marginTop: 6 }}>
                      <tbody>
                        {shown.drivers.map((d) => (
                          <tr key={d.feature}>
                            <td className="mono" style={{ fontSize: 11, padding: '4px 0' }}>{d.feature}</td>
                            <td className="num" style={{ padding: '4px 0' }}>{d.impact.toFixed(3)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}

                <div className="eyebrow" style={{ marginTop: 16 }}>ground truth</div>
                <p className="note" style={{ marginTop: 4, marginBottom: 0 }}>
                  {shown.is_fraud
                    ? <>Fraudulent — <span className="mono">{shown.attack_id}</span></>
                    : 'Legitimate. This escalation is a false positive, and it costs friction.'}
                </p>
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head"><h2>Queue</h2><div className="spacer" />
            <span className="eyebrow">{alerts.length} open</span>
          </div>
          <div className="panel-body flush scroll" style={{ maxHeight: 260 }}>
            {alerts.length === 0 ? (
              <div className="empty">Nothing escalated in the current window.</div>
            ) : (
              alerts.map((a) => (
                <div
                  key={a.event_id}
                  className="stream-row"
                  style={{ gridTemplateColumns: '1fr auto auto', cursor: 'pointer' }}
                  onClick={() => setFocus(a)}
                >
                  <span className="payee">{a.reasons?.[0] ?? a.payee}</span>
                  <span className="mono">{inr(a.amount)}</span>
                  <span className={`chip ${a.action}`}>{a.action.replace('_', ' ')}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
