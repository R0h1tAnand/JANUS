import { useEffect, useState } from 'react'
import { useFetch } from './api'
import Overview from './views/Overview'
import Atlas from './views/Atlas'
import Console from './views/Console'
import Evidence from './views/Evidence'

/**
 * The four stages are the navigation AND the argument.
 *
 * Janus's claim is that identify / generate / defend is a cycle rather than a
 * pipeline - the attacks you simulate become the training ground, and what the
 * defence misses becomes the next attack. So the nav is drawn as a closed loop
 * with an explicit wrap-around, not as four unrelated tabs.
 */
const STAGES = [
  { key: 'overview', n: '00', label: 'Overview', accent: 'var(--text-dim)', view: Overview },
  { key: 'identify', n: '01', label: 'Identify', accent: 'var(--red)', view: Atlas },
  { key: 'defend', n: '02', label: 'Defend', accent: 'var(--blue)', view: Console },
  { key: 'adapt', n: '03', label: 'Adapt', accent: 'var(--violet)', view: Evidence },
]

const stageFromHash = () => {
  const key = window.location.hash.replace('#', '')
  return STAGES.some((s) => s.key === key) ? key : 'overview'
}

export default function App() {
  // Hash routing rather than component state alone: a stage you can link to is a stage you can
  // put in a demo script, a bug report, or a slide. It also means the browser back button does
  // what the reader expects.
  const [stage, setStage] = useState(stageFromHash)

  useEffect(() => {
    const onHashChange = () => setStage(stageFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const goTo = (key) => {
    window.location.hash = key
    setStage(key)
  }
  const [health] = useFetch('/api/health')
  const active = STAGES.find((s) => s.key === stage) ?? STAGES[0]
  const View = active.view

  const ready = health?.ready
  const statusText = health
    ? ready
      ? `model ${String(health.model_version).slice(0, 10)} · ${health.scoring_latency_us?.toFixed(0)}µs/event`
      : 'backend not ready'
    : 'connecting'

  return (
    <div className="app">
      <header className="masthead">
        <div className="wordmark">JANUS<span>.</span></div>
        <div className="tagline">payment fraud, from both sides</div>
        <div className="spacer" />
        <div className={`status-pip${ready ? '' : ' down'}`}>
          <i />
          <span>{statusText}</span>
        </div>
      </header>

      <nav className="loop" aria-label="Pipeline stages">
        {STAGES.map((s) => (
          <button
            key={s.key}
            className="loop-stage"
            style={{ '--accent': s.accent }}
            aria-current={s.key === stage}
            onClick={() => goTo(s.key)}
          >
            <span className="k">{s.n}</span>
            <span className="v">{s.label}</span>
          </button>
        ))}
        <div className="loop-wrap" title="What the defence misses becomes the next attack">
          <svg width="26" height="14" viewBox="0 0 26 14" fill="none" aria-hidden="true">
            <path d="M25 9c0-4-4-7-9-7H6" stroke="var(--line)" strokeWidth="1" strokeDasharray="2 2" />
            <path d="M9 12 5.5 9 9 6" stroke="var(--line)" strokeWidth="1" />
          </svg>
          <span>feeds back</span>
        </div>
      </nav>

      <main>
        {health && !ready ? (
          <div className="panel">
            <div className="empty">
              <p style={{ marginTop: 0 }}>The backend is up but has nothing to serve yet.</p>
              <p className="mono" style={{ fontSize: 12 }}>{health.error}</p>
              <p className="note" style={{ maxWidth: 460, margin: '14px auto 0' }}>
                Generate a world and train the defence first:<br />
                <code className="mono">uv run janus generate run</code><br />
                <code className="mono">uv run janus defend train</code>
              </p>
            </div>
          </div>
        ) : (
          <View onOpenStage={goTo} />
        )}
      </main>
    </div>
  )
}
