import { useState } from 'react'
import { useFetch } from '../api'

/**
 * The kill-chain matrix is the signature view.
 *
 * Columns are phases; the meter under each phase is the share of steps in it that
 * generative AI materially enables. The gradient across that row - saturated at the
 * left, empty at the right - is the atlas's central finding, so it is drawn once,
 * large, and never repeated as a decorative motif elsewhere.
 */
export default function Atlas() {
  const [matrix] = useFetch('/api/atlas/matrix')
  const [atlas] = useFetch('/api/atlas')
  const [selected, setSelected] = useState(null)
  const [card] = useFetch(selected ? `/api/atlas/${selected}` : null, [selected])

  const grid = matrix?.matrix ?? {}
  const intensity = matrix?.genai_intensity ?? {}
  const phases = Object.keys(grid).filter((p) => grid[p]?.length)

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <div className="panel-head">
          <h2>Kill-chain matrix</h2>
          <div className="spacer" />
          <span className="eyebrow">
            {atlas?.coverage.total_cards} techniques · bar = share of steps GenAI enables
          </span>
        </div>
        <div className="panel-body">
          <div className="matrix">
            {phases.map((phase) => {
              const share = intensity[phase] ?? 0
              return (
                <div className="mx-col" key={phase}>
                  <div className="mx-head">
                    <div className="mx-phase">{phase.replace('_', ' ')}</div>
                    <div className="mx-meter">
                      <i style={{ width: `${Math.round(share * 100)}%` }} />
                    </div>
                    <div className="mx-pct">
                      {Math.round(share * 100)}% · {grid[phase].length}
                    </div>
                  </div>
                  {grid[phase].map((t) => (
                    <button
                      key={t.id}
                      className={`mx-cell${t.genai_used_here ? ' genai' : ''}${t.simulated ? ' sim' : ''}`}
                      onClick={() => setSelected(t.id)}
                      title={t.name}
                    >
                      {t.id.replace('VY-', '')}
                    </button>
                  ))}
                </div>
              )
            })}
          </div>
          <p className="note" style={{ marginBottom: 0, marginTop: 14, maxWidth: 780 }}>
            Read the meters left to right. GenAI is doing nearly all the work in
            <strong> pretext, resource development and trust-building</strong>, and almost none
            in <strong>monetise and launder</strong> — those are still governed by the payment
            rail's own mechanics. The bottleneck that generative models removed was human
            persuasion at scale, which is why the front of the chain is where detection has to
            move.
          </p>
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Techniques</h2>
            <div className="spacer" />
            <span className="eyebrow">{atlas?.coverage.simulated_cards} simulated</span>
          </div>
          <div className="panel-body flush scroll" style={{ maxHeight: 460 }}>
            <table>
              <thead>
                <tr>
                  <th>id</th><th>technique</th><th>status</th><th className="num">risk</th>
                </tr>
              </thead>
              <tbody>
                {(atlas?.cards ?? []).map((c) => (
                  <tr
                    key={c.id}
                    onClick={() => setSelected(c.id)}
                    style={{ cursor: 'pointer', background: selected === c.id ? 'var(--surface-2)' : undefined }}
                  >
                    <td className="mono" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
                      {c.id.replace('VY-', '')}
                    </td>
                    <td>
                      {c.name}{' '}
                      {c.simulated && <span className="tag sim">sim</span>}
                    </td>
                    <td><span className="tag">{c.status}</span></td>
                    <td className="num">{c.risk_score.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>{selected ? selected : 'Select a technique'}</h2>
          </div>
          <div className="panel-body scroll" style={{ maxHeight: 460 }}>
            {!card?.kill_chain ? (
              <div className="empty">
                Pick a cell in the matrix or a row in the table to read the full technique.
              </div>
            ) : (
              <>
                <h3 style={{ fontSize: 15, marginBottom: 6 }}>{card.name}</h3>
                <p className="note" style={{ marginTop: 0 }}>{card.summary}</p>

                <div className="eyebrow" style={{ marginTop: 16 }}>kill chain</div>
                <div style={{ marginTop: 8 }}>
                  {card.kill_chain.map((s, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'grid', gridTemplateColumns: '120px 1fr', gap: 10,
                        padding: '6px 0', borderBottom: '1px solid var(--line-soft)',
                      }}
                    >
                      <span
                        className="mono"
                        style={{ fontSize: 10.5, color: s.genai_used ? 'var(--red)' : 'var(--text-faint)' }}
                      >
                        {s.genai_used ? '▮ ' : '  '}{s.phase}
                      </span>
                      <span style={{ fontSize: 12.5, color: 'var(--text-dim)' }}>{s.description}</span>
                    </div>
                  ))}
                </div>

                <div className="eyebrow" style={{ marginTop: 16 }}>what a defender can see</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
                  {card.observables.map((o) => <span className="tag" key={o}>{o}</span>)}
                </div>

                {card.mitigations?.length > 0 && (
                  <>
                    <div className="eyebrow" style={{ marginTop: 16 }}>mitigations</div>
                    <ul className="note" style={{ paddingLeft: 18 }}>
                      {card.mitigations.map((m) => <li key={m}>{m}</li>)}
                    </ul>
                  </>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
