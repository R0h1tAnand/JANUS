import { useFetch, useStream, inr, pct } from '../api'
import Score from '../components/Score'

function Stat({ label, value, sub, tone }) {
  return (
    <div className="panel">
      <div className="panel-body stat">
        <div className="eyebrow">{label}</div>
        <div className={`n ${tone ?? ''}`} style={{ marginTop: 6 }}>{value}</div>
        {sub && <div className="sub">{sub}</div>}
      </div>
    </div>
  )
}

export default function Overview({ onOpenStage }) {
  const [atlas] = useFetch('/api/atlas')
  const [reports] = useFetch('/api/reports')
  const { events, connected } = useStream({ limit: 22, rate: 25 })

  const cov = atlas?.coverage
  const det = reports?.detection
  const loao = reports?.loao
  const fid = reports?.fidelity?.references?.[0]

  // Comes from the API so the console and RESULTS.md quote the same number. Folds with too
  // few test events are excluded there, not here.
  const loaoSummary = reports?.loao_summary
  const meanUnseen = loaoSummary?.mean_recall_unseen ?? null

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="grid cols-4">
        <Stat
          label="attack atlas"
          value={cov ? cov.total_cards : '—'}
          sub={cov ? `${cov.simulated_cards} with working simulators` : ''}
        />
        <Stat
          label="recall @ 0.1% FPR"
          value={det ? pct(det.metrics['recall@fpr0.001'], 1) : '—'}
          sub={det ? `PR-AUC ${det.metrics.pr_auc} · on families it has seen` : ''}
          tone="blue"
        />
        <Stat
          label="unseen-family recall"
          value={meanUnseen != null ? pct(meanUnseen, 1) : '—'}
          sub={loaoSummary
            ? `${loaoSummary.families_above_50pct}/${loaoSummary.families} families above 50% · gap +${(loaoSummary.generalisation_gap * 100).toFixed(0)}pts`
            : 'mean across leave-one-attack-out folds'}
          tone="red"
        />
        <Stat
          label="net benefit"
          value={det ? inr(det.best_policy?.net_benefit) : '—'}
          sub={det ? `${pct(det.best_policy?.legit_challenged_rate, 2)} of good payments challenged` : ''}
          tone="money"
        />
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Live decisions</h2>
            <div className="spacer" />
            <span className="eyebrow">{connected ? 'streaming' : 'offline'}</span>
          </div>
          <div className="panel-body flush scroll" style={{ minHeight: 430, maxHeight: 430 }}>
            {events.length === 0 ? (
              <div className="empty">Waiting for the stream…</div>
            ) : (
              events.map((e) => (
                <div key={e.event_id} className={`stream-row${e.action === 'block' ? ' flag' : ''}`}>
                  <span className="mono" style={{ color: 'var(--text-faint)', fontSize: 11 }}>
                    {e.rail}
                  </span>
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
          <div className="panel-head" style={{ borderTop: '1px solid var(--line-soft)', borderBottom: 0 }}>
            <button className="ctl" onClick={() => onOpenStage('defend')}>open console →</button>
          </div>
        </div>

        <div className="grid" style={{ gap: 14, alignContent: 'start' }}>
          <div className="panel">
            <div className="panel-head"><h2>What the numbers rest on</h2></div>
            <div className="panel-body">
              <p className="note" style={{ marginTop: 0 }}>
                Every figure here comes from data this system generated and then had to detect.
                Three things make that worth believing:
              </p>
              <ul className="note" style={{ paddingLeft: 18, marginBottom: 0 }}>
                <li>
                  <strong>Fidelity is measured, not asserted.</strong>{' '}
                  A classifier trying to separate our synthetic payments from real public data
                  reaches AUC{' '}
                  <span className="mono">{fid ? fid.discriminator.auc : '—'}</span>{' '}
                  — reported whether or not it flatters us.
                </li>
                <li>
                  <strong>Detection is tested on unseen attacks.</strong>{' '}
                  Every family is held out in turn and the model retrained without it.
                </li>
                <li>
                  <strong>Features cannot see the future.</strong>{' '}
                  Rolling aggregates are recomputed on truncated history and must come back
                  bit-identical.
                </li>
              </ul>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head"><h2>Where GenAI actually bites</h2></div>
            <div className="panel-body">
              <p className="note" style={{ marginTop: 0 }}>
                Across the atlas, generative AI enables almost every step of pretext and
                preparation — and almost none of the mechanics of moving money. The defence has
                to shift left, toward the part of the kill chain that used to be too expensive
                for attackers to run at scale.
              </p>
              <button className="ctl" onClick={() => onOpenStage('identify')}>
                see the kill-chain matrix →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
