import { useFetch, inr, pct } from '../api'
import {
  Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

const AXIS = { stroke: '#5C6484', fontSize: 10, fontFamily: 'IBM Plex Mono' }
const TOOLTIP = {
  contentStyle: {
    background: '#1A2036', border: '1px solid #2A3350', borderRadius: 3,
    fontFamily: 'IBM Plex Mono', fontSize: 11,
  },
  labelStyle: { color: '#8C94AE' },
}

export default function Evidence() {
  const [reports] = useFetch('/api/reports')
  const loao = reports?.loao ?? []
  const arena = (reports?.arena ?? []).map((r) => ({
    round: Number(r.round),
    value_through_pct: Number(r.value_through_pct),
    red_evasion_rate: Number(r.red_evasion_rate),
    blue_recall: Number(r.blue_recall),
    realised_fpr: Number(r.realised_fpr),
  }))
  const fidelity = reports?.fidelity
  const det = reports?.detection

  // From the API, not recomputed here: folds with too few test events are excluded there so
  // the console and RESULTS.md quote the same figure. Averaging locally reported 19.9% against
  // the honest 15.9%.
  const summary = reports?.loao_summary
  const meanUnseen = summary?.mean_recall_unseen ?? null

  // Folds with only a handful of test events land on 0.0 or 1.0 by luck. They are excluded
  // from the mean, so plotting them at full height next to real results would mislead - a
  // viewer would see "one family at 100%" and take it for a success.
  const MIN_RELIABLE = summary?.min_reliable_events ?? 15
  const allFolds = loao
    .filter((r) => r.recall_unseen != null && r.recall_unseen !== '')
    .map((r) => ({
      family: String(r.family).replace('VY-', ''),
      unseen: Number(r.recall_unseen),
      seen: Number(r.recall_seen_families),
      n: Number(r.n_held_out_events),
    }))
  const excluded = allFolds.filter((r) => r.n < MIN_RELIABLE)
  const loaoData = allFolds
    .filter((r) => r.n >= MIN_RELIABLE)
    .sort((a, b) => b.unseen - a.unseen)

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="panel">
        <div className="panel-head">
          <h2>Recall on attacks the model has never seen</h2>
          <div className="spacer" />
          <span className="eyebrow">
            leave-one-attack-out · {summary?.families ?? loaoData.length} families ·
            {' '}{loaoData.length} plotted · 0.1% FPR
          </span>
        </div>
        <div className="panel-body">
          <p className="note" style={{ marginTop: 0, maxWidth: 820 }}>
            Each bar is a family the detector was trained <em>without</em>, then tested against.
            This is the closest honest proxy for emerging fraud, and it is the number that should
            be read first — aggregate AUC on families the model has already been taught says
            very little.
            {summary && (
              <> Mean across folds: <strong>{pct(meanUnseen)}</strong>, against{' '}
              {pct(summary.mean_recall_seen)} on families it has seen — a{' '}
              <strong>{(summary.generalisation_gap * 100).toFixed(1)}-point</strong> gap.{' '}
              Only {summary.families_above_50pct} of {summary.families} clear 50%.</>
            )}
          </p>
          {loaoData.length === 0 ? (
            <div className="empty">
              No LOAO results yet. Run <code className="mono">uv run janus defend loao</code>.
            </div>
          ) : (
            <div style={{ height: 290, marginTop: 10 }}>
              <ResponsiveContainer>
                <BarChart data={loaoData} margin={{ top: 4, right: 8, bottom: 46, left: -18 }}>
                  <CartesianGrid stroke="#1E2640" vertical={false} />
                  <XAxis dataKey="family" angle={-45} textAnchor="end" interval={0} {...AXIS} />
                  <YAxis domain={[0, 1]} {...AXIS} />
                  <Tooltip {...TOOLTIP} formatter={(v) => pct(v)} />
                  <Bar dataKey="unseen" fill="#D6453F" name="unseen family" radius={[2, 2, 0, 0]}
                       isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          {excluded.length > 0 && (
            <p className="note" style={{ marginBottom: 0, marginTop: 6 }}>
              Not plotted, and excluded from the mean: {excluded
                .map((r) => `${r.family} (n=${r.n})`)
                .join(', ')} — too few test events for the recall to mean anything.
            </p>
          )}
        </div>
      </div>

      <div className="grid cols-2">
        <div className="panel">
          <div className="panel-head">
            <h2>Red versus Blue</h2>
            <div className="spacer" />
            <span className="eyebrow">value that got through, per round</span>
          </div>
          <div className="panel-body">
            {arena.length === 0 ? (
              <div className="empty">
                No arena run yet. Run <code className="mono">uv run janus arena run</code>.
              </div>
            ) : (
              <>
                <div style={{ height: 220 }}>
                  <ResponsiveContainer>
                    <LineChart data={arena} margin={{ top: 6, right: 10, bottom: 4, left: -22 }}>
                      <CartesianGrid stroke="#1E2640" vertical={false} />
                      <XAxis dataKey="round" {...AXIS} />
                      <YAxis {...AXIS} />
                      <Tooltip {...TOOLTIP} />
                      <Line type="monotone" dataKey="value_through_pct" stroke="#E8A33D"
                            strokeWidth={2} dot={{ r: 3 }} name="attack value through"
                            isAnimationActive={false} />
                      <Line type="monotone" dataKey="red_evasion_rate" stroke="#D6453F"
                            strokeWidth={2} dot={{ r: 3 }} name="Red evasion" isAnimationActive={false} />
                      <Line type="monotone" dataKey="blue_recall" stroke="#5B7CFA"
                            strokeWidth={2} dot={{ r: 3 }} name="Blue recall" isAnimationActive={false} />
                      <Line type="monotone" dataKey="realised_fpr" stroke="#8C94AE"
                            strokeWidth={1} strokeDasharray="3 3" dot={false} name="FPR"
                            isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="note" style={{ marginBottom: 0 }}>
                  Red optimises for the <span style={{ color: 'var(--marigold)' }}>share of
                  attack value that gets through</span>, not evasion rate — an attacker who
                  evades everything by sending nothing is not a threat. The dashed line is the
                  false-positive rate: it must stay flat, or Blue is simply buying recall with
                  customer friction rather than getting better.
                </p>
              </>
            )}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Fidelity of the simulation</h2>
            <div className="spacer" />
            <span className="eyebrow">0.50 = indistinguishable from real data</span>
          </div>
          <div className="panel-body">
            {!fidelity ? (
              <div className="empty">
                No fidelity report. Run <code className="mono">uv run janus generate fidelity</code>.
              </div>
            ) : (
              <>
                <table>
                  <thead>
                    <tr>
                      <th>reference</th><th className="num">discriminator AUC</th>
                      <th className="num">TSTR ratio</th><th>verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fidelity.references.map((r) => (
                      <tr key={r.reference}>
                        <td className="mono">{r.reference}</td>
                        <td className="num">{r.discriminator.auc}</td>
                        <td className="num">{r.tstr?.transfer_ratio ?? '—'}</td>
                        <td style={{ color: 'var(--text-dim)', fontSize: 12 }}>
                          {r.discriminator.verdict}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="eyebrow" style={{ marginTop: 16 }}>stated limitations</div>
                <ul className="note" style={{ paddingLeft: 18, marginBottom: 0 }}>
                  {fidelity.known_limitations.map((l, i) => <li key={i}>{l}</li>)}
                </ul>
              </>
            )}
          </div>
        </div>
      </div>

      {det && (
        <div className="panel">
          <div className="panel-head">
            <h2>What the operating point costs and saves</h2>
            <div className="spacer" />
            <span className="eyebrow">within a 2% legitimate-challenge ceiling</span>
          </div>
          <div className="panel-body grid cols-4">
            <div className="stat">
              <div className="eyebrow">fraud prevented</div>
              <div className="n money">{inr(det.best_policy.fraud_prevented)}</div>
              <div className="sub">of {inr(det.best_policy.fraud_exposure)} exposed</div>
            </div>
            <div className="stat">
              <div className="eyebrow">friction cost</div>
              <div className="n">{inr(det.best_policy.friction_cost)}</div>
              <div className="sub">challenges, reviews, false blocks</div>
            </div>
            <div className="stat">
              <div className="eyebrow">net benefit</div>
              <div className="n money">{inr(det.best_policy.net_benefit)}</div>
              <div className="sub">prevented minus friction</div>
            </div>
            <div className="stat">
              <div className="eyebrow">good payments challenged</div>
              <div className="n blue">{pct(det.best_policy.legit_challenged_rate, 2)}</div>
              <div className="sub">the constraint that binds in practice</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
