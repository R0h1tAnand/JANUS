/**
 * Risk score cell.
 *
 * Two decisions worth stating. First, the bar is drawn on a square-root scale: calibrated
 * fraud probabilities for legitimate traffic sit in the 1e-4 range, and on a linear scale
 * every honest payment renders as an empty track - a display that tells the viewer nothing
 * about whether the model is running. Square root keeps the ordering exact while making
 * small-but-nonzero risk visible.
 *
 * Second, the number is formatted by magnitude rather than to fixed decimals, so a genuinely
 * negligible score reads as "<0.001" instead of a flat "0.000" that looks like a broken feed.
 */
export function scoreLabel(score) {
  if (score >= 0.01) return score.toFixed(3)
  if (score >= 0.001) return score.toFixed(4)
  if (score > 0) return '<0.001'
  return '0'
}

export default function Score({ value }) {
  const width = Math.max(Math.sqrt(Math.max(value, 0)) * 100, value > 0 ? 1.5 : 0)
  return (
    <span className="score-cell">
      <span className="score">
        <i className={value > 0.5 ? 'hot' : ''} style={{ width: `${Math.min(width, 100)}%` }} />
      </span>
      <span className="score-num mono">{scoreLabel(value)}</span>
    </span>
  )
}
