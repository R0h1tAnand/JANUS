"""FastAPI backend for the Janus console.

Design constraint that shapes everything here: the dashboard must feel live while the heavy
work - simulating a world, building features, training a model - takes tens of seconds. So the
API loads a persisted defence bundle and a pre-simulated event stream at startup, scores the
whole stream once, and then *replays* it over a WebSocket at a controllable rate.

That is not a shortcut for the demo's benefit. It mirrors how a real deployment separates the
offline path (training, backtesting, threshold calibration) from the online path (scoring a
single authorisation in single-digit milliseconds), and the replay lets the console show the
online path honestly - including the per-event latency, which is measured rather than claimed.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from janus.defend import persist, policy
from janus.defend.explain import Explainer
from janus.defend.features import build_features
from janus.identify.loader import atlas, by_id, coverage
from janus.identify.matrix import genai_intensity, matrix
from janus.identify.signals import FAMILY_DESCRIPTIONS, SignalFamily, observables_for

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DEFAULT_EVENTS = ROOT / "data" / "synthetic" / "events.parquet"


@dataclass
class AppState:
    """Everything loaded once at startup and shared across requests."""

    bundle: persist.DefenceBundle | None = None
    events: pd.DataFrame | None = None
    features: pd.DataFrame | None = None
    fraud_scores: np.ndarray | None = None
    novelty_scores: np.ndarray | None = None
    actions: np.ndarray | None = None
    explainer: Explainer | None = None
    thresholds: policy.Thresholds = field(
        default_factory=lambda: policy.Thresholds(0.15, 0.20, 0.27)
    )
    threshold_source: str = "default"
    scoring_latency_us: float = 0.0
    load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.bundle is not None and self.events is not None


state = AppState()


def _prepare() -> None:
    """Load the model and stream, score everything, and measure per-event latency."""
    try:
        state.bundle = persist.load()
    except FileNotFoundError as exc:
        state.load_error = str(exc)
        return
    if not DEFAULT_EVENTS.exists():
        state.load_error = f"No events at {DEFAULT_EVENTS}. Run: janus generate run"
        return

    events = pd.read_parquet(DEFAULT_EVENTS).sort_values("ts", kind="stable")
    state.events = events.reset_index(drop=True)
    state.features = build_features(state.events)

    detector = state.bundle.detector
    state.fraud_scores = detector.score(state.features)
    if state.bundle.novelty is not None:
        state.novelty_scores = state.bundle.novelty.score(state.features)
    else:
        state.novelty_scores = np.zeros(len(state.features))

    # Prefer the operating point the evaluation actually selected, so the console and the
    # reported numbers describe the same system. Falls back to the defaults on a fresh clone.
    detection = REPORTS / "detection.json"
    if detection.exists():
        try:
            best = json.loads(detection.read_text())["best_policy"]
            state.thresholds = policy.Thresholds(
                step_up=float(best["step_up"]), hold=float(best["hold"]),
                block=float(best["block"]),
            )
            state.threshold_source = "reports/detection.json"
        except (KeyError, ValueError, TypeError):
            pass

    state.actions = policy.decide(state.fraud_scores, state.novelty_scores, state.thresholds)
    state.explainer = Explainer(detector)

    # Measure the online path the way a production service would run it: the raw booster on a
    # preallocated float array, one payment at a time. Scoring through the pandas/sklearn
    # convenience wrapper instead would report ~7,700us, essentially all of it framework
    # overhead that a real scoring service pays once at startup rather than per authorisation.
    sample = state.features.iloc[:500][detector.features].to_numpy(dtype=np.float32)
    detector.score_fast(sample[:1])  # warm the booster
    start = time.perf_counter()
    for i in range(len(sample)):
        detector.score_fast(sample[i : i + 1])
    state.scoring_latency_us = (time.perf_counter() - start) / len(sample) * 1e6


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, _prepare)
    yield


app = FastAPI(
    title="Project Janus",
    description="Adversarial red-team/blue-team lab for payment fraud.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _report(name: str) -> Any:
    path = REPORTS / name
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return pd.read_csv(path).to_dict("records")


# --- status -----------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {
        "ready": state.ready,
        "error": state.load_error,
        "model_version": state.bundle.version if state.bundle else None,
        "n_events": int(len(state.events)) if state.events is not None else 0,
        "scoring_latency_us": round(state.scoring_latency_us, 1),
        "thresholds": {
            "step_up": state.thresholds.step_up,
            "hold": state.thresholds.hold,
            "block": state.thresholds.block,
            "source": state.threshold_source,
        },
    }


# --- pillar 1: identify -----------------------------------------------------------------

@app.get("/api/atlas")
def get_atlas() -> dict:
    cards = [
        {
            "id": c.id, "name": c.name, "summary": c.summary, "family": c.family,
            "rails": [str(r) for r in c.rails],
            "genai_enablers": [str(g) for g in c.genai_enablers],
            "monetization": [str(m) for m in c.monetization],
            "status": str(c.status), "severity": c.severity, "scalability": c.scalability,
            "detectability": c.detectability, "risk_score": c.risk_score,
            "simulated": c.simulated,
            "signal_families": sorted(str(f) for f in c.signal_families),
        }
        for c in atlas()
    ]
    return {"cards": cards, "coverage": coverage()}


@app.get("/api/atlas/matrix")
def get_matrix() -> dict:
    return {"matrix": matrix(), "genai_intensity": genai_intensity()}


@app.get("/api/atlas/signals")
def get_signals() -> list[dict]:
    cards = atlas()
    return [
        {
            "family": str(f),
            "description": FAMILY_DESCRIPTIONS[f],
            "observables": observables_for(f),
            "n_cards": sum(1 for c in cards if f in c.signal_families),
        }
        for f in SignalFamily
    ]


@app.get("/api/atlas/ideate")
def get_ideas(top: int = 15, extrapolate: bool = False) -> dict:
    from janus.identify.ideator import coverage_ratio, propose, search_space_size

    cands = propose(top, min_extrapolation=0.5 if extrapolate else 0.0)
    return {
        "search_space": search_space_size(),
        "coverage_ratio": round(coverage_ratio(), 4),
        "candidates": [
            {
                "title": c.title(), "tier": c.tier, "priority": c.priority,
                "plausibility": c.plausibility, "impact": c.impact,
                "extrapolation": c.extrapolation, "nearest_card": c.nearest_card,
                "rail": str(c.rail), "enabler": str(c.enabler),
                "surface": str(c.surface), "monetization": str(c.monetization),
            }
            for c in cands
        ],
    }


@app.get("/api/atlas/{card_id}")
def get_card(card_id: str) -> dict:
    try:
        c = by_id(card_id.upper())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "id": c.id, "name": c.name, "summary": c.summary, "status": str(c.status),
        "rails": [str(r) for r in c.rails],
        "genai_enablers": [str(g) for g in c.genai_enablers],
        "surfaces": [str(s) for s in c.surfaces],
        "monetization": [str(m) for m in c.monetization],
        "kill_chain": [
            {"phase": str(s.phase), "description": s.description, "genai_used": s.genai_used}
            for s in c.kill_chain
        ],
        "observables": c.observables,
        "mitigations": c.mitigations,
        "severity": c.severity, "scalability": c.scalability,
        "detectability": c.detectability, "risk_score": c.risk_score,
        "simulated": c.simulated, "injector": c.injector,
        "references": c.references,
        "signal_families": sorted(str(f) for f in c.signal_families),
    }


# --- pillar 2 & 3: reports ---------------------------------------------------------------

@app.get("/api/reports")
def get_reports() -> dict:
    """Reports, plus a pre-computed LOAO summary.

    The summary is computed here rather than in the browser so that the console and
    RESULTS.md cannot disagree. They did: the dashboard averaged every fold including ones
    with three test events, and reported 19.9% where the honest figure is 15.9%.
    """
    loao = _report("loao.csv")
    summary = None
    if loao:
        import pandas as pd

        from janus.defend.evaluate import loao_summary

        df = pd.DataFrame(loao)
        for col in ("recall_unseen", "recall_seen_families", "n_held_out_events"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        summary = loao_summary(df)

    return {
        "fidelity": _report("fidelity.json"),
        "detection": _report("detection.json"),
        "loao": loao,
        "loao_summary": summary,
        "arena": _report("arena.csv"),
    }


@app.get("/api/artifacts/sample")
def sample_artifacts(n: int = 8, seed: int = 0) -> list[dict]:
    """Sample GenAI artefacts from the committed corpus."""
    from janus.generate.artifacts import generate_batch

    return [
        {
            "text": a.text, "is_scam": a.is_scam, "card": a.card,
            "channel": a.channel, "language": a.language,
            "tells": list(a.tells), "source_id": a.source_id,
        }
        for a in generate_batch(n, seed=seed)
    ]


# --- live console ------------------------------------------------------------------------

def _event_payload(i: int) -> dict:
    ev = state.events.iloc[i]
    payload = {
        "event_id": int(ev.event_id),
        "ts": pd.Timestamp(ev.ts).isoformat(),
        "rail": ev.rail,
        "channel": ev.channel,
        "amount": float(ev.amount),
        "payer": f"cust{int(ev.payer_id):06d}",
        "payee": ev.payee_handle,
        "payee_type": ev.payee_type,
        "city": ev.payer_city,
        "fraud_score": round(float(state.fraud_scores[i]), 4),
        "novelty_score": round(float(state.novelty_scores[i]), 4),
        "action": str(state.actions[i]),
        "is_fraud": int(ev.is_fraud),
        "attack_id": ev.attack_id or None,
    }
    if payload["action"] != "allow" and state.explainer is not None:
        try:
            expl = state.explainer.explain(state.features.iloc[i : i + 1], top=4)[0]
            payload["reasons"] = expl.reasons[:3]
            payload["drivers"] = [{"feature": f, "impact": round(v, 4)}
                                  for f, v in expl.top_features[:4]]
        except Exception:  # noqa: BLE001 - an explanation failure must not drop the event
            payload["reasons"] = []
    return payload


@app.get("/api/stream/snapshot")
def stream_snapshot(limit: int = 60, alerts_only: bool = False) -> list[dict]:
    if not state.ready:
        raise HTTPException(status_code=503, detail=state.load_error or "not ready")
    idx = range(len(state.events))
    if alerts_only:
        idx = np.flatnonzero(state.actions != "allow")
    picked = list(idx)[-limit:]
    return [_event_payload(int(i)) for i in picked]


@app.websocket("/ws/stream")
async def stream(websocket: WebSocket) -> None:
    """Replay the scored event stream. Client may send {"rate": n} to change speed."""
    await websocket.accept()
    if not state.ready:
        await websocket.send_json({"type": "error", "detail": state.load_error})
        await websocket.close()
        return

    rate = 12.0          # events per second
    cursor = int(len(state.events) * 0.85)   # start in the held-out region

    # Send a backlog before going live. Without it the console sits empty for the first few
    # seconds - the worst possible first impression for the view that is supposed to show the
    # system working - and an analyst opening a console mid-shift expects to see what just
    # happened, not to wait for it. These are the same scored events, just already past.
    backlog = 45
    for i in range(max(0, cursor - backlog), cursor):
        await websocket.send_json({"type": "event", "data": _event_payload(i), "backlog": True})

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0 / rate)
                command = json.loads(msg)
                rate = float(command.get("rate", rate))
                if command.get("reset"):
                    cursor = int(len(state.events) * 0.85)
            except TimeoutError:
                pass

            if cursor >= len(state.events):
                cursor = int(len(state.events) * 0.85)
            await websocket.send_json({"type": "event", "data": _event_payload(cursor)})
            cursor += 1
    except WebSocketDisconnect:
        return
