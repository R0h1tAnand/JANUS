"""API contract tests.

The dashboard is a mandated deliverable, so its backend needs the same guarantees as the rest
of the system: endpoints that exist, shapes that do not drift, and - most importantly - no
label leakage into anything a viewer can read. The event payload the console renders must never
carry the ground-truth label in a field the UI could accidentally key off.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from janus.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_always_answers(client):
    """Health must respond even when no model or data has been produced yet."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body


def test_atlas_endpoint_matches_the_atlas(client):
    from janus.identify.loader import atlas

    body = client.get("/api/atlas").json()
    assert len(body["cards"]) == len(atlas())
    assert body["coverage"]["total_cards"] == len(atlas())
    card = body["cards"][0]
    for key in ("id", "name", "rails", "simulated", "signal_families"):
        assert key in card


def test_matrix_endpoint_carries_genai_intensity(client):
    body = client.get("/api/atlas/matrix").json()
    assert body["matrix"]
    intensity = body["genai_intensity"]
    assert intensity["resource_dev"] > intensity["monetize"], (
        "the atlas's central finding must survive the API layer"
    )


def test_card_detail_and_404(client):
    ok = client.get("/api/atlas/VY-SOC-001")
    assert ok.status_code == 200
    assert ok.json()["kill_chain"]
    assert client.get("/api/atlas/VY-NOPE-999").status_code == 404


def test_signals_endpoint_lists_every_family(client):
    from janus.identify.signals import SignalFamily

    body = client.get("/api/atlas/signals").json()
    assert len(body) == len(list(SignalFamily))
    assert all(row["observables"] for row in body)


def test_ideator_endpoint_ranks_candidates(client):
    body = client.get("/api/atlas/ideate?top=5").json()
    assert len(body["candidates"]) == 5
    priorities = [c["priority"] for c in body["candidates"]]
    assert priorities == sorted(priorities, reverse=True)


def test_artifact_sampling_returns_both_classes(client):
    rows = client.get("/api/artifacts/sample?n=40").json()
    labels = {r["is_scam"] for r in rows}
    assert labels == {True, False}
    assert all(r["text"] for r in rows)


def test_reports_endpoint_tolerates_missing_files(client):
    """A fresh clone has no reports yet; the console must degrade, not error."""
    body = client.get("/api/reports").json()
    assert {"fidelity", "detection", "loao", "loao_summary", "arena"} <= set(body)


def test_loao_summary_is_computed_server_side(client):
    """The console must not recompute the LOAO mean itself.

    It did once, averaging every fold including three-event ones, and reported 19.9% where
    RESULTS.md said 15.9%. Two surfaces quoting different headline numbers is worse than
    either being slightly off, so the summary is computed once, server-side, with the same
    exclusion rule the CLI uses.
    """
    summary = client.get("/api/reports").json()["loao_summary"]
    if summary is None:
        pytest.skip("no LOAO results generated yet")
    for key in ("mean_recall_unseen", "mean_recall_seen", "generalisation_gap",
                "families_above_50pct", "min_reliable_events"):
        assert key in summary
    assert 0.0 <= summary["mean_recall_unseen"] <= 1.0
    assert summary["mean_recall_seen"] > summary["mean_recall_unseen"], (
        "if unseen recall ever matches seen recall, suspect leakage rather than success"
    )
