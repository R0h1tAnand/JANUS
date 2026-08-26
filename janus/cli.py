"""Janus command line: one entry point per pillar."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Project Janus - adversarial red-team/blue-team lab for payment fraud.",
)
atlas_app = typer.Typer(no_args_is_help=True, help="Pillar 1: identify. Inspect the attack atlas.")
app.add_typer(atlas_app, name="atlas")
gen_app = typer.Typer(no_args_is_help=True, help="Pillar 2: generate. Simulate the payment world and its attacks.")
app.add_typer(gen_app, name="generate")
defend_app = typer.Typer(no_args_is_help=True, help="Pillar 3: defend. Train and evaluate the detector.")
app.add_typer(defend_app, name="defend")
arena_app = typer.Typer(no_args_is_help=True, help="The closed loop: Red evolves, Blue retrains.")
app.add_typer(arena_app, name="arena")

console = Console()


@atlas_app.command("validate")
def atlas_validate(
    check_injectors: bool = typer.Option(
        True, help="Also assert every simulated card has a working injector module."
    ),
) -> None:
    """Strictly validate every attack card. Non-zero exit on any inconsistency."""
    from janus.identify.loader import AtlasError, load_cards

    try:
        cards = load_cards(check_injectors=check_injectors)
    except AtlasError as exc:
        console.print(f"[bold red]Atlas invalid[/]\n{exc}")
        raise typer.Exit(1) from exc
    sim = sum(c.simulated for c in cards)
    console.print(f"[bold green]Atlas valid[/] - {len(cards)} cards, {sim} simulated")


@atlas_app.command("stats")
def atlas_stats(as_json: bool = typer.Option(False, "--json")) -> None:
    """Coverage of the atlas across every axis."""
    from janus.identify.loader import coverage

    cov = coverage()
    if as_json:
        console.print_json(json.dumps(cov))
        return

    console.print(
        f"\n[bold]{cov['total_cards']} attack cards[/]  "
        f"({cov['simulated_cards']} simulated, {cov['distinct_observables']} distinct observables)\n"
    )
    for axis in ("families", "status", "rails", "genai_enablers", "monetization"):
        table = Table(title=axis.replace("_", " ").title(), title_justify="left", show_edge=False)
        table.add_column("value", style="cyan")
        table.add_column("cards", justify="right")
        for key, count in sorted(cov[axis].items(), key=lambda kv: -kv[1]):
            style = "dim" if count == 0 else ""
            table.add_row(key, str(count), style=style)
        console.print(table)
        console.print()


@atlas_app.command("matrix")
def atlas_matrix() -> None:
    """Render the ATT&CK-style kill-chain matrix."""
    from janus.identify.matrix import genai_intensity, matrix

    grid = matrix()
    intensity = genai_intensity()

    table = Table(title="Janus payment-fraud matrix", show_lines=False)
    table.add_column("phase", style="bold cyan")
    table.add_column("n", justify="right")
    table.add_column("GenAI", justify="right")
    table.add_column("techniques")

    for phase, cards in grid.items():
        if not cards:
            continue
        ids = " ".join(c["id"].replace("VY-", "") for c in cards)
        pct = intensity[phase]
        colour = "red" if pct >= 0.6 else "yellow" if pct >= 0.3 else "green"
        table.add_row(phase, str(len(cards)), f"[{colour}]{pct:.0%}[/]", ids)
    console.print(table)
    console.print(
        "\n[dim]GenAI column: fraction of kill-chain steps in that phase that generative AI "
        "materially enables.\nIt concentrates where the bottleneck used to be human effort.[/]\n"
    )


@atlas_app.command("show")
def atlas_show(card_id: str) -> None:
    """Print one attack card in full."""
    from janus.identify.loader import by_id

    try:
        card = by_id(card_id.upper())
    except KeyError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    console.print(f"\n[bold cyan]{card.id}[/]  [bold]{card.name}[/]")
    console.print(f"[dim]{card.summary}[/]\n")
    console.print(f"  rails       {', '.join(str(r) for r in card.rails)}")
    console.print(f"  genai       {', '.join(str(g) for g in card.genai_enablers)}")
    console.print(f"  monetize    {', '.join(str(m) for m in card.monetization)}")
    console.print(
        f"  status      {card.status}  severity={card.severity} "
        f"scalability={card.scalability} detectability={card.detectability} "
        f"risk={card.risk_score}"
    )
    console.print(f"  simulated   {card.simulated}" + (f"  ({card.injector})" if card.injector else ""))
    console.print("\n  [bold]kill chain[/]")
    for step in card.kill_chain:
        marker = "[red]*[/]" if step.genai_used else " "
        console.print(f"   {marker} [cyan]{step.phase:<18}[/] {step.description}")
    console.print("\n  [bold]observables[/]")
    for obs in card.observables:
        console.print(f"     - {obs}")
    if card.mitigations:
        console.print("\n  [bold]mitigations[/]")
        for m in card.mitigations:
            console.print(f"     - {m}")
    console.print()


@atlas_app.command("ideate")
def atlas_ideate(
    top: int = typer.Option(15, help="How many candidates to return."),
    extrapolate: bool = typer.Option(
        False, "--extrapolate", help="Return vectors that reach beyond attested pairs."
    ),
) -> None:
    """Search the attack product space for vectors the atlas does not yet cover."""
    from janus.identify.ideator import coverage_ratio, propose, search_space_size

    cands = propose(top, min_extrapolation=0.5 if extrapolate else 0.0)
    console.print(
        f"\n[dim]product space = {search_space_size():,} cells; "
        f"atlas explicitly realises {coverage_ratio():.1%}[/]\n"
    )
    table = Table(show_edge=False)
    table.add_column("priority", justify="right", style="bold")
    table.add_column("tier", style="magenta")
    table.add_column("plaus", justify="right")
    table.add_column("impact", justify="right")
    table.add_column("proposed vector")
    table.add_column("nearest", style="dim")
    for c in cands:
        table.add_row(
            f"{c.priority:.3f}", c.tier, f"{c.plausibility:.2f}",
            f"{c.impact:.2f}", c.title(), c.nearest_card,
        )
    console.print(table)
    console.print()

@atlas_app.command("signals")
def atlas_signals() -> None:
    """Show the signal families the atlas requires the defence to implement."""
    from janus.identify.loader import atlas as load_atlas
    from janus.identify.signals import FAMILY_DESCRIPTIONS, SignalFamily, observables_for

    cards = load_atlas()
    table = Table(title="Signal families (Pillar 1 -> Pillar 3 contract)", show_edge=False)
    table.add_column("family", style="bold cyan")
    table.add_column("obs", justify="right")
    table.add_column("cards", justify="right")
    table.add_column("what it measures", style="dim")
    for fam in SignalFamily:
        n_cards = sum(1 for c in cards if fam in c.signal_families)
        table.add_row(str(fam), str(len(observables_for(fam))), str(n_cards),
                      FAMILY_DESCRIPTIONS[fam])
    console.print(table)
    console.print()


@gen_app.command("run")
def generate_run(
    customers: int = typer.Option(10_000, help="Population size of the synthetic economy."),
    days: int = typer.Option(60, help="Length of the simulation window."),
    seed: int = typer.Option(42),
    intensity: float = typer.Option(1.0, help="Attack campaign volume multiplier."),
    fraud_rate: float = typer.Option(0.006, help="Target fraud rate; set 0 to leave uncalibrated."),
    out: str = typer.Option("data/synthetic/events.parquet", help="Where to write the events."),
) -> None:
    """Generate a full simulation and write it to parquet."""
    from pathlib import Path

    from janus.generate.simulate import simulate
    from janus.generate.world import WorldConfig

    cfg = WorldConfig(n_customers=customers, days=days, seed=seed)
    with console.status("simulating..."):
        res = simulate(cfg, intensity=intensity,
                       target_fraud_rate=fraud_rate if fraud_rate > 0 else None)
    console.print(f"[bold green]{res.summary()}[/]")

    table = Table(title="Attack families generated", show_edge=False)
    table.add_column("attack", style="cyan")
    table.add_column("events", justify="right")
    table.add_column("campaigns", justify="right")
    table.add_column("median amt", justify="right")
    table.add_column("total value", justify="right")
    for aid, row in res.per_attack.iterrows():
        table.add_row(aid, f"{int(row.events):,}", f"{int(row.campaigns):,}",
                      f"{row.median_amount:,.0f}", f"{row.total_value / 1e5:,.1f}L")
    console.print(table)

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    res.events.to_parquet(path, index=False)
    console.print(f"\n[dim]wrote {path} ({path.stat().st_size / 1e6:.0f} MB)[/]")


@gen_app.command("fidelity")
def generate_fidelity(
    events_path: str = typer.Option("data/synthetic/events.parquet", "--events"),
    seed: int = typer.Option(0),
) -> None:
    """Measure how closely the synthetic data resembles real public payment data."""
    from pathlib import Path

    import pandas as pd

    from janus.generate.fidelity import report as fidelity_report

    path = Path(events_path)
    if not path.exists():
        console.print(f"[red]{path} not found. Run `janus generate run` first.[/]")
        raise typer.Exit(1)

    events = pd.read_parquet(path)
    with console.status("measuring fidelity..."):
        rep = fidelity_report.build(events, seed=seed)

    for entry in rep["references"]:
        d = entry["discriminator"]
        colour = ("green" if d["auc"] < 0.65 else "yellow" if d["auc"] < 0.80 else "red")
        console.print(f"\n[bold]{entry['reference'].upper()}[/]  "
                      f"[dim]compared on {', '.join(entry['compared_on'])}[/]")
        t = Table(show_edge=False, box=None)
        t.add_column("metric", style="cyan")
        t.add_column("value", justify="right")
        t.add_column("note", style="dim")
        for k in entry["ks"]:
            t.add_row(f"KS {k['column']}", f"{k['statistic']:.4f}", k["verdict"])
        t.add_row("correlation delta", f"{entry['correlation_delta']:.4f}", "0 = identical joint structure")
        t.add_row("Benford MAD (synth)", f"{entry['benford_synthetic']['mad']:.4f}",
                  entry["benford_synthetic"]["verdict"])
        t.add_row("Benford MAD (real)", f"{entry['benford_reference']['mad']:.4f}",
                  entry["benford_reference"]["verdict"])
        sh = entry["shape"]
        t.add_row("log-amount sd", f"{sh['synthetic_sd']:.2f}", f"reference {sh['reference_sd']:.2f}")
        t.add_row("log-amount skew", f"{sh['synthetic_skew']:+.2f}", f"reference {sh['reference_skew']:+.2f}")
        console.print(t)
        console.print(f"  [bold {colour}]DISCRIMINATOR AUC {d['auc']:.4f}[/] "
                      f"[{colour}]({d['verdict']})[/]  [dim]0.50 = indistinguishable[/]")
        if entry["tstr"].get("available"):
            ts = entry["tstr"]
            console.print(f"  TSTR {ts['tstr_auc']} vs TRTR {ts['trtr_auc']} "
                          f"-> transfer ratio [bold]{ts['transfer_ratio']}[/]")

    console.print("\n[bold]Known limitations[/]")
    for lim in rep["known_limitations"]:
        console.print(f"  [dim]- {lim}[/]")

    out = fidelity_report.write_json(rep)
    console.print(f"\n[dim]wrote {out}[/]\n")


@defend_app.command("train")
def defend_train(
    events_path: str = typer.Option("data/synthetic/events.parquet", "--events"),
    seed: int = typer.Option(0),
    out: str = typer.Option("models/defence.joblib"),
) -> None:
    """Train the full defence and persist it for the API and dashboard."""
    from pathlib import Path

    import pandas as pd

    from janus.defend import novelty, persist, supervised
    from janus.defend.evaluate import temporal_split

    events = pd.read_parquet(events_path)
    with console.status("building features..."):
        split = temporal_split(events, train_fraction=0.85)
    with console.status("training supervised layer..."):
        detector = supervised.train(split.X_train, split.y_train, seed=seed)
    with console.status("training novelty layer..."):
        nov = novelty.train(split.X_train, split.y_train, seed=seed)

    path = persist.save(detector, nov, metadata={
        "events_path": events_path,
        "n_train_events": int(len(split.X_train)),
        "train_fraud_rate": float(split.y_train.mean()),
        "seed": seed,
    }, path=Path(out))
    console.print(f"[bold green]Defence trained[/] - {len(detector.features)} features, "
                  f"{len(split.X_train):,} training events")
    console.print(f"[dim]wrote {path} ({path.stat().st_size / 1e6:.1f} MB)[/]")


@defend_app.command("evaluate")
def defend_evaluate(
    events_path: str = typer.Option("data/synthetic/events.parquet", "--events"),
    seed: int = typer.Option(0),
    out: str = typer.Option("reports/detection.json"),
) -> None:
    """Train on a temporal split and report detection efficacy."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from janus.defend import novelty, policy, supervised
    from janus.defend.evaluate import temporal_split

    events = pd.read_parquet(events_path)
    with console.status("building features..."):
        split = temporal_split(events)
    with console.status("training supervised detector..."):
        detector = supervised.train(split.X_train, split.y_train, seed=seed)
    with console.status("training novelty layer..."):
        nov = novelty.train(split.X_train, split.y_train, seed=seed)

    # Two scores, two jobs. Ranking and thresholding use the uncalibrated model output, which
    # has full resolution; the policy engine uses the calibrated probability, because its
    # arithmetic is denominated in rupees and needs a real probability rather than a rank.
    rank_scores = detector.rank_score(split.X_test)
    calibrated_scores = detector.score(split.X_test)
    novelty_scores = nov.score(split.X_test)
    metrics = supervised.evaluate(detector, split.X_test, split.y_test, scores=rank_scores)

    console.print(f"\n[bold]Detection efficacy[/]  [dim]temporal holdout, "
                  f"{metrics['n']:,} events, {metrics['n_fraud']:,} fraud "
                  f"({metrics['base_rate']:.3%})[/]\n")
    t = Table(show_edge=False, box=None)
    t.add_column("metric", style="cyan")
    t.add_column("value", justify="right", style="bold")
    t.add_column("", style="dim")
    for key in ("roc_auc", "pr_auc"):
        t.add_row(key.upper().replace("_", "-"), f"{metrics[key]:.4f}", "")
    for fpr in supervised.TARGET_FPRS:
        realised = metrics.get(f"realised_fpr@{fpr}")
        t.add_row(f"recall @ {fpr:.2%} FPR", f"{metrics[f'recall@fpr{fpr}']:.4f}",
                  f"realised FPR {realised}")
    console.print(t)

    console.print("\n[bold]Novelty layer[/] [dim](trained on legitimate traffic only)[/]")
    from sklearn.metrics import roc_auc_score

    auc_nov = float(roc_auc_score(split.y_test, novelty_scores))
    console.print(f"  standalone ROC-AUC {auc_nov:.4f} [dim](supervised: {metrics['roc_auc']:.4f})[/]")

    console.print("\n[bold]Per-family recall[/] [dim]@ 0.1% FPR, families the model has seen[/]")
    threshold = supervised.threshold_at_fpr(split.y_test, rank_scores, 0.001)
    test = split.test_events
    fam_rows = []
    for aid in sorted(test.loc[test.is_fraud == 1, "attack_id"].unique()):
        mask = (test.attack_id == aid).to_numpy() & (split.y_test == 1)
        if mask.sum():
            fam_rows.append((aid, int(mask.sum()), float((rank_scores[mask] >= threshold).mean())))
    ft = Table(show_edge=False, box=None)
    ft.add_column("family", style="cyan")
    ft.add_column("n", justify="right")
    ft.add_column("recall", justify="right")
    for aid, n, r in sorted(fam_rows, key=lambda x: -x[2]):
        colour = "green" if r >= 0.8 else "yellow" if r >= 0.5 else "red"
        ft.add_row(aid, str(n), f"[{colour}]{r:.3f}[/]")
    console.print(ft)

    console.print("\n[bold]Operating point[/] [dim](best net benefit within a 2% "
                  "legitimate-challenge ceiling)[/]")
    amounts = split.test_events.amount.to_numpy()
    sweep = policy.sweep_thresholds(calibrated_scores, novelty_scores, split.y_test, amounts)
    best = sweep.iloc[0]
    console.print(f"  thresholds  step-up {best.step_up:.2f} / hold {best.hold:.2f} / block {best.block:.2f}")
    console.print(f"  fraud prevented  Rs {best.fraud_prevented/1e5:,.1f}L of "
                  f"Rs {best.fraud_exposure/1e5:,.1f}L exposure ({best.prevention_rate:.1%})")
    console.print(f"  friction cost    Rs {best.friction_cost/1e5:,.2f}L")
    console.print(f"  [bold green]net benefit      Rs {best.net_benefit/1e5:,.1f}L[/]")
    console.print(f"  legitimate payments challenged: {best.legit_challenged_rate:.3%} "
                  f"[dim](ceiling 2.000%)[/]")
    unconstrained = sweep.sort_values("net_benefit", ascending=False).iloc[0]
    if unconstrained.legit_challenged_rate > 0.02:
        console.print(f"  [dim]unconstrained optimum would net Rs "
                      f"{unconstrained.net_benefit/1e5:,.1f}L but challenge "
                      f"{unconstrained.legit_challenged_rate:.1%} of legitimate payments - "
                      f"rejected as unshippable.[/]")

    console.print("\n[bold]Top features[/]")
    for row in supervised.feature_importance(detector, 12).itertuples():
        bar = "#" * int(row.importance * 260)
        console.print(f"  {row.feature:34s} {row.importance:.3f} [dim]{bar}[/]")

    report = {
        "metrics": metrics,
        "novelty_auc": round(auc_nov, 4),
        "per_family_recall": {a: round(r, 4) for a, n, r in fam_rows},
        "best_policy": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in best.to_dict().items() if k != "action_counts"},
        "importance": supervised.feature_importance(detector, 20).to_dict("records"),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    console.print(f"\n[dim]wrote {out}[/]\n")


@defend_app.command("loao")
def defend_loao(
    customers: int = typer.Option(10_000),
    days: int = typer.Option(60),
    seed: int = typer.Option(42),
    out: str = typer.Option("reports/loao.csv"),
) -> None:
    """Leave-one-attack-out: how well does the defence catch families it never trained on?"""
    from pathlib import Path

    from janus.defend.evaluate import loao_summary, run_loao
    from janus.generate.world import WorldConfig

    cfg = WorldConfig(n_customers=customers, days=days, seed=seed)
    console.print("[dim]Each fold trains a detector on a world where that family does not "
                  "exist, then tests against a different world where it does.[/]\n")
    df = run_loao(cfg)

    t = Table(title="Leave-one-attack-out recall @ 0.1% FPR", show_edge=False)
    t.add_column("held-out family", style="cyan")
    t.add_column("events", justify="right")
    t.add_column("recall (unseen)", justify="right")
    t.add_column("recall (seen families)", justify="right")
    t.add_column("gap", justify="right")
    for row in df.itertuples():
        r = row.recall_unseen
        colour = "green" if r and r >= 0.7 else "yellow" if r and r >= 0.3 else "red"
        t.add_row(row.family, str(row.n_held_out_events),
                  f"[{colour}]{r:.3f}[/]" if r is not None else "-",
                  f"{row.recall_seen_families:.3f}" if row.recall_seen_families else "-",
                  f"{row.generalisation_gap:+.3f}" if row.generalisation_gap == row.generalisation_gap else "-")
    console.print(t)

    s = loao_summary(df)
    console.print(f"\n  mean recall on unseen families   [bold]{s['mean_recall_unseen']:.3f}[/]")
    console.print(f"  median                           {s['median_recall_unseen']:.3f}")
    console.print(f"  mean recall on seen families     {s['mean_recall_seen']:.3f}")
    console.print(f"  families above 50% recall        {s['families_above_50pct']}/{s['families']}")
    console.print(f"  families below 20% recall        [red]{s['families_below_20pct']}[/]/{s['families']}")
    console.print(f"  generalisation gap               [bold red]{s['generalisation_gap']:+.3f}[/] "
                  f"[dim](seen minus unseen)[/]")
    if s["n_low_volume_folds_excluded"]:
        console.print(f"  [dim]{s['n_low_volume_folds_excluded']} fold(s) with <"
                      f"{s['min_reliable_events']} test events excluded from the mean as "
                      f"statistically meaningless.[/]")
    console.print(
        "\n[dim]The gap is the point. Supervised detection largely does not transfer to attack\n"
        "families it was never trained on - which is why this system also ships an unsupervised\n"
        "novelty layer and an adversarial loop. Read `janus arena run` next.[/]\n"
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"\n[dim]wrote {out}[/]\n")


@arena_app.command("run")
def arena_run(
    profile: str = typer.Option(
        "full", help="'full' for the reported numbers (~1h), 'fast' for a demo (~15 min)."
    ),
    rounds: int = typer.Option(0, help="Override the profile's round count."),
    generations: int = typer.Option(0, help="Override generations per Red search."),
    population: int = typer.Option(0, help="Override Red population size."),
    customers: int = typer.Option(0, help="Override world size."),
    days: int = typer.Option(0),
    seed: int = typer.Option(5),
    out: str = typer.Option("reports/arena.csv"),
) -> None:
    """Run the adversarial loop: Red searches for evasive variants, Blue learns from them.

    Cost grows as rounds x generations x population x world size, because every candidate
    triggers a fresh simulation and feature build. The 'fast' profile shrinks all four so the
    loop fits in a demo; note that its smaller world cannot reach a realistic fraud rate, so
    read it as a demonstration of the mechanism rather than as a measurement.
    """
    import json
    from pathlib import Path

    from janus.generate.world import WorldConfig
    from janus.loop.arena import PROFILES, rounds_to_detect, run_arena

    if profile not in PROFILES:
        console.print(f"[red]Unknown profile {profile!r}. Choose one of: {list(PROFILES)}[/]")
        raise typer.Exit(1)
    preset = PROFILES[profile]
    rounds = rounds or preset["rounds"]
    generations = generations or preset["generations"]
    population = population or preset["population"]
    customers = customers or preset["customers"]
    days = days or preset["days"]

    cfg = WorldConfig(n_customers=customers, n_merchants=400, days=days, seed=seed)
    console.print(f"[dim]profile '{profile}': {rounds} rounds x {generations} generations x "
                  f"population {population}, {customers:,} customers over {days} days[/]")
    console.print(
        "\n[dim]Red optimises rupees-through, not evasion rate: an attacker who evades "
        "everything\nby sending nothing is not a threat. Blue retrains on what Red finds, "
        "at a fixed 0.1% FPR\nso it cannot win by simply becoming more aggressive.[/]\n"
    )
    result = run_arena(cfg, rounds=rounds, generations=generations, population=population)

    df = result.to_frame()
    t = Table(title="Red vs Blue", show_edge=False)
    t.add_column("round", justify="right", style="bold")
    t.add_column("Red evasion", justify="right")
    t.add_column("value through", justify="right")
    t.add_column("", width=0)
    t.add_column("Blue recall", justify="right")
    t.add_column("FPR", justify="right")
    t.add_column("base rate", justify="right", style="dim")
    t.add_column("Red's adaptation", style="dim")
    for row in df.itertuples():
        ev_colour = "red" if row.red_evasion_rate > 0.5 else "yellow" if row.red_evasion_rate > 0.25 else "green"
        t.add_row(
            str(row.round),
            f"[{ev_colour}]{row.red_evasion_rate:.1%}[/]",
            f"{row.value_through_pct:.1%}",
            "",
            f"{row.blue_recall:.3f}",
            f"{row.realised_fpr:.4f}",
            f"{row.train_base_rate:.3%}",
            (row.adaptation[:46] + "...") if len(row.adaptation) > 46 else row.adaptation,
        )
    console.print(t)

    s = result.summary()
    console.print(f"\n  Red evasion   {s['evasion_first']:.1%} -> [bold]{s['evasion_last']:.1%}[/]")
    console.print(f"  value through {s['value_through_first_pct']:.1%} -> "
                  f"[bold]{s['value_through_last_pct']:.1%}[/] [dim]of attack value[/]")
    console.print(f"  Blue recall   {s['recall_first']:.3f} -> [bold]{s['recall_last']:.3f}[/]")
    console.print(f"  FPR drift     {s['fpr_drift']:+.5f} [dim](should be ~0: Blue must not "
                  f"buy recall with friction)[/]")
    console.print(f"  base rate     {s['base_rate_first']:.3%} -> {s['base_rate_last']:.3%} "
                  f"[dim](must stay flat: a replay buffer that inflates the fraud share "
                  f"breaks the model rather than hardening it)[/]")
    rtd = rounds_to_detect(result)
    console.print(f"  rounds-to-detect (<50% evasion): "
                  f"[bold]{rtd if rtd else 'not reached'}[/]")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    # asdict(), not __dict__: RoundResult is a slots dataclass and has no instance dict.
    from dataclasses import asdict

    Path(out).with_suffix(".json").write_text(json.dumps(
        {"summary": s, "rounds": [asdict(r) for r in result.rounds]}, indent=2, default=str))
    console.print(f"\n[dim]wrote {out}[/]\n")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (development)."),
) -> None:
    """Serve the API that backs the web console."""
    import uvicorn

    from janus.defend import persist

    if not persist.exists():
        console.print("[yellow]No trained defence found.[/] The console will start but show "
                      "a setup message until you run:")
        console.print("  [cyan]uv run janus generate run[/]")
        console.print("  [cyan]uv run janus defend train[/]\n")
    console.print(f"[bold]API on http://{host}:{port}[/]  "
                  f"[dim](start the UI with: cd web && npm run dev)[/]\n")
    uvicorn.run("janus.api.app:app", host=host, port=port, reload=reload)


@app.command("status")
def status() -> None:
    """What has been built so far, and what is still missing."""
    from pathlib import Path

    from janus.defend import persist
    from janus.identify.loader import coverage

    cov = coverage()
    checks = [
        ("attack atlas", True, f"{cov['total_cards']} cards, {cov['simulated_cards']} simulated"),
        ("synthetic events", Path("data/synthetic/events.parquet").exists(),
         "data/synthetic/events.parquet"),
        ("reference data", Path("data/raw/sparkov.csv").exists(), "data/raw/*.csv"),
        ("trained defence", persist.exists(), "models/defence.joblib"),
        ("fidelity report", Path("reports/fidelity.json").exists(), "reports/fidelity.json"),
        ("detection report", Path("reports/detection.json").exists(), "reports/detection.json"),
        ("LOAO results", Path("reports/loao.csv").exists(), "reports/loao.csv"),
        ("arena results", Path("reports/arena.csv").exists(), "reports/arena.csv"),
        ("web dependencies", Path("web/node_modules").exists(), "web/node_modules"),
    ]
    t = Table(show_edge=False, box=None)
    t.add_column("", width=3)
    t.add_column("component", style="cyan")
    t.add_column("detail", style="dim")
    for name, ok, detail in checks:
        t.add_row("[green]OK[/]" if ok else "[red]--[/]", name, detail)
    console.print()
    console.print(t)
    missing = [n for n, ok, _ in checks if not ok]
    if missing:
        console.print(f"\n[dim]Missing: {', '.join(missing)}. Run `make all` to build everything.[/]\n")
    else:
        console.print("\n[bold green]Everything is built.[/] Run `janus serve`.\n")


if __name__ == "__main__":
    app()
