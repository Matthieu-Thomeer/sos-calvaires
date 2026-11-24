#!/usr/bin/env python3
"""Suivi des votes de vote.soscalvaires.org (appel à projets « Croix du Pape »).

Toutes les 10 minutes : lit le nombre de voix de chaque lieu, l'ajoute à votes.csv
et redessine deux graphiques :
  - votes_cumul.png : nombre de voix de chaque lieu au fil du temps ;
  - votes_delta.png : voix gagnées par tranche de 10 minutes ;
  - votes_normalise.png et dashboard.html : croissance de chaque lieu en % de ses voix (voir dashboard.py).

Les compteurs viennent de l'API publique que la page du vote lit elle-même
(Supabase, clé « publishable » publiée dans https://vote.soscalvaires.org/config.js).
La page la relit toutes les 15 s ; un relevé toutes les 10 min est bien plus léger.

Usage :
  python3 suivi_votes.py            # boucle infinie, un relevé toutes les 10 min, calé sur l'horloge (hh:00, hh:10…)
  python3 suivi_votes.py --once     # un seul relevé (pour cron / systemd)
  python3 suivi_votes.py --plot     # redessine seulement, sans relevé
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "votes.csv"
COLORS_PATH = HERE / "couleurs.json"
CONFIG_URL = "https://vote.soscalvaires.org/config.js"
# Valeurs relevées dans config.js le 2026-11-23 ; relues à chaque démarrage au cas où elles changent.
DEFAULT_URL = "https://jdptavibfllmjcuhocbs.supabase.co"
DEFAULT_KEY = "sb_publishable_G1uYrZxCniHpewFb_AL6PQ_g0JHitym"
INTERVAL_S = 600
TOP_N = 8
FIELDS = ["timestamp_utc", "id", "slug", "name", "commune", "departement", "votes"]

# Palette catégorielle validée (8 teintes, ordre fixe), surface claire.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#e8e7e2"


def load_config():
    """Relit l'URL et la clé publique depuis config.js ; garde les valeurs par défaut sinon."""
    try:
        txt = requests.get(CONFIG_URL, timeout=20).text
        cfg = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
        return cfg["SUPABASE_URL"], cfg["SUPABASE_PUBLISHABLE_KEY"]
    except Exception as e:  # noqa: BLE001
        print(f"[config] lecture impossible ({e}), valeurs par défaut", file=sys.stderr)
        return DEFAULT_URL, DEFAULT_KEY


def fetch(base, key):
    r = requests.get(
        f"{base}/rest/v1/candidates",
        params={"select": "id,slug,name,commune,departement,votes_count", "published": "eq.true"},
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def record(rows):
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({"timestamp_utc": ts, "id": r["id"], "slug": r.get("slug") or r["id"],
                        "name": r.get("name") or "", "commune": r.get("commune") or "",
                        "departement": r.get("departement") or "", "votes": r["votes_count"]})
    total = sum(r["votes_count"] for r in rows)
    print(f"[{ts}] {len(rows)} lieux, {total} voix au total", flush=True)


def stable_colors(top_ids):
    """Une couleur reste attachée à un lieu ; un lieu qui sort du top libère sa couleur."""
    try:
        mapping = json.loads(COLORS_PATH.read_text())
    except Exception:  # noqa: BLE001
        mapping = {}
    mapping = {k: v for k, v in mapping.items() if k in top_ids}
    free = [i for i in range(len(SERIES)) if i not in mapping.values()]
    for pid in top_ids:
        if pid not in mapping:
            mapping[pid] = free.pop(0)
    COLORS_PATH.write_text(json.dumps(mapping, indent=1))
    return {pid: SERIES[i] for pid, i in mapping.items()}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    if not CSV_PATH.exists():
        print("Pas encore de données.")
        return
    df = pd.read_csv(CSV_PATH)
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert("Europe/Paris")
    df["label"] = df["name"].fillna(df["slug"]).str.slice(0, 40)
    wide = df.pivot_table(index="t", columns="id", values="votes", aggfunc="last").sort_index()
    # Ignore un relevé pris moins de 3 min après le précédent gardé (ex. --once puis boucle) :
    # ramené à 10 min, son écart serait très amplifié.
    keep, prev = [], None
    for t in wide.index:
        if prev is None or t - prev >= pd.Timedelta(minutes=3):
            keep.append(t)
            prev = t
    wide = wide.loc[keep]
    labels = df.drop_duplicates("id", keep="last").set_index("id")["label"]
    last = wide.ffill().iloc[-1].sort_values(ascending=False)
    top = list(last.index[:TOP_N])
    colors = stable_colors(top)
    tz = df["t"].dt.tz

    def dates(ax, n=7):
        loc = mdates.AutoDateLocator(tz=tz, minticks=2, maxticks=n)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=tz, show_offset=False))
    stamp = wide.index[-1].strftime("%d/%m/%Y %Hh%M")

    # Voix gagnées, ramenées à une tranche de 10 min (corrige les relevés manqués).
    dt_min = wide.index.to_series().diff().dt.total_seconds() / 60
    delta = wide.diff().div(dt_min, axis=0) * 10
    delta = delta.iloc[1:]

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2})

    # 1) cumul
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=SURFACE)
    style(ax)
    for pid in wide.columns:
        if pid not in top:
            ax.plot(wide.index, wide[pid], color=MUTED, linewidth=0.8, alpha=0.6, zorder=1)
    for rank, pid in enumerate(top, 1):
        s = wide[pid].dropna()
        ax.plot(s.index, s, color=colors[pid], linewidth=2, zorder=3,
                label=f"{rank}. {labels[pid]} ({int(s.iloc[-1])})")
        ax.annotate(f"{rank}", (s.index[-1], s.iloc[-1]), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    dates(ax)
    ax.set_ylabel("voix")
    ax.set_title(f"Voix par lieu (top {TOP_N} en couleur, autres en gris) · relevé du {stamp}",
                 loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "votes_cumul.png", dpi=120)
    plt.close(fig)

    # 2) différences sur 10 min : total + petits multiples du top
    fig = plt.figure(figsize=(14, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1, 1], hspace=0.55, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, :])
    style(ax0)
    width = (delta.index.to_series().diff().median() or pd.Timedelta(minutes=10)) * 0.8
    if len(delta):
        ax0.bar(delta.index, delta.sum(axis=1, min_count=1), width=width, color=INK2)
    ax0.set_title("Tous les lieux : voix gagnées par tranche de 10 min", loc="left", fontsize=11)
    dates(ax0, 10)
    ymax = max(1.0, float(delta[top].max().max()) if len(delta) else 1.0)
    for k, pid in enumerate(top):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4], sharex=ax0)
        style(ax)
        if len(delta):
            ax.bar(delta.index, delta[pid], width=width, color=colors[pid])
        ax.set_ylim(0, ymax * 1.05)
        ax.set_title(f"{k + 1}. {labels[pid][:34]}", loc="left", fontsize=8.5)
        dates(ax, 4)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Voix gagnées par tranche de 10 min · relevé du {stamp} (même échelle pour le top {TOP_N})",
                 x=0.01, ha="left", fontsize=12)
    fig.savefig(HERE / "votes_delta.png", dpi=120, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    # 3) tableau de bord HTML + graphique normalisé (pentes comparables)
    import dashboard
    dashboard.build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="un seul relevé puis quitte")
    ap.add_argument("--plot", action="store_true", help="redessine seulement")
    ap.add_argument("--interval", type=int, default=INTERVAL_S, help="secondes entre relevés")
    a = ap.parse_args()
    if a.plot:
        plot()
        return
    base, key = load_config()
    if not a.once:
        wait_next_tick(a.interval)
    while True:
        try:
            record(fetch(base, key))
            plot()
        except Exception as e:  # noqa: BLE001 — une erreur réseau ne doit pas arrêter le suivi
            print(f"[erreur] {e}", file=sys.stderr, flush=True)
        if a.once:
            return
        wait_next_tick(a.interval)


def wait_next_tick(interval):
    """Attend le prochain multiple de `interval` secondes sur l'horloge (300 s → hh:00, hh:05, hh:10…).

    Calé sur l'heure ronde plutôt que sur l'heure de lancement : les relevés tombent toujours aux mêmes minutes,
    et un relevé lent (graphiques, prévision) ne fait pas dériver les suivants. Les fuseaux horaires étant décalés
    d'heures entières, c'est vrai en heure de Paris comme en UTC.
    """
    now = time.time()
    nxt = (now // interval + 1) * interval
    print(f"[suivi] prochain relevé à {datetime.fromtimestamp(nxt).strftime('%H:%M:%S')}", flush=True)
    time.sleep(nxt - now)


if __name__ == "__main__":
    main()
