#!/usr/bin/env python3
"""Tableau de bord des votes SOS Calvaires.

Lit votes.csv et produit :
  - dashboard.html        : tableau de bord interactif autonome (données intégrées) ;
  - votes_normalise.png   : croissance de chaque lieu en % de ses propres voix (pentes comparables) ;
  - votes_total.png       : total des voix et voix gagnées d'une heure à l'autre.
La prévision (modèle non linéaire, voir prevision.py) est intégrée au tableau de bord.

Usage :
  python dashboard.py
Appelé aussi automatiquement par suivi_votes.py après chaque relevé.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "votes.csv"
TEMPLATE = HERE / "dashboard_template.html"
OUT_HTML = HERE / "dashboard.html"
OUT_PNG = HERE / "votes_normalise.png"
OUT_TOTAL_PNG = HERE / "votes_total.png"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#e8e7e2"


def load():
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    df["t"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    wide = df.pivot_table(index="t", columns="id", values="votes", aggfunc="last").sort_index()
    # Un relevé pris moins de 3 min après le précédent gardé est ignoré (doublon de démarrage).
    keep, prev = [], None
    for t in wide.index:
        if prev is None or t - prev >= pd.Timedelta(minutes=3):
            keep.append(t)
            prev = t
    wide = wide.loc[keep].ffill().bfill()
    meta = df.drop_duplicates("id", keep="last").set_index("id")
    return wide, meta


# La prévision (deux modèles + tests rétrospectifs) coûte 1 à 2 min de calcul sur tous les cœurs : on ne la refait
# qu'une fois par heure de relevés et on réutilise la dernière entre-temps (la page recale ses projections sur le
# dernier relevé). SOS_PREVISION_FORCE=1 force le recalcul.
FC_CACHE = HERE / "prevision_cache.json"
FC_REFRESH_MIN = 55


def forecast_safe(wide):
    """Prévision (prevision.py) ; une erreur de fit ne doit jamais empêcher de produire le tableau de bord."""
    import os
    t_last = int(wide.index[-1].timestamp() * 1000)
    if not os.environ.get("SOS_PREVISION_FORCE"):
        try:
            cached = json.loads(FC_CACHE.read_text(encoding="utf-8"))
            if (cached.get("ok") and cached.get("models") and 0 <= t_last - cached["at"] < FC_REFRESH_MIN * 60e3
                    and set(cached["models"]["jn"]["places"]) == set(wide.columns)):
                cached["fresh"] = False
                return cached
        except Exception:  # noqa: BLE001 — pas de cache ou cache illisible : on recalcule
            pass
    try:
        from prevision import forecast
        fc = forecast(wide)
    except Exception as e:  # noqa: BLE001
        print(f"[prévision] échec : {e!r}")
        return {"ok": False, "why": "Prévision indisponible pour ce relevé."}
    fc["at"] = t_last
    if fc.get("ok"):
        try:
            FC_CACHE.write_text(json.dumps(fc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        except OSError as e:
            print(f"[prévision] cache non écrit : {e!r}")
    fc["fresh"] = True
    return fc


def history_safe(fc, wide):
    """Enregistre la prévision du moment (une fois par heure) et renvoie tout l'historique pour la page Bilan."""
    try:
        from prevision import load_history, record
        if fc.get("ok") and fc.get("models") and fc.get("fresh"):   # une prévision réutilisée est déjà enregistrée
            record(fc, wide)
        return load_history()
    except Exception as e:  # noqa: BLE001
        print(f"[historique des prévisions] échec : {e!r}")
        return []


def build_html(wide, meta):
    places = []
    for pid in wide.columns:
        m = meta.loc[pid]
        places.append({
            "id": pid,
            "name": str(m["name"]) if isinstance(m["name"], str) and m["name"] else str(m["slug"]),
            "commune": m["commune"] if isinstance(m["commune"], str) else "",
            "dept": m["departement"] if isinstance(m["departement"], str) else "",
            "v": [int(x) for x in wide[pid].tolist()],
        })
    fc = forecast_safe(wide)
    data = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "t": [int(ts.timestamp() * 1000) for ts in wide.index],
        "places": places,
        "fc": fc,
        "hist": history_safe(fc, wide),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/{}", payload)
    OUT_HTML.write_text(html, encoding="utf-8")


def build_png(wide, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    tz = "Europe/Paris"
    w = wide.copy()
    w.index = w.index.tz_convert(tz)
    pct = (w - w.iloc[0]) / w.iloc[0] * 100          # croissance en % de ses propres voix
    hours = (w.index[-1] - w.index[0]).total_seconds() / 3600 or 1
    order = pct.iloc[-1].sort_values(ascending=False).index
    labels = meta["name"].fillna(meta["slug"])
    total = w.sum(axis=1)
    gaps = w.index.to_series().diff() > pd.Timedelta(minutes=25)

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2})
    ncols = 6
    nrows = -(-len(order) // ncols)
    fig = plt.figure(figsize=(18, 3.0 + 2.05 * nrows), facecolor=SURFACE)
    gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[2.6] + [1] * nrows, hspace=0.75, wspace=0.18)

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=7.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def dates(ax, n=6):
        loc = mdates.AutoDateLocator(tz=w.index.tz, minticks=2, maxticks=n)
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=w.index.tz, show_offset=False))

    def shade_gaps(ax):
        for i in range(1, len(w.index)):
            if gaps.iloc[i]:
                ax.axvspan(w.index[i - 1], w.index[i], color=GRID, alpha=0.6, lw=0, zorder=0)

    # Panneau principal : top 8 en couleur, les autres en gris.
    ax0 = fig.add_subplot(gs[0, :])
    style(ax0)
    shade_gaps(ax0)
    top = list(order[:8])
    for pid in order:
        if pid not in top:
            ax0.plot(pct.index, pct[pid], color=MUTED, lw=0.8, alpha=0.7, zorder=1)
    for k, pid in enumerate(top):
        s = pct[pid]
        ax0.plot(s.index, s, color=SERIES[k], lw=2, zorder=3,
                 label=f"{labels[pid][:42]}  +{s.iloc[-1]:.2f} %  ({s.iloc[-1] / hours:.2f} %/h)")
        ax0.plot(s.index[-1], s.iloc[-1], "o", color=SERIES[k], ms=5, zorder=4, mec=SURFACE, mew=1.5)
    ax0.set_ylabel("croissance depuis le 1er relevé (%)")
    ax0.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f} %"))
    dates(ax0, 10)
    ax0.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    t0 = w.index[0].strftime("%d/%m %Hh%M")
    t1 = w.index[-1].strftime("%d/%m %Hh%M")
    ax0.set_title(f"Croissance de chaque lieu, en % de ses propres voix · {t0} → {t1} · "
                  f"{int(total.iloc[-1]):,} voix au total".replace(",", " "),
                  loc="left", fontsize=12.5, color=INK, pad=10)

    # Petits multiples, même échelle : la pente se compare d'un lieu à l'autre.
    ymax = max(0.1, float(pct.max().max()) * 1.08)
    for k, pid in enumerate(order):
        ax = fig.add_subplot(gs[1 + k // ncols, k % ncols], sharex=ax0)
        style(ax)
        shade_gaps(ax)
        s = pct[pid]
        col = SERIES[top.index(pid)] if pid in top else INK2
        ax.fill_between(s.index, 0, s, color=col, alpha=0.12, lw=0)
        ax.plot(s.index, s, color=col, lw=1.6)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} %" if v >= 1 or v == 0 else f"{v:.1f} %"))
        ax.set_title(f"{k + 1}. {labels[pid][:30]}", loc="left", fontsize=8.5, color=INK)
        ax.text(0.02, 0.86, f"{int(w[pid].iloc[-1])} voix · +{int(w[pid].iloc[-1] - w[pid].iloc[0])}  ({s.iloc[-1]:.2f} %)",
                transform=ax.transAxes, fontsize=7.5, color=INK2)
        dates(ax, 4)
        ax.tick_params(labelsize=6.5)
    fig.text(0.01, 0.005, "Zones grisées : pas de relevé (trou de suivi). Même échelle verticale pour tous les petits graphiques.",
             fontsize=8, color=INK2)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def build_total_png(wide):
    """Total des voix au fil du temps + voix gagnées par heure, avec l'évolution d'une heure à l'autre."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    total = wide.sum(axis=1)
    t = total.index
    ts = np.array([x.timestamp() for x in t])  # secondes, quelle que soit l'unité interne de pandas
    gap = pd.Timedelta(minutes=25)

    # Périodes d'une heure calées sur l'horloge ; total interpolé aux bornes (estimé si trou de relevé).
    step = pd.Timedelta(hours=1)
    rows = []
    b = t[0].floor("h")
    while b < t[-1]:
        a, e = max(b, t[0]), min(b + step, t[-1])
        if e - a >= pd.Timedelta(minutes=1):
            va, ve = np.interp([a.timestamp(), e.timestamp()], ts, total.values)
            est = any((t[i] - t[i - 1] > gap) and t[i - 1] < e and t[i] > a for i in range(1, len(t)))
            hours = (e - a).total_seconds() / 3600
            rows.append(dict(b=b, a=a, e=e, gain=ve - va, rate=(ve - va) / hours, est=est,
                             full=e - a >= step - pd.Timedelta(minutes=1)))
        b += step
    per = pd.DataFrame(rows)
    per["delta"] = per["rate"].pct_change() * 100

    tz = "Europe/Paris"
    plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), facecolor=SURFACE, height_ratios=[1, 1.15])
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    tl = t.tz_convert(tz)
    for i in range(1, len(t)):
        seg = slice(i - 1, i + 1)
        if t[i] - t[i - 1] > gap:
            ax1.axvspan(tl[i - 1], tl[i], color=GRID, alpha=0.6, lw=0)
            ax1.plot(tl[seg], total.values[seg], color=SERIES[0], lw=1.5, ls=(0, (3, 3)))
        else:
            ax1.plot(tl[seg], total.values[seg], color=SERIES[0], lw=2, solid_capstyle="round")
    ax1.plot(tl[-1], total.iloc[-1], "o", color=SERIES[0], ms=5, mec=SURFACE, mew=1.5)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    loc = mdates.AutoDateLocator(tz=tl.tz, minticks=3, maxticks=10)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=tl.tz, show_offset=False))
    g = total.iloc[-1] - total.iloc[0]
    fmt = lambda v: f"{int(v):,}".replace(",", " ")
    ax1.set_title(f"Total des voix : {fmt(total.iloc[0])} → {fmt(total.iloc[-1])} (+{int(g)}, +{g / total.iloc[0] * 100:.2f} %)"
                  .replace(".", ",") + " · axe non partant de zéro · trait pointillé : pas de relevé",
                  loc="left", fontsize=11.5, color=INK)

    x = np.arange(len(per))
    colors = [MUTED if r.est else SERIES[0] for r in per.itertuples()]
    alphas = [1 if r.full else 0.45 for r in per.itertuples()]
    for xi, r, c, al in zip(x, per.itertuples(), colors, alphas):
        ax2.bar(xi, r.gain, width=0.78, color=c, alpha=al)
        label = f"{r.gain:.0f}"
        if not np.isnan(r.delta):
            label += f"\n{'▲' if r.delta >= 0 else '▼'} {abs(r.delta):.0f} %"
        ax2.text(xi, r.gain, label, ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.3)
    ax2.set_xticks(x, [r.b.tz_convert(tz).strftime("%Hh") for r in per.itertuples()])
    ax2.set_ylim(0, per["gain"].max() * 1.3)
    ax2.set_ylabel("voix gagnées dans l'heure")
    ax2.set_title("Voix gagnées par heure (tous lieux) · ▲▼ : rythme par rapport à l'heure précédente · "
                  "gris : estimé (trou de relevé) · pâle : heure incomplète", loc="left", fontsize=11.5, color=INK)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(OUT_TOTAL_PNG, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def build():
    if not CSV_PATH.exists():
        print("Pas encore de données.")
        return
    wide, meta = load()
    if len(wide) < 2:
        print("Il faut au moins deux relevés.")
        return
    build_html(wide, meta)
    build_png(wide, meta)
    build_total_png(wide)


if __name__ == "__main__":
    build()
