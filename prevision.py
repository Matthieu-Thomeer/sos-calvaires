#!/usr/bin/env python3
"""Prévision des votes : fond + boost qui s'amortit, modulé par un cycle horaire.

Deux modèles sont ajustés et comparés par test rétrospectif :
  - « jour/nuit » : le cycle est une seule vague de 24 h (ci-dessous) ;
  - « jour/nuit/boulot » : le cycle est un profil horaire libre mais lissé (pointes du matin et du soir possibles),
    voir la section « modèle jour/nuit/boulot » plus bas.

Modèle du débit de voix d'un lieu (voix/heure) :

    dN/dt = (r0 + A·exp(-t/τ)) · D(t),    D(t) = 1 + a·cos(2π(h(t) − φ)/24)

  - r0 : rythme de fond, qui reste quand l'effet des appels au vote est retombé ;
  - A, τ : boost lié à un appel au vote, qui s'amortit (demi-vie = τ·ln 2) ;
  - D : cycle jour/nuit (h = heure de Paris, φ = heure de pointe, a = amplitude), commun à tous les lieux
    et estimé sur le total des voix, car un seul lieu n'a pas assez de voix pour le mesurer.

Le fit porte sur les voix gagnées entre deux relevés (vraisemblance de Poisson, avec la surdispersion mesurée),
pas sur le cumul : le cumul lisse tout et donne une fausse impression de précision. Les trous de relevé
(la nuit par exemple) sont pris en compte naturellement : on compare les voix gagnées pendant le trou au débit
intégré sur sa durée.

Incertitude : approximation de Laplace (paramètres tirés autour du meilleur fit) + bruit de comptage.
Le modèle est confronté à la projection linéaire par un test rétrospectif : fit sans les dernières heures,
puis comparaison avec ce qui s'est réellement passé.

Usage :
  python prevision.py            # affiche le bilan dans le terminal
Appelé aussi par dashboard.py, qui intègre la prévision au tableau de bord.
"""
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Fin du vote : lue sur le site (table publique « settings », champ closes_at), ou forcée par la variable
# d'environnement SOS_FIN_VOTE (ISO 8601, heure de Paris si pas de fuseau). Sinon, projection sur HORIZON_H heures.
FIN_VOTE_DEFAUT = "2026-11-30T22:59:59+00:00"   # relevé sur le site le 2026-11-24
HORIZON_H = 72
STEP_H = 1 / 12                 # pas d'intégration du débit (5 min)
TAU_BOUNDS = (0.5, 400.0)       # heures
N_DRAWS = 400
RNG = np.random.default_rng(12345)
TZ = "Europe/Paris"


# ---------------------------------------------------------------- outils
def _local_hours(t0_s, g_h):
    """Heure de Paris (décimale) aux instants t0 + g heures."""
    idx = pd.to_datetime(t0_s + g_h * 3600, unit="s", utc=True).tz_convert(TZ)
    return np.asarray(idx.hour + idx.minute / 60 + idx.second / 3600, dtype=float)


class Grid:
    """Grille fine de temps (heures depuis le 1er relevé) sur laquelle on intègre le débit."""

    def __init__(self, t0_s, t_obs_h, t_end_h):
        self.t0 = t0_s
        self.g = np.arange(0, t_end_h + STEP_H, STEP_H)
        self.hloc = _local_hours(t0_s, self.g)
        self.t_obs = t_obs_h

    def diurnal(self, a, phi):
        return 1 + a * np.cos(2 * np.pi * (self.hloc - phi) / 24)

    def cum(self, r0, A, tau, D):
        rate = (r0 + A * np.exp(-self.g / tau)) * D
        c = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2 * STEP_H)])
        return c

    def at(self, c, t_h):
        return np.interp(t_h, self.g, c)


def _nll(mu, y):
    mu = np.maximum(mu, 1e-9)
    return float(np.sum(mu - y * np.log(mu)))


def _hessian(f, x, h=1e-3):
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.eye(n)[i] * h
            ej = np.eye(n)[j] * h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h**2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h**2)
    return H


def _cov(H, disp, active):
    """Covariance de Laplace sur les paramètres libres ; ceux qui butent sur une borne restent fixés."""
    n = len(H)
    cov = np.zeros((n, n))
    idx = np.flatnonzero(active)
    if len(idx):
        h = H[np.ix_(idx, idx)]
        h = (h + h.T) / 2
        w, V = np.linalg.eigh(h)
        w = np.maximum(w, 0.5)       # direction presque plate : écart-type plafonné (≈ ×4 en log)
        cov[np.ix_(idx, idx)] = disp * (V / w) @ V.T
    return cov


# ---------------------------------------------------------------- fit d'une série
class Fit:
    def __init__(self, grid, v, D, a_phi=None):
        self.grid, self.v, self.D = grid, np.asarray(v, float), D
        self.y = np.diff(self.v)
        self.a_phi = a_phi
        self.fit_diurnal = D is None
        hours = grid.t_obs[-1] - grid.t_obs[0]
        self.r_mean = max((self.v[-1] - self.v[0]) / max(hours, 1e-6), 0.05)

    def unpack(self, x):
        r0, A, tau = math.exp(x[0]), math.exp(x[1]), math.exp(x[2])
        if self.fit_diurnal:
            a = 0.95 / (1 + math.exp(-x[3]))
            phi = x[4] % 24
            D = self.grid.diurnal(a, phi)
        else:
            a, phi = self.a_phi
            D = self.D
        return r0, A, tau, a, phi, D

    def mu(self, x):
        r0, A, tau, a, phi, D = self.unpack(x)
        c = self.grid.cum(r0, A, tau, D)
        return np.diff(self.grid.at(c, self.grid.t_obs))

    def nll(self, x):
        return _nll(self.mu(x), self.y)

    def run(self):
        lr = math.log(self.r_mean)
        lo_tau, hi_tau = math.log(TAU_BOUNDS[0]), math.log(TAU_BOUNDS[1])
        bounds = [(lr - 9, lr + 4), (lr - 9, lr + 5), (lo_tau, hi_tau)]
        starts = []
        for tau0 in (3, 12, 48):
            for split in ((0.3, 0.7) if self.fit_diurnal else (0.5,)):
                s = [lr + math.log(split), lr + math.log(1 - split) + 0.3, math.log(tau0)]
                if self.fit_diurnal:
                    for phi0 in (2, 8, 14, 20):
                        starts.append(s + [0.5, phi0])
                else:
                    starts.append(s)
        if self.fit_diurnal:
            bounds += [(-6, 6), (-48, 72)]
        self.bounds = np.array(bounds)
        best = None
        for s in starts:
            r = minimize(self.nll, np.array(s), method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.x = best.x
        mu = self.mu(self.x)
        dof = max(len(self.y) - len(self.x), 1)
        self.disp = max(1.0, float(np.sum((self.y - mu) ** 2 / np.maximum(mu, 0.5))) / dof)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        active = (self.x - lo > 1e-2) & (hi - self.x > 1e-2)
        if self.fit_diurnal:
            active[4] = True                      # φ est périodique : pas de vraie borne
        self.cov = _cov(_hessian(self.nll, self.x), self.disp, active)
        return self

    def params(self):
        r0, A, tau, a, phi, _ = self.unpack(self.x)
        return dict(r0=r0, A=A, tau=tau, a=a, phi=phi, disp=self.disp)

    def fitted(self):
        """Cumul ajusté aux instants observés, calé sur le premier relevé."""
        r0, A, tau, a, phi, D = self.unpack(self.x)
        c = self.grid.cum(r0, A, tau, D)
        return self.v[0] + self.grid.at(c, self.grid.t_obs)

    def rate_at(self, t_h, x=None):
        r0, A, tau, a, phi, D = self.unpack(self.x if x is None else x)
        return float(np.interp(t_h, self.grid.g, (r0 + A * np.exp(-self.grid.g / tau)) * D))

    def project(self, t_future_h, draws=N_DRAWS, future=None):
        """Cumul projeté aux instants futurs : projection centrale + quantiles 10/50/90 (paramètres + comptage).

        future : grille couvrant l'horizon (sinon on en construit une) ; le cycle jour/nuit y est recalculé.
        """
        t_last = self.grid.t_obs[-1]
        fg = future or Grid(self.grid.t0, self.grid.t_obs, float(np.max(t_future_h)) + 0.1)

        def curve(x):
            r0, A, tau, a, phi, _ = self.unpack(x)
            c = fg.cum(r0, A, tau, fg.diurnal(a, phi))
            return fg.at(c, t_future_h) - fg.at(c, t_last)

        central = self.v[-1] + curve(self.x)
        out = np.empty((draws, len(t_future_h)))
        if draws > 1:
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            xs = RNG.multivariate_normal(self.x, self.cov, size=draws, check_valid="ignore")
            for d, xd in enumerate(xs):
                gain = curve(np.clip(xd, lo, hi))
                noise = RNG.standard_normal(len(gain)) * np.sqrt(self.disp * np.maximum(gain, 0))
                out[d] = self.v[-1] + np.maximum(gain + noise, 0)
            out = np.maximum.accumulate(out, axis=1)       # un cumul ne redescend pas
            q10, q50, q90 = np.percentile(out, [10, 50, 90], axis=0)
            self.end_draws = out[:, -1]                    # tirages au dernier instant (duels : chances d'être devant)
        else:
            q10 = q50 = q90 = central
        return central, q10, q50, q90


# ---------------------------------------------------------------- ensemble
def _times(wide):
    ts = np.array([x.timestamp() for x in wide.index])
    return ts[0], (ts - ts[0]) / 3600


def fit_all(wide):
    """Fit du total (donne le cycle jour/nuit) puis de chaque lieu avec ce cycle fixé."""
    t0, th = _times(wide)
    grid = Grid(t0, th, th[-1])
    tot = Fit(grid, wide.sum(axis=1).values, None).run()
    p = tot.params()
    D = grid.diurnal(p["a"], p["phi"])
    places = {pid: Fit(grid, wide[pid].values, D, (p["a"], p["phi"])).run() for pid in wide.columns}
    return grid, tot, places


def _linear(v, th, basis_h=None):
    """Rythme linéaire (voix/h) sur tout le suivi, ou sur les basis_h dernières heures."""
    j = 0 if basis_h is None else int(np.searchsorted(th, th[-1] - basis_h))
    j = min(j, len(th) - 2)
    return (v[-1] - v[j]) / max(th[-1] - th[j], 1e-6)


def backtest(wide, holdout_h=None, fitter=None):
    """Fit sans les dernières heures, puis erreur sur les voix réellement gagnées ensuite (tous lieux).

    fitter(train) -> (grid, total, lieux) : fit_all (jour/nuit) par défaut, fit_all_b pour jour/nuit/boulot
    (qui choisit alors son lissage sur les seules données d'entraînement : le test reste honnête)."""
    t0, th = _times(wide)
    span = th[-1]
    if holdout_h is None:
        holdout_h = max(3.0, 0.25 * span)
    cut = int(np.searchsorted(th, th[-1] - holdout_h, side="right")) - 1
    if cut < 6 or span < 6:
        return None
    train = wide.iloc[: cut + 1]
    t_test = th[-1]
    _, tot, places = (fitter or fit_all)(train)
    _, th_tr = _times(train)
    fg = Grid(t0, th_tr, t_test + 0.1)
    err = {"modele": [], "lineaire": [], "lineaire_3h": []}
    rows = []
    for pid in wide.columns:
        v_tr, actual = train[pid].values, float(wide[pid].iloc[-1])
        dt = t_test - th_tr[-1]
        central = places[pid].project(np.array([t_test]), draws=1, future=fg)[0][0]
        lin = v_tr[-1] + _linear(v_tr, th_tr) * dt
        lin3 = v_tr[-1] + _linear(v_tr, th_tr, 3) * dt
        gain = actual - v_tr[-1]
        err["modele"].append(central - actual)
        err["lineaire"].append(lin - actual)
        err["lineaire_3h"].append(lin3 - actual)
        rows.append((pid, gain, central - v_tr[-1], lin - v_tr[-1], lin3 - v_tr[-1]))
    mae = {k: float(np.mean(np.abs(e))) for k, e in err.items()}
    bias = {k: float(np.mean(e)) for k, e in err.items()}
    return dict(hours=float(t_test - th_tr[-1]), cut=int(wide.index[cut].timestamp() * 1000), mae=mae, bias=bias,
                gain=float(sum(r[1] for r in rows)), rows=rows)


# ================================================================ modèle jour/nuit/boulot
# Même structure (fond + boost amorti), mais le cycle horaire n'est plus une seule vague : c'est un profil libre
#     D(h) = exp(Σ_k b_k·cos(2πkh/24) + c_k·sin(2πkh/24)),  k = 1..4  (périodes 24, 12, 8 et 6 h),
# qui peut avoir une pointe le matin, un creux à midi, une pointe le soir. Une pénalité de courbure (λ·Σ k⁴·(b²+c²))
# l'empêche d'onduler sans raison : une heure jamais observée est interpolée en douceur entre ses voisines.
# Chaque lieu peut s'écarter du profil commun (k = 1..2), mais l'écart est rappelé vers zéro (κ) : un lieu avec
# peu de voix garde le profil commun, un lieu avec beaucoup de voix peut avoir le sien.
# λ et κ sont choisis par test rétrospectif : on garde le réglage qui prévoit le mieux les dernières heures.
K_TOT, K_LIEU = 4, 2
LAMBDAS = (0.3, 3.0, 30.0, 300.0)
KAPPAS = (None, 30.0, 3.0)       # None : pas d'écart propre au lieu
H24 = np.arange(0, 24, STEP_H)   # heures d'une journée, pour normaliser le profil (moyenne = 1)


def _basis(h, K):
    k = np.arange(1, K + 1)
    a = 2 * np.pi * np.outer(h, k) / 24
    return np.column_stack([np.cos(a), np.sin(a)]).reshape(len(h), 2, K).transpose(0, 2, 1).reshape(len(h), 2 * K)


def _pen_w(K):
    return np.repeat(np.arange(1, K + 1, dtype=float) ** 4, 2)


def _integ_matrix(grid):
    """mu = M @ débit : voix attendues entre deux relevés (intégrale trapèze puis interpolation aux relevés)."""
    n = len(grid.g)
    C = np.zeros((n, n))
    for i in range(1, n):
        C[i] = C[i - 1]
        C[i, i - 1] += STEP_H / 2
        C[i, i] += STEP_H / 2
    L = np.zeros((len(grid.t_obs), n))
    for r, t in enumerate(grid.t_obs):
        j = min(int(np.searchsorted(grid.g, t, side="right")) - 1, n - 2)
        f = (t - grid.g[j]) / STEP_H
        L[r, j], L[r, j + 1] = 1 - f, f
    return np.diff(L @ C, axis=0)


class Cycle:
    """Profil horaire normalisé (moyenne 1 sur 24 h) : exp(B·θ) éventuellement multiplié par un profil commun."""

    def __init__(self, K, theta, base=None):
        self.K, self.theta, self.base = K, np.asarray(theta, float), base
        self.norm = float(np.mean(self.raw(H24)))

    def raw(self, h):
        d = np.exp(_basis(h, self.K) @ self.theta) if self.K else np.ones(len(h))
        return d * (self.base(h) if self.base else 1)

    def __call__(self, h):
        return self.raw(h) / self.norm

    def peaks(self):
        """Heures des pointes (maxima locaux nets) du profil."""
        h = np.arange(0, 24, 0.25)
        d = self(h)
        out = [h[i] for i in range(len(h)) if d[i] >= d[i - 1] and d[i] > d[(i + 1) % len(h)] and d[i] > 1.05]
        return [float(x) for x in out]


class FitB:
    """Fit fond + boost × cycle libre, avec gradient analytique (bien plus rapide que les différences finies)."""

    def __init__(self, grid, M, v, K, pen, base=None):
        self.grid, self.M, self.v = grid, M, np.asarray(v, float)
        self.y = np.diff(self.v)
        self.K, self.base = K, base
        self.B = _basis(grid.hloc, K) if K else np.zeros((len(grid.g), 0))
        self.E0 = base(grid.hloc) if base else np.ones(len(grid.g))
        self.pen = np.asarray(pen, float)
        hours = grid.t_obs[-1] - grid.t_obs[0]
        self.r_mean = max((self.v[-1] - self.v[0]) / max(hours, 1e-6), 0.05)

    def _parts(self, x, g=None, B=None, E0=None):
        g = self.grid.g if g is None else g
        B = self.B if B is None else B
        E0 = self.E0 if E0 is None else E0
        r0, A, tau = math.exp(x[0]), math.exp(x[1]), math.exp(x[2])
        E = E0 * np.exp(B @ x[3:]) if self.K else E0
        boost = A * np.exp(-g / tau)
        return r0, boost, tau, E, (r0 + boost) * E

    def obj(self, x):
        r0, boost, tau, E, rate = self._parts(x)
        mu = np.maximum(self.M @ rate, 1e-9)
        th = x[3:]
        f = float(np.sum(mu - self.y * np.log(mu))) + float(np.sum(self.pen * th * th))
        w = (1 - self.y / mu) @ self.M                       # d nll / d débit
        g = np.empty_like(x)
        g[0] = w @ (r0 * E)
        g[1] = w @ (boost * E)
        g[2] = w @ (boost * self.grid.g / tau * E)
        if self.K:
            g[3:] = (w * rate) @ self.B + 2 * self.pen * th
        return f, g

    def run(self):
        lr = math.log(self.r_mean)
        lo_tau, hi_tau = math.log(TAU_BOUNDS[0]), math.log(TAU_BOUNDS[1])
        m = 2 * self.K
        self.bounds = np.array([(lr - 9, lr + 4), (lr - 9, lr + 5), (lo_tau, hi_tau)] + [(-5, 5)] * m)
        best = None
        for tau0 in (3, 12, 48):
            s = np.array([lr + math.log(0.5), lr + math.log(0.5) + 0.3, math.log(tau0)] + [0.0] * m)
            r = minimize(self.obj, s, jac=True, method="L-BFGS-B", bounds=self.bounds)
            if best is None or r.fun < best.fun:
                best = r
        self.x = best.x
        _, _, _, _, rate = self._parts(self.x)
        mu = self.M @ rate
        dof = max(len(self.y) - 3, 1)
        self.disp = max(1.0, float(np.sum((self.y - mu) ** 2 / np.maximum(mu, 0.5))) / dof)
        # Hessien de l'objectif pénalisé (la pénalité joue le rôle d'un a priori), par différences du gradient.
        n, h = len(self.x), 1e-4
        H = np.zeros((n, n))
        for i in range(n):
            e = np.zeros(n); e[i] = h
            H[:, i] = (self.obj(self.x + e)[1] - self.obj(self.x - e)[1]) / (2 * h)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        active = (self.x - lo > 1e-2) & (hi - self.x > 1e-2)
        self.cov = _cov((H + H.T) / 2, self.disp, active)
        return self

    def cycle(self, x=None):
        x = self.x if x is None else x
        return Cycle(self.K, x[3:], self.base)

    def params(self):
        r0, A, tau = (math.exp(v) for v in self.x[:3])
        n = self.cycle().norm                      # débit rapporté à une journée moyenne (profil de moyenne 1)
        return dict(r0=r0 * n, A=A * n, tau=tau, disp=self.disp)

    def fitted(self):
        _, _, _, _, rate = self._parts(self.x)
        return self.v[0] + np.concatenate([[0], np.cumsum(self.M @ rate)])

    def rate_at(self, t_h):
        _, _, _, _, rate = self._parts(self.x)
        return float(np.interp(t_h, self.grid.g, rate))

    def project(self, t_future_h, draws=N_DRAWS, future=None):
        t_last = self.grid.t_obs[-1]
        fg = future or Grid(self.grid.t0, self.grid.t_obs, float(np.max(t_future_h)) + 0.1)
        Bf = _basis(fg.hloc, self.K) if self.K else np.zeros((len(fg.g), 0))
        E0f = self.base(fg.hloc) if self.base else np.ones(len(fg.g))

        def curve(x):
            *_, rate = self._parts(x, fg.g, Bf, E0f)
            c = np.concatenate([[0], np.cumsum((rate[1:] + rate[:-1]) / 2 * STEP_H)])
            return fg.at(c, t_future_h) - fg.at(c, t_last)

        central = self.v[-1] + curve(self.x)
        if draws <= 1:
            return central, central, central, central
        out = np.empty((draws, len(t_future_h)))
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        xs = RNG.multivariate_normal(self.x, self.cov, size=draws, check_valid="ignore")
        for d, xd in enumerate(xs):
            gain = curve(np.clip(xd, lo, hi))
            noise = RNG.standard_normal(len(gain)) * np.sqrt(self.disp * np.maximum(gain, 0))
            out[d] = self.v[-1] + np.maximum(gain + noise, 0)
        out = np.maximum.accumulate(out, axis=1)
        q10, q50, q90 = np.percentile(out, [10, 50, 90], axis=0)
        self.end_draws = out[:, -1]
        return central, q10, q50, q90

    def empirical(self):
        """Cycle observé, heure par heure : voix gagnées ÷ voix attendues sans cycle (relevés espacés de 1 h au plus)."""
        r0, boost, *_ = self._parts(self.x)
        base = self.M @ ((r0 + boost) * self.cycle().norm)
        num, den = np.zeros(24), np.zeros(24)
        t = self.grid.t_obs
        for i in range(len(self.y)):
            if t[i + 1] - t[i] > 1.0 + 1e-6:
                continue
            h = int(_local_hours(self.grid.t0, np.array([(t[i] + t[i + 1]) / 2]))[0]) % 24
            num[h] += self.y[i]
            den[h] += base[i]
        return num, den


def fit_all_b(wide, lam=None, kap="auto"):
    """Profil commun sur le total, puis chaque lieu (fond, boost, écart au profil rappelé vers zéro)."""
    if lam is None:
        lam, kap = select_b(wide)
    t0, th = _times(wide)
    grid = Grid(t0, th, th[-1])
    M = _integ_matrix(grid)
    tot = FitB(grid, M, wide.sum(axis=1).values, K_TOT, lam * _pen_w(K_TOT)).run()
    common = tot.cycle()
    K = K_LIEU if kap is not None else 0
    pen = (kap * _pen_w(K_LIEU)) if kap is not None else []
    places = {pid: FitB(grid, M, wide[pid].values, K, pen, base=common).run() for pid in wide.columns}
    tot.lam, tot.kap = lam, kap
    return grid, tot, places


def _holdout(wide, holdout_h=None):
    t0, th = _times(wide)
    if holdout_h is None:
        holdout_h = max(3.0, 0.25 * th[-1])
    cut = int(np.searchsorted(th, th[-1] - holdout_h, side="right")) - 1
    return cut if cut >= 6 and th[-1] >= 6 else None


def _place_errors(wide, cut, places, grid_end):
    t0, th = _times(wide)
    train = wide.iloc[: cut + 1]
    _, th_tr = _times(train)
    fg = Grid(t0, th_tr, grid_end + 0.1)
    return {pid: places[pid].project(np.array([th[-1]]), draws=1, future=fg)[0][0] - float(wide[pid].iloc[-1])
            for pid in wide.columns}


def select_b(wide):
    """Choix de λ puis κ : le réglage qui prévoit le mieux les dernières heures (sans les avoir vues)."""
    cut = _holdout(wide)
    if cut is None:
        return 30.0, 30.0
    train = wide.iloc[: cut + 1]
    _, th = _times(wide)
    score = lambda pl: float(np.mean(np.abs(list(_place_errors(wide, cut, pl, th[-1]).values()))))
    res = {}
    for lam in LAMBDAS:
        _, _, pl = fit_all_b(train, lam, None)
        res[lam] = score(pl)
    lam = min(res, key=res.get)
    best, kap = res[lam], None
    for k in KAPPAS[1:]:
        _, _, pl = fit_all_b(train, lam, k)
        s = score(pl)
        if s < best - 1e-9:
            best, kap = s, k
    return lam, kap


def fin_vote():
    """Date de clôture du vote : variable d'environnement, sinon site, sinon valeur relevée le 2026-11-24."""
    if os.environ.get("SOS_FIN_VOTE"):
        return os.environ["SOS_FIN_VOTE"]
    try:
        import requests
        from suivi_votes import load_config

        base, key = load_config()
        r = requests.get(f"{base}/rest/v1/settings", params={"select": "closes_at", "id": "eq.1"},
                         headers={"apikey": key, "Authorization": f"Bearer {key}"}, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if rows and rows[0].get("closes_at"):
            return rows[0]["closes_at"]
    except Exception as e:  # noqa: BLE001
        print(f"[prévision] date de clôture illisible sur le site ({e!r}), valeur par défaut")
    return FIN_VOTE_DEFAUT


def horizon_h(wide):
    t0, th = _times(wide)
    fin = fin_vote()
    if fin:
        end = pd.Timestamp(fin)
        end = end.tz_localize(TZ) if end.tzinfo is None else end
        h = (end.timestamp() - t0) / 3600
        if h > th[-1]:
            return h, True
    return th[-1] + HORIZON_H, False


MODELES = {"jn": "Jour/nuit", "jnb": "Jour/nuit/boulot"}


def forecast(wide):
    """Tout ce que le tableau de bord affiche, pour les deux modèles : fits, projections, test rétrospectif."""
    t0, th = _times(wide)
    if len(th) < 8 or th[-1] < 3:
        return {"ok": False, "why": "Il faut au moins 3 h de relevés pour ajuster le modèle."}
    t_end, is_end = horizon_h(wide)
    fg = Grid(t0, th, t_end + 0.1)
    step = max(0.5, (t_end - th[-1]) / 150)
    fut = np.unique(np.concatenate([np.arange(th[-1], t_end, step), [t_end]]))
    to_ms = lambda h: [int((t0 + x * 3600) * 1000) for x in h]
    hh = np.arange(0, 24.01, 0.5)

    def pack(f, v, with_draws=False):
        central, q10, q50, q90 = f.project(fut, future=fg)
        lin_r = _linear(v, th)
        p = f.params()
        out = {
            "fit": [round(float(x), 1) for x in f.fitted()],
            "c": [round(float(x), 1) for x in central],
            "q10": [round(float(x), 1) for x in q10],
            "q50": [round(float(x), 1) for x in q50],
            "q90": [round(float(x), 1) for x in q90],
            "lin": [round(float(v[-1] + lin_r * (x - th[-1])), 1) for x in fut],
            "linRate": round(float(lin_r), 3),
            "r0": round(p["r0"], 3), "A": round(p["A"], 3), "tau": round(p["tau"], 2),
            "rateNow": round(f.rate_at(th[-1]), 3),
            "baseNow": round(p["r0"] + p["A"] * math.exp(-th[-1] / p["tau"]), 3),
            "disp": round(p["disp"], 2),
        }
        if with_draws:
            out["draws"] = [int(round(x)) for x in f.end_draws]
        return out

    def model(key, fitter):
        _, tot, places = fitter(wide)
        m = {"name": MODELES[key],
             "total": pack(tot, wide.sum(axis=1).values),
             "places": {pid: pack(f, wide[pid].values, with_draws=True) for pid, f in places.items()}}
        if key == "jn":
            a, phi = tot.params()["a"], tot.params()["phi"] % 24
            m["diurnal"] = {"a": round(a, 3), "peak": round(phi, 2)}
            m["cycle"] = [round(float(1 + a * math.cos(2 * math.pi * (h - phi) / 24)), 3) for h in hh]
        else:
            cyc = tot.cycle()
            m["cycle"] = [round(float(x), 3) for x in cyc(hh)]
            m["peaks"] = [round(x, 2) for x in cyc.peaks()]
            m["lam"], m["kap"] = tot.lam, tot.kap
            num, den = tot.empirical()
            m["emp"] = [[h, round(float(num[h] / den[h]), 3), round(float(den[h]), 1)] for h in range(24) if den[h] > 0.5]
        bt = backtest(wide, fitter=fitter)
        if bt:
            m["bt"] = {k: bt[k] for k in ("hours", "cut", "mae", "bias", "gain")}
        return m

    models = {"jn": model("jn", fit_all)}
    try:
        models["jnb"] = model("jnb", fit_all_b)
    except Exception as e:  # noqa: BLE001  — le modèle jour/nuit reste disponible
        print(f"[prévision] modèle jour/nuit/boulot indisponible : {e!r}")
    mae = {k: m["bt"]["mae"]["modele"] for k, m in models.items() if "bt" in m}
    return {
        "ok": True,
        "grid": to_ms(fut),
        "end": int((t0 + t_end * 3600) * 1000),
        "isEnd": is_end,
        "hours": round(float(th[-1]), 2),
        "models": models,
        "best": min(mae, key=mae.get) if mae else "jn",
    }


# ---------------------------------------------------------------- terminal
def main():
    from dashboard import load

    wide, meta = load()
    fc = forecast(wide)
    if not fc["ok"]:
        print(fc["why"])
        return
    end = datetime.fromtimestamp(fc["end"] / 1000, timezone.utc).astimezone()
    names = meta["name"].fillna(meta["slug"])
    hm = lambda x: f"{int(x)}h{int(round(x % 1 * 60)):02d}"
    for key, m in fc["models"].items():
        print(f"\n======== Modèle {m['name']}{'  (recommandé par le test rétrospectif)' if key == fc['best'] else ''}")
        if key == "jn":
            d = m["diurnal"]
            print(f"Cycle : amplitude ±{d['a'] * 100:.0f} %, pointe vers {hm(d['peak'])} (Paris)")
        else:
            print(f"Cycle : pointes vers {', '.join(hm(x) for x in m['peaks']) or '—'} (Paris) · lissage λ = {m['lam']:g} · "
                  f"écart propre aux lieux : {'aucun' if m['kap'] is None else 'κ = ' + format(m['kap'], 'g')}")
            print("Profil horaire (×) : " + " ".join(f"{h}h:{m['cycle'][2 * h]:.2f}" for h in range(0, 24, 2)))
        t = m["total"]
        print(f"Total : {int(t['fit'][-1])} → {t['q50'][-1]:.0f} [{t['q10'][-1]:.0f} – {t['q90'][-1]:.0f}] "
              f"le {end:%d/%m %Hh%M} (linéaire : {t['lin'][-1]:.0f}) · demi-vie du boost {t['tau'] * math.log(2):.1f} h")
        if "bt" in m:
            b = m["bt"]
            print(f"Test rétrospectif : fit sans les {b['hours']:.1f} dernières heures ({b['gain']:.0f} voix gagnées ensuite, tous lieux)")
            for k, lbl in (("modele", "modèle"), ("lineaire", "linéaire (tout le suivi)"), ("lineaire_3h", "linéaire (3 dernières h)")):
                print(f"  {lbl:26s} erreur moyenne par lieu {b['mae'][k]:6.1f} voix · biais {b['bias'][k]:+6.1f}")
        print(f"{'Lieu':42s} {'voix':>6s} {'prévu':>7s} {'[p10 – p90]':>15s} {'linéaire':>8s} {'demi-vie':>9s}")
        for pid, p in sorted(m["places"].items(), key=lambda kv: -kv[1]["q50"][-1]):
            print(f"{str(names[pid])[:42]:42s} {wide[pid].iloc[-1]:6.0f} {p['q50'][-1]:7.0f} "
                  f"{'[' + format(p['q10'][-1], '.0f') + ' – ' + format(p['q90'][-1], '.0f') + ']':>15s} {p['lin'][-1]:8.0f} "
                  f"{p['tau'] * math.log(2):8.1f}h")


if __name__ == "__main__":
    main()
