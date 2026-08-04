"""Meta-study: which method wins in each ``alpha`` x ``beta`` regime.

Run::

    python -m pdf_compare.meta_study                 # full study
    python pdf_compare/meta_study.py --quick         # coarse grid

Three sections:

**A. Accuracy over the alpha-beta plane.**  Done in ``d == 1``, the only setting
with an *exact* reference for every ``beta``: a spectral measure with atoms at
``+-1`` realises a univariate S1 law, so ``alpha_stable.pdf`` (an independent
implementation) gives ground truth.  ``d == 1`` also makes the projection
method's sphere quadrature exact, isolating the density kernels themselves.  The
sampler is antithetic so the empirical skewness is exact -- otherwise the shared
Monte-Carlo error swamps the comparison near ``alpha = 1`` (see section C).

**B. Cost in ``d == 2``.**  The realistic setting.  The FFT pays a fixed build
and then interpolates; the projection method pays per point.  The crossover in
query count is what decides the practical winner.

**C. Shared sensitivity near ``alpha = 1``.**  The S1 characteristic function
multiplies the skewness by ``tan(pi alpha / 2)``, which diverges as
``alpha -> 1``, so Monte-Carlo error in the estimated skewness is amplified by
that factor.  This hits *both* methods identically and decays only like
``1 / sqrt(N)``; it is a property of the shared spectral-measure estimate, not of
either density method.

Results are written as JSON next to the plots so they can be reported verbatim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.pdf_compare.estimators import FFTEstimator, ProjectionEstimator  # type: ignore
else:
    from .estimators import FFTEstimator, ProjectionEstimator

from aub_htp import IsotropicSampler, alpha_stable
from aub_htp.pdf.multivariate import MultivariateStableDensity
from aub_htp.random.spectral_measure_sampler import BaseSpectralMeasureSampler
from aub_htp.random.util import get_random_state_generator


# --------------------------------------------------------------------------- #
# 1-D spectral measure realising an exact (alpha, beta)
# --------------------------------------------------------------------------- #
class Atoms1D(BaseSpectralMeasureSampler):
    """Atoms at ``+-1``; ``beta = (wp - wm) / (wp + wm)``, ``sigma = 1``."""

    def __init__(self, beta, alpha, sigma=1.0, antithetic=True):
        total = sigma ** alpha
        self.wp = 0.5 * total * (1.0 + beta)
        self.wm = 0.5 * total * (1.0 - beta)
        self._mass = self.wp + self.wm
        self._p_plus = self.wp / self._mass
        self._antithetic = antithetic

    def sample(self, number_of_samples, random_state=None):
        if self._antithetic:
            n_plus = int(round(self._p_plus * number_of_samples))
            signs = np.concatenate([np.ones(n_plus),
                                    -np.ones(number_of_samples - n_plus)])
        else:
            rng = get_random_state_generator(random_state)
            signs = np.where(rng.random(number_of_samples) <= self._p_plus,
                             1.0, -1.0)
        return signs.reshape(-1, 1)

    def dimensions(self):
        return 1

    def mass(self):
        return float(self._mass)


def bulk_window(alpha, beta):
    """Mode and half-width of the reference density's bulk (2% of the peak)."""
    span = 40.0 if alpha > 0.9 else 80.0
    x = np.linspace(-span, span, 20_001)
    f = np.nan_to_num(alpha_stable.pdf(x, alpha, beta, loc=0.0, scale=1.0))
    mode = float(x[np.argmax(f)])
    kept = x[f > 0.02 * f.max()]
    return mode, max(float(np.max(np.abs(kept - mode))), 2.0)


# --------------------------------------------------------------------------- #
# Section A: accuracy over the alpha-beta plane
# --------------------------------------------------------------------------- #
def accuracy_cell(alpha, beta, n_spectral, seed):
    sampler = Atoms1D(beta, alpha)
    mode, half = bulk_window(alpha, beta)
    xs = np.linspace(-half, half, 301)
    ref = np.nan_to_num(alpha_stable.pdf(xs + mode, alpha, beta,
                                        loc=0.0, scale=1.0))
    keep = ref > 0.02 * ref.max()

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        extent = max(6.0 * half, 40.0)
        dx = min(half / 60.0, 0.05)
        grid = int(2 * round(extent / dx)) + 1
        t0 = time.perf_counter()
        fft = FFTEstimator(number_of_spectral_samples=n_spectral,
                          grid_size=grid, dt=np.pi / extent)
        fft.setup(alpha, sampler, shift=-mode, random_state=seed)
        got_f = fft.evaluate(xs.reshape(-1, 1))
        out["fft_time"] = time.perf_counter() - t0
        out["fft_grid"] = grid

        t0 = time.perf_counter()
        model = MultivariateStableDensity(
            alpha, sampler, shift=-mode,
            number_of_spectral_samples=n_spectral, random_state=seed)
        got_p = model.pdf(xs.reshape(-1, 1))
        out["proj_time"] = time.perf_counter() - t0

    out["fft_err"] = float(np.median(np.abs(got_f - ref)[keep] / ref[keep]))
    out["proj_err"] = float(np.median(np.abs(got_p - ref)[keep] / ref[keep]))
    out["mode"] = mode
    out["half"] = half
    return out


def run_accuracy(alphas, betas, *, n_spectral, seed):
    print("\n" + "=" * 86)
    print("A. ACCURACY over the alpha-beta plane  (d=1, exact reference, "
          f"N_spectral={n_spectral})")
    print("=" * 86)
    print(f"{'alpha':>6} {'beta':>6} {'mode':>9} {'half':>7} "
          f"{'proj_err':>10} {'fft_err':>10} {'winner':>10} {'ratio':>8}")
    rows = []
    for alpha in alphas:
        for beta in betas:
            c = accuracy_cell(alpha, beta, n_spectral, seed)
            better = "proj" if c["proj_err"] < c["fft_err"] else "fft"
            hi = max(c["proj_err"], c["fft_err"])
            lo = min(c["proj_err"], c["fft_err"])
            ratio = hi / max(lo, 1e-12)
            if ratio < 2.0:
                better = "tie"
            print(f"{alpha:>6} {beta:>6} {c['mode']:>9.3f} {c['half']:>7.2f} "
                  f"{c['proj_err']:>10.5f} {c['fft_err']:>10.5f} "
                  f"{better:>10} {ratio:>8.1f}")
            rows.append(dict(alpha=alpha, beta=beta, winner=better,
                             ratio=ratio, **c))
    return rows


# --------------------------------------------------------------------------- #
# Section B: cost in d = 2
# --------------------------------------------------------------------------- #
def run_cost(alphas, *, n_spectral, seed, point_counts):
    print("\n" + "=" * 86)
    print(f"B. COST in d=2 (isotropic, N_spectral={n_spectral})")
    print("=" * 86)
    rows = []
    for alpha in alphas:
        sampler = IsotropicSampler(2, alpha, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fft = FFTEstimator(number_of_spectral_samples=n_spectral,
                               grid_size=101, dt=0.3)
            fft.setup(alpha, sampler, random_state=seed)
            proj = ProjectionEstimator(number_of_spectral_samples=n_spectral)
            proj.setup(alpha, sampler, random_state=seed)
        rng = np.random.default_rng(seed)
        print(f"  alpha={alpha}: build  fft={fft.setup_time:.3f}s  "
              f"proj={proj.setup_time:.3f}s")
        per = {}
        for n in point_counts:
            pts = rng.uniform(-6, 6, size=(n, 2))
            fft.evaluate(pts)
            proj.evaluate(pts)
            per[n] = (fft.eval_time, proj.eval_time)
            print(f"      n={n:<7} fft_eval={fft.eval_time:.4f}s  "
                  f"proj_eval={proj.eval_time:.4f}s  "
                  f"proj_total/fft_total="
                  f"{(proj.setup_time+proj.eval_time)/(fft.setup_time+fft.eval_time):.2f}")
        rows.append(dict(alpha=alpha, fft_build=fft.setup_time,
                         proj_build=proj.setup_time,
                         per_n={str(k): v for k, v in per.items()}))
    return rows


# --------------------------------------------------------------------------- #
# Section C: shared sensitivity near alpha = 1
# --------------------------------------------------------------------------- #
def run_alpha1_sensitivity(*, seed, sample_counts):
    print("\n" + "=" * 86)
    print("C. SHARED sensitivity near alpha=1 (spurious skew x tan(pi a/2))")
    print("=" * 86)
    rows = []
    print(f"{'alpha':>6} {'tan':>9} {'N':>9} {'sampler':>11} {'beta_hat':>10} "
          f"{'|b*tan|':>9} {'proj_err':>10} {'fft_err':>10}")
    for alpha in (0.9, 0.95, 0.99, 1.0):
        tan = float(np.tan(np.pi * alpha / 2.0)) if alpha != 1.0 else float("inf")
        xs = np.linspace(-4.0, 4.0, 121)
        ref = np.nan_to_num(alpha_stable.pdf(xs, alpha, 0.0, loc=0.0, scale=1.0))
        keep = ref > 0.02 * ref.max()
        for antithetic in (True, False):
            for n in sample_counts:
                sampler = Atoms1D(0.0, alpha, antithetic=antithetic)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = MultivariateStableDensity(
                        alpha, sampler, number_of_spectral_samples=n,
                        random_state=seed)
                    gp = model.pdf(xs.reshape(-1, 1))
                    fft = FFTEstimator(number_of_spectral_samples=n,
                                       grid_size=6001, dt=np.pi / 60.0)
                    fft.setup(alpha, sampler, random_state=seed)
                    gf = fft.evaluate(xs.reshape(-1, 1))
                bhat = float(np.max(np.abs(model.beta)))
                amp = bhat * (abs(tan) if np.isfinite(tan) else 0.0)
                ep = float(np.median(np.abs(gp - ref)[keep] / ref[keep]))
                ef = float(np.median(np.abs(gf - ref)[keep] / ref[keep]))
                label = "antithetic" if antithetic else "iid"
                print(f"{alpha:>6} {tan:>9.2f} {n:>9} {label:>11} {bhat:>10.5f} "
                      f"{amp:>9.4f} {ep:>10.4f} {ef:>10.4f}")
                rows.append(dict(alpha=alpha, tan=tan if np.isfinite(tan) else None,
                                 n=n, antithetic=antithetic, beta_hat=bhat,
                                 amplified=amp, proj_err=ep, fft_err=ef))
    return rows


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_accuracy(rows, alphas, betas, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

    A, B = len(alphas), len(betas)
    proj = np.full((A, B), np.nan)
    fft = np.full((A, B), np.nan)
    win = np.zeros((A, B))
    idx = {(r["alpha"], r["beta"]): r for r in rows}
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            r = idx[(a, b)]
            proj[i, j] = r["proj_err"]
            fft[i, j] = r["fft_err"]
            win[i, j] = {"proj": -1.0, "tie": 0.0, "fft": 1.0}[r["winner"]]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    vmin = max(min(np.nanmin(proj), np.nanmin(fft)), 1e-6)
    vmax = max(np.nanmax(proj), np.nanmax(fft))
    for ax, data, title in ((axes[0], proj, "projection"), (axes[1], fft, "inverse FFT")):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xticks(range(B)); ax.set_xticklabels(betas)
        ax.set_yticks(range(A)); ax.set_yticklabels(alphas)
        ax.set_xlabel("beta"); ax.set_ylabel("alpha")
        ax.set_title(f"median relative error -- {title}")
        for i in range(A):
            for j in range(B):
                ax.text(j, i, f"{data[i, j]:.0e}", ha="center", va="center",
                        fontsize=7, color="w")
        fig.colorbar(im, ax=ax)

    cmap = ListedColormap(["#2c7fb8", "#d9d9d9", "#d95f0e"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im = axes[2].imshow(win, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    axes[2].set_xticks(range(B)); axes[2].set_xticklabels(betas)
    axes[2].set_yticks(range(A)); axes[2].set_yticklabels(alphas)
    axes[2].set_xlabel("beta"); axes[2].set_ylabel("alpha")
    axes[2].set_title("more accurate method (>2x)")
    cb = fig.colorbar(im, ax=axes[2], ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["projection", "tie", "FFT"])

    fig.suptitle("Accuracy over the alpha-beta plane (d=1, exact reference)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = Path(outdir) / "meta_accuracy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("  -> saved", path)


def plot_cost(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in rows:
        ns = sorted(int(k) for k in r["per_n"])
        fft_total = [r["fft_build"] + r["per_n"][str(n)][0] for n in ns]
        proj_total = [r["proj_build"] + r["per_n"][str(n)][1] for n in ns]
        ax.loglog(ns, fft_total, "o-", label=f"FFT a={r['alpha']}")
        ax.loglog(ns, proj_total, "s--", label=f"proj a={r['alpha']}")
    ax.set_xlabel("query points")
    ax.set_ylabel("total wall-clock (build + eval), s")
    ax.set_title("Cost in d=2: fixed FFT build vs per-point projection")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize="small", ncol=2)
    fig.tight_layout()
    path = Path(outdir) / "meta_cost.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("  -> saved", path)


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",
                   default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        alphas = [0.5, 1.0, 1.5]
        betas = [0.0, 1.0]
        n_spectral = 2_000
        cost_alphas = [1.5]
        point_counts = [100, 1_000, 10_000]
        sample_counts = [2_000]
    else:
        alphas = [0.4, 0.6, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5, 1.8, 1.95]
        betas = [0.0, 0.25, 0.5, 0.75, 1.0]
        n_spectral = 8_000
        cost_alphas = [0.6, 1.0, 1.5]
        point_counts = [100, 1_000, 10_000, 100_000]
        sample_counts = [2_000, 50_000]

    acc = run_accuracy(alphas, betas, n_spectral=n_spectral, seed=args.seed)
    cost = run_cost(cost_alphas, n_spectral=n_spectral, seed=args.seed,
                    point_counts=point_counts)
    sens = run_alpha1_sensitivity(seed=args.seed, sample_counts=sample_counts)

    plot_accuracy(acc, alphas, betas, outdir)
    plot_cost(cost, outdir)

    payload = dict(alphas=alphas, betas=betas, n_spectral=n_spectral,
                   accuracy=acc, cost=cost, alpha1_sensitivity=sens)
    jpath = outdir / "meta_study.json"
    jpath.write_text(json.dumps(payload, indent=2))
    print("\nJSON ->", jpath)

    # ---- headline summary ---- #
    wins = {"proj": 0, "fft": 0, "tie": 0}
    for r in acc:
        wins[r["winner"]] += 1
    print("\nAccuracy winners:", wins)
    worst_fft = max(acc, key=lambda r: r["fft_err"])
    worst_proj = max(acc, key=lambda r: r["proj_err"])
    print(f"worst FFT  cell: alpha={worst_fft['alpha']} beta={worst_fft['beta']} "
          f"err={worst_fft['fft_err']:.4f}")
    print(f"worst proj cell: alpha={worst_proj['alpha']} beta={worst_proj['beta']} "
          f"err={worst_proj['proj_err']:.4f}")


if __name__ == "__main__":
    main()
