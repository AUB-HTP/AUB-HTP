"""Unified benchmark harness comparing the projection and inverse-FFT methods.

Run directly (from the repo root)::

    python -m pdf_compare.benchmark              # default (d=2 focus)
    python pdf_compare/benchmark.py --full       # heavier sweeps
    python pdf_compare/benchmark.py --sections correctness performance

It prints console tables for **correctness** (each method vs independent
references and vs each other), **convergence** (error vs Monte-Carlo /
grid / sphere resolution), and **performance** (wall-clock), and writes PNG
plots to ``--outdir`` (default ``testground/pdf_compare/results``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from benchmarks.pdf_compare import FFTEstimator, ProjectionEstimator, cases as cases_mod, metrics

from . import cases as cases_mod, metrics

# Allow both "python -m testground.pdf_compare.benchmark" and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks.pdf_compare import (  # type: ignore
        references,
    )
else:
    from . import references
    from .estimators import FFTEstimator, ProjectionEstimator


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def default_fft_params(d):
    """Grid size / spacing per dimension (memory- and extent-aware)."""
    if d == 2:
        return dict(grid_size=101, dt=0.3)
    return dict(grid_size=41, dt=0.45)


def anchor_points(d, radii=(1.0, 2.0, 4.0)):
    """A small set of representative evaluation points: origin, axis, diagonal."""
    pts = [np.zeros(d)]
    e0 = np.zeros(d); e0[0] = 1.0
    diag = np.ones(d) / np.sqrt(d)
    for r in radii:
        pts.append(e0 * r)
        pts.append(diag * r)
    return np.array(pts)


def make_axes(d, half_width, n):
    """``d`` identical coordinate axes over ``[-half_width, half_width]``."""
    axis = np.linspace(-half_width, half_width, n)
    return [axis] * d, axis[1] - axis[0]


# --------------------------------------------------------------------------- #
# Correctness
# --------------------------------------------------------------------------- #
def run_correctness(cases, *, n_spectral, seed, grid_n=41, half_width=9.0):
    print("\n" + "=" * 78)
    print("CORRECTNESS  (n_spectral=%d, projection grid=%dx.., seed=%d)"
          % (n_spectral, grid_n, seed))
    print("=" * 78)

    header = ("%-22s %-11s %9s %9s %10s %10s"
              % ("case", "method", "norm", "neg_mass", "marg_relm", "xmethod_relm"))
    print(header)
    print("-" * 78)

    for case in cases:
        sampler = case.sampler()
        d = case.d
        samples = references.spectral_samples(sampler, n_spectral, seed)

        # Shared 1D marginal reference along axis 0.
        beta0, scale0 = references.axis_marginal_params(sampler, samples,
                                                        case.alpha, axis=0)
        shift0 = float(np.broadcast_to(case.shift, (d,))[0])

        fft_anchor = proj_anchor = None
        anchors = anchor_points(d)

        # ---- FFT (symmetric only) ---- #
        if case.both_methods:
            fft = FFTEstimator(number_of_spectral_samples=n_spectral,
                               **default_fft_params(d))
            fft.setup(case.alpha, sampler, shift=case.shift, random_state=seed)
            norm = fft.normalization()
            neg = fft.negative_mass()
            marg = metrics.grid_marginal(fft.pdf_grid, fft.dx, axis=0) \
                if d == 2 else None
            marg_relm = float("nan")
            if marg is not None and case.alpha != 1.0:
                ref = references.axis_marginal_pdf(fft.axis, case.alpha, beta0,
                                                   scale0, loc=shift0)
                marg_relm = metrics.error_summary(marg, ref)["median_rel"]
            fft_anchor = fft.evaluate(anchors)
            print("%-22s %-11s %9.4f %9.2e %10s %10s"
                  % (case.name, "fft", norm, neg,
                     _fmt(marg_relm), "-"))

        # ---- Gaussian exact reference (alpha == 2) ---- #
        if case.alpha == 2.0 and case.both_methods:
            cov = references.gaussian_covariance(sampler, samples)
            ref_vals = references.gaussian_pdf(anchors, cov)
            g = metrics.error_summary(fft_anchor, ref_vals)
            print("%-22s %-11s %9s %9s %10s %10s   <- vs Gaussian: med_rel=%.3g max_rel=%.3g"
                  % ("", "  (exact)", "-", "-", "-", "-",
                     g["median_rel"], g["max_rel"]))

        # ---- Projection (kernel is singular at alpha == 2) ---- #
        if case.alpha == 2.0:
            print("%-22s %-11s %9s %9s %10s %10s   <- N/A (projection kernel"
                  " singular at alpha=2)"
                  % ("", "projection", "-", "-", "-", "-"))
            print("-" * 78)
            continue

        proj = ProjectionEstimator(number_of_spectral_samples=n_spectral)
        proj.setup(case.alpha, sampler, shift=case.shift, random_state=seed)
        proj_anchor = proj.evaluate(anchors)

        norm_p = float("nan")
        marg_relm_p = float("nan")
        if d == 2:  # 2D grid integration is affordable; 3D is not
            axes, dx = make_axes(d, half_width, grid_n)
            grid = proj.joint_on_grid(axes)
            norm_p = metrics.integrate_grid(grid, dx, d)
            if case.alpha != 1.0:
                axis = axes[0]
                marg = metrics.grid_marginal(grid, dx, axis=0)
                ref = references.axis_marginal_pdf(axis, case.alpha, beta0,
                                                   scale0, loc=shift0)
                marg_relm_p = metrics.error_summary(marg, ref)["median_rel"]

        xmethod = float("nan")
        if case.both_methods and fft_anchor is not None:
            xmethod = metrics.error_summary(proj_anchor, fft_anchor)["median_rel"]

        print("%-22s %-11s %9s %9s %10s %10s"
              % ("", "projection",
                 _fmt(norm_p), "-", _fmt(marg_relm_p), _fmt(xmethod)))
        print("-" * 78)


def _fmt(x):
    return "nan" if x != x else ("%.4f" % x)


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
def run_convergence(case, *, seed, spectral_list, dt_list, sphere_list,
                    hi_spectral, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 78)
    print("CONVERGENCE  (case=%s)" % case.name)
    print("=" * 78)

    sampler = case.sampler()
    d = case.d
    anchors = anchor_points(d)
    fftp = default_fft_params(d)

    # ---- High-fidelity self-references (each method to itself) ---- #
    fft_hi = FFTEstimator(number_of_spectral_samples=hi_spectral,
                          grid_size=fftp["grid_size"] + 40 if d == 2 else fftp["grid_size"],
                          dt=fftp["dt"] * 0.6 if d == 2 else fftp["dt"])
    fft_hi.setup(case.alpha, sampler, random_state=seed + 99)
    fft_ref = fft_hi.evaluate(anchors)

    proj_hi = ProjectionEstimator(number_of_spectral_samples=hi_spectral)
    proj_hi.setup(case.alpha, sampler, random_state=seed + 99)
    proj_ref = proj_hi.evaluate(anchors)

    # ---- 1) error vs Monte-Carlo sample count (both methods) ---- #
    fft_err, proj_err, xmethod = [], [], []
    for n in spectral_list:
        fft = FFTEstimator(number_of_spectral_samples=n, **fftp)
        fft.setup(case.alpha, sampler, random_state=seed)
        fv = fft.evaluate(anchors)
        proj = ProjectionEstimator(number_of_spectral_samples=n)
        proj.setup(case.alpha, sampler, random_state=seed)
        pv = proj.evaluate(anchors)
        fft_err.append(metrics.error_summary(fv, fft_ref)["median_rel"])
        proj_err.append(metrics.error_summary(pv, proj_ref)["median_rel"])
        xmethod.append(metrics.error_summary(pv, fv)["median_rel"])
        print("  n_spectral=%-7d fft_selferr=%.4f proj_selferr=%.4f xmethod=%.4f"
              % (n, fft_err[-1], proj_err[-1], xmethod[-1]))

    # ---- 2) FFT accuracy vs grid spacing dt (normalization + marginal) ---- #
    beta0, scale0 = references.axis_marginal_params(
        sampler, references.spectral_samples(sampler, hi_spectral, seed),
        case.alpha, axis=0)
    dt_norm, dt_marg = [], []
    for dt in dt_list:
        fft = FFTEstimator(number_of_spectral_samples=hi_spectral,
                           grid_size=fftp["grid_size"], dt=dt)
        fft.setup(case.alpha, sampler, random_state=seed)
        dt_norm.append(abs(fft.normalization() - 1.0))
        if d == 2 and case.alpha != 1.0:
            marg = metrics.grid_marginal(fft.pdf_grid, fft.dx, axis=0)
            ref = references.axis_marginal_pdf(fft.axis, case.alpha, beta0, scale0)
            dt_marg.append(metrics.error_summary(marg, ref)["median_rel"])
        else:
            dt_marg.append(float("nan"))
        print("  dt=%-6.3f |norm-1|=%.4f marg_relm=%.4f"
              % (dt, dt_norm[-1], dt_marg[-1]))

    # ---- 3) projection stability vs sphere-quadrature nodes ---- #
    sph_err = []
    sph_hi = ProjectionEstimator(number_of_spectral_samples=hi_spectral,
                                 number_of_sphere_points=max(sphere_list) * 2)
    sph_hi.setup(case.alpha, sampler, random_state=seed)
    sph_ref = sph_hi.evaluate(anchors)
    for m in sphere_list:
        proj = ProjectionEstimator(number_of_spectral_samples=hi_spectral,
                                   number_of_sphere_points=m)
        proj.setup(case.alpha, sampler, random_state=seed)
        pv = proj.evaluate(anchors)
        sph_err.append(metrics.error_summary(pv, sph_ref)["median_rel"])
        print("  sphere_points=%-6d proj_relm=%.4f" % (m, sph_err[-1]))

    # ---- plots ---- #
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].loglog(spectral_list, fft_err, "o-", label="FFT self-err")
    axes[0].loglog(spectral_list, proj_err, "s-", label="projection self-err")
    axes[0].loglog(spectral_list, xmethod, "^--", label="cross-method")
    ref_rate = np.array(spectral_list, float) ** -0.5
    ref_rate *= (fft_err[0] / ref_rate[0]) if fft_err[0] == fft_err[0] else 1.0
    axes[0].loglog(spectral_list, ref_rate, "k:", alpha=0.6, label=r"$N^{-1/2}$")
    axes[0].set_xlabel("spectral samples N")
    axes[0].set_ylabel("median relative error")
    axes[0].set_title("Monte-Carlo convergence")
    axes[0].legend(fontsize="small"); axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(dt_list, dt_norm, "o-", label="|norm - 1|")
    axes[1].loglog(dt_list, dt_marg, "s-", label="marginal median rel")
    axes[1].set_xlabel("FFT grid spacing dt")
    axes[1].set_ylabel("error")
    axes[1].set_title("FFT grid convergence")
    axes[1].legend(fontsize="small"); axes[1].grid(alpha=0.3, which="both")

    axes[2].semilogx(sphere_list, sph_err, "o-")
    axes[2].set_xlabel("sphere quadrature nodes")
    axes[2].set_ylabel("median relative error")
    axes[2].set_title("Projection sphere convergence")
    axes[2].grid(alpha=0.3, which="both")

    fig.suptitle("Convergence -- %s (alpha=%.2f, d=%d)"
                 % (case.name, case.alpha, case.d))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(outdir) / f"convergence_{case.name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> saved", out)


# --------------------------------------------------------------------------- #
# Performance
# --------------------------------------------------------------------------- #
def run_performance(case, *, seed, n_spectral, point_counts, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 78)
    print("PERFORMANCE  (case=%s, n_spectral=%d)" % (case.name, n_spectral))
    print("=" * 78)

    sampler = case.sampler()
    d = case.d
    fftp = default_fft_params(d)
    rng = np.random.default_rng(seed)

    # Build cost is paid once by FFT; projection has no build step.
    fft = FFTEstimator(number_of_spectral_samples=n_spectral, **fftp)
    fft.setup(case.alpha, sampler, random_state=seed)
    fft_build = fft.setup_time
    print("  FFT build (grid %s, dt=%.3f): %.3f s"
          % (fft.pdf_grid.shape, fft.dt, fft_build))

    proj = ProjectionEstimator(number_of_spectral_samples=n_spectral)
    proj.setup(case.alpha, sampler, random_state=seed)

    print("  %-10s %12s %14s %16s" %
          ("n_points", "fft_eval(s)", "proj_eval(s)", "proj/fft_amortized"))
    fft_times, proj_times = [], []
    for npts in point_counts:
        pts = rng.uniform(-6, 6, size=(npts, d))
        fft.evaluate(pts)
        proj.evaluate(pts)
        fft_times.append(fft.eval_time)
        proj_times.append(proj.eval_time)
        amort = proj.eval_time / (fft_build + fft.eval_time)
        print("  %-10d %12.4f %14.4f %16.2f"
              % (npts, fft.eval_time, proj.eval_time, amort))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(point_counts, np.array(fft_times) + fft_build, "o-",
              label="FFT (build + eval)")
    ax.loglog(point_counts, fft_times, "o--", alpha=0.5, label="FFT (eval only)")
    ax.loglog(point_counts, proj_times, "s-", label="projection (eval)")
    ax.set_xlabel("number of query points")
    ax.set_ylabel("wall-clock seconds")
    ax.set_title("Evaluation cost -- %s (alpha=%.2f, d=%d)"
                 % (case.name, case.alpha, case.d))
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = Path(outdir) / f"performance_{case.name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> saved", out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+",
                        default=["correctness", "convergence", "performance"],
                        choices=["correctness", "convergence", "performance"])
    parser.add_argument("--outdir",
                        default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="smaller sizes for a fast smoke run")
    parser.add_argument("--full", action="store_true",
                        help="larger sizes / more cases")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        n_spectral = 8_000
        spectral_list = [1_000, 4_000, 16_000]
        dt_list = [0.6, 0.4, 0.25]
        sphere_list = [90, 360, 720]
        hi_spectral = 20_000
        point_counts = [50, 500, 2_000]
        corr_cases = [cases_mod.get_case("iso2d_a1.5"),
                      cases_mod.get_case("gauss2d")]
    elif args.full:
        n_spectral = 40_000
        spectral_list = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000]
        dt_list = [0.8, 0.6, 0.45, 0.3, 0.22, 0.15]
        sphere_list = [45, 90, 180, 360, 720, 1440]
        hi_spectral = 80_000
        point_counts = [50, 200, 1_000, 5_000, 20_000]
        corr_cases = cases_mod.ALL_CASES
    else:
        n_spectral = 20_000
        spectral_list = [1_000, 4_000, 16_000, 32_000]
        dt_list = [0.6, 0.45, 0.3, 0.2]
        sphere_list = [90, 180, 360, 720]
        hi_spectral = 40_000
        point_counts = [50, 500, 2_000, 8_000]
        # Default set excludes the stiff small-alpha projection grids (alpha<1),
        # which are only tractable in --full.
        corr_cases = [cases_mod.get_case(n) for n in (
            "iso2d_a1.2", "iso2d_a1.5", "iso2d_a1.8",
            "elliptic2d_a1.5", "iso3d_a1.5", "gauss2d",
            "skew_star2d_a1.5", "shifted_iso2d_a1.5",
        )]

    if "correctness" in args.sections:
        run_correctness(corr_cases, n_spectral=n_spectral, seed=args.seed)

    if "convergence" in args.sections:
        run_convergence(cases_mod.get_case("iso2d_a1.5"), seed=args.seed,
                        spectral_list=spectral_list, dt_list=dt_list,
                        sphere_list=sphere_list, hi_spectral=hi_spectral,
                        outdir=outdir)

    if "performance" in args.sections:
        run_performance(cases_mod.get_case("iso2d_a1.5"), seed=args.seed,
                        n_spectral=n_spectral, point_counts=point_counts,
                        outdir=outdir)

    print("\nDone. Plots in", outdir)


if __name__ == "__main__":
    main()
