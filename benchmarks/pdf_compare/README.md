# Multivariate α-stable PDF — unified testing ground

Two independent implementations estimate the joint density of an α-stable random
vector from samples of its **spectral measure**:

| method | source | evaluates | valid for |
|---|---|---|---|
| **projection** | `aub_htp/pdf/multivariate.py` (`MultivariateStableDensity`) | point-wise (sphere integral per point) | all α∈(0,2), all β∈[-1,1], skew + shift |
| **inverse FFT** | `pdf_fft.py` (repo root) | a whole grid at once (MC characteristic function → `ifftn`) | all α∈(0,2), all β∈[-1,1], skew + shift, **and** α=2 |

Both `pdf_compare/` and `pdf_fft.py` are standalone top-level modules at the
repo root (not part of the installed `aub_htp` package).

## Layout

```
pdf_fft.py        the inverse-FFT implementation under test (repo root)
pdf_compare/
  estimators.py   DensityEstimator / ProjectionEstimator / FFTEstimator (common API, timed)
  references.py   independent ground truths (Gaussian limit, 1D stable marginal, Monte-Carlo)
  metrics.py      error_summary / integrate_grid / l1_distance / grid_marginal
  cases.py        standard scenarios (symmetric, Gaussian-limit, skewed/shifted)
  benchmark.py    correctness / convergence / performance harness -> tables + plots
  meta_study.py   which method wins in each alpha-beta regime -> tables + plots + JSON
  _render_report.py  renders META_STUDY.html tables from results/meta_study.json
  META_STUDY.html    the written-up meta-study (open in a browser)
  results/        generated plots and JSON (git-ignored)
```

## Verdict (see `META_STUDY.html` for the full study)

| when | use |
|---|---|
| α ≲ 1.2, any β | **projection** — FFT is 15–626× less accurate |
| α ≳ 1.8, any β | either — indistinguishable (~1e-4); pick on workload |
| ≳ 10⁵ points, or a full grid | **inverse FFT** — 3–5× cheaper once amortised |
| ≲ 10⁴ scattered points | **projection** — smaller build *and* more accurate |
| α = 2 | **inverse FFT** (or the exact Gaussian); projection refuses |
| α near 1 with skew | fix the sampler, not the method — see below |

Across 50 α×β cells the projection method wins 37 and ties 13; it never loses.

Tests:

```
tests/multivariate_kernel_test.py       kernel g_{alpha,d} vs its defining integral
tests/multivariate_alpha_beta_test.py   full alpha x beta coverage, both methods
tests/multivariate_pdf_compare_test.py  cross-method + reference guards in 2-D
```

## Common interface

```python
from pdf_compare import FFTEstimator, ProjectionEstimator
from aub_htp import IsotropicSampler

sampler = IsotropicSampler(2, 1.5, 1.0)

fft = FFTEstimator(number_of_spectral_samples=20_000, grid_size=101, dt=0.3)
fft.setup(alpha=1.5, sampler=sampler, random_state=0)   # one-off grid build
values = fft.evaluate(points)                           # cheap interpolation

proj = ProjectionEstimator(number_of_spectral_samples=20_000)
proj.setup(alpha=1.5, sampler=sampler, random_state=0)  # tabulates the kernel
values = proj.evaluate(points)                          # ~O(points)
```

Both are driven from the same sampler and seed, so measured differences are
genuine method differences. `setup_time` / `eval_time` record each phase.
`ProjectionEstimator(exact=True)` bypasses the kernel spline (slower, used as the
accuracy reference).

## References (how correctness is judged)

* **Exact univariate law (all β).** A 1-D spectral measure with atoms at ±1
  realises a univariate S1 law with `σ^α = w₊+w₋`, `β = (w₊−w₋)/(w₊+w₋)`, so
  `alpha_stable.pdf` is ground truth for *every* β — unreachable with a
  symmetric multivariate measure. `d=1` also makes the projection method's
  sphere quadrature exact, isolating the kernels.
* **Analytic α=1 isotropic 2-D.** `f(x) = (1/2π)(1+|x|²)^(−3/2)`.
* **Gaussian limit (α=2).** `Σ = 2·mass·mean(V Vᵀ)` — exact, FFT only.
* **1-D marginal** of a 2-D law, **normalization**, **non-negativity**,
  **radial symmetry**, and **cross-method agreement**.

## Running

```bash
pytest tests/multivariate_kernel_test.py        # kernel accuracy   (~40 s)
pytest tests/multivariate_pdf_compare_test.py   # 2-D guards        (~90 s)
pytest tests/multivariate_alpha_beta_test.py    # alpha x beta grid (~8 min)

python -m pdf_compare.benchmark  [--quick|--full]
python -m pdf_compare.meta_study [--quick]
```

## Notes and gotchas

These are all things that bit us; they are recorded so they do not again.

**Projection method**

* `alpha == 1` cannot use Zolotarev's (B) parameterization: it carries
  `exp(-x/β)` and is singular as β→0. Because MC noise makes β≈1e-3 rather than
  0, *symmetric* α=1 silently took that branch and was wrong by ~3×. α=1 is now
  evaluated from the definition (panel quadrature for |v|≤30, a closed-form
  large-|v| series beyond).
* `|β| = 1` is an endpoint singularity of the finite-interval kernel; β is
  clamped just inside. For α<1 with **even d** the kernel is unusable there and
  the defining integral is used instead. Only even d matters: odd d reads
  `Re h`, even d reads `Im h`, and the defect is confined to `Im h`.
* There is **no** Theorem 2.1(c) "extra term". Adding it double-counts and
  doubled `g` at β=1 for even d.
* `alpha == 2` is refused — the kernel's `tan(πα/2)` diverges. Use the Gaussian.
* Cost is `O(points × sphere nodes)`. The kernel is tabulated once as a spline
  in `arcsinh(v)` (2-D in `(v, β)` when β(s) varies), which is what makes
  evaluation ~120 µs/point instead of ~140 ms/point. Evaluating the spline nodes
  in `arcsinh(v)` bands to localise `quad_vec` refinement was measured **3×
  slower** — the per-call setup dominates. One call is correct.

**FFT method**

* `ifftn` uses the `+i` convention, so it reconstructs `f(−x)`. The estimator
  feeds the **conjugate** CF to compensate. Without that, skewed and shifted
  densities come out **mirrored** — invisible for symmetric laws, which is
  exactly why it went unnoticed.
* `estimate_cf` builds a dense `(#grid × #samples)` matrix; `FFTEstimator` calls
  it in grid batches (`max_matrix_elements`) to stay within memory (it OOM'd at
  5 GiB otherwise).
* Small α needs a fine grid *and* a wide window (sharp peak, heavy tail), so it
  is resolution-limited exactly where the point-wise method is not.

**Shared**

* Near α=1 the S1 CF multiplies β by `tan(πα/2)` (12.7 at α=0.95, 63.7 at
  α=0.99). Monte-Carlo error in the estimated skewness is amplified by that
  factor, so *both* methods degrade identically — and only as `1/√N`. Use a
  symmetry-respecting (antithetic) spectral sampler rather than more samples.
* `alpha_stable.pdf` is accurate in the bulk but **truncates its tails to zero**
  (e.g. returns 0 at x=500 where the true density is 2.6e-6), so use it as a
  bulk reference only.
