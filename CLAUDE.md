# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Two notebooks, no `src/`, no `tests/`. `pyproject.toml` declares `numpy` and `matplotlib` as runtime deps; `ipywidgets` is installed in `.venv` but not declared (add it before relying on the interactive cells from a clean checkout). `description` and `authors` in `pyproject.toml` are still empty stubs.

- [trilateration.ipynb](trilateration.ipynb) — 3D position estimation from biased ranges (the original `example.ipynb`, expanded).
- [distance.ipynb](distance.ipynb) — single-link multi-tone phase ranging with a multipath channel model.

## Environment

A Python 3.12 virtualenv lives at `.venv/`. Activate with `source .venv/bin/activate` or invoke directly via `.venv/bin/python` / `.venv/bin/pip`. `pyproject.toml` requires Python `>=3.10`.

## Common commands

Install in editable mode with dev extras (once `src/` exists or after the package layout is decided):

```
.venv/bin/pip install -e ".[dev]"
```

Lint / format check (ruff is configured with `line-length = 100`):

```
.venv/bin/ruff check .
.venv/bin/ruff format .
```

Run tests (pytest is configured to discover under `tests/`):

```
.venv/bin/pytest                    # all tests
.venv/bin/pytest tests/test_x.py    # one file
.venv/bin/pytest -k name_substring  # filter by name
```

## Domain context — [trilateration.ipynb](trilateration.ipynb)

**3D localization from biased range measurements with an unknown common bias.**

Model: `d_i = ||x - a_i|| + b + n_i`, where `b` is an unknown common offset and `n_i ~ N(0, sigma^2)`.

Two MLE solvers, mathematically equivalent and verified to agree:

1. **`localize`** — *cancels* the bias by differencing each range against a chosen reference anchor, then runs Gauss-Newton / LM on the differenced residuals. Differencing introduces correlated noise; the correct weight is `Sigma^{-1}` with `Sigma = sigma^2 * (I + 11^T)` (the off-diagonal `1`s come from the shared `n_ref`). Bootstrapped from `linear_init`, a closed-form Schau-Robinson / spherical-intersection initializer that linearizes by squaring `||x - a_i|| = ||x - a_ref|| + delta_i`.
2. **`localize_joint`** — does *not* cancel the bias; treats `b` as a 4th unknown, minimizes raw-range residuals with i.i.d. weights. Cleaner noise model, larger Jacobian.

Key invariants when modifying:
- Both solvers require `N >= 4` anchors (3 independent differences for 3D, or 4 unknowns in the joint form).
- In `localize`, the `ref` anchor index threads through `build_differences`, the residuals, the Jacobian, and the difference covariance — they must agree.
- The differenced-form Jacobian sign convention follows `residual = delta - (||x - a_i|| - ||x - a_ref||)`, giving `dr/dx = -u_other + u_ref`.
- The Monte Carlo block validates implementations by comparing empirical RMSE against `sqrt(diag(cov))` from the final `(J^T W J)^{-1}`. Preserve this sanity check when refactoring.

## Domain context — [distance.ipynb](distance.ipynb)

**Single Tx-Rx distance estimation from multi-tone phasors.** N orthogonal tones at spacing `delta_f` are transmitted; the per-tone phase ramp `-2π f τ` encodes the propagation delay `τ = d/c`.

Pipeline, factored into composable stages — keep this separation when extending:

- **`transmit(delay, n_tones)`** — frequency-domain Tx spectrum (N tones with random LO-phase noise, padded to M = OSR·N bins).
- **`channel(d_true, delay_spread, n_paths, los)`** — multipath frequency response on the FFT grid. `delay_spread = 0` collapses to single-path LOS regardless of `los`. `los=False` removes the deterministic LOS tap (NLOS), biasing first-arrival estimators.
- **`simulate(...)`** — runs Tx → channel → IFFT → LO offset → AWGN → FFT → phase-vs-frequency → distance estimate. Returns `(d_hat, phases, phasors, slope, intercept)`.

Two distance estimators, switchable via `method=`:

- `'line_fit'` — slope of unwrapped phase vs frequency. Robust to uniform LO offset (rotates every tone equally → cancels in the slope). Default.
- `'music'` — `music_distance()` runs MUSIC on the per-tone phasors (forward spatial smoothing → eigendecomp → noise-subspace pseudo-spectrum on a distance grid). Resolves multipath; resolution is grid-limited (default `c/(2·delta_f·n_grid) ≈ 37 mm`).

Key invariants when modifying:
- `OSR = 8` and `delta_f = 1 MHz` set sample rate `fs = OSR·N·delta_f` and unambiguous range `c/(2·delta_f) = 150 m`. Anything beyond wraps in the line-fit estimator.
- `transmit`/`channel`/`simulate` all accept an `n_tones` override; when given, `M`, `fs`, `t`, `freqs` are recomputed *for that call only* and must stay consistent across the three.
- MUSIC requires `P + 1 <= L < N` where `L` defaults to `N // 2`.
- The interactive cell renders matplotlib figures into an `Image` widget rather than using `interactive()` + inline backend — VS Code + ipywidgets duplicates rows otherwise. Do not "simplify" this back to `interactive()`.
