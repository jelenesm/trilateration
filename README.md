# loc

Notebooks exploring localization and ranging signal processing — multi-anchor
position estimation from biased/noisy ranges, multi-tone phase-based distance
estimation under multipath, and a full RF Tx → channel → Rx chain at carrier
rate.

## Notebooks

- **[trilateration.ipynb](trilateration.ipynb)** — 3D position estimation from
  biased range measurements with an unknown common bias `b`. Two MLE solvers
  (`localize` cancels `b` by differencing against a reference anchor;
  `localize_joint` treats `b` as a 4th unknown), bootstrapped from a closed-form
  Schau-Robinson initializer. Validated by Monte-Carlo against
  `sqrt(diag(cov))` from the final Fisher information.

- **[anchor_cal.ipynb](anchor_cal.ipynb) / [anchor_cal.py](anchor_cal.py)** —
  3D multi-anchor calibration from pairwise distances (`N ≤ 8`). Closed-form
  classical-MDS init via double-centering + top-3 eigendecomposition, then
  trust-region Levenberg-Marquardt polish on the weighted distance residuals
  (Gaussian-noise MLE) with explicit dense Jacobian. Frame-fixing utilities
  (`procrustes_align` for evaluation, `gauge_fix` for deployment via 4
  reference anchors). RMSE-vs-σ Monte-Carlo sweeps on log-log axes for
  `N ∈ {4, 10, 16}`.

- **[distance.ipynb](distance.ipynb)** — Single Tx-Rx distance estimation from
  multi-tone phasors (`N` orthogonal tones at spacing `Δf`, baseband only).
  Composable pipeline (`transmit` → `channel` → `simulate`); two estimators
  (`'line_fit'` slope-of-phase and `'music'` MUSIC pseudo-spectrum). Multipath
  channel model with LOS/NLOS toggle and configurable delay spread, sequential
  CW phase-ranging cell with pilot-based LO compensation, RMSE-vs-SNR sweeps
  per radio (BLE 1M/2M, WiFi 20–160 MHz, TGn-C/E profiles).

- **[tx_rx_chain.ipynb](tx_rx_chain.ipynb)** — Discrete-time RF Tx → channel →
  Rx end-to-end simulation at full carrier rate (`f_LO = 2.4 GHz`,
  `fs = 10 GHz`). I/Q upconversion, propagation delay applied as a spectral
  phase ramp, downconvert + LPF, analytical-prediction verifier, MUSIC distance
  estimate. Closing analysis derives why LO clock skew (`t_TX`, `t_RX`) doesn't
  affect the distance estimate (it absorbs into the per-tone phase intercept,
  not the slope) and confirms it empirically.
