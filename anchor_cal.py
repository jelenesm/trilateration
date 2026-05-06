"""
3D anchor localization from noisy pairwise distance measurements.

Pipeline:
    1. Classical MDS for an initial guess.
    2. Weighted nonlinear least squares (Levenberg-Marquardt) for the MLE
       under Gaussian range noise.
    3. Procrustes alignment for evaluation against ground truth, and a
       gauge-fixing utility for deployment.

The unknowns are the N x 3 coordinates X. Pairwise distances determine X
only up to a rigid motion + reflection (7 DOF in 3D), so the answer is
canonical only after we fix a frame.

Sized for small N (typically 4-7 anchors): dense Jacobian, no sparse
machinery, full pairwise connectivity assumed.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist, squareform


# ---------- Step 1: classical MDS initialization ----------

def mds_init(D, dim=3):
    """Classical multidimensional scaling.

    Args:
        D:   (N, N) symmetric matrix of (possibly noisy) distances, zero diag.
        dim: embedding dimension (3 for our problem).

    Returns:
        X0 of shape (N, dim) -- the centered configuration whose pairwise
        distances best match D in the squared-error sense.
    """
    N = D.shape[0]
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    B = 0.5 * (B + B.T)  # symmetrize against numerical drift

    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    idx = np.argsort(eigvals)[::-1][:dim]
    L = np.maximum(eigvals[idx], 0.0)     # clip negatives from noise
    V = eigvecs[:, idx]
    return V * np.sqrt(L)


# ---------- Step 2: weighted nonlinear least squares ----------

def _residuals(x_flat, pairs, d_meas, w, N):
    X = x_flat.reshape(N, 3)
    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    dist = np.linalg.norm(diffs, axis=1)
    return w * (dist - d_meas)


def _jacobian(x_flat, pairs, d_meas, w, N):
    """Dense Jacobian: each row has 6 nonzeros (3 at anchor i, 3 at j).
    For N < 8 the matrix is at most 21x21, so sparse storage is overkill."""
    X = x_flat.reshape(N, 3)
    M = pairs.shape[0]
    diffs = X[pairs[:, 0]] - X[pairs[:, 1]]
    dist = np.maximum(np.linalg.norm(diffs, axis=1), 1e-12)
    u = diffs / dist[:, None]                  # unit vectors along links

    J = np.zeros((M, 3 * N))
    for k, (i, j) in enumerate(pairs):
        J[k, 3 * i:3 * i + 3] =  w[k] * u[k]
        J[k, 3 * j:3 * j + 3] = -w[k] * u[k]
    return J


def estimate_positions(D_meas, sigma=None, dim=3, verbose=False, polish=True):
    """Estimate N anchor positions from a noisy distance matrix.

    Args:
        D_meas: (N, N) symmetric measured distances, zero diag, fully
                connected (every pair has a measurement).
        sigma:  None  -> uniform unit weights
                float -> uniform sigma (1/sigma weight on every residual).
        polish: bool, default True. If False, return only the closed-form
                MDS init (skip the LM step). Useful for comparing the
                closed-form initial guess against the polished MLE.

    Returns:
        (X_hat, res). X_hat is (N, 3); frame is arbitrary -- gauge-fix or
        Procrustes-align downstream. res is the SciPy OptimizeResult, or
        None when polish=False.
    """
    N = D_meas.shape[0]
    pairs = np.column_stack(np.triu_indices(N, k=1))
    d_meas = D_meas[pairs[:, 0], pairs[:, 1]]
    w = np.ones_like(d_meas) if sigma is None else np.full_like(d_meas, 1.0 / sigma)

    X0 = mds_init(D_meas, dim=dim)

    if not polish:
        return X0, None

    res = least_squares(
        _residuals, X0.ravel(),
        jac=_jacobian,
        args=(pairs, d_meas, w, N),
        method="trf",          # trust-region; handles the 7-DOF gauge null-space
        x_scale="jac",
        verbose=2 if verbose else 0,
    )
    return res.x.reshape(N, 3), res

# ---------- Step 3: frame fixing ----------

def procrustes_align(X, X_ref, allow_reflection=True):
    """Rigid-align X to X_ref. Returns the aligned copy of X.

    Solves min_{R, t} ||X R + t - X_ref||_F over R in O(3) (or SO(3) if
    allow_reflection=False). Closed-form via SVD.
    """
    cX = X.mean(0)
    cR = X_ref.mean(0)
    A = X - cX
    B = X_ref - cR
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return A @ R + cR


def gauge_fix(X, ref=(0, 1, 2, 3)):
    """Pin a deterministic frame using four reference anchors.

    Anchor ref[0] -> origin
    Anchor ref[1] -> +x axis
    Anchor ref[2] -> xy-plane with y > 0
    Anchor ref[3] -> z > 0   (resolves reflection)
    """
    i0, i1, i2, i3 = ref
    X = X - X[i0]

    # Rotate ref[1] to +x
    v = X[i1]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        e1 = v / nrm
        # Build any orthonormal basis with e1 as first column
        # Pick a helper not parallel to e1
        helper = np.array([0., 1., 0.]) if abs(e1[0]) > 0.9 else np.array([1., 0., 0.])
        e2 = helper - (helper @ e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        R = np.column_stack([e1, e2, e3])  # columns map world -> our basis
        X = X @ R                          # express in new basis

    # Rotate about x to put ref[2] in xy with y > 0
    _, y, z = X[i2]
    theta = np.arctan2(z, y)
    c, s = np.cos(theta), np.sin(theta)
    Rx = np.array([[1, 0, 0],
                   [0,  c, s],
                   [0, -s, c]])
    X = X @ Rx.T

    # Flip z so ref[3] has z > 0
    if X[i3, 2] < 0:
        X[:, 2] *= -1
    return X


# ---------- Demo / sanity check ----------

def _simulate(N=10, sigma=0.10, seed=0, scale=5.0):
    rng = np.random.default_rng(seed)
    X_true = rng.normal(size=(N, 3)) * scale
    D_true = squareform(pdist(X_true))
    noise = rng.normal(scale=sigma, size=(N, N))
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    D_meas = np.maximum(D_true + noise, 0.0)
    return X_true, D_true, D_meas


def _rmse(X, X_ref):
    return float(np.sqrt(np.mean(np.sum((X - X_ref) ** 2, axis=1))))


def _noisy_distances(X_true, sigma, seed):
    """Symmetric Gaussian range noise on the pairwise distance matrix."""
    rng = np.random.default_rng(seed)
    D_true = squareform(pdist(X_true))
    noise = rng.normal(scale=sigma, size=D_true.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    return np.maximum(D_true + noise, 0.0)


def _scenario_sweep(X_true, label, sigmas=(0.01, 0.05, 0.1, 0.25, 0.5), n_seeds=20):
    print(f"\n--- {label}  (N = {len(X_true)}) ---")
    print("  true anchors:")
    for i, x in enumerate(X_true):
        print(f"    {i:2d}   ({x[0]:+7.3f}, {x[1]:+7.3f}, {x[2]:+7.3f})")
    for s in sigmas:
        errs = []
        example = None
        for seed in range(n_seeds):
            Dm = _noisy_distances(X_true, s, seed)
            Xh, _ = estimate_positions(Dm, sigma=s)
            Xh_a = procrustes_align(Xh, X_true)
            if seed == 0:
                example = Xh_a
            errs.append(_rmse(Xh_a, X_true))
        print(f"  sigma = {s:.2f}  ->  RMSE = {np.mean(errs):.4f} ± {np.std(errs):.4f}")
        print(f"    seed=0 estimated:")
        for i, h in enumerate(example):
            print(f"      {i:2d}   ({h[0]:+7.3f}, {h[1]:+7.3f}, {h[2]:+7.3f})")


if __name__ == "__main__":
    N = 4
    sigma = 0.10
    X_true, D_true, D_meas = _simulate(N=N, sigma=sigma, seed=42)

    # MDS only
    X_mds = mds_init(D_meas, dim=3)
    X_mds_aligned = procrustes_align(X_mds, X_true, allow_reflection=True)

    # Full pipeline
    X_hat, res = estimate_positions(D_meas, sigma=sigma)
    X_hat_aligned = procrustes_align(X_hat, X_true, allow_reflection=True)

    print(f"N = {N}, sigma = {sigma}")
    print(f"MDS-only RMSE     : {_rmse(X_mds_aligned, X_true):.4f}")
    print(f"MDS + LM RMSE     : {_rmse(X_hat_aligned, X_true):.4f}")
    print(f"LM iterations     : {res.nfev}")
    print(f"Final cost (½‖r‖²): {res.cost:.4f}")

    # Quick sweep over noise levels
    print("\nNoise sweep (RMSE after MDS+LM, averaged over 20 seeds):")
    for s in [0.01, 0.05, 0.1, 0.25, 0.5]:
        errs = []
        example = None
        for seed in range(20):
            Xt, _, Dm = _simulate(N=N, sigma=s, seed=seed)
            Xh, _ = estimate_positions(Dm, sigma=s)
            Xh_a = procrustes_align(Xh, Xt)
            if seed == 0:
                example = (Xt, Xh_a)
            errs.append(_rmse(Xh_a, Xt))
        print(f"  sigma = {s:.2f}  ->  RMSE = {np.mean(errs):.4f} ± {np.std(errs):.4f}")
        Xt_e, Xh_e = example
        print(f"    example (seed=0):")
        print(f"       i        true (x, y, z)                 est (x, y, z)")
        for i, (t, h) in enumerate(zip(Xt_e, Xh_e)):
            print(f"      {i:2d}   ({t[0]:+7.3f}, {t[1]:+7.3f}, {t[2]:+7.3f})   "
                  f"({h[0]:+7.3f}, {h[1]:+7.3f}, {h[2]:+7.3f})")

    # Cuboid placement scenarios (cube of side L). The alternating-corner set
    # is the regular tetrahedron inscribed in the cube (vertices with even
    # XOR of binary coordinates).
    L = 10.0
    corners = np.array([
        [0, 0, 0], [L, 0, 0], [0, L, 0], [L, L, 0],
        [0, 0, L], [L, 0, L], [0, L, L], [L, L, L],
    ], dtype=float)
    alt_idx = [0, 3, 5, 6]  # (0,0,0), (L,L,0), (L,0,L), (0,L,L)
    _scenario_sweep(corners,           "Scenario 1: 8 anchors at all cuboid corners")
    _scenario_sweep(corners[alt_idx],  "Scenario 2: 4 anchors at alternating corners")
        