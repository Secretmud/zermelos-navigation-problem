import numpy as np
from adiabatic_computation import YvesConfig


def fid_calc(psi, gs_idx, f_thr):
    psi = np.asarray(psi).reshape(-1).astype(np.complex128)
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("psi has zero norm")
    psi = psi / nrm
    return (np.abs(psi[gs_idx]) ** 2) - f_thr


def yield_bisection(f, args: YvesConfig, f_thr=0.9, epsilon=1e-2, max_runs=40):
    """Bisection search for the anneal time at which fidelity crosses f_thr.

    Yields (t, fidelity) pairs during the search so callers can stream results.

    f        : callable(YvesConfig) → state psi
    args     : YvesConfig with t_range = [t_min, ..., t_max] search range
    f_thr    : fidelity threshold (default 0.9)
    epsilon  : stopping criterion on interval width
    max_runs : maximum bisection iterations
    """
    a = args.t_range[0]
    b = args.t_range[-1]

    gs_idx = np.argmin(np.diagonal(args.H_f))

    a_args = YvesConfig(H_f=args.H_f, H_i=args.H_i, n=args.n, t=a)
    b_args = YvesConfig(H_f=args.H_f, H_i=args.H_i, n=args.n, t=b)

    f_a = fid_calc(f(a_args), gs_idx, f_thr)
    yield a, f_a + f_thr
    f_b = fid_calc(f(b_args), gs_idx, f_thr)
    yield b, f_b + f_thr

    if np.sign(f_a) == np.sign(f_b):
        raise ValueError(
            f"For n={args.n}, no sign difference at endpoints; cannot bisect."
        )

    i = 0
    while True:
        if b - a <= epsilon or i == max_runs:
            return
        mid = (a + b) / 2
        m_args = YvesConfig(H_f=args.H_f, H_i=args.H_i, n=args.n, t=mid)
        f_mid = fid_calc(f(m_args), gs_idx, f_thr)
        yield mid, f_mid + f_thr

        if np.sign(f_a) == np.sign(f_mid):
            a = mid
            f_a = f_mid
        else:
            b = mid
            f_b = f_mid
        i += 1
