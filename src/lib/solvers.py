import numpy as np
from lib import yvesData


def fid_calc(psi, gs_idx, f_thr):
    psi = np.asarray(psi).reshape(-1).astype(np.complex128)
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("psi has zero norm")
    psi = psi / nrm
    return (np.abs(psi[gs_idx])**2) - f_thr

def yield_bisection(f, args, f_thr=0.9, epsilon=1e-3):
    """
    This function uses bisection to find roots, default threshold is set to 0.9, it is however
    fully updateable at run time when you invoke the bisection method.

    This implementation uses yield so that we can get real-time data for the search.
    
    f : function returning state psi for given args
    args : yvesData object containing Hf, Hi, ts, etc.
    f_thr : fidelity threshold
    epsilon : stopping criterion for bisection
    """

    # Indices in ts array
    a = args.ts[0]
    b = args.ts[-1]

    gs_idx = np.argmin(np.diagonal(args.Hf))

    # Evaluate fidelity at endpoints
    a_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=a)
    b_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=b)
    psi = f(a_args)
    print("norm:", np.linalg.norm(np.asarray(psi).reshape(-1)))
    print("F:", np.abs(np.asarray(psi).reshape(-1)[gs_idx])**2)

    psi = f(b_args)
    print("norm:", np.linalg.norm(np.asarray(psi).reshape(-1)))
    print("F:", np.abs(np.asarray(psi).reshape(-1)[gs_idx])**2)

    f_a = fid_calc(f(a_args), gs_idx, f_thr)
    yield a, f_a + f_thr
    f_b = fid_calc(f(b_args), gs_idx, f_thr)
    yield b, f_b + f_thr
    if np.sign(f_a) == np.sign(f_b):
        raise ValueError(f"For n={args.n}, no sign difference at endpoints; cannot bisect.")

    # Main bisection loop
    while 1:
        if b - a <= 1:
            return
        mid = (a + b) // 2
        m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=mid)
        f_mid = fid_calc(f(m_args), gs_idx, f_thr)
        yield mid, f_mid + f_thr

        s_a = np.sign(f_a)
        s_m = np.sign(f_mid)
        if s_a == s_m:
            a = mid
            f_a = f_mid
        else:
            b = mid
            f_b = f_mid

