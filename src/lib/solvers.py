import numpy as np
from lib import yvesData

def fid_calc(psi, gs_idx, f_thr):
    """
    Compute fidelity difference from threshold.
    """
    return np.abs(psi[gs_idx])**2 - f_thr


def bisection(f, args, f_thr=0.9, epsilon=1e-3):
    """
    Find the value of t where fidelity crosses f_thr using bisection.
    
    f : function returning state psi for given args
    args : yvesData object containing Hf, Hi, ts, etc.
    f_thr : fidelity threshold
    epsilon : stopping criterion for bisection
    """

    # Indices in ts array
    a = 0
    b = len(args.ts) - 1

    # Ground-state index from sparse diagonal
    gs_idx = np.argmin(np.diagonal(args.Hf))
    print(gs_idx)

    # Evaluate fidelity at endpoints
    a_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[a])
    b_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[b])

    f_a = fid_calc(f(a_args), gs_idx, f_thr)
    print(f_a)
    f_b = fid_calc(f(b_args), gs_idx, f_thr)
    print(f_b)
    if np.sign(f_a) == np.sign(f_b):
        raise ValueError("No sign difference at endpoints; cannot bisect.")

    # Main bisection loop
    while (b - a) > 1:
        mid = (a + b) // 2
        m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[mid])
        f_mid = fid_calc(f(m_args), gs_idx, f_thr)

        s_a = np.sign(f_a)
        s_m = np.sign(f_mid)

        print(s_a, s_m)
        if s_a == s_m:
            a = mid
            f_a = f_mid
        else:
            b = mid
            f_b = f_mid

    # Return t value and fidelity at midpoint
    mid = (a + b) // 2
    m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[mid])
    f_mid = fid_calc(f(m_args), gs_idx, f_thr)
    return args.ts[mid], f_mid + f_thr


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
    a = 0
    b = len(args.ts) - 1

    gs_idx = np.argmin(np.diagonal(args.Hf))

    # Evaluate fidelity at endpoints
    a_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[a])
    b_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[b])

    f_a = fid_calc(f(a_args), gs_idx, f_thr)
    yield args.ts[a], f_a + f_thr
    f_b = fid_calc(f(b_args), gs_idx, f_thr)
    yield args.ts[b], f_b + f_thr
    if np.sign(f_a) == np.sign(f_b):
        raise ValueError(f"For n={args.n}, no sign difference at endpoints; cannot bisect.")

    # Main bisection loop
    while 1:
        if b - a <= 1:
            return
        mid = (a + b) // 2
        m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[mid])
        f_mid = fid_calc(f(m_args), gs_idx, f_thr)
        yield args.ts[mid], f_mid + f_thr

        s_a = np.sign(f_a)
        s_m = np.sign(f_mid)
        if s_a == s_m:
            a = mid
            f_a = f_mid
        else:
            b = mid
            f_b = f_mid



def secant():
    """
    TODO: Implement this
    """
    pass
