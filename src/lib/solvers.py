import numpy as np
from lib import yvesData

def fid_calc(psi, gs_idx, f_thr):
    return np.abs(psi[gs_idx])**2 - f_thr

def bisection(f, args, f_thr=0.9):
    """
    Let us construct a method that searches for a value using bisection
    """

    a = 0
    b = len(args.ts) - 1

    gs_idx = np.argmin(np.diag(args.Hf))

    a_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[a])
    b_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[b])

    f_a = fid_calc(f(a_args), gs_idx, f_thr)
    f_b = fid_calc(f(b_args), gs_idx, f_thr)
    epsilon = 10e-3

    if np.sign(f_a) == np.sign(f_b):
        raise ValueError("No sign diff")
    
    mid = (a + b) // 2
    m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[mid])
    f_mid = fid_calc(f(m_args), gs_idx, f_thr)
    while np.abs(f_a - f_b) > epsilon:
        mid = (a + b) // 2
        m_args = yvesData(Hf=args.Hf, Hi=args.Hi, n=args.n, t=args.ts[mid])
        f_mid = fid_calc(f(m_args), gs_idx, f_thr)
        if np.sign(f_a) == np.sign(f_mid):
            b = mid
            f_b = f_mid
        else:
            a = mid
            f_a = f_mid

        print(f_a, f_b, f_mid)
        


    return args.ts[mid], f_mid + f_thr
        
        

    