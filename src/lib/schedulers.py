

def alpha(t, T_x, T_pen):
    if t < T_x:
        return alpha_max * (t / T_x)
    elif t < T_x + T_pen:
        return alpha_max
    elif t < 2*T_x + T_pen:
        return alpha_max * (1 - (t - (T_x + T_pen)) / T_x)
    else:
        return 0

def beta(t, T_x, T_pen):
    if t < T_x:
        return 0
    elif t < T_x + T_pen:
        return beta_max * (t - T_x) / T_pen
    else:
        return beta_max