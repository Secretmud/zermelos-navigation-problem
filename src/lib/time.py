import numpy as np
from lib import N, smax

def gaussian(n, N, smax, sigma=None):
    if sigma is None:
        sigma = N / 4  # Default spread
    return smax * np.exp(-((n - N/2)**2) / (2 * sigma**2)) + smax * 1.8  # Slight boost to first and last values


#time = gaussian(np.linspace(0, N, N), N, smax)
time = [1, 3, 3, 2, 1, 1]
