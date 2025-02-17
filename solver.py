import numpy as np
from scipy.optimize import fsolve

def computeDt(D1, D2, mu1, mu2, mu_t, L_ctc, g_target, Dt_initial_guess):    
    # Define the function f(Dt)
    def f(Dt):
        if Dt >= min(D1, D2) or Dt <= 0:
            return 1e10  # Return a large value if Dt is out of valid range
        
        L1 = np.sqrt(D1**2 - Dt**2) / 2
        L2 = np.sqrt(D2**2 - Dt**2) / 2
        Lt = L_ctc - (L1 + L2)
        
        a1 = 4 / (D1**3 * np.pi**2)
        b1 = 2 * D1 * L1 / (D1**2 - 4 * L1**2) + np.arctanh(2 * L1 / D1)
        F1 = a1 * b1
        
        a2 = 4 / (D2**3 * np.pi**2)
        b2 = 2 * D2 * L2 / (D2**2 - 4 * L2**2) + np.arctanh(2 * L2 / D2)
        F2 = a2 * b2
        
        Ft = Lt / ((np.pi / 4 * Dt**2)**2)
        
        S1 = 1 / (8 * np.pi**3 * F1)
        S2 = 1 / (8 * np.pi**3 * F2)
        St = 1 / (8 * np.pi**3 * Ft)
        
        g1 = S1 / mu1
        g2 = S2 / mu2
        gt = St / mu_t
        
        g_total = 1 / (1/g1 + 1/gt + 1/g2)
        
        return g_total - g_target

    # Find Dt
    Dt_solution = fsolve(f, Dt_initial_guess)
        
    return Dt_solution




