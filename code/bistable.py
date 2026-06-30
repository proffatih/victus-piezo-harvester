"""
bistable.py
==================================================================
Non-dimensional bistable piezoelectric energy harvester (Duffing /
piezomagnetoelastic) following the canonical normalized form of
Erturk-Inman / Daqaq et al. The double-well restoring force is
  f(x) = -1/2 x (1 - x^2),  wells at x = +/-1.

Dimensionless coupled equations (overdot = d/dtau):
  x'' + 2 zeta x' - 1/2 x (1 - x^2) + chi*v = f0 cos(Omega tau)
  v' + alpha v + kappa x' = 0
where
  zeta  : mechanical damping ratio
  chi   : forward (mechanical<-electrical) coupling
  kappa : backward (electrical<-mechanical) coupling
  alpha : reciprocal dimensionless time constant = 1/(Omega_e), here
          alpha = 1/(R_L C_p w_n)  -> set via dimensionless load
  f0    : dimensionless forcing amplitude (proportional to base accel)
  Omega : dimensionless excitation frequency (=1 near a single well's
          linearized resonance)

This dimensionless system is numerically benign and is the standard
model used to demonstrate broadband snap-through harvesting.
==================================================================
"""
import numpy as np
from scipy.integrate import solve_ivp

def simulate(Omega, f0, zeta=0.01, chi=0.05, kappa=0.5, alpha=0.05,
             t_end=400.0, y0=None, n_keep=20000):
    """Integrate the dimensionless system; return steady-state RMS
    voltage/power proxy and the final state for continuation."""
    def rhs(t, y):
        x, xd, v = y
        xdd = -2*zeta*xd + 0.5*x*(1 - x*x) - chi*v + f0*np.cos(Omega*t)
        vd  = -alpha*v - kappa*xd
        return [xd, xdd, vd]
    if y0 is None:
        y0 = [1.0, 0.0, 0.0]      # start in a well
    t_eval = np.linspace(0.6*t_end, t_end, n_keep)  # steady-state window
    sol = solve_ivp(rhs, [0, t_end], y0, t_eval=t_eval,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=0.05)
    v = sol.y[2]; x = sol.y[0]
    Vrms = np.sqrt(np.mean(v**2))
    # dimensionless power proxy ~ alpha * v^2 (power dissipated in load)
    P = alpha*np.mean(v**2)
    xpp = x.max() - x.min()
    yend = [sol.y[0, -1], sol.y[1, -1], sol.y[2, -1]]
    return dict(Vrms=Vrms, P=P, xpp=xpp, yend=yend)

def linear_reference(Omega, f0, zeta=0.01, chi=0.05, kappa=0.5,
                     alpha=0.05, t_end=400.0, n_keep=20000):
    """Single-well linearized harvester (positive stiffness 1) at the
    same coupling/load, for bandwidth comparison."""
    def rhs(t, y):
        x, xd, v = y
        xdd = -2*zeta*xd - x - chi*v + f0*np.cos(Omega*t)
        vd  = -alpha*v - kappa*xd
        return [xd, xdd, vd]
    t_eval = np.linspace(0.6*t_end, t_end, n_keep)
    sol = solve_ivp(rhs, [0, t_end], [0, 0, 0], t_eval=t_eval,
                    method="RK45", rtol=1e-7, atol=1e-9, max_step=0.05)
    v = sol.y[2]
    return alpha*np.mean(v**2)

if __name__ == "__main__":
    # quick sanity check: deep, well-driven snap-through at moderate forcing
    for f0 in [0.02, 0.08, 0.15, 0.3]:
        r = simulate(0.8, f0)
        print("f0=%.2f  P=%.4e  xpp=%.3f (snap if xpp>1.5)" %
              (f0, r['P'], r['xpp']))
