"""
run_bistable_sweep.py
==================================================================
Frequency-sweep study of the dimensionless bistable harvester with
solution continuation (up/down) to reveal the broadband high-energy
orbit and hysteresis. Writes results to ../results/.
==================================================================
"""
import numpy as np, csv, os, json
from bistable import simulate, linear_reference

RES = os.path.join(os.path.dirname(__file__), "..", "results")
f0 = 0.10          # forcing strong enough to access the high-energy orbit
zeta, chi, kappa, alpha = 0.01, 0.05, 0.5, 0.05
Om = np.linspace(0.3, 1.6, 80)

# --- bistable: sweep UP with continuation ---
P_up, xpp_up = [], []
yend = [1.0, 0.0, 0.0]
for O in Om:
    r = simulate(O, f0, zeta, chi, kappa, alpha, t_end=350.0, y0=yend)
    P_up.append(r['P']); xpp_up.append(r['xpp']); yend = r['yend']

# --- bistable: sweep DOWN with continuation ---
P_dn, xpp_dn = [], []
yend = [1.0, 0.0, 0.0]
for O in Om[::-1]:
    r = simulate(O, f0, zeta, chi, kappa, alpha, t_end=350.0, y0=yend)
    P_dn.append(r['P']); xpp_dn.append(r['xpp']); yend = r['yend']
P_dn = P_dn[::-1]; xpp_dn = xpp_dn[::-1]

# --- linear single-well reference at same parameters ---
P_lin = [linear_reference(O, f0, zeta, chi, kappa, alpha, t_end=350.0)
         for O in Om]

# normalize all powers to the linear peak for a clean comparison
Pmax_lin = max(P_lin)
def norm(a): return [x/Pmax_lin for x in a]

with open(os.path.join(RES, "S7_bistable_sweep.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Omega", "P_up_norm", "P_down_norm", "P_linear_norm",
                "xpp_up", "xpp_down"])
    for i, O in enumerate(Om):
        w.writerow([O, P_up[i]/Pmax_lin, P_dn[i]/Pmax_lin,
                    P_lin[i]/Pmax_lin, xpp_up[i], xpp_dn[i]])

def bandwidth(O, P, thr_frac=0.5):
    P = np.asarray(P); O = np.asarray(O)
    thr = P.max()*thr_frac
    above = O[P >= thr]
    return (above.max()-above.min()) if len(above) > 1 else 0.0

# Linear half-power bandwidth (analytic, lightly damped SDOF): ~2*zeta
# in normalized frequency about Omega=1. The coarse sweep cannot resolve
# this narrow peak, so we use the exact small-damping value.
bw_lin = 2.0*zeta
bw_bi  = bandwidth(Om, np.maximum(P_up, P_dn))
gain = bw_bi/bw_lin if bw_lin > 0 else float("nan")

summary = {
  "forcing_f0": f0, "zeta": zeta, "chi": chi, "kappa": kappa, "alpha": alpha,
  "linear_bandwidth_Omega": float(bw_lin),
  "bistable_bandwidth_Omega": float(bw_bi),
  "bandwidth_gain_x": float(gain),
  "max_norm_power_bistable": float(np.max(np.maximum(P_up, P_dn))/Pmax_lin),
}
with open(os.path.join(RES, "S7_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
