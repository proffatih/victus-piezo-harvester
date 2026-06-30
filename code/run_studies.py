"""
run_studies.py
==================================================================
Parametric power studies for the unimorph piezoelectric harvester and
the nonlinear bistable (Duffing) broadband extension. All outputs are
written as CSV to ../results/ for figure generation and the manuscript.

Studies:
  S1  Voltage & power FRF vs frequency at several load resistances.
  S2  Power vs load resistance at resonance (optimal-R identification).
  S3  Power & resonance vs tip (proof) mass -> frequency tunability.
  S4  Power vs substructure thickness ratio (geometry optimization).
  S5  Power vs overhang length.
  S6  2-D optimization map: power over (tip mass, load resistance).
  S7  Nonlinear bistable Duffing frequency sweep (up & down) showing
      broadband hysteresis vs the linear counterpart.
==================================================================
"""
import numpy as np, csv, os, json
from harvester_model import PiezoHarvester, duffing_bistable

RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RES, exist_ok=True)
G = 9.81           # 1 g base acceleration amplitude
BASE = PiezoHarvester(zeta=0.01)

def wcsv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", name, len(rows), "rows")

# ---------------- S1: FRF vs frequency at several loads ----------------
f = np.linspace(40, 56, 800)
loads = [1e3, 4.4e3, 1e4, 4.18e4, 1e6]
rows = []
frf_cache = {RL: BASE.frf(f, RL, n_modes=3, base_accel=G) for RL in loads}
for i, fi in enumerate(f):
    row = [fi]
    for RL in loads:
        row += [np.abs(frf_cache[RL]['V'][i]), frf_cache[RL]['P'][i]*1e3]
    rows.append(row)
hdr = ["freq_Hz"]
for RL in loads:
    hdr += ["V_%g_ohm_V" % RL, "P_%g_ohm_mW" % RL]
wcsv("S1_frf_vs_freq.csv", hdr, rows)

# wideband (3 modes) for mode-shape / multi-resonance figure
fw = np.linspace(20, 2200, 4000)
rw = BASE.frf(fw, 1e4, n_modes=3, base_accel=G)
wcsv("S1b_frf_wideband.csv", ["freq_Hz", "V_V", "P_mW"],
     [[fw[i], abs(rw['V'][i]), rw['P'][i]*1e3] for i in range(len(fw))])

# ---------------- S2: power vs load at resonance ----------------
Ropt, Pmax, Rg, Pg, ft = BASE.optimal_load(base_accel=G)
wcsv("S2_power_vs_load.csv", ["R_ohm", "P_mW"],
     [[Rg[i], Pg[i]*1e3] for i in range(len(Rg))])
# also at open-circuit resonance
Ropt2, Pmax2, Rg2, Pg2, foc = BASE.optimal_load(
    f_target=BASE.open_circuit_freq(), base_accel=G)

# ---------------- S3: tip mass sweep ----------------
masses = np.linspace(0.0, 0.02, 41)     # 0..20 g
rows = []
for Mt in masses:
    h = PiezoHarvester(zeta=0.01, Mt=Mt)
    fsc = h.short_circuit_freq()
    Ropt_m, Pmax_m, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([Mt*1e3, fsc, Ropt_m, Pmax_m*1e3])
wcsv("S3_tipmass.csv",
     ["tip_mass_g", "f_sc_Hz", "R_opt_ohm", "P_max_mW"], rows)

# ---------------- S4: thickness ratio (geometry) ----------------
# keep total thickness fixed; vary substructure fraction
htot = 0.9e-3
fracs = np.linspace(0.2, 0.85, 40)
rows = []
for fr in fracs:
    hs = fr*htot; hp = (1-fr)*htot
    h = PiezoHarvester(hs=hs, hp=hp, zeta=0.01)
    fsc = h.short_circuit_freq()
    Ropt_t, Pmax_t, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([fr, hs*1e3, hp*1e3, fsc, Ropt_t, Pmax_t*1e3])
wcsv("S4_thickness.csv",
     ["sub_fraction", "hs_mm", "hp_mm", "f_sc_Hz", "R_opt_ohm", "P_max_mW"], rows)

# ---------------- S5: overhang length sweep ----------------
lengths = np.linspace(60e-3, 160e-3, 40)
rows = []
for L in lengths:
    h = PiezoHarvester(L=L, zeta=0.01)
    fsc = h.short_circuit_freq()
    Ropt_L, Pmax_L, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([L*1e3, fsc, Ropt_L, Pmax_L*1e3])
wcsv("S5_length.csv",
     ["length_mm", "f_sc_Hz", "R_opt_ohm", "P_max_mW"], rows)

# ---------------- S6: 2D map power(tip mass, R) ----------------
mass_g = np.linspace(0, 15e-3, 31)
Rgrid = np.logspace(2.5, 6, 60)
Z = np.zeros((len(mass_g), len(Rgrid)))
for i, Mt in enumerate(mass_g):
    h = PiezoHarvester(zeta=0.01, Mt=Mt)
    fsc = h.short_circuit_freq()
    for j, RL in enumerate(Rgrid):
        Z[i, j] = h.frf([fsc], RL, n_modes=3, base_accel=G)['P'][0]*1e3
np.savez(os.path.join(RES, "S6_map.npz"),
         mass_g=mass_g*1e3, Rgrid=Rgrid, P_mW=Z)
print("wrote S6_map.npz", Z.shape)

# ---------------- S7: nonlinear bistable frequency sweep ----------------
# Build single-mode parameters from the fundamental mode of BASE.
mc = BASE.modal_coupling(1)
wn = mc['wr'][0]; theta = abs(mc['theta'][0]); Cp = BASE.Cp
RLn = 1e5
x0 = 0.6e-3            # half-distance between potential wells [m]
params = dict(wn=wn, zeta=0.01, x0=x0, theta=theta, Cp=Cp, RL=RLn)

# linear reference FRF (single mode) at same RL for comparison
f_lin = np.linspace(20, 75, 120)
P_lin = []
for fi in f_lin:
    P_lin.append(BASE.frf([fi], RLn, n_modes=1, base_accel=0.5*G)['P'][0]*1e3)

# bistable: sweep up and sweep down (continuation to capture hysteresis)
f_sweep = np.linspace(20, 75, 60)
acc = 0.5*G
P_up = []; y_prev = [x0*0.9, 0.0, 0.0]
for fi in f_sweep:
    r = duffing_bistable(params, fi, acc, t_end=40.0, dt=2e-4, y0=y_prev)
    P_up.append(r['Prms']*1e3)
    y_prev = [r['X'][-1], 0.0, r['V'][-1]]
P_down = []; y_prev = [x0*0.9, 0.0, 0.0]
for fi in f_sweep[::-1]:
    r = duffing_bistable(params, fi, acc, t_end=40.0, dt=2e-4, y0=y_prev)
    P_down.append(r['Prms']*1e3)
    y_prev = [r['X'][-1], 0.0, r['V'][-1]]
P_down = P_down[::-1]

wcsv("S7_bistable_sweep.csv",
     ["freq_Hz", "P_up_mW", "P_down_mW"],
     [[f_sweep[i], P_up[i], P_down[i]] for i in range(len(f_sweep))])
wcsv("S7_linear_ref.csv", ["freq_Hz", "P_mW"],
     [[f_lin[i], P_lin[i]] for i in range(len(f_lin))])

# bandwidth metrics (half-power) for linear vs bistable
def bandwidth(fvec, pvec):
    pvec = np.asarray(pvec); fvec = np.asarray(fvec)
    pk = pvec.max(); thr = pk/2
    above = fvec[pvec >= thr]
    return (above.max()-above.min()) if len(above) > 1 else 0.0
bw_lin = bandwidth(f_lin, P_lin)
bw_bi  = bandwidth(f_sweep, np.maximum(P_up, P_down))

# ---------------- summary ----------------
summary = {
  "optimal_R_ohm_scres": float(Ropt),
  "peak_power_1g_mW_scres": float(Pmax*1e3),
  "resonance_sc_Hz": float(ft),
  "optimal_R_ohm_ocres": float(Ropt2),
  "peak_power_1g_mW_ocres": float(Pmax2*1e3),
  "resonance_oc_Hz": float(foc),
  "tipmass_max_power_mW": float(max(r[3] for r in
        __import__('csv'))) if False else None,
  "linear_halfpower_bandwidth_Hz": float(bw_lin),
  "bistable_halfpower_bandwidth_Hz": float(bw_bi),
  "bandwidth_gain_x": float(bw_bi/bw_lin) if bw_lin>0 else None,
}
# recompute clean tip-mass best power
import csv as _csv
with open(os.path.join(RES, "S3_tipmass.csv")) as fp:
    rd = list(_csv.reader(fp))[1:]
    pm = [(float(r[0]), float(r[3])) for r in rd]
    best = max(pm, key=lambda t: t[1])
    summary["tipmass_max_power_mW"] = best[1]
    summary["tipmass_at_max_power_g"] = best[0]
    summary["baseline_power_mW"] = pm[0][1]
    summary["tipmass_power_gain_x"] = best[1]/pm[0][1]

with open(os.path.join(RES, "summary.json"), "w") as fp:
    json.dump(summary, fp, indent=2)
print(json.dumps(summary, indent=2))
