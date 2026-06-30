"""Re-run the load-window-sensitive linear studies (S3 tip mass, S4
thickness, S5 length, S6 map) with the corrected resonance-tracking
window. Fast (no time integration)."""
import numpy as np, csv, os, json
from harvester_model import PiezoHarvester

RES = os.path.join(os.path.dirname(__file__), "..", "results")
G = 9.81

def wcsv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", name, len(rows))

# S3 tip mass
rows = []
for Mt in np.linspace(0.0, 0.02, 41):
    h = PiezoHarvester(zeta=0.01, Mt=Mt)
    fsc = h.short_circuit_freq()
    Ropt, Pmax, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([Mt*1e3, fsc, Ropt, Pmax*1e3])
wcsv("S3_tipmass.csv", ["tip_mass_g","f_sc_Hz","R_opt_ohm","P_max_mW"], rows)

# S4 thickness
htot = 0.9e-3; rows = []
for fr in np.linspace(0.2, 0.85, 40):
    hs = fr*htot; hp = (1-fr)*htot
    h = PiezoHarvester(hs=hs, hp=hp, zeta=0.01)
    fsc = h.short_circuit_freq()
    Ropt, Pmax, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([fr, hs*1e3, hp*1e3, fsc, Ropt, Pmax*1e3])
wcsv("S4_thickness.csv",
     ["sub_fraction","hs_mm","hp_mm","f_sc_Hz","R_opt_ohm","P_max_mW"], rows)

# S5 length
rows = []
for L in np.linspace(60e-3, 160e-3, 40):
    h = PiezoHarvester(L=L, zeta=0.01)
    fsc = h.short_circuit_freq()
    Ropt, Pmax, *_ = h.optimal_load(f_target=fsc, base_accel=G)
    rows.append([L*1e3, fsc, Ropt, Pmax*1e3])
wcsv("S5_length.csv", ["length_mm","f_sc_Hz","R_opt_ohm","P_max_mW"], rows)

# S6 map
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
print("wrote S6_map.npz")

# refresh summary tip-mass numbers
with open(os.path.join(RES, "S3_tipmass.csv")) as fp:
    rd = list(csv.reader(fp))[1:]
    pm = [(float(r[0]), float(r[1]), float(r[3])) for r in rd]
best = max(pm, key=lambda t: t[2])
out = {"tipmass_at_max_power_g": best[0], "f_sc_at_best_Hz": best[1],
       "tipmass_max_power_mW": best[2], "baseline_power_mW": pm[0][2],
       "tipmass_power_gain_x": best[2]/pm[0][2]}
with open(os.path.join(RES, "S3_summary.json"), "w") as fp:
    json.dump(out, fp, indent=2)
print(json.dumps(out, indent=2))
