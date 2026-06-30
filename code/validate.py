"""
validate.py
==================================================================
Validation of the distributed-parameter harvester model.

(A) Geometric / modal validation against the Erturk-Inman (2008)
    unimorph PZT-5A benchmark: composite bending stiffness, mass per
    length, fundamental short-circuit resonance (target 47.8 Hz).

(B) Internal analytical validation: the multi-mode coupled voltage &
    power FRFs must collapse onto the CLOSED-FORM single-mode
    analytical solution of Erturk & Inman near the fundamental
    resonance. This is an exact self-consistency benchmark.

(C) Matched-impedance check: optimal load near 1/(omega_1 Cp).

Outputs a JSON record to results/validation.json
==================================================================
"""
import numpy as np, json, os
from harvester_model import PiezoHarvester

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

h = PiezoHarvester(zeta=0.01)
mc = h.modal_coupling(3)

# ---------------- (A) geometric / modal ----------------
f_sc = h.short_circuit_freq()
f_oc = h.open_circuit_freq()
benchmark_sc = 47.8     # Hz, Erturk-Inman 2008 Table 2
err_sc = 100*abs(f_sc-benchmark_sc)/benchmark_sc

# ---------------- (B) closed-form single-mode FRF ----------------
# Single-mode analytical voltage FRF (Erturk-Inman closed form):
#   V/(-w^2 Y0) = (-i w theta_1 phi'... ) ; we build the single-mode
#   coupled solution explicitly and compare to the multimode solver.
def closed_form_single_mode(freqs_hz, RL, base_accel=1.0):
    wr = mc['wr'][0]; th = mc['theta'][0]; Cp = h.Cp
    F1 = h.modal_forcing(0)
    phiL = mc['phiL'][0]
    w = 2*np.pi*np.asarray(freqs_hz)
    Dr = (wr**2 - w**2 + 1j*2*h.zeta*wr*w)
    Yel = 1.0/RL + 1j*w*Cp
    V = (1j*w*th*F1*base_accel/Dr)/(Yel + 1j*w*th**2/Dr)
    P = np.abs(V)**2/RL
    return V, P

f = np.linspace(44, 52, 1600)
RL = 1e4
Vmm = h.frf(f, RL, n_modes=1, base_accel=9.81)['V']
Vcf, _ = closed_form_single_mode(f, RL, base_accel=9.81)
rel_err = np.max(np.abs(Vmm-Vcf))/np.max(np.abs(Vcf))

# 3-mode vs 1-mode difference near f1 (higher-mode contribution)
V3 = h.frf(f, RL, n_modes=3, base_accel=9.81)['V']
mode_corr = np.max(np.abs(V3-Vmm))/np.max(np.abs(V3))

# ---------------- (C) matched impedance ----------------
Ropt, Pmax, Rg, Pg, ft = h.optimal_load(base_accel=9.81)
R_match = 1.0/(2*np.pi*f_sc*h.Cp)

rec = {
 "benchmark": "Erturk & Inman (2008) unimorph PZT-5A, JVA 130 041002",
 "composite_YI_Nm2": h.YI,
 "mass_per_length_kg_m": h.m,
 "Cp_nF": h.Cp*1e9,
 "f_sc_model_Hz": f_sc,
 "f_sc_benchmark_Hz": benchmark_sc,
 "f_sc_rel_error_pct": err_sc,
 "f_oc_model_Hz": f_oc,
 "closed_form_single_mode_relerr": rel_err,
 "three_mode_vs_one_mode_relerr": mode_corr,
 "optimal_R_ohm": Ropt,
 "peak_power_at_1g_mW": Pmax*1e3,
 "matched_impedance_R_ohm": R_match,
 "resonance_Hz": ft,
}
with open(os.path.join(OUT, "validation.json"), "w") as fp:
    json.dump(rec, fp, indent=2)

print(json.dumps(rec, indent=2))
print("\nVALIDATION SUMMARY")
print(" SC freq error vs Erturk-Inman: %.3f %%" % err_sc)
print(" Closed-form single-mode FRF match: max rel err = %.2e" % rel_err)
print(" Higher-mode correction at f1: %.2e" % mode_corr)
print(" Optimal R = %.3e ohm  (matched 1/wCp = %.3e ohm)" % (Ropt, R_match))
