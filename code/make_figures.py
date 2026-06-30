"""
make_figures.py
==================================================================
Publication-quality figures (vector PDF + 300 dpi PNG), colour-blind
safe palette, labelled units. Reads CSV/NPZ from ../results/ and the
analytical model for mode shapes and the schematic.
==================================================================
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from harvester_model import PiezoHarvester

RES = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG, exist_ok=True)

# ---- global style ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})
# Wong colour-blind-safe palette
CB = ["#000000", "#0072B2", "#D55E00", "#009E73",
      "#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]

def loadcsv(name):
    with open(os.path.join(RES, name)) as f:
        r = list(csv.reader(f))
    hdr = r[0]; data = np.array(r[1:], dtype=float)
    return hdr, data

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)

# ============================================================
# Fig 1 -- device schematic
# ============================================================
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(-2.2, 11); ax.set_ylim(-2.4, 2.6); ax.axis("off")
    # clamp / base
    ax.add_patch(Rectangle((-2.0, -2.2), 1.4, 4.4, fc="#888888", ec="k"))
    for yy in np.linspace(-2.1, 2.1, 9):
        ax.plot([-2.0, -2.3], [yy, yy-0.25], color="k", lw=0.8)
    # substructure (brass)
    ax.add_patch(Rectangle((-0.6, -0.18), 9.4, 0.36, fc="#d9b46a",
                            ec="k", label="substructure"))
    # piezo layer
    ax.add_patch(Rectangle((-0.6, 0.18), 9.4, 0.42, fc="#5a8fd6", ec="k"))
    # electrodes
    ax.plot([-0.6, 8.8], [0.62, 0.62], color="k", lw=2.2)
    ax.plot([-0.6, 8.8], [0.16, 0.16], color="k", lw=2.2)
    # tip mass
    ax.add_patch(Rectangle((8.8, -0.55), 0.9, 1.2, fc="#444444", ec="k"))
    ax.text(9.25, 0.95, r"$M_t$", ha="center", fontsize=11)
    # load resistor
    ax.plot([8.8, 10.3], [0.62, 0.62], color="k", lw=1.4)
    ax.plot([8.8, 10.3], [0.16, 0.16], color="k", lw=1.4)
    ax.plot([10.3, 10.3], [0.16, 0.62], color="k", lw=1.4)
    ax.add_patch(Rectangle((10.15, 0.28), 0.3, 0.22, fc="white", ec="k"))
    ax.text(10.75, 0.39, r"$R_L$", va="center", fontsize=11)
    # base excitation arrow
    ax.add_patch(FancyArrowPatch((-1.3, -1.6), (-1.3, -0.6),
                 arrowstyle="<->", mutation_scale=14, color=CB[2]))
    ax.text(-1.0, -1.1, r"$\ddot{w}_b(t)$", color=CB[2], fontsize=11)
    # labels
    ax.text(4.0, 0.95, "PZT piezoceramic layer", ha="center",
            color=CB[1], fontsize=10)
    ax.text(4.0, -0.62, "metallic substructure", ha="center",
            color="#a07b1f", fontsize=10)
    ax.text(4.0, -1.7, r"length $L$,  width $b$", ha="center", fontsize=10)
    ax.annotate("", xy=(8.8, -1.3), xytext=(-0.6, -1.3),
                arrowprops=dict(arrowstyle="<->", lw=0.9))
    save(fig, "fig1_schematic")

# ============================================================
# Fig 2 -- voltage & power FRF vs frequency (several loads)
# ============================================================
def fig_frf():
    hdr, d = loadcsv("S1_frf_vs_freq.csv")
    f = d[:, 0]
    loads = [1e3, 4.4e3, 1e4, 4.18e4, 1e6]
    labels = [r"$1\,\mathrm{k\Omega}$", r"$4.4\,\mathrm{k\Omega}$ (opt.)",
              r"$10\,\mathrm{k\Omega}$", r"$41.8\,\mathrm{k\Omega}$",
              r"$1\,\mathrm{M\Omega}$"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    for i, lab in enumerate(labels):
        V = d[:, 1 + 2*i]; P = d[:, 2 + 2*i]
        a1.plot(f, V, color=CB[i], label=lab)
        a2.plot(f, P, color=CB[i], label=lab)
    a1.set_xlabel("Frequency (Hz)"); a1.set_ylabel(r"$|V|$ (V) per 1$g$")
    a2.set_xlabel("Frequency (Hz)"); a2.set_ylabel("Power (mW) per 1$g$")
    a1.set_title("(a) Voltage FRF"); a2.set_title("(b) Power FRF")
    a1.legend(fontsize=8); a2.legend(fontsize=8)
    a1.set_xlim(f.min(), f.max()); a2.set_xlim(f.min(), f.max())
    fig.tight_layout()
    save(fig, "fig2_frf_frequency")

# ============================================================
# Fig 3 -- power vs load resistance (optimal R)
# ============================================================
def fig_load():
    hdr, d = loadcsv("S2_power_vs_load.csv")
    R = d[:, 0]; P = d[:, 1]
    iopt = np.argmax(P)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.semilogx(R, P, color=CB[1])
    ax.plot(R[iopt], P[iopt], "o", color=CB[2], ms=8)
    ax.annotate(r"$R_{\mathrm{opt}}=%.1f\,\mathrm{k\Omega}$,  $P=%.1f\,\mathrm{mW}$"
                % (R[iopt]/1e3, P[iopt]),
                xy=(R[iopt], P[iopt]), xytext=(R[iopt]*1.3, P[iopt]*0.72),
                arrowprops=dict(arrowstyle="->", color=CB[2]), fontsize=9)
    ax.set_xlabel(r"Load resistance $R_L$ ($\Omega$)")
    ax.set_ylabel("Power (mW) at resonance, 1$g$")
    fig.tight_layout()
    save(fig, "fig3_power_vs_load")

# ============================================================
# Fig 4 -- tip mass: resonance + power tuning
# ============================================================
def fig_tipmass():
    hdr, d = loadcsv("S3_tipmass.csv")
    m = d[:, 0]; fsc = d[:, 1]; P = d[:, 3]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    l1, = ax.plot(m, fsc, color=CB[1], label="resonance")
    ax.set_xlabel("Tip (proof) mass (g)")
    ax.set_ylabel("Short-circuit resonance (Hz)", color=CB[1])
    ax.tick_params(axis="y", labelcolor=CB[1])
    ax2 = ax.twinx(); ax2.grid(False)
    l2, = ax2.plot(m, P, color=CB[2], label="peak power")
    ax2.set_ylabel("Peak power (mW), 1$g$", color=CB[2])
    ax2.tick_params(axis="y", labelcolor=CB[2])
    ax.legend([l1, l2], ["resonance", "peak power"], fontsize=9, loc="center right")
    fig.tight_layout()
    save(fig, "fig4_tipmass")

# ============================================================
# Fig 5 -- geometry: thickness ratio & length
# ============================================================
def fig_geometry():
    _, dt = loadcsv("S4_thickness.csv")
    _, dl = loadcsv("S5_length.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    fr = dt[:, 0]; Pt = dt[:, 5]
    iopt = np.argmax(Pt)
    a1.plot(fr, Pt, color=CB[1])
    a1.plot(fr[iopt], Pt[iopt], "o", color=CB[2], ms=7)
    a1.set_xlabel(r"Substructure thickness fraction $h_s/h_{\mathrm{tot}}$")
    a1.set_ylabel("Peak power (mW), 1$g$")
    a1.set_title("(a) Thickness ratio ($h_{\\mathrm{tot}}=0.9$ mm)")
    a1.annotate(r"opt. $\approx$%.2f" % fr[iopt],
                xy=(fr[iopt], Pt[iopt]), xytext=(fr[iopt]+0.05, Pt[iopt]*0.8),
                arrowprops=dict(arrowstyle="->", color=CB[2]), fontsize=9)
    L = dl[:, 0]; PL = dl[:, 3]; fL = dl[:, 1]
    a2.plot(L, PL, color=CB[1], label="peak power")
    a2.set_xlabel("Overhang length (mm)")
    a2.set_ylabel("Peak power (mW), 1$g$", color=CB[1])
    a2.tick_params(axis="y", labelcolor=CB[1])
    a2b = a2.twinx(); a2b.grid(False)
    a2b.plot(L, fL, color=CB[2], ls="--")
    a2b.set_ylabel("Resonance (Hz)", color=CB[2])
    a2b.tick_params(axis="y", labelcolor=CB[2])
    a2.set_title("(b) Overhang length")
    fig.tight_layout()
    save(fig, "fig5_geometry")

# ============================================================
# Fig 6 -- mode shapes
# ============================================================
def fig_modeshapes():
    h = PiezoHarvester()
    x = np.linspace(0, h.L, 400)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for r in range(3):
        phi, _, _ = h.mode_shape(r, x)
        phi = phi/np.max(np.abs(phi))
        fr = h.natural_freqs_sc(3)[r]/2/np.pi
        ax.plot(x*1e3, phi, color=CB[r+1],
                label=r"mode %d ($f_%d=%.1f$ Hz)" % (r+1, r+1, fr))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Position along beam (mm)")
    ax.set_ylabel("Normalized mode shape $\\phi_r(x)$")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "fig6_modeshapes")

# ============================================================
# Fig 7 -- 2D optimization map power(tip mass, R)
# ============================================================
def fig_map():
    d = np.load(os.path.join(RES, "S6_map.npz"))
    m = d["mass_g"]; R = d["Rgrid"]; Z = d["P_mW"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cf = ax.contourf(R, m, Z, levels=30, cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel(r"Load resistance $R_L$ ($\Omega$)")
    ax.set_ylabel("Tip mass (g)")
    cb = fig.colorbar(cf, ax=ax); cb.set_label("Power (mW), 1$g$ at resonance")
    # mark global max
    i, j = np.unravel_index(np.argmax(Z), Z.shape)
    ax.plot(R[j], m[i], "*", color="red", ms=15)
    fig.tight_layout()
    save(fig, "fig7_optimization_map")

# ============================================================
# Fig 8 -- nonlinear bistable broadband sweep + hysteresis
# ============================================================
def fig_bistable():
    # columns: Omega, P_up_norm, P_down_norm, P_linear_norm, xpp_up, xpp_down
    _, ds = loadcsv("S7_bistable_sweep.csv")
    O = ds[:, 0]; Pu = ds[:, 1]; Pd = ds[:, 2]; Pl = ds[:, 3]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(O, Pl, color=CB[0], ls=":", lw=2.0,
            label="linear (single-well)")
    ax.plot(O, Pu, "-o", color=CB[1], ms=3.5, label="bistable, sweep up")
    ax.plot(O, Pd, "-s", color=CB[2], ms=3.5, label="bistable, sweep down")
    ax.set_xlabel(r"Normalized excitation frequency $\Omega$")
    ax.set_ylabel("Power normalized to linear peak")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Broadband snap-through orbit vs linear resonance")
    fig.tight_layout()
    save(fig, "fig8_bistable")

if __name__ == "__main__":
    fig_schematic()
    fig_frf()
    fig_load()
    fig_tipmass()
    fig_geometry()
    fig_modeshapes()
    fig_map()
    fig_bistable()
    print("ALL FIGURES DONE")
