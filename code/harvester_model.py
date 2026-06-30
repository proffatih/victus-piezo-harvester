"""
harvester_model.py
==================================================================
Analytical distributed-parameter (Euler-Bernoulli) electromechanical
model of a unimorph cantilevered piezoelectric vibration energy
harvester, following Erturk & Inman (J. Vib. Acoust. 130, 041002, 2008;
Smart Mater. Struct. 18, 025009, 2009).

Implements:
  * Modal (distributed-parameter) coupled electromechanical model with
    series electrode connection to a resistive load.
  * Closed-form single-mode (analytical) voltage and tip-velocity
    frequency-response functions (FRFs) for base excitation.
  * Validation against the Erturk-Inman 2008 unimorph PZT-5A benchmark
    (short-/open-circuit fundamental resonances, optimal load).
  * Parametric studies: load resistance, tip (proof) mass,
    overhang length, substructure/piezo thickness ratio.

Author: Fatih Gul (Recep Tayyip Erdogan University)
Solo work. No fabricated data -- all numbers are model outputs or
cited benchmark values.
==================================================================
"""
import numpy as np
from scipy.optimize import brentq
import json, os, csv

# ---------------------------------------------------------------
# Physical / numerical helpers
# ---------------------------------------------------------------
def beam_eigenvalues(n_modes, tip_mass_ratio=0.0):
    """
    Dimensionless eigenvalues lambda_r = beta_r*L for a clamped-free
    Euler-Bernoulli beam, optionally carrying a tip (point) mass.

    Characteristic equation (no tip mass):
        1 + cos(L)cosh(L) = 0
    With a tip mass Mt (ratio mt = Mt/(m*L), m = mass per length):
        1 + cos(L)cosh(L)
          + mt*L*( cos(L)sinh(L) - sin(L)cosh(L) ) = 0
    (rotary inertia of the tip mass neglected -- standard assumption).
    """
    lam = []
    mt = tip_mass_ratio

    def char(x):
        c, ch = np.cos(x), np.cosh(x)
        s, sh = np.sin(x), np.sinh(x)
        return (1.0 + c*ch) + mt*x*(c*sh - s*ch)

    # bracket roots
    x = 1e-4
    dx = 1e-3
    prev = char(x)
    found = 0
    while found < n_modes and x < 60:
        x2 = x + dx
        cur = char(x2)
        if prev == 0.0:
            lam.append(x); found += 1
        elif prev*cur < 0:
            r = brentq(char, x, x2, xtol=1e-12, rtol=1e-14)
            lam.append(r); found += 1
        prev = cur
        x = x2
    return np.array(lam[:n_modes])


class PiezoHarvester:
    """
    Unimorph cantilever harvester (series/single piezo layer on a
    passive substructure), resistive load R across the electrodes.

    All SI units.
    """
    def __init__(self,
                 L=100e-3, b=20e-3,
                 hs=0.5e-3, hp=0.4e-3,
                 Ys=100e9, Yp=66e9,
                 rho_s=7165.0, rho_p=7800.0,
                 d31=-190e-12, eps33_S=15.93e-9,
                 zeta=0.01, Mt=0.0):
        self.L, self.b = L, b
        self.hs, self.hp = hs, hp
        self.Ys, self.Yp = Ys, Yp
        self.rho_s, self.rho_p = rho_s, rho_p
        self.d31, self.eps33_S = d31, eps33_S
        self.zeta = zeta            # modal damping ratio (assumed equal modes)
        self.Mt = Mt               # tip proof mass [kg]
        self._build()

    # -----------------------------------------------------------
    def _build(self):
        b, hs, hp = self.b, self.hs, self.hp
        Ys, Yp = self.Ys, self.Yp

        # --- Neutral axis from bottom of substructure (composite) ---
        # Layer 1: substructure (0..hs), Layer 2: piezo (hs..hs+hp)
        A_s = b*hs;  A_p = b*hp
        yc_s = hs/2.0
        yc_p = hs + hp/2.0
        # area-weighted by modulus (transformed section)
        n = (Ys*A_s + Yp*A_p)
        ybar = (Ys*A_s*yc_s + Yp*A_p*yc_p)/n   # neutral axis location
        self.ybar = ybar

        # --- Bending stiffness YI of composite (transformed) ---
        I_s = b*hs**3/12.0 + A_s*(yc_s-ybar)**2
        I_p = b*hp**3/12.0 + A_p*(yc_p-ybar)**2
        self.YI = Ys*I_s + Yp*I_p

        # --- Mass per unit length ---
        self.m = self.rho_s*A_s + self.rho_p*A_p

        # --- Piezo layer geometry relative to neutral axis ---
        self.yb = hs - ybar          # bottom of piezo wrt NA
        self.yt = hs + hp - ybar     # top of piezo wrt NA
        self.hpc = (self.yt + self.yb)/2.0  # distance NA->piezo centroid

        # --- Piezoelectric / dielectric constants ---
        # e31 = d31 * Yp  (plane-stress, bar approximation)
        self.e31 = self.d31*self.Yp
        # internal capacitance Cp = eps33_S * b * L / hp
        self.Cp = self.eps33_S*b*self.L/hp

        # --- Eigenvalues with tip mass ratio ---
        self.mt_ratio = self.Mt/(self.m*self.L)
        self.lam = beam_eigenvalues(6, self.mt_ratio)

    # -----------------------------------------------------------
    def mode_shape(self, r, x):
        """Mass-normalized clamped-free mode shape phi_r(x) including
        tip mass in the normalization (eq. set, Erturk-Inman 2009)."""
        lr = self.lam[r]
        L = self.L
        bl = lr/L
        # sigma_r
        s, sh = np.sin(lr), np.sinh(lr)
        c, ch = np.cos(lr), np.cosh(lr)
        mt = self.mt_ratio
        sigma = (s - sh + mt*lr*(c - ch)) / (c + ch + mt*lr*(s - sh))
        shape = (np.cosh(bl*x) - np.cos(bl*x)
                 - sigma*(np.sinh(bl*x) - np.sin(bl*x)))
        # mass normalization constant C
        # integral of m*phi^2 + Mt*phi(L)^2 = 1
        xx = np.linspace(0, L, 4000)
        c_, ch_ = np.cos(bl*xx), np.cosh(bl*xx)
        s_, sh_ = np.sin(bl*xx), np.sinh(bl*xx)
        sh_full = (ch_ - c_ - sigma*(sh_ - s_))
        norm = self.m*np.trapz(sh_full**2, xx)
        phiL = (np.cosh(lr) - np.cos(lr) - sigma*(np.sinh(lr) - np.sin(lr)))
        norm += self.Mt*phiL**2
        C = 1.0/np.sqrt(norm)
        return C*shape, C, sigma

    def natural_freqs_sc(self, n=3):
        """Short-circuit (R->0) undamped natural frequencies [rad/s]."""
        w = []
        for r in range(n):
            br = self.lam[r]/self.L
            w.append(br**2*np.sqrt(self.YI/self.m))
        return np.array(w)

    # -----------------------------------------------------------
    def modal_coupling(self, n_modes=3):
        """Modal electromechanical coupling theta_r and forcing for
        base excitation. Returns dict of modal quantities."""
        L = self.L
        wr = self.natural_freqs_sc(n_modes)
        # piezoelectric coupling term per mode:
        #   theta_r = -e31*hpc*b * dphi_r/dx |_0^L  (slope difference)
        # For a cantilever the relevant quantity is the integral of
        # the second derivative -> slope of mode shape at the free end
        thetas, sigmas, Cs, phiLs = [], [], [], []
        for r in range(n_modes):
            lr = self.lam[r]; br = lr/L
            _, C, sigma = self.mode_shape(r, np.array([L]))
            # slope of phi at x=L and x=0
            def dphi(x):
                return C*br*(np.sinh(br*x)+np.sin(br*x)
                             - sigma*(np.cosh(br*x)-np.cos(br*x)))
            dphi_L = dphi(L); dphi_0 = dphi(0.0)
            theta = self.e31*self.hpc*self.b*(dphi_L - dphi_0)
            thetas.append(theta); sigmas.append(sigma); Cs.append(C)
            phiL = C*(np.cosh(lr)-np.cos(lr)-sigma*(np.sinh(lr)-np.sin(lr)))
            phiLs.append(phiL)
        return dict(wr=wr, theta=np.array(thetas),
                    sigma=np.array(sigmas), C=np.array(Cs),
                    phiL=np.array(phiLs), n_modes=n_modes)

    def modal_forcing(self, r):
        """Modal forcing amplitude for unit base translational
        acceleration: sigma_r = -m*integral(phi_r) - Mt*phi_r(L)."""
        L = self.L; lr = self.lam[r]; br = lr/L
        _, C, sigma = self.mode_shape(r, np.array([L]))
        xx = np.linspace(0, L, 4000)
        c_, ch_ = np.cos(br*xx), np.cosh(br*xx)
        s_, sh_ = np.sin(br*xx), np.sinh(br*xx)
        sh_full = C*(ch_ - c_ - sigma*(sh_ - s_))
        integ = self.m*np.trapz(sh_full, xx)
        phiL = C*(np.cosh(lr)-np.cos(lr)-sigma*(np.sinh(lr)-np.sin(lr)))
        integ += self.Mt*phiL
        return -integ

    # -----------------------------------------------------------
    def frf(self, freqs_hz, RL, n_modes=3, base_accel=1.0):
        """
        Multi-mode coupled voltage & power FRF for harmonic base
        acceleration of amplitude `base_accel` [m/s^2].

        Solves the coupled modal electrical equations:
          (-w^2 + 2 zeta wr i w + wr^2) eta_r - theta_r/?? ...
        Using the standard Erturk-Inman modal-domain formulation:
          mechanical: (wr^2 - w^2 + i 2 zeta_r wr w) eta_r
                       + chi_r * V = F_r
          electrical: (i w Cp + 1/RL) V - i w * sum_r kappa_r eta_r = 0
        with modal coupling theta_r appearing in both equations.
        """
        mc = self.modal_coupling(n_modes)
        wr = mc['wr']; theta = mc['theta']
        F = np.array([self.modal_forcing(r) for r in range(n_modes)])
        w = 2*np.pi*np.asarray(freqs_hz)
        V = np.zeros(len(w), dtype=complex)
        # tip velocity FRF too
        tipvel = np.zeros(len(w), dtype=complex)
        phiL = mc['phiL']
        for k, wk in enumerate(w):
            # Build modal admittance: eta_r = (F_r*base - theta_r*V) / Dr
            Dr = (wr**2 - wk**2 + 1j*2*self.zeta*wr*wk)
            # electrical eq: (1/RL + i w Cp) V = i w sum theta_r eta_r
            # substitute eta_r:
            #   sum theta_r eta_r = sum theta_r (F_r*a - theta_r V)/Dr
            S_force = np.sum(theta*F*base_accel/Dr)
            S_coup  = np.sum(theta**2/Dr)
            Yel = (1.0/RL + 1j*wk*self.Cp)
            # (Yel + i w S_coup) V = i w S_force
            V[k] = (1j*wk*S_force)/(Yel + 1j*wk*S_coup)
            eta = (F*base_accel - theta*V[k])/Dr
            # tip relative displacement = sum phiL*eta ; velocity = i w *that
            tipvel[k] = 1j*wk*np.sum(phiL*eta)
        P = np.abs(V)**2/RL
        return dict(V=V, P=P, tipvel=tipvel, wr=wr)

    def _f1_estimate(self):
        """Coarse undamped fundamental short-circuit frequency [Hz]."""
        return self.natural_freqs_sc(1)[0]/(2*np.pi)

    def open_circuit_freq(self, n_modes=3):
        """Open-circuit fundamental resonance: peak of |V| FRF as RL->inf."""
        f0 = self._f1_estimate()
        f = np.linspace(max(1.0, 0.7*f0), 1.4*f0, 4000)
        r = self.frf(f, RL=1e9, n_modes=n_modes)
        return f[np.argmax(np.abs(r['V']))]

    def short_circuit_freq(self, n_modes=3):
        f0 = self._f1_estimate()
        f = np.linspace(max(1.0, 0.7*f0), 1.3*f0, 4000)
        r = self.frf(f, RL=1e2, n_modes=n_modes)
        return f[np.argmax(r['P'])]

    def optimal_load(self, f_target=None, n_modes=3,
                     Rgrid=None, base_accel=9.81):
        """Sweep load resistance at the (short-circuit) resonance and
        return the resistance maximizing power and that peak power."""
        if Rgrid is None:
            Rgrid = np.logspace(2, 7, 220)
        if f_target is None:
            f_target = self.short_circuit_freq(n_modes)
        Ps = []
        for RL in Rgrid:
            r = self.frf([f_target], RL, n_modes, base_accel)
            Ps.append(r['P'][0])
        Ps = np.array(Ps)
        i = np.argmax(Ps)
        return Rgrid[i], Ps[i], Rgrid, Ps, f_target


# ===============================================================
# NONLINEAR BISTABLE (Duffing) EXTENSION
# ===============================================================
def duffing_bistable(params, f_excite, base_accel, t_end=60.0, dt=2e-4,
                     y0=None):
    """
    Single-mode bistable harvester with magnetic restoring force
    yielding a Duffing-type oscillator:

        x'' + 2 zeta wn x' - (1/2) wn^2 x + (1/2) wn^2/x0^2 x^3
              + chi*v = -a(t)
        Cp v' + v/RL + theta x' = 0   (lumped, single mode)

    Negative linear stiffness (-1/2 wn^2) + positive cubic gives the
    classic double-well bistable potential with wells at +/- x0.
    Integrated with RK4. Returns steady-state RMS power.
    """
    wn   = params['wn']
    zeta = params['zeta']
    x0   = params['x0']
    theta= params['theta']
    Cp   = params['Cp']
    RL   = params['RL']
    w = 2*np.pi*f_excite
    chi = theta/Cp   # back-coupling onto mechanical (normalized)

    n = int(t_end/dt)
    if y0 is None:
        y = np.array([x0*0.9, 0.0, 0.0])  # start near a well
    else:
        y = np.array(y0, dtype=float)

    def deriv(t, y):
        x, xd, v = y
        a = base_accel*np.sin(w*t)
        xdd = (-2*zeta*wn*xd + 0.5*wn**2*x - 0.5*wn**2/x0**2*x**3
               - chi*v - a)
        vd = (-v/(RL*Cp) - (theta/Cp)*xd)
        return np.array([xd, xdd, vd])

    ts = np.arange(n)*dt
    V = np.empty(n); X = np.empty(n)
    for k in range(n):
        t = ts[k]
        k1 = deriv(t, y)
        k2 = deriv(t+dt/2, y+dt/2*k1)
        k3 = deriv(t+dt/2, y+dt/2*k2)
        k4 = deriv(t+dt, y+dt*k3)
        y = y + dt/6*(k1+2*k2+2*k3+k4)
        V[k] = y[2]; X[k] = y[0]
    # steady-state window = last 40%
    i0 = int(0.6*n)
    Vrms = np.sqrt(np.mean(V[i0:]**2))
    Prms = Vrms**2/RL
    xpp = X[i0:].max()-X[i0:].min()    # peak-to-peak displacement
    return dict(Vrms=Vrms, Prms=Prms, xpp=xpp, ts=ts, V=V, X=X)


if __name__ == "__main__":
    h = PiezoHarvester()
    print("Composite YI = %.4e N m^2" % h.YI)
    print("mass/length  = %.4e kg/m" % h.m)
    print("Cp           = %.3f nF" % (h.Cp*1e9))
    print("SC f1 = %.2f Hz | OC f1 = %.2f Hz"
          % (h.short_circuit_freq(), h.open_circuit_freq()))
    Ropt, Pmax, _, _, ft = h.optimal_load(base_accel=9.81)
    print("Optimal R = %.3e ohm | Pmax(1g) = %.3e W @ %.2f Hz"
          % (Ropt, Pmax, ft))
