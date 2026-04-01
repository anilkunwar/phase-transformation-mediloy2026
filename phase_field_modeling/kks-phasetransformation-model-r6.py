# =============================================================================
# MEDILOY γ-FCC → ε-HCP PHASE TRANSFORMATION (KKS + MOELANS + ALLEN‑CAHN)
# =============================================================================
# - η (HCP order parameter) evolves via Allen‑Cahn (non‑conserved)
# - c (Co concentration) evolves via Cahn‑Hilliard (conserved)
# - Kim‑Kim‑Suzuki (KKS) condition: equal chemical potentials at interface
# - Moelans interpolation: φ_ε = (1‑η)² / (η² + (1‑η)²)
# - Parabolic phase free energies: f^α(c) = ½ K_α (c – c_α^eq)² + E_α
# - Temperature: 950°C (1223.15 K) – pseudo‑binary Co–M_y model
# =============================================================================

import numpy as np
from numba import njit, prange
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
from io import BytesIO

# -----------------------------------------------------------------------------
# PHYSICAL SCALES
# -----------------------------------------------------------------------------

class PhysicalScalesMediloy:
    """Physical unit conversion for phase‑field simulations."""
    def __init__(self, T_celsius=950.0, V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        self.R = 8.314462618          # J/(mol·K)
        self.T = T_celsius + 273.15   # K
        self.T_celsius = T_celsius
        self.V_m = V_m_m3mol          # m³/mol
        self.D_b = D_b_m2s            # m²/s (Cr diffusion in Co matrix)

        self.L0 = 2.0e-9              # m – reference length (interface width)
        delta_G_mol = 400.0           # J/mol – chemical driving force at 950°C
        self.E0 = delta_G_mol / self.V_m   # J/m³ – energy density scale

        self.t0 = self.L0**2 / self.D_b    # s – diffusion time scale
        self.M0 = self.D_b / self.E0       # m⁵/(J·s) – chemical mobility scale
        self.M0_eta = self.M0 * 10.0       # (kept for compatibility, not used directly)

        print(f"Mediloy scales initialized at {self.T_celsius}°C ({self.T:.1f} K)")
        print(f"  L0 = {self.L0*1e9:.2f} nm, t0 = {self.t0:.2e} s")
        print(f"  E0 = {self.E0:.2e} J/m³, M0 = {self.M0:.2e} m⁵/(J·s)")

    def dim_to_phys(self, W_dim, kappa_c_dim, kappa_eta_dim, M_c_dim,
                    L_struct_dim, dt_dim, dx_dim=1.0):
        W_phys = W_dim * self.E0
        kappa_c_phys = kappa_c_dim * self.E0 * self.L0**2
        kappa_eta_phys = kappa_eta_dim * self.E0 * self.L0**2
        M_c_phys = M_c_dim * self.M0
        # Allen‑Cahn mobility L: units m³/(J·s)  -> scaling = L0²/(t0·E0)
        L_struct_phys = L_struct_dim * (self.L0**2 / (self.t0 * self.E0))
        dt_phys = dt_dim * self.t0
        dx_phys = dx_dim * self.L0
        return (W_phys, kappa_c_phys, kappa_eta_phys,
                M_c_phys, L_struct_phys, dt_phys, dx_phys)

    def phys_to_interface_width(self, kappa_phys, W_phys):
        if W_phys <= 0 or kappa_phys <= 0:
            return 2.0e-9
        return np.sqrt(kappa_phys / W_phys)

    def format_time(self, t_seconds):
        if not np.isfinite(t_seconds) or t_seconds < 0:
            return "0 s"
        if t_seconds < 1e-9: return f"{t_seconds*1e12:.2f} ps"
        elif t_seconds < 1e-6: return f"{t_seconds*1e9:.2f} ns"
        elif t_seconds < 1e-3: return f"{t_seconds*1e6:.2f} μs"
        elif t_seconds < 1.0: return f"{t_seconds*1e3:.2f} ms"
        elif t_seconds < 60: return f"{t_seconds:.3f} s"
        elif t_seconds < 3600: return f"{t_seconds/60:.2f} min"
        elif t_seconds < 86400: return f"{t_seconds/3600:.3f} h"
        else: return f"{t_seconds/86400:.3f} d"

    def format_length(self, L_meters):
        if not np.isfinite(L_meters) or L_meters < 0:
            return "0 nm"
        if L_meters < 1e-10: return f"{L_meters*1e12:.2f} pm"
        elif L_meters < 1e-9: return f"{L_meters*1e10:.2f} Å"
        elif L_meters < 1e-6: return f"{L_meters*1e9:.2f} nm"
        elif L_meters < 1e-3: return f"{L_meters*1e6:.2f} μm"
        elif L_meters < 1.0: return f"{L_meters*1e3:.2f} mm"
        else: return f"{L_meters:.3f} m"


# =============================================================================
# NUMBA KERNELS – KKS + MOELANS + ALLEN‑CAHN
# =============================================================================

@njit(fastmath=True, cache=True)
def _clip_scalar(val, lo, hi):
    if val < lo:
        return lo
    elif val > hi:
        return hi
    return val

@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian_2d(field, dx):
    nx, ny = field.shape
    lap = np.empty_like(field)
    for i in prange(nx):
        for j in prange(ny):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            lap[i, j] = (field[ip1, j] + field[im1, j] +
                         field[i, jp1] + field[i, jm1] -
                         4.0 * field[i, j]) / (dx * dx)
    return lap

@njit(parallel=True, fastmath=True, cache=True)
def compute_gradient_divergence_2d(flux_x, flux_y, dx):
    nx, ny = flux_x.shape
    div = np.empty_like(flux_x)
    for i in prange(nx):
        for j in prange(ny):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            div_x = (flux_x[ip1, j] - flux_x[im1, j]) / (2.0 * dx)
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
            div[i, j] = div_x + div_y
    return div

@njit(parallel=True, fastmath=True, cache=True)
def update_kks_phase_transformation(c, eta, dt, dx,
                                    K_gamma, K_epsilon,
                                    c_gamma_eq, c_epsilon_eq,
                                    E_gamma, E_epsilon,
                                    W_barrier, kappa_eta, L_struct,
                                    M_c):
    """
    One time step of KKS phase transformation:
    - c (Co fraction) evolves via Cahn‑Hilliard (conserved)
    - η (HCP order) evolves via Allen‑Cahn (non‑conserved)

    Uses:
      - Moelans phase fractions: φ_ε = (1-η)² / (η² + (1-η)²)
      - KKS condition: μ = ∂f/∂c (common chemical potential)
      - Parabolic phase free energies
    """
    nx, ny = c.shape
    # Pre‑allocate arrays
    phi_epsilon = np.empty_like(c)
    mu = np.empty_like(c)
    c_gamma = np.empty_like(c)
    c_epsilon = np.empty_like(c)

    # ---- 1. Phase fractions (Moelans) ----
    for i in prange(nx):
        for j in prange(ny):
            et = eta[i, j]
            S = et * et + (1.0 - et) * (1.0 - et)
            if S > 1e-12:
                phi_epsilon[i, j] = (1.0 - et) * (1.0 - et) / S
            else:
                phi_epsilon[i, j] = 0.0 if et < 0.5 else 1.0

    # ---- 2. KKS: common μ and phase compositions ----
    for i in prange(nx):
        for j in prange(ny):
            phi_e = phi_epsilon[i, j]
            phi_g = 1.0 - phi_e
            # Weighted average of equilibrium compositions
            c_eq_avg = phi_g * c_gamma_eq + phi_e * c_epsilon_eq
            # Harmonic mean of curvatures (for the denominator)
            denom = phi_g / K_gamma + phi_e / K_epsilon
            if denom > 1e-12:
                mu[i, j] = (c[i, j] - c_eq_avg) / denom
            else:
                mu[i, j] = 0.0
            c_gamma[i, j] = c_gamma_eq + mu[i, j] / K_gamma
            c_epsilon[i, j] = c_epsilon_eq + mu[i, j] / K_epsilon

    # ---- 3. Cahn‑Hilliard for global c (conserved) ----
    # Compute flux from μ gradient
    flux_c_x = np.empty_like(c)
    flux_c_y = np.empty_like(c)
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_mu_x = (mu[ip1, j] - mu[im1, j]) / (2.0 * dx)
            grad_mu_y = (mu[i, jp1] - mu[i, jm1]) / (2.0 * dx)
            flux_c_x[i, j] = -M_c * grad_mu_x
            flux_c_y[i, j] = -M_c * grad_mu_y
    div_flux_c = compute_gradient_divergence_2d(flux_c_x, flux_c_y, dx)
    c_new = c + dt * div_flux_c

    # ---- 4. Allen‑Cahn for η (non‑conserved) ----
    lap_eta = compute_laplacian_2d(eta, dx)
    df_deta = np.empty_like(eta)
    for i in prange(nx):
        for j in prange(ny):
            et = eta[i, j]
            phi_e = phi_epsilon[i, j]
            phi_g = 1.0 - phi_e

            # Phase free energies (using phase compositions from KKS)
            f_g = 0.5 * K_gamma * (c_gamma[i, j] - c_gamma_eq) ** 2 + E_gamma
            f_e = 0.5 * K_epsilon * (c_epsilon[i, j] - c_epsilon_eq) ** 2 + E_epsilon

            # Derivative of φ_ε w.r.t η (Moelans, 2‑phase case)
            # φ_ε = (1-η)² / (η² + (1-η)²)
            # ∂φ_ε/∂η = 2η(1-η)(1-2η) / (η²+(1-η)²)²   (derived)
            denom_sq = (et * et + (1.0 - et) * (1.0 - et)) ** 2 + 1e-12
            dphi_deta = 2.0 * et * (1.0 - et) * (1.0 - 2.0 * et) / denom_sq

            # Bulk driving force = ∂φ/∂η * (f_ε – f_γ)
            df_bulk = dphi_deta * (f_e - f_g)

            # Double‑well barrier term: ∂/∂η [W η²(1-η)²]
            df_barrier = 2.0 * W_barrier * et * (1.0 - et) * (1.0 - 2.0 * et)

            # Total variational derivative
            df_deta[i, j] = df_bulk + df_barrier - kappa_eta * lap_eta[i, j]

    # Allen‑Cahn update (L is structural mobility)
    eta_new = eta - dt * L_struct * df_deta

    # Physical bounds
    c_new = np.clip(c_new, 0.01, 0.99)
    eta_new = np.clip(eta_new, 0.0, 1.0)

    return c_new, eta_new


# =============================================================================
# MAIN SIMULATION CLASS (KKS + Allen‑Cahn)
# =============================================================================

class MediloyKKSPhaseTransformation:
    """
    2D KKS phase‑field simulation of γ-FCC ↔ ε-HCP transformation in Mediloy.
    - η (HCP order) evolves via Allen‑Cahn (non‑conserved)
    - c (Co fraction) evolves via Cahn‑Hilliard (conserved)
    - KKS condition for correct solute partitioning
    """
    def __init__(self, nx=256, ny=256, T_celsius=950.0,
                 K_gamma_Jm3=2.0e10, K_epsilon_Jm3=2.0e10,
                 c_gamma_eq=0.61, c_epsilon_eq=0.575,
                 E_epsilon_Jm3=-400.0/6.7e-6,   # -400 J/mol → J/m³
                 D_b_m2s=5.0e-15, V_m_m3mol=6.7e-6):
        self.nx = nx
        self.ny = ny
        self.dx_dim = 1.0

        # Physical constants (in SI)
        self.T_celsius = T_celsius
        self.T_K = T_celsius + 273.15
        self.V_m = V_m_m3mol
        self.D_b = D_b_m2s

        # Phase‑specific parameters (physical units)
        self.K_gamma = K_gamma_Jm3
        self.K_epsilon = K_epsilon_Jm3
        self.c_gamma_eq = c_gamma_eq
        self.c_epsilon_eq = c_epsilon_eq
        self.E_gamma = 0.0
        self.E_epsilon = E_epsilon_Jm3

        # Dimensionless parameters (to be scaled)
        self.W_barrier_dim = 1.0          # double‑well height
        self.kappa_eta_dim = 1.0          # η gradient coefficient
        self.M_c_dim = 1.0                # chemical mobility (c)
        self.L_struct_dim = 1.0           # structural mobility (Allen‑Cahn)
        self.dt_dim = 0.005               # time step

        # Initialize physical scales
        self.scales = PhysicalScalesMediloy(
            T_celsius=T_celsius,
            V_m_m3mol=V_m_m3mol,
            D_b_m2s=D_b_m2s
        )
        self._update_physical_params()

        # Fields
        self.c = np.full((nx, ny), 0.61, dtype=np.float64)   # Co fraction
        self.eta = np.zeros((nx, ny), dtype=np.float64)       # η=0: FCC

        # Time tracking
        self.time_phys = 0.0
        self.step = 0

        # History
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [],
            'hcp_fraction': [], 'fcc_fraction': [],
            'total_energy': []
        }
        self.update_history()

    def _update_physical_params(self):
        """Convert dimensionless parameters to physical units."""
        (self.W_barrier, self.kappa_c, self.kappa_eta,
         self.M_c, self.L_struct, self.dt_phys, self.dx_phys) = \
            self.scales.dim_to_phys(
                self.W_barrier_dim, 0.0, self.kappa_eta_dim,   # kappa_c not used
                self.M_c_dim, self.L_struct_dim, self.dt_dim, self.dx_dim
            )
        # kappa_c (for c) is not used in KKS, but we keep it for compatibility

    def set_physical_parameters(self, **kwargs):
        """Update physical parameters directly."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        # If any dimensional parameter changed, recompute scaled ones
        self._update_physical_params()

    def initialize_random(self, c0=0.61, eta0=0.0, noise_c=0.02, noise_eta=0.02, seed=42):
        np.random.seed(seed)
        self.c = np.clip(c0 + noise_c * (2*np.random.random((self.nx, self.ny)) - 1), 0.01, 0.99)
        self.eta = np.clip(eta0 + noise_eta * (2*np.random.random((self.nx, self.ny)) - 1), 0.0, 1.0)
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        self.update_history()

    def initialize_fcc_with_random_hcp_seeds(self, num_seeds=12, radius_grid=5,
                                             seed_co_fraction=0.58, seed=42):
        np.random.seed(seed)
        self.c = np.full((self.nx, self.ny), 0.61, dtype=np.float64)
        self.eta = np.zeros((self.nx, self.ny), dtype=np.float64)

        for s in range(num_seeds):
            cx = np.random.randint(radius_grid + 5, self.nx - radius_grid - 5)
            cy = np.random.randint(radius_grid + 5, self.ny - radius_grid - 5)
            for i in range(-radius_grid*2, radius_grid*2 + 1):
                for j in range(-radius_grid*2, radius_grid*2 + 1):
                    r = np.sqrt(i**2 + j**2)
                    if r <= radius_grid:
                        ii = (cx + i) % self.nx
                        jj = (cy + j) % self.ny
                        weight = min(1.0, r / radius_grid)
                        self.eta[ii, jj] = 1.0 * (1.0 - weight)
                        self.c[ii, jj] = seed_co_fraction * (1.0 - weight) + 0.61 * weight
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        self.update_history()

    def initialize_from_arrays(self, c_array, eta_array, reset_time=True):
        self.c = np.clip(np.array(c_array, dtype=np.float64), 0.01, 0.99)
        self.eta = np.clip(np.array(eta_array, dtype=np.float64), 0.0, 1.0)
        if reset_time:
            self.time_phys = 0.0
            self.step = 0
            self.clear_history()
            self.update_history()

    def clear_history(self):
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [],
            'hcp_fraction': [], 'fcc_fraction': [],
            'total_energy': []
        }

    def update_history(self):
        self.history['time_phys'].append(self.time_phys)
        self.history['eta_mean'].append(float(np.mean(self.eta)))
        self.history['eta_std'].append(float(np.std(self.eta)))
        self.history['c_mean'].append(float(np.mean(self.c)))
        self.history['c_std'].append(float(np.std(self.c)))
        self.history['hcp_fraction'].append(float(np.sum(self.eta > 0.5) / (self.nx * self.ny)))
        self.history['fcc_fraction'].append(float(np.sum(self.eta < 0.5) / (self.nx * self.ny)))

        # Free energy (optional, compute every 10 steps to save time)
        if self.step % 10 == 0:
            try:
                self.history['total_energy'].append(self.compute_total_free_energy())
            except Exception:
                self.history['total_energy'].append(np.nan)
        else:
            self.history['total_energy'].append(np.nan)

    def compute_total_free_energy(self):
        """Return total free energy in Joules."""
        # Phase fractions
        S = self.eta**2 + (1 - self.eta)**2
        phi_epsilon = (1 - self.eta)**2 / (S + 1e-12)
        phi_gamma = 1.0 - phi_epsilon

        # Chemical potential (KKS) – for energy, we need phase compositions
        mu = np.empty_like(self.c)
        c_gamma = np.empty_like(self.c)
        c_epsilon = np.empty_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                phi_e = phi_epsilon[i, j]
                phi_g = 1.0 - phi_e
                c_eq_avg = phi_g * self.c_gamma_eq + phi_e * self.c_epsilon_eq
                denom = phi_g / self.K_gamma + phi_e / self.K_epsilon
                if denom > 1e-12:
                    mu[i, j] = (self.c[i, j] - c_eq_avg) / denom
                else:
                    mu[i, j] = 0.0
                c_gamma[i, j] = self.c_gamma_eq + mu[i, j] / self.K_gamma
                c_epsilon[i, j] = self.c_epsilon_eq + mu[i, j] / self.K_epsilon

        # Bulk energy density
        f_gamma = 0.5 * self.K_gamma * (c_gamma - self.c_gamma_eq)**2 + self.E_gamma
        f_epsilon = 0.5 * self.K_epsilon * (c_epsilon - self.c_epsilon_eq)**2 + self.E_epsilon
        f_bulk = phi_gamma * f_gamma + phi_epsilon * f_epsilon

        # Gradient energy (only η contributes – c gradient is implicitly handled by μ)
        grad_eta_x = np.zeros_like(self.eta)
        grad_eta_y = np.zeros_like(self.eta)
        for i in range(self.nx):
            for j in range(self.ny):
                ip1 = (i + 1) % self.nx
                im1 = (i - 1) % self.nx
                jp1 = (j + 1) % self.ny
                jm1 = (j - 1) % self.ny
                grad_eta_x[i, j] = (self.eta[ip1, j] - self.eta[im1, j]) / (2.0 * self.dx_phys)
                grad_eta_y[i, j] = (self.eta[i, jp1] - self.eta[i, jm1]) / (2.0 * self.dx_phys)
        grad_eta_sq = grad_eta_x**2 + grad_eta_y**2
        f_gradient = 0.5 * self.kappa_eta * grad_eta_sq

        total_energy = np.sum(f_bulk + f_gradient) * (self.dx_phys**2)
        return float(total_energy)

    def run_step(self):
        self.c, self.eta = update_kks_phase_transformation(
            self.c, self.eta,
            self.dt_phys, self.dx_phys,
            self.K_gamma, self.K_epsilon,
            self.c_gamma_eq, self.c_epsilon_eq,
            self.E_gamma, self.E_epsilon,
            self.W_barrier, self.kappa_eta, self.L_struct,
            self.M_c
        )
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()

    def run_steps(self, n_steps, progress_callback=None):
        for step_idx in range(n_steps):
            self.run_step()
            if progress_callback is not None and step_idx % 10 == 0:
                progress_callback(step_idx + 1, n_steps)

    def get_statistics(self):
        domain_size_m = self.nx * self.dx_phys
        interface_width_eta = self.scales.phys_to_interface_width(self.kappa_eta, self.W_barrier)
        return {
            'time_phys': self.time_phys,
            'time_formatted': self.scales.format_time(self.time_phys),
            'step': self.step,
            'domain_size_m': domain_size_m,
            'domain_size_formatted': self.scales.format_length(domain_size_m),
            'interface_width_eta_nm': interface_width_eta * 1e9,
            'eta_mean': float(np.mean(self.eta)),
            'eta_std': float(np.std(self.eta)),
            'eta_min': float(np.min(self.eta)),
            'eta_max': float(np.max(self.eta)),
            'c_mean': float(np.mean(self.c)),
            'c_std': float(np.std(self.c)),
            'c_min': float(np.min(self.c)),
            'c_max': float(np.max(self.c)),
            'hcp_fraction': float(np.sum(self.eta > 0.5) / (self.nx * self.ny)),
            'fcc_fraction': float(np.sum(self.eta < 0.5) / (self.nx * self.ny)),
            'W_barrier': self.W_barrier,
            'M_c': self.M_c,
            'L_struct': self.L_struct,
            'dt_phys': self.dt_phys,
        }

    def get_time_series(self, key):
        if key not in self.history:
            raise ValueError(f"Unknown history key: {key}")
        return np.array(self.history['time_phys']), np.array(self.history[key])


# =============================================================================
# STREAMLIT APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="Mediloy γ→ε KKS Phase Transformation",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 12px; border-radius: 8px; margin: 5px 0;}
    .stButton>button {width: 100%; border-radius: 6px;}
    .phase-fcc {color: #2ecc71; font-weight: bold;}
    .phase-hcp {color: #e74c3c; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

    st.title("⚙️ Mediloy γ-FCC → ε-HCP Phase Transformation")
    st.markdown("""
    **Kim‑Kim‑Suzuki (KKS) multiphase‑field model with Allen‑Cahn dynamics for η**

    - **c** (Co fraction) evolves via **Cahn‑Hilliard** (conserved)
    - **η** (HCP order) evolves via **Allen‑Cahn** (non‑conserved)
    - **KKS condition**: equal chemical potentials across the interface → correct solute partitioning
    - **Moelans interpolation**: φ_ε = (1‑η)² / (η² + (1‑η)²)
    - Parabolic phase free energies: f^α(c) = ½ K_α (c – c_α^eq)² + E_α

    **Temperature:** 950°C (1223 K) – pseudo‑binary Co–M<sub>y</sub> alloy
    """, unsafe_allow_html=True)

    if 'sim' not in st.session_state:
        st.session_state.sim = MediloyKKSPhaseTransformation(
            nx=256, ny=256, T_celsius=950.0,
            K_gamma_Jm3=2.0e10, K_epsilon_Jm3=2.0e10,
            c_gamma_eq=0.61, c_epsilon_eq=0.575,
            E_epsilon_Jm3=-400.0/6.7e-6,
            D_b_m2s=5.0e-15
        )
        st.session_state.sim.initialize_fcc_with_random_hcp_seeds(
            num_seeds=12, radius_grid=5, seed_co_fraction=0.58, seed=42
        )

    sim = st.session_state.sim

    # -------------------------------------------------------------------------
    # SIDEBAR
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ Control Panel")

        st.subheader("⏱️ Time Stepping")
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            steps_input = st.number_input("Steps per update", 1, 5000, 100, 10)
        with col_run2:
            if st.button("▶️ Run", type="primary", use_container_width=True):
                with st.spinner(f"Computing {steps_input} steps..."):
                    sim.run_steps(steps_input)
                st.rerun()

        col_stop1, col_stop2 = st.columns(2)
        with col_stop1:
            if st.button("⏸️ Pause", use_container_width=True):
                st.rerun()
        with col_stop2:
            if st.button("⏭️ Step", use_container_width=True):
                sim.run_step()
                st.rerun()

        st.divider()

        st.subheader("🎲 Initial Conditions")
        init_type = st.radio("Initialization", ["Random noise", "HCP seeds in FCC", "Uniform FCC"])
        if init_type == "Random noise":
            noise_c = st.slider("Co noise amplitude", 0.0, 0.1, 0.02, 0.01)
            noise_eta = st.slider("η noise amplitude", 0.0, 0.1, 0.02, 0.01)
            if st.button("🔄 Initialize with Random Noise", use_container_width=True):
                sim.initialize_random(c0=0.61, eta0=0.0, noise_c=noise_c, noise_eta=noise_eta, seed=int(time.time()))
                st.rerun()
        elif init_type == "HCP seeds in FCC":
            num_seeds = st.slider("Number of HCP seeds", 1, 50, 12, 1)
            seed_radius = st.slider("Seed radius (grid units)", 3, 15, 5, 1)
            if st.button("🌱 Initialize with HCP Seeds", use_container_width=True):
                sim.initialize_fcc_with_random_hcp_seeds(
                    num_seeds=num_seeds, radius_grid=seed_radius, seed_co_fraction=0.58, seed=42
                )
                st.rerun()
        else:  # Uniform FCC
            if st.button("🧊 Initialize Uniform FCC", use_container_width=True):
                sim.c = np.full((sim.nx, sim.ny), 0.61, dtype=np.float64)
                sim.eta = np.zeros((sim.nx, sim.ny), dtype=np.float64)
                sim.time_phys = 0.0
                sim.step = 0
                sim.clear_history()
                sim.update_history()
                st.rerun()

        st.divider()

        st.subheader("⚙️ Model Parameters (Physical Units)")
        st.caption("KKS + Allen‑Cahn parameters")

        # Show current interface width
        xi_eta_nm = sim.scales.phys_to_interface_width(sim.kappa_eta, sim.W_barrier) * 1e9

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            new_W = st.number_input("W_barrier (J/m³)", 1e4, 1e8, float(sim.W_barrier), format="%.2e")
            new_kappa_eta = st.number_input("κ_η (J/m)", 1e-13, 1e-9, float(sim.kappa_eta), format="%.2e")
            new_L = st.number_input("L_struct (m³/(J·s))", 1e-14, 1e-6, float(sim.L_struct), format="%.2e")
        with col_p2:
            new_M_c = st.number_input("M_c (m⁵/(J·s))", 1e-25, 1e-17, float(sim.M_c), format="%.2e")
            new_dt = st.number_input("Δt (s)", 1e-12, 1e-5, float(sim.dt_phys), format="%.2e")

        # Phase‑specific parameters
        st.markdown("**Phase free energies (parabolic)**")
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            new_K_gamma = st.number_input("K_gamma (J/m³ per (Δc)²)", 1e8, 1e12, float(sim.K_gamma), format="%.2e")
            new_c_gamma_eq = st.number_input("c_gamma_eq (at% Co)", 0.5, 0.7, float(sim.c_gamma_eq), 0.01)
        with col_k2:
            new_K_epsilon = st.number_input("K_epsilon (J/m³ per (Δc)²)", 1e8, 1e12, float(sim.K_epsilon), format="%.2e")
            new_c_epsilon_eq = st.number_input("c_epsilon_eq (at% Co)", 0.5, 0.7, float(sim.c_epsilon_eq), 0.01)

        if st.button("Apply Parameters", use_container_width=True):
            sim.set_physical_parameters(
                W_barrier=new_W,
                kappa_eta=new_kappa_eta,
                L_struct=new_L,
                M_c=new_M_c,
                dt_phys=new_dt,
                K_gamma=new_K_gamma,
                K_epsilon=new_K_epsilon,
                c_gamma_eq=new_c_gamma_eq,
                c_epsilon_eq=new_c_epsilon_eq,
            )
            st.rerun()

        if xi_eta_nm < 1.5:
            st.warning(f"⚠️ Interface width ({xi_eta_nm:.2f} nm) < 1.5 nm: under‑resolved")
        elif xi_eta_nm > 15.0:
            st.info(f"ℹ️ Interface width ({xi_eta_nm:.1f} nm) is quite diffuse")

        st.divider()

        stats = sim.get_statistics()
        st.subheader("📊 Live Statistics")
        st.markdown(f"""
        <div class="metric-card"><b>⏱️ Physical Time:</b> {stats['time_formatted']}</div>
        <div class="metric-card"><b>🔢 Simulation Step:</b> {stats['step']:,}</div>
        <div class="metric-card"><b>📐 Domain Size:</b> {stats['domain_size_formatted']}</div>
        <div class="metric-card"><b>🔲 Interface Width (η):</b> {stats['interface_width_eta_nm']:.2f} nm</div>
        """, unsafe_allow_html=True)

        st.markdown("**Phase Distribution**")
        col_ph1, col_ph2 = st.columns(2)
        with col_ph1:
            st.metric("<span class='phase-hcp'>ε-HCP Fraction</span>", f"{stats['hcp_fraction']*100:.1f}%")
        with col_ph2:
            st.metric("<span class='phase-fcc'>γ-FCC Fraction</span>", f"{stats['fcc_fraction']*100:.1f}%")

        st.markdown("**Fields**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("⟨c_Co⟩", f"{stats['c_mean']:.3f}")
            st.metric("⟨η⟩", f"{stats['eta_mean']:.3f}")
        with col_c2:
            st.metric("σ(c)", f"{stats['c_std']:.3f}")
            st.metric("σ(η)", f"{stats['eta_std']:.3f}")

    # -------------------------------------------------------------------------
    # MAIN CONTENT: VISUALIZATIONS
    # -------------------------------------------------------------------------

    extent_um_x = [0, sim.nx * sim.dx_phys * 1e6]
    extent_um_y = [0, sim.ny * sim.dx_phys * 1e6]

    # Row 1: η and c fields
    col_viz1, col_viz2 = st.columns(2)

    with col_viz1:
        st.subheader("ε-HCP Order Parameter η (Allen‑Cahn)")
        st.caption(f"η = 0 (FCC) → η = 1 (HCP) | t = {stats['time_formatted']} | Non‑conserved")
        fig_eta = go.Figure(data=go.Heatmap(
            z=sim.eta.T,
            x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
            y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
            colorscale='RdYlBu_r',
            zmin=0, zmax=1,
            colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP']),
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br><br>η: %{z:.3f}<extra></extra>'
        ))
        fig_eta.update_layout(
            title="HCP Order Parameter (Allen‑Cahn)",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=600, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_eta, use_container_width=True)

    with col_viz2:
        st.subheader("Co Concentration c_Co (Cahn‑Hilliard)")
        st.caption("Nominal composition: c₀ = 0.61 (61 at.% Co) | Conserved")
        fig_c = go.Figure(data=go.Heatmap(
            z=sim.c.T,
            x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
            y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
            colorscale='Viridis',
            zmin=0.55, zmax=0.67,
            colorbar=dict(title="c_Co"),
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>c: %{z:.3f}<extra></extra>'
        ))
        fig_c.update_layout(
            title="Cobalt Mole Fraction",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=600, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_c, use_container_width=True)

    # Row 2: Overlay + histograms
    col_overlay, col_hist = st.columns([2, 1])

    with col_overlay:
        st.subheader("Phase + Composition Overlay")
        st.caption("HCP regions (red contours) with Co depletion (blue) at interfaces")
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Heatmap(
            z=sim.c.T,
            x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
            y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
            colorscale='Viridis', zmin=0.55, zmax=0.67,
            opacity=0.7, colorbar=dict(title="c_Co", x=1.02), showscale=True
        ))
        for level in [0.3, 0.5, 0.7]:
            fig_overlay.add_trace(go.Contour(
                z=sim.eta.T,
                x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
                y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
                contours=dict(start=level, end=level, size=0.1),
                line=dict(color='red', width=1.5), showscale=False,
                hoverinfo='skip', name=f'η = {level:.1f}'
            ))
        fig_overlay.update_layout(
            title="HCP Phase Boundaries on Co Concentration",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=700, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

    with col_hist:
        st.subheader("Field Distributions")
        fig_hist = make_subplots(rows=2, cols=1, subplot_titles=("η Distribution", "c_Co Distribution"))
        fig_hist.add_trace(go.Histogram(x=sim.eta.flatten(), nbinsx=40, name="η", marker_color='#e74c3c', opacity=0.7), row=1, col=1)
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        fig_hist.add_trace(go.Histogram(x=sim.c.flatten(), nbinsx=40, name="c_Co", marker_color='#2ecc71', opacity=0.7), row=2, col=1)
        fig_hist.add_vline(x=0.61, line_dash="dash", line_color="gray", row=2, col=1)
        fig_hist.update_layout(height=500, width=400, showlegend=False, margin=dict(l=40, r=20, t=60, b=40))
        fig_hist.update_xaxes(title_text="Value", row=2, col=1)
        fig_hist.update_xaxes(title_text="Value", row=1, col=1)
        fig_hist.update_yaxes(title_text="Frequency", row=1, col=1)
        fig_hist.update_yaxes(title_text="Frequency", row=2, col=1)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Row 3: Kinetics
    st.divider()
    st.subheader("📈 Transformation Kinetics (Allen‑Cahn η + Cahn‑Hilliard c)")

    if len(sim.history['time_phys']) > 3:
        times_s = np.array(sim.history['time_phys'])
        times_min = times_s / 60.0

        fig_kin = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Phase Fractions", "Standard Deviations", "Mean Values"),
            shared_xaxes=True, x_title="Time (minutes)"
        )

        # Phase fractions
        fig_kin.add_trace(go.Scatter(x=times_min, y=np.array(sim.history['hcp_fraction'])*100,
                                     mode='lines', name='HCP %', line=dict(color='#e74c3c', width=2.5)), row=1, col=1)
        fig_kin.add_trace(go.Scatter(x=times_min, y=np.array(sim.history['fcc_fraction'])*100,
                                     mode='lines', name='FCC %', line=dict(color='#2ecc71', width=2)), row=1, col=1)
        fig_kin.update_yaxes(title_text="Phase fraction (%)", row=1, col=1)

        # Standard deviations
        fig_kin.add_trace(go.Scatter(x=times_min, y=sim.history['eta_std'],
                                     mode='lines', name='σ(η)', line=dict(color='#9b59b6', width=2)), row=1, col=2)
        fig_kin.add_trace(go.Scatter(x=times_min, y=sim.history['c_std'],
                                     mode='lines', name='σ(c)', line=dict(color='#f39c12', width=1.5, dash='dash')), row=1, col=2)
        fig_kin.update_yaxes(title_text="Standard deviation", row=1, col=2)

        # Mean values
        fig_kin.add_trace(go.Scatter(x=times_min, y=sim.history['eta_mean'],
                                     mode='lines', name='⟨η⟩', line=dict(color='#e74c3c', width=2)), row=1, col=3)
        fig_kin.add_trace(go.Scatter(x=times_min, y=sim.history['c_mean'],
                                     mode='lines', name='⟨c⟩', line=dict(color='#2ecc71', width=2)), row=1, col=3)
        fig_kin.update_yaxes(title_text="Mean value", row=1, col=3)

        fig_kin.update_layout(height=450, width=1200, showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_kin, use_container_width=True)

        # Free energy evolution
        with st.expander("🔋 Free Energy Evolution (click to expand)"):
            valid_energy = [e for e in sim.history['total_energy'] if not np.isnan(e)]
            valid_times = [t for t, e in zip(times_min, sim.history['total_energy']) if not np.isnan(e)]
            if len(valid_energy) > 2:
                fig_fe = go.Figure()
                fig_fe.add_trace(go.Scatter(x=valid_times, y=valid_energy, mode='lines+markers',
                                            name='Total free energy', line=dict(color='#8e44ad', width=2), marker=dict(size=3)))
                fig_fe.update_layout(title="Free Energy Minimization", xaxis_title="Time (minutes)", yaxis_title="Total free energy (J)", height=400)
                st.plotly_chart(fig_fe, use_container_width=True)
            else:
                st.info("Run more steps to compute free energy evolution.")
    else:
        st.info("📊 Run simulation for at least 4 steps to display kinetics plots.")

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("💾 Export Results")

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        if st.button("📸 Save η Snapshot", use_container_width=True):
            fig_snap = go.Figure(data=go.Heatmap(
                z=sim.eta.T,
                x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
                y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
                colorscale='RdYlBu_r', zmin=0, zmax=1,
                colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP'])
            ))
            fig_snap.update_layout(title=f"Mediloy HCP – t = {stats['time_formatted']}",
                                   xaxis_title="x (μm)", yaxis_title="y (μm)", width=800, height=700)
            img_bytes = fig_snap.to_image(format="png", width=800, height=700, scale=2)
            st.download_button(label="⬇️ Download PNG", data=img_bytes,
                               file_name=f"Mediloy_KKS_eta_t{sim.time_phys:.2e}s.png",
                               mime="image/png", use_container_width=True)

    with col_exp2:
        if st.button("📊 Save Kinetics Data", use_container_width=True):
            csv_lines = ["time_s,time_min,eta_mean,eta_std,c_mean,c_std,hcp_frac,fcc_frac,energy_J"]
            for i in range(len(sim.history['time_phys'])):
                t_s = sim.history['time_phys'][i]
                line = f"{t_s:.6e},{t_s/60:.6e},"
                line += f"{sim.history['eta_mean'][i]:.6f},{sim.history['eta_std'][i]:.6f},"
                line += f"{sim.history['c_mean'][i]:.6f},{sim.history['c_std'][i]:.6f},"
                line += f"{sim.history['hcp_fraction'][i]:.6f},{sim.history['fcc_fraction'][i]:.6f},"
                line += f"{sim.history['total_energy'][i]:.6e}"
                csv_lines.append(line)
            csv_content = "\n".join(csv_lines)
            st.download_button(label="⬇️ Download CSV", data=csv_content,
                               file_name="mediloy_kks_kinetics.csv", mime="text/csv", use_container_width=True)

    with col_exp3:
        if st.button("⚙️ Save Simulation State", use_container_width=True):
            npz_buf = BytesIO()
            np.savez_compressed(npz_buf,
                concentration=sim.c, order_parameter=sim.eta,
                time_phys=sim.time_phys, step=sim.step,
                params={
                    'T_celsius': sim.T_celsius,
                    'K_gamma': sim.K_gamma, 'K_epsilon': sim.K_epsilon,
                    'c_gamma_eq': sim.c_gamma_eq, 'c_epsilon_eq': sim.c_epsilon_eq,
                    'E_gamma': sim.E_gamma, 'E_epsilon': sim.E_epsilon,
                    'W_barrier': sim.W_barrier, 'kappa_eta': sim.kappa_eta,
                    'M_c': sim.M_c, 'L_struct': sim.L_struct,
                    'dt_phys': sim.dt_phys, 'dx_phys': sim.dx_phys
                }
            )
            npz_buf.seek(0)
            st.download_button(label="⬇️ Download NPZ", data=npz_buf.getvalue(),
                               file_name=f"Mediloy_KKS_state_t{sim.time_phys:.2e}s.npz",
                               mime="application/octet-stream", use_container_width=True)

    # -------------------------------------------------------------------------
    # PHYSICS GUIDE
    # -------------------------------------------------------------------------
    with st.expander("ℹ️ Physics Guide: KKS + Allen‑Cahn Model for Mediloy", expanded=False):
        st.markdown("""
        ## ⚙️ KKS Phase‑Field Model for γ-FCC → ε-HCP Transformation

        This simulation implements a **Kim‑Kim‑Suzuki (KKS) multiphase‑field model** that combines:

        - **Allen‑Cahn** dynamics for the non‑conserved HCP order parameter η
        - **Cahn‑Hilliard** dynamics for the conserved Co concentration c
        - **Moelans interpolation** for phase fractions φ_α = η_α² / Σ η_ρ²
        - **KKS condition**: equal chemical potentials across the interface

        ### Governing Equations

        **Concentration field (conserved):**
        ```
        ∂c/∂t = ∇·[ M_c ∇μ ]
        ```
        where μ is the common chemical potential obtained from the KKS mixture rule.

        **Order parameter (non‑conserved):**
        ```
        ∂η/∂t = -L ( δF/δη - κ ∇²η )
        ```

        **Phase fractions (Moelans, 2‑phase case):**
        ```
        φ_ε = (1-η)² / (η² + (1-η)²),    φ_γ = 1 - φ_ε
        ```

        **Phase free energies (parabolic):**
        ```
        f^γ(c) = ½ K_γ (c – c_γ^eq)² + E_γ
        f^ε(c) = ½ K_ε (c – c_ε^eq)² + E_ε
        ```

        **KKS mixture rule (two‑phase):**
        ```
        μ = (c – [φ_γ c_γ^eq + φ_ε c_ε^eq]) / (φ_γ/K_γ + φ_ε/K_ε)
        c^γ = c_γ^eq + μ/K_γ,    c^ε = c_ε^eq + μ/K_ε
        ```

        **Total bulk free energy density:**
        ```
        f_bulk = φ_γ f^γ(c^γ) + φ_ε f^ε(c^ε)
        ```

        ### Why KKS?

        - Eliminates artificial solute drag at interfaces
        - Correctly predicts equilibrium partitioning (Cr enrichment in HCP)
        - Allows using a diffuse interface without spurious effects
        - Numerically stable and widely validated for binary alloys

        ### Key Parameters

        | Parameter | Physical meaning | Typical value (Mediloy) |
        |-----------|------------------|--------------------------|
        | K_γ, K_ε | Curvature of free energy parabola | 1–5×10¹⁰ J/m³ |
        | c_γ^eq, c_ε^eq | Equilibrium compositions at 950°C | 0.61, 0.575 |
        | E_ε – E_γ | Chemical driving force (ΔG) | -400 J/mol |
        | W_barrier | Double‑well barrier height | ~1×10⁷ J/m³ |
        | κ_η | Gradient energy coefficient | ~1×10⁻⁹ J/m |
        | L | Structural mobility (Allen‑Cahn) | ~1×10⁻⁹ m³/(J·s) |
        | M_c | Chemical mobility (Cahn‑Hilliard) | ~5×10⁻²⁰ m⁵/(J·s) |

        ### Numerical Stability

        - Interface width should be resolved by at least 3 grid points
        - Time step must satisfy: Δt ≲ min( Δx²/(4 M_c K), Δx²/(4 L κ_η) )
        - For typical parameters, Δt ≈ 1×10⁻⁷ … 1×10⁻⁵ s

        ### Interpreting Results

        - **⟨η⟩ increases** over time (HCP grows) – because η is **non‑conserved**
        - **⟨c⟩ stays constant** (conserved)
        - **Cr enrichment** (low c) appears inside HCP regions (blue areas in overlay)
        - Free energy decreases monotonically

        ### References

        1. Kim, S.G., Kim, W.T., Suzuki, T. (1999). *Phys. Rev. E* **60**, 7186.
        2. Moelans, N., Blanpain, B., Wollants, P. (2008). *CALPHAD* **32**, 268.
        3. Steinbach, I. (2009). *Model. Simul. Mater. Sci. Eng.* **17**, 073001.
        4. Yamanaka, K. et al. (2018). *Dent. Mater. J.* **37**, 1.
        """)

    # -------------------------------------------------------------------------
    # AUTO-RUN
    # -------------------------------------------------------------------------
    st.sidebar.divider()
    with st.sidebar.expander("🔄 Auto-run Settings"):
        auto_run = st.checkbox("Enable auto-run", value=False)
        auto_speed = st.slider("Speed (steps/second)", 1, 500, 50)
        auto_max_steps = st.number_input("Max steps (0 = unlimited)", 0, 200000, 0)
        if auto_run:
            stop_auto = st.button("⏹️ Stop Auto-run", type="secondary")
            if not stop_auto:
                steps_this_frame = min(auto_speed,
                                       auto_max_steps - sim.step if auto_max_steps > 0 else auto_speed)
                if steps_this_frame > 0 and (auto_max_steps == 0 or sim.step < auto_max_steps):
                    with st.spinner(f"Auto-running {steps_this_frame} steps..."):
                        sim.run_steps(steps_this_frame)
                    st.rerun()
                elif auto_max_steps > 0 and sim.step >= auto_max_steps:
                    st.success(f"✓ Reached max steps: {sim.step:,}")

    # Footer
    st.markdown("---")
    st.caption(
        "Mediloy γ→ε KKS Phase Transformation | Allen‑Cahn (η) + Cahn‑Hilliard (c) | "
        f"Physical units: m, s, J/m³ | T = {sim.T_celsius}°C | "
        f"Pseudo‑binary Co–M<sub>y</sub> (c₀ = 0.61) | Visualized with Plotly"
    )


if __name__ == "__main__":
    print("⚙️ Starting Mediloy KKS Phase Transformation Simulator...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Numba: JIT compilation enabled")
    print(f"   Temperature: 950°C (1223.15 K)")
    print(f"   Model: KKS + Moelans + Allen‑Cahn (η) + Cahn‑Hilliard (c)")
    print(f"   Initial condition: FCC matrix + random HCP seeds")
    main()
