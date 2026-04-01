# =============================================================================
# MEDILOY γ-FCC → ε-HCP PHASE TRANSFORMATION SIMULATOR
# Mixed Dynamics: c (Cahn-Hilliard) + η (Allen-Cahn) + KKS Interface Model
# Temperature: 950°C (1223.15 K) - Pseudo-binary Co-M_y model
# =============================================================================
# Key Features:
#   - η: NON-CONSERVED order parameter → Allen-Cahn: ∂η/∂t = -L·(δF/δη)
#   - c: CONSERVED concentration → Cahn-Hilliard: ∂c/∂t = ∇·[M_c ∇(δF/δc)]
#   - KKS model: μ^γ(c^γ) = μ^ε(c^ε) = μ at interface (no artificial solute drag)
#   - Moelans interpolation: φ_α = η_α² / Ση_ρ² for smooth phase fractions
#   - Parabolic free energies for numerical stability at high T
#   - All Numba functions use type inference (no explicit scalar signatures)
# =============================================================================

import numpy as np
from numba import njit, prange, config
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
from io import BytesIO

# Optional: Uncomment to disable JIT for debugging
# config.DISABLE_JIT = True


# =============================================================================
# PHYSICAL SCALES FOR MEDILOY (Co-Cr-Mo-Si-W at 950°C)
# =============================================================================

class PhysicalScalesMediloy:
    """
    Physical unit conversion for Mediloy phase-field simulations.
    
    Mediloy is modeled as a pseudo-binary Co-M_y alloy where:
    - Co: 61 at.% (base concentration c0 = 0.61)
    - M_y: collective effect of Cr (~25%), Mo, Si, W, etc.
    
    At T = 950°C (1223.15 K):
    - Near equilibrium T0 for γ-FCC ↔ ε-HCP transformation
    - Small chemical driving force: ΔG_chem ≈ 200-600 J/mol
    - Phase transformation via coupled CH (solute) + AC (structure) dynamics
    """
    
    def __init__(self, T_celsius=950.0, V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        self.R = 8.314462618          # J/(mol·K)
        self.T = T_celsius + 273.15   # K
        self.T_celsius = T_celsius    # °C
        self.V_m = V_m_m3mol          # m³/mol
        self.D_b = D_b_m2s            # m²/s - Cr diffusion in Co matrix
        
        self.L0 = 2.0e-9              # m - Reference length (interface width)
        delta_G_mol = 400.0           # J/mol - Chemical driving force at 950°C
        self.E0 = delta_G_mol / self.V_m   # J/m³ - Energy density scale
        
        self.t0 = self.L0**2 / self.D_b    # s - Diffusion time scale
        self.M0 = self.D_b / self.E0        # m⁵/(J·s) - Chemical mobility scale
        
        # Structural mobility scale for Allen-Cahn η evolution (typically faster)
        self.L0_struct = 1.0e-10      # m³/(J·s) reference for Allen-Cahn mobility
        
        print(f"Mediloy KKS scales initialized at {self.T_celsius}°C ({self.T:.1f} K)")
        print(f"  L0 = {self.L0*1e9:.2f} nm, t0 = {self.t0:.2e} s")
        print(f"  E0 = {self.E0:.2e} J/m³, M0_c = {self.M0:.2e} m⁵/(J·s), L0_η = {self.L0_struct:.2e} m³/(J·s)")
    
    def dim_to_phys(self, W_dim, kappa_c_dim, kappa_eta_dim, M_c_dim, L_eta_dim, dt_dim, dx_dim=1.0):
        """Convert dimensionless parameters to physical SI units."""
        W_phys = W_dim * self.E0
        kappa_c_phys = kappa_c_dim * self.E0 * self.L0**2
        kappa_eta_phys = kappa_eta_dim * self.E0 * self.L0**2
        M_c_phys = M_c_dim * self.M0
        L_eta_phys = L_eta_dim * self.L0_struct  # Allen-Cahn mobility
        dt_phys = dt_dim * self.t0
        dx_phys = dx_dim * self.L0
        return W_phys, kappa_c_phys, kappa_eta_phys, M_c_phys, L_eta_phys, dt_phys, dx_phys
    
    def phys_to_interface_width(self, kappa_phys, W_phys):
        """Estimate interface width from gradient coefficient and barrier."""
        if W_phys <= 0 or kappa_phys <= 0:
            return 2.0e-9
        return np.sqrt(kappa_phys / W_phys)
    
    def format_time(self, t_seconds):
        """Human-readable time formatting."""
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
        """Human-readable length formatting."""
        if not np.isfinite(L_meters) or L_meters < 0:
            return "0 nm"
        if L_meters < 1e-10: return f"{L_meters*1e12:.2f} pm"
        elif L_meters < 1e-9: return f"{L_meters*1e10:.2f} Å"
        elif L_meters < 1e-6: return f"{L_meters*1e9:.2f} nm"
        elif L_meters < 1e-3: return f"{L_meters*1e6:.2f} μm"
        elif L_meters < 1.0: return f"{L_meters*1e3:.2f} mm"
        else: return f"{L_meters:.3f} m"


# =============================================================================
# NUMBA KERNELS – KKS + Moelans + Mixed CH/AC Dynamics
# =============================================================================

@njit(fastmath=True, cache=True)
def _clip_scalar(val, lo, hi):
    """Scalar clip helper for nopython mode compatibility."""
    if val < lo:
        return lo
    elif val > hi:
        return hi
    return val


@njit(fastmath=True, cache=True)
def parabolic_free_energy(c, c_eq, K, E_offset):
    """
    Parabolic free energy for phase α: f^α(c) = ½K(c - c_eq)² + E_offset
    Works with scalars AND arrays via Numba type inference.
    """
    return 0.5 * K * (c - c_eq)**2 + E_offset


@njit(fastmath=True, cache=True)
def d_parabolic_f_dc(c, c_eq, K):
    """Chemical potential for parabolic free energy: ∂f/∂c = K(c - c_eq)"""
    return K * (c - c_eq)


@njit(fastmath=True, cache=True)
def moelans_phase_fraction(eta_val):
    """
    Moelans interpolation for 2-phase system (γ-FCC, ε-HCP).
    φ_ε = (1-η)² / [η² + (1-η)²], φ_γ = η² / [η² + (1-η)²]
    Returns (phi_gamma, phi_epsilon).
    """
    eta_sq = eta_val * eta_val
    one_minus_eta_sq = (1.0 - eta_val) * (1.0 - eta_val)
    S = eta_sq + one_minus_eta_sq
    if S < 1e-12:
        return 1.0 if eta_val < 0.5 else 0.0, 0.0 if eta_val < 0.5 else 1.0
    phi_gamma = eta_sq / S
    phi_epsilon = one_minus_eta_sq / S
    return phi_gamma, phi_epsilon


@njit(fastmath=True, cache=True)
def d_phi_epsilon_deta(eta_val):
    """
    Derivative ∂φ_ε/∂η for Moelans 2-phase interpolation.
    Used in Allen-Cahn variational derivative.
    """
    eta_sq = eta_val * eta_val
    one_minus_eta_sq = (1.0 - eta_val) * (1.0 - eta_val)
    S = eta_sq + one_minus_eta_sq
    if S < 1e-12:
        return 0.0
    # d/dη [(1-η)² / S] = [2(1-η)(-1)·S - (1-η)²·dS/dη] / S²
    # dS/dη = 2η - 2(1-η) = 4η - 2
    dS_deta = 4.0 * eta_val - 2.0
    numerator = -2.0 * (1.0 - eta_val) * S - one_minus_eta_sq * dS_deta
    return numerator / (S * S)


@njit(parallel=True, fastmath=True, cache=True)
def compute_laplacian_2d(field, dx):
    """5-point stencil Laplacian with periodic BCs."""
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
    """Divergence: ∇·J = ∂Jx/∂x + ∂Jy/∂y with periodic BCs."""
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
    One time step of KKS-coupled phase transformation:
    
    ∂c/∂t = ∇·[ M_c ∇μ ]                    (Cahn-Hilliard, conserved)
    ∂η/∂t = -L_struct · (∂f/∂η - κ_η∇²η)  (Allen-Cahn, non-conserved)
    
    KKS condition at interface:
      μ^γ(c^γ) = μ^ε(c^ε) = μ  (common chemical potential)
      c = φ_γ·c^γ + φ_ε·c^ε    (mixture rule)
    
    Free energy: F = ∫[ Σ φ_α·f^α(c^α) + W·g(η) + (κ_η/2)|∇η|² ] dV
    """
    nx, ny = c.shape
    
    # ========== STEP 1: Compute Moelans phase fractions ==========
    phi_epsilon = np.empty_like(c)
    for i in prange(nx):
        for j in prange(ny):
            phi_g, phi_e = moelans_phase_fraction(eta[i, j])
            phi_epsilon[i, j] = phi_e  # φ_ε (HCP fraction)
    
    # ========== STEP 2: Solve KKS for common μ, c^γ, c^ε ==========
    mu = np.empty_like(c)
    c_gamma_local = np.empty_like(c)
    c_epsilon_local = np.empty_like(c)
    
    for i in prange(nx):
        for j in prange(ny):
            phi_e = phi_epsilon[i, j]
            phi_g = 1.0 - phi_e
            
            # KKS closed-form solution for 2-phase parabolic free energies:
            # μ = [c - (φ_γ·c_γ^eq + φ_ε·c_ε^eq)] / [φ_γ/K_γ + φ_ε/K_ε]
            denom = phi_g / K_gamma + phi_e / K_epsilon
            weighted_c_eq = phi_g * c_gamma_eq + phi_e * c_epsilon_eq
            
            if denom > 1e-12:
                mu[i, j] = (c[i, j] - weighted_c_eq) / denom
            else:
                mu[i, j] = 0.0
            
            # Phase-specific compositions
            c_gamma_local[i, j] = c_gamma_eq + mu[i, j] / K_gamma
            c_epsilon_local[i, j] = c_epsilon_eq + mu[i, j] / K_epsilon
    
    # ========== STEP 3: Cahn-Hilliard step for global c ==========
    # Flux: J_c = -M_c · ∇μ
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
    
    # ========== STEP 4: Allen-Cahn step for η ==========
    lap_eta = compute_laplacian_2d(eta, dx)
    df_deta = np.empty_like(eta)
    
    for i in prange(nx):
        for j in prange(ny):
            # Phase fractions and derivative
            phi_e = phi_epsilon[i, j]
            phi_g = 1.0 - phi_e
            dphi_e_deta = d_phi_epsilon_deta(eta[i, j])
            
            # Bulk free energies evaluated at phase compositions (KKS)
            f_gamma = parabolic_free_energy(c_gamma_local[i, j], c_gamma_eq, K_gamma, E_gamma)
            f_epsilon = parabolic_free_energy(c_epsilon_local[i, j], c_epsilon_eq, K_epsilon, E_epsilon)
            
            # Driving force from bulk energy difference (Moelans chain rule)
            df_bulk = dphi_e_deta * (f_epsilon - f_gamma)
            
            # Double-well barrier term: g(η) = η²(1-η)² → ∂g/∂η = 2η(1-η)(1-2η)
            df_barrier = 2.0 * W_barrier * eta[i, j] * (1.0 - eta[i, j]) * (1.0 - 2.0 * eta[i, j])
            
            # Gradient energy contribution
            df_gradient = -kappa_eta * lap_eta[i, j]
            
            # Total variational derivative
            df_deta[i, j] = df_bulk + df_barrier + df_gradient
    
    # Allen-Cahn evolution (non-conserved)
    eta_new = eta - dt * L_struct * df_deta
    
    # ========== STEP 5: Physical bounds clipping ==========
    for i in prange(nx):
        for j in prange(ny):
            c_new[i, j] = _clip_scalar(c_new[i, j], 0.01, 0.99)
            eta_new[i, j] = _clip_scalar(eta_new[i, j], 0.0, 1.0)
    
    return c_new, eta_new, mu, c_gamma_local, c_epsilon_local, phi_epsilon


# =============================================================================
# MAIN SIMULATION CLASS: KKS Phase Transformation (CH + AC)
# =============================================================================

class MediloyKKSPhaseTransformation:
    """
    2D phase-field simulation of γ-FCC → ε-HCP transformation
    in Mediloy (Co-Cr-Mo) at 950°C using:
    
    - Cahn-Hilliard for conserved Co concentration c
    - Allen-Cahn for non-conserved HCP order parameter η  
    - Kim-Kim-Suzuki (KKS) model for interface partitioning
    - Moelans interpolation for smooth phase fractions
    
    η = 0 → γ-FCC (austenite), η = 1 → ε-HCP (martensite)
    """
    
    def __init__(self, nx=256, ny=256, T_celsius=950.0, 
                 V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        self.nx = nx
        self.ny = ny
        self.dx_dim = 1.0
        
        # Dimensionless model parameters (tuned for numerical stability)
        self.W_dim = 1.0              # Structural double-well barrier
        self.kappa_c_dim = 2.0        # Concentration gradient coefficient  
        self.kappa_eta_dim = 1.0      # Structural gradient coefficient
        self.M_c_dim = 1.0            # Chemical mobility for c (diffusion-limited)
        self.L_eta_dim = 100.0        # Allen-Cahn mobility for η (typically >> M_c)
        self.dt_dim = 0.005           # Dimensionless time step
        
        # Material parameters - parabolic free energy approximation
        self.T_celsius = T_celsius
        self.V_m = V_m_m3mol
        self.D_b = D_b_m2s
        
        # Equilibrium compositions (pseudo-binary Co-M_y)
        self.c_gamma_eq = 0.610       # Co fraction in FCC at equilibrium
        self.c_epsilon_eq = 0.575     # Co fraction in HCP (Cr-enriched)
        
        # Parabola curvatures (stiffness of free energy wells)
        self.K_gamma = 2.0e10         # J/m³ per (mole fraction)²
        self.K_epsilon = 2.0e10
        
        # Energy offsets (chemical driving force)
        self.E_gamma = 0.0
        self.E_epsilon = -400.0 / self.V_m  # ~ -6e7 J/m³ driving force
        
        # Initialize physical scales
        self.scales = PhysicalScalesMediloy(
            T_celsius=T_celsius,
            V_m_m3mol=V_m_m3mol,
            D_b_m2s=D_b_m2s
        )
        
        self._update_physical_params()
        
        # Initialize fields with explicit float64 dtype
        self.c = np.full((nx, ny), 0.61, dtype=np.float64)   # Co fraction
        self.eta = np.zeros((nx, ny), dtype=np.float64)       # η=0: FCC
        
        # Time tracking
        self.time_phys = 0.0
        self.step = 0
        
        # History for analysis
        self.history = {
            'time_phys': [],
            'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [],
            'hcp_fraction': [], 'fcc_fraction': [],
            'total_energy': [],
            'interface_mu': []  # Track chemical potential at interface
        }
        
        self.update_history()
    
    def _update_physical_params(self):
        """Convert dimensionless parameters to physical SI units."""
        (self.W_phys, self.kappa_c, self.kappa_eta, 
         self.M_c, self.L_struct, self.dt_phys, self.dx_phys) = \
            self.scales.dim_to_phys(
                self.W_dim, self.kappa_c_dim, self.kappa_eta_dim,
                self.M_c_dim, self.L_eta_dim, self.dt_dim, self.dx_dim
            )
        
        self.T_K = self.T_celsius + 273.15
    
    def set_physical_parameters(self, W_Jm3=None, kappa_c_Jm=None, kappa_eta_Jm=None,
                                M_c_m5Js=None, L_eta_m3Js=None, dt_s=None,
                                K_gamma=None, K_epsilon=None,
                                c_gamma_eq=None, c_epsilon_eq=None,
                                driving_force_Jmol=None):
        """Set physical parameters directly."""
        if driving_force_Jmol is not None:
            self.E_epsilon = -driving_force_Jmol / self.V_m
        
        if K_gamma is not None:
            self.K_gamma = K_gamma
        if K_epsilon is not None:
            self.K_epsilon = K_epsilon
        if c_gamma_eq is not None:
            self.c_gamma_eq = c_gamma_eq
        if c_epsilon_eq is not None:
            self.c_epsilon_eq = c_epsilon_eq
        
        if W_Jm3 is not None and self.scales.E0 > 0:
            self.W_dim = W_Jm3 / self.scales.E0
        if kappa_c_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_c_dim = kappa_c_Jm / (self.scales.E0 * self.scales.L0**2)
        if kappa_eta_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_eta_dim = kappa_eta_Jm / (self.scales.E0 * self.scales.L0**2)
        if M_c_m5Js is not None and self.scales.M0 > 0:
            self.M_c_dim = M_c_m5Js / self.scales.M0
        if L_eta_m3Js is not None and self.scales.L0_struct > 0:
            self.L_eta_dim = L_eta_m3Js / self.scales.L0_struct
        if dt_s is not None and self.scales.t0 > 0:
            self.dt_dim = dt_s / self.scales.t0
        
        self._update_physical_params()
    
    def initialize_random(self, c0=0.61, eta0=0.0, noise_c=0.02, noise_eta=0.02, seed=42):
        """Initialize with random noise around nominal values."""
        np.random.seed(seed)
        self.c = np.clip(c0 + noise_c * (2*np.random.random((self.nx, self.ny)) - 1), 0.01, 0.99)
        self.eta = np.clip(eta0 + noise_eta * (2*np.random.random((self.nx, self.ny)) - 1), 0.0, 1.0)
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
        self.update_history()
    
    def initialize_fcc_with_random_hcp_seeds(self, num_seeds=12, radius_grid=5,
                                             seed_co_fraction=0.58, seed=42):
        """Initialize with FCC matrix + random circular HCP seeds."""
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
    
    def clear_history(self):
        """Clear all history tracking arrays."""
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [],
            'hcp_fraction': [], 'fcc_fraction': [],
            'total_energy': [], 'interface_mu': []
        }
    
    def update_history(self):
        """Record current state to history arrays."""
        self.history['time_phys'].append(self.time_phys)
        self.history['eta_mean'].append(float(np.mean(self.eta)))
        self.history['eta_std'].append(float(np.std(self.eta)))
        self.history['c_mean'].append(float(np.mean(self.c)))
        self.history['c_std'].append(float(np.std(self.c)))
        self.history['hcp_fraction'].append(float(np.sum(self.eta > 0.5) / (self.nx * self.ny)))
        self.history['fcc_fraction'].append(float(np.sum(self.eta < 0.5) / (self.nx * self.ny)))
        
        # Track interface chemical potential (sample at η ≈ 0.5)
        interface_mask = (self.eta > 0.4) & (self.eta < 0.6)
        if np.any(interface_mask):
            self.history['interface_mu'].append(float(np.mean(self._compute_mu()[interface_mask])))
        else:
            self.history['interface_mu'].append(np.nan)
        
        # Compute total free energy (optional, expensive)
        if self.step % 10 == 0:
            try:
                energy = self.compute_total_free_energy()
                self.history['total_energy'].append(energy)
            except Exception:
                self.history['total_energy'].append(np.nan)
        else:
            self.history['total_energy'].append(np.nan)
    
    def _compute_mu(self):
        """Compute chemical potential field using KKS (helper for history)."""
        mu = np.empty_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                phi_g, phi_e = moelans_phase_fraction(self.eta[i, j])
                denom = phi_g / self.K_gamma + phi_e / self.K_epsilon
                weighted_c_eq = phi_g * self.c_gamma_eq + phi_e * self.c_epsilon_eq
                if denom > 1e-12:
                    mu[i, j] = (self.c[i, j] - weighted_c_eq) / denom
                else:
                    mu[i, j] = 0.0
        return mu
    
    def compute_total_free_energy(self):
        """
        Compute total free energy with KKS: 
        F = ∫[ Σ φ_α·f^α(c^α) + W·g(η) + (κ_η/2)|∇η|² ] dV
        Returns energy in Joules.
        """
        # Phase fractions
        phi_gamma = np.empty_like(self.c)
        phi_epsilon = np.empty_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                phi_gamma[i,j], phi_epsilon[i,j] = moelans_phase_fraction(self.eta[i,j])
        
        # Solve KKS for phase compositions
        mu = self._compute_mu()
        c_gamma_local = self.c_gamma_eq + mu / self.K_gamma
        c_epsilon_local = self.c_epsilon_eq + mu / self.K_epsilon
        
        # Bulk free energy (KKS: evaluate at phase compositions)
        f_gamma = parabolic_free_energy(c_gamma_local, self.c_gamma_eq, self.K_gamma, self.E_gamma)
        f_epsilon = parabolic_free_energy(c_epsilon_local, self.c_epsilon_eq, self.K_epsilon, self.E_epsilon)
        f_bulk = phi_gamma * f_gamma + phi_epsilon * f_epsilon
        
        # Barrier term
        f_barrier = self.W_phys * self.eta**2 * (1.0 - self.eta)**2
        
        # Gradient energy for η
        grad_eta_x = np.zeros_like(self.eta, dtype=np.float64)
        grad_eta_y = np.zeros_like(self.eta, dtype=np.float64)
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
        
        total_F = np.sum(f_bulk + f_barrier + f_gradient) * (self.dx_phys**2)
        return float(total_F)
    
    def run_step(self):
        """Execute one time step of KKS-coupled phase transformation."""
        self.c, self.eta, mu_field, c_g, c_e, phi_e = update_kks_phase_transformation(
            self.c, self.eta,
            self.dt_phys, self.dx_phys,
            self.K_gamma, self.K_epsilon,
            self.c_gamma_eq, self.c_epsilon_eq,
            self.E_gamma, self.E_epsilon,
            self.W_phys, self.kappa_eta, self.L_struct,
            self.M_c
        )
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()
    
    def run_steps(self, n_steps, progress_callback=None):
        """Execute multiple time steps with optional progress reporting."""
        for step_idx in range(n_steps):
            self.run_step()
            if progress_callback is not None and step_idx % 10 == 0:
                progress_callback(step_idx + 1, n_steps)
    
    def get_statistics(self):
        """Compute comprehensive simulation statistics."""
        domain_size_m = self.nx * self.dx_phys
        interface_width_eta = self.scales.phys_to_interface_width(self.kappa_eta, self.W_phys)
        
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
            'W_phys': self.W_phys,
            'M_c': self.M_c,
            'L_struct': self.L_struct,
            'dt_phys': self.dt_phys,
            'driving_force_Jm3': self.E_epsilon - self.E_gamma,
        }
    
    def get_time_series(self, key):
        """Retrieve time series data from history."""
        if key not in self.history:
            raise ValueError(f"Unknown history key: {key}")
        return np.array(self.history['time_phys']), np.array(self.history[key])


# =============================================================================
# STREAMLIT APPLICATION: Interactive KKS Phase Transformation Simulator
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    
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
    
    st.title("⚙️ Mediloy γ-FCC → ε-HCP Phase Transformation (KKS + Allen-Cahn)")
    st.markdown(f"""
    **Kim-Kim-Suzuki interface model with non-conserved structural order parameter**
    
    Pseudo-binary Co–M<sub>y</sub> phase-field simulation at {950}°C (1223 K)
    - η = 0 → γ-FCC (austenite), η = 1 → ε-HCP (martensite)
    - **c**: CONSERVED → Cahn-Hilliard (solute diffusion)
    - **η**: NON-CONSERVED → Allen-Cahn (interface motion)
    - **KKS**: μ^γ = μ^ε at interface (correct partitioning, no artificial drag)
    - Moelans interpolation for smooth phase fractions
    """)
    
    if 'sim' not in st.session_state:
        st.session_state.sim = MediloyKKSPhaseTransformation(
            nx=256, ny=256,
            T_celsius=950.0,
            V_m_m3mol=6.7e-6,
            D_b_m2s=5.0e-15
        )
        st.session_state.sim.initialize_fcc_with_random_hcp_seeds(
            num_seeds=12, radius_grid=5, seed_co_fraction=0.58, seed=42
        )
    
    sim = st.session_state.sim
    
    # =============================================================================
    # SIDEBAR: Control Panel
    # =============================================================================
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        st.subheader("⏱️ Time Stepping")
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            steps_input = st.number_input("Steps per update", 1, 5000, 100)
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
            seed_radius = st.slider("Seed radius (grid units)", 3, 50, 5, 1)
            if st.button("🌱 Initialize with HCP Seeds", use_container_width=True):
                sim.initialize_fcc_with_random_hcp_seeds(
                    num_seeds=num_seeds, radius_grid=seed_radius, seed=42
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
        
        st.subheader("⚙️ Model Parameters (KKS)")
        st.caption("Physical parameters for phase transformation")
        
        xi_eta_nm = sim.scales.phys_to_interface_width(sim.kappa_eta, sim.W_phys) * 1e9
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            W_phys = st.number_input("W (J/m³)", 1e4, 1e8, float(sim.W_phys), format="%.2e")
            kappa_eta_phys = st.number_input("κ_η (J/m)", 1e-13, 1e-9, float(sim.kappa_eta), format="%.2e")
            driving_force = st.number_input("ΔG (J/mol)", -2000, 2000, 400, 50)
        with col_p2:
            L_eta_phys = st.number_input("L_η (m³/J·s)", 1e-15, 1e-5, float(sim.L_struct), format="%.2e")
            dt_phys = st.number_input("Δt (s)", 1e-12, 1e-5, float(sim.dt_phys), format="%.2e")
            c_epsilon_eq = st.number_input("c_ε^eq", 0.50, 0.61, float(sim.c_epsilon_eq), 0.005)
        
        if st.button("Apply Parameters", use_container_width=True):
            sim.set_physical_parameters(
                W_Jm3=W_phys,
                kappa_eta_Jm=kappa_eta_phys,
                L_eta_m3Js=L_eta_phys,
                dt_s=dt_phys,
                driving_force_Jmol=driving_force,
                c_epsilon_eq=c_epsilon_eq
            )
            st.rerun()
        
        if xi_eta_nm < 1.5:
            st.warning(f"⚠️ Interface width ({xi_eta_nm:.2f} nm) < 1.5 nm: under-resolved")
        elif xi_eta_nm > 15.0:
            st.info(f"ℹ️ Interface width ({xi_eta_nm:.1f} nm) is quite diffuse")
        
        st.divider()
        
        stats = sim.get_statistics()
        st.subheader("📊 Live Statistics")
        
        st.markdown(f"""
        <div class="metric-card">
        <b>⏱️ Physical Time:</b> {stats['time_formatted']}
        </div>
        <div class="metric-card">
        <b>🔢 Simulation Step:</b> {stats['step']:,}
        </div>
        <div class="metric-card">
        <b>📐 Domain Size:</b> {stats['domain_size_formatted']}
        </div>
        <div class="metric-card">
        <b>🔲 Interface Width:</b> {stats['interface_width_eta_nm']:.2f} nm
        </div>
        <div class="metric-card">
        <b>⚡ Driving Force:</b> {stats['driving_force_Jm3']:.2e} J/m³
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown(f"**Phase Distribution**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("<span class='phase-hcp'>ε-HCP Fraction</span>", f"{stats['hcp_fraction']*100:.1f}%")
        with col_p2:
            st.metric("<span class='phase-fcc'>γ-FCC Fraction</span>", f"{stats['fcc_fraction']*100:.1f}%")
        
        st.markdown(f"**Field Statistics**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("⟨c_Co⟩", f"{stats['c_mean']:.3f}")
            st.metric("⟨η⟩", f"{stats['eta_mean']:.3f}")
        with col_c2:
            st.metric("σ(c)", f"{stats['c_std']:.3f}")
            st.metric("σ(η)", f"{stats['eta_std']:.3f}")
    
    # =============================================================================
    # MAIN CONTENT: Plotly Visualizations
    # =============================================================================
    
    extent_um_x = [0, sim.nx * sim.dx_phys * 1e6]
    extent_um_y = [0, sim.ny * sim.dx_phys * 1e6]
    
    # Row 1: Structural order parameter + Concentration field
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("ε-HCP Order Parameter η (Allen-Cahn)")
        st.caption(f"η = 0 (FCC) → η = 1 (HCP) | t = {stats['time_formatted']} | Non-conserved dynamics")
        
        fig_eta = go.Figure(data=go.Heatmap(
            z=sim.eta.T,
            x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
            y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
            colorscale='RdYlBu_r',
            zmin=0, zmax=1,
            colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP']),
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>η: %{z:.3f}<extra></extra>'
        ))
        fig_eta.update_layout(
            title="HCP Phase Distribution (Allen-Cahn)",
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            width=600, height=550,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_eta, use_container_width=True)
    
    with col_viz2:
        st.subheader("Co Concentration c_Co (Cahn-Hilliard)")
        st.caption("Nominal: c₀ = 0.61 | KKS partitioning: c^ε < c^γ at interface")
        
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
            title="Cobalt Mole Fraction (KKS Model)",
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            width=600, height=550,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_c, use_container_width=True)
    
    # Row 2: Overlay + Distribution
    col_overlay, col_hist = st.columns([2, 1])
    
    with col_overlay:
        st.subheader("Phase + Composition Overlay (KKS)")
        st.caption("HCP regions (red contours) with Cr enrichment (low c) inside plates")
        
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Heatmap(
            z=sim.c.T,
            x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
            y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
            colorscale='Viridis',
            zmin=0.55, zmax=0.67,
            opacity=0.7,
            colorbar=dict(title="c_Co", x=1.02),
            showscale=True
        ))
        levels = [0.3, 0.5, 0.7]
        for level in levels:
            fig_overlay.add_trace(go.Contour(
                z=sim.eta.T,
                x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
                y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
                contours=dict(start=level, end=level, size=0.1),
                line=dict(color='red', width=1.5),
                showscale=False,
                hoverinfo='skip',
                name=f'η = {level:.1f}'
            ))
        fig_overlay.update_layout(
            title="HCP Phase Boundaries on Co Concentration",
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            width=700, height=550,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    with col_hist:
        st.subheader("Field Distributions")
        
        fig_hist = make_subplots(rows=2, cols=1, subplot_titles=("η Distribution", "c_Co Distribution"))
        
        fig_hist.add_trace(
            go.Histogram(x=sim.eta.flatten(), nbinsx=40, name="η", marker_color='#e74c3c', opacity=0.7),
            row=1, col=1
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        
        fig_hist.add_trace(
            go.Histogram(x=sim.c.flatten(), nbinsx=40, name="c_Co", marker_color='#2ecc71', opacity=0.7),
            row=2, col=1
        )
        fig_hist.add_vline(x=0.61, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig_hist.update_layout(
            height=500, width=400,
            showlegend=False,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        fig_hist.update_xaxes(title_text="Value", row=2, col=1)
        fig_hist.update_xaxes(title_text="Value", row=1, col=1)
        fig_hist.update_yaxes(title_text="Frequency", row=1, col=1)
        fig_hist.update_yaxes(title_text="Frequency", row=2, col=1)
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 3: Kinetics plots
    st.divider()
    st.subheader("📈 Transformation Kinetics (KKS + Allen-Cahn)")
    
    if len(sim.history['time_phys']) > 3:
        times_s = np.array(sim.history['time_phys'])
        times_min = times_s / 60
        
        fig_kin = make_subplots(
            rows=1, cols=3,
            subplot_titles=("HCP Fraction Growth", 
                            "Field Std. Dev. (Coarsening)",
                            "Mean Composition"),
            shared_xaxes=True,
            x_title="Time (minutes)"
        )
        
        # Plot 1: HCP fraction (should INCREASE with Allen-Cahn)
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=np.array(sim.history['hcp_fraction'])*100,
                       mode='lines', name='HCP %',
                       line=dict(color='#e74c3c', width=2.5)),
            row=1, col=1
        )
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=np.array(sim.history['fcc_fraction'])*100,
                       mode='lines', name='FCC %',
                       line=dict(color='#2ecc71', width=2)),
            row=1, col=1
        )
        fig_kin.update_yaxes(title_text="Phase fraction (%)", row=1, col=1)
        
        # Plot 2: Standard deviations
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_std'],
                       mode='lines', name='σ(η)',
                       line=dict(color='#9b59b6', width=2)),
            row=1, col=2
        )
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['c_std'],
                       mode='lines', name='σ(c)',
                       line=dict(color='#f39c12', width=1.5, dash='dash')),
            row=1, col=2
        )
        fig_kin.update_yaxes(title_text="Standard deviation", row=1, col=2)
        
        # Plot 3: Mean values (⟨η⟩ now CHANGES with Allen-Cahn!)
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_mean'],
                       mode='lines', name='⟨η⟩',
                       line=dict(color='#e74c3c', width=2)),
            row=1, col=3
        )
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['c_mean'],
                       mode='lines', name='⟨c⟩ (conserved)',
                       line=dict(color='#2ecc71', width=2)),
            row=1, col=3
        )
        fig_kin.update_yaxes(title_text="Mean value", row=1, col=3)
        
        fig_kin.update_layout(
            height=450, width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_kin, use_container_width=True)
        
        # Free energy evolution
        with st.expander("🔋 Free Energy Evolution (click to expand)"):
            valid_energy = [e for e in sim.history['total_energy'] if not np.isnan(e)]
            valid_times = [t for t, e in zip(times_min, sim.history['total_energy']) if not np.isnan(e)]
            
            if len(valid_energy) > 2:
                fig_fe = go.Figure()
                fig_fe.add_trace(go.Scatter(
                    x=valid_times, y=valid_energy,
                    mode='lines+markers', name='Total free energy',
                    line=dict(color='#8e44ad', width=2),
                    marker=dict(size=3)
                ))
                fig_fe.update_layout(
                    title="Free Energy Minimization (KKS + Allen-Cahn)",
                    xaxis_title="Time (minutes)",
                    yaxis_title="Total free energy (J)",
                    height=400
                )
                st.plotly_chart(fig_fe, use_container_width=True)
            else:
                st.info("Run more steps to compute free energy evolution.")
    else:
        st.info("📊 Run simulation for at least 4 steps to display kinetics plots.")
    
    # =============================================================================
    # Export Section
    # =============================================================================
    st.divider()
    st.subheader("💾 Export Results")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button("📸 Save η Snapshot", use_container_width=True):
            fig_snap = go.Figure(data=go.Heatmap(
                z=sim.eta.T,
                x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
                y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
                colorscale='RdYlBu_r',
                zmin=0, zmax=1,
                colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP'])
            ))
            fig_snap.update_layout(
                title=f"Mediloy HCP (Allen-Cahn η) – t = {stats['time_formatted']}",
                xaxis_title="x (μm)",
                yaxis_title="y (μm)",
                width=800, height=700
            )
            img_bytes = fig_snap.to_image(format="png", width=800, height=700, scale=2)
            st.download_button(
                label="⬇️ Download PNG",
                data=img_bytes,
                file_name=f"Mediloy_eta_AC_t{sim.time_phys:.2e}s.png",
                mime="image/png",
                use_container_width=True
            )
    
    with col_exp2:
        if st.button("📊 Save Kinetics Data", use_container_width=True):
            csv_lines = ["time_s,time_min,eta_mean,eta_std,c_mean,c_std,hcp_frac,fcc_frac,energy_J,interface_mu"]
            for i in range(len(sim.history['time_phys'])):
                t_s = sim.history['time_phys'][i]
                line = f"{t_s:.6e},"
                line += f"{t_s/60:.6e},"
                line += f"{sim.history['eta_mean'][i]:.6f},"
                line += f"{sim.history['eta_std'][i]:.6f},"
                line += f"{sim.history['c_mean'][i]:.6f},"
                line += f"{sim.history['c_std'][i]:.6f},"
                line += f"{sim.history['hcp_fraction'][i]:.6f},"
                line += f"{sim.history['fcc_fraction'][i]:.6f},"
                line += f"{sim.history['total_energy'][i]:.6e},"
                line += f"{sim.history['interface_mu'][i]:.6e}"
                csv_lines.append(line)
            
            csv_content = "\n".join(csv_lines)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_content,
                file_name="mediloy_kks_kinetics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        if st.button("⚙️ Save Simulation State", use_container_width=True):
            npz_buf = BytesIO()
            np.savez_compressed(
                npz_buf,
                concentration=sim.c,
                order_parameter=sim.eta,
                time_phys=sim.time_phys,
                step=sim.step,
                params={
                    'T_celsius': sim.T_celsius,
                    'K_gamma': sim.K_gamma,
                    'K_epsilon': sim.K_epsilon,
                    'c_gamma_eq': sim.c_gamma_eq,
                    'c_epsilon_eq': sim.c_epsilon_eq,
                    'E_gamma': sim.E_gamma,
                    'E_epsilon': sim.E_epsilon,
                    'W_phys': sim.W_phys,
                    'kappa_c': sim.kappa_c,
                    'kappa_eta': sim.kappa_eta,
                    'M_c': sim.M_c,
                    'L_struct': sim.L_struct,
                    'dt_phys': sim.dt_phys,
                    'dx_phys': sim.dx_phys,
                }
            )
            npz_buf.seek(0)
            filename = f"Mediloy_KKS_state_t{sim.time_phys:.2e}s.npz"
            st.download_button(
                label="⬇️ Download NPZ",
                data=npz_buf.getvalue(),
                file_name=filename,
                mime="application/octet-stream",
                use_container_width=True
            )
    
    # =============================================================================
    # Physics Guide & Documentation
    # =============================================================================
    with st.expander("ℹ️ Physics Guide: KKS + Allen-Cahn for Mediloy", expanded=False):
        st.markdown("""
        ## ⚙️ Mediloy γ-FCC → ε-HCP: KKS Interface Model + Allen-Cahn Dynamics
        
        This simulation implements a **state-of-the-art phase-field model** for 
        diffusional phase transformation with correct interface thermodynamics.
        
        ### Governing Equations
        
        ```
        ∂c/∂t = ∇·[ M_c ∇μ ]                    (Cahn-Hilliard, conserved)
        ∂η/∂t = -L_struct · (∂f/∂η - κ_η∇²η)  (Allen-Cahn, non-conserved)
        ```
        
        ### Kim-Kim-Suzuki (KKS) Interface Model
        
        At every point in the diffuse interface, the model enforces:
        
        1. **Equal chemical potentials**: μ^γ(c^γ) = μ^ε(c^ε) = μ
        2. **Mixture rule**: c = φ_γ·c^γ + φ_ε·c^ε
        
        This ensures:
        - ✅ Correct Cr/Mo partitioning: HCP is Cr-rich (lower c_Co)
        - ✅ No artificial solute drag at interfaces
        - ✅ Accurate interface kinetics even with wide interfaces
        
        ### Free Energy Functional (KKS + Moelans)
        
        ```
        F = ∫[ φ_γ·f^γ(c^γ) + φ_ε·f^ε(c^ε) + W·η²(1-η)² + (κ_η/2)|∇η|² ] dV
        ```
        
        | Term | Expression | Meaning |
        |------|-----------|---------|
        | f^α(c^α) | ½K_α(c^α - c_α^eq)² + E_α | Parabolic bulk energy per phase |
        | φ_α | η_α² / Ση_ρ² | Moelans phase fraction (smooth, vanishing derivatives) |
        | W·η²(1-η)² | Double-well barrier | Controls interface energy |
        
        ### Key Differences from Previous Dual Cahn-Hilliard Model
        
        | Feature | Dual Cahn-Hilliard | KKS + Allen-Cahn (This Model) |
        |---------|-------------------|-------------------------------|
        | η evolution | ∂η/∂t = ∇·[M_η∇(δF/δη)] | ∂η/∂t = -L·(δF/δη) |
        | η conservation | ✅ Globally conserved | ❌ Non-conserved (grows/shrinks) |
        | Interface composition | c^γ = c^ε = c (artificial) | c^γ ≠ c^ε (KKS partitioning) |
        | Physical regime | Spinodal decomposition | Classic phase transformation |
        | ⟨η⟩ vs time | Constant | Increases with driving force |
        
        ### When to Use This Model?
        
        ✓ Modeling **γ→ε martensitic/diffusional transformation** in Mediloy  
        ✓ Studying **interface-controlled growth** of HCP plates  
        ✓ When **solute partitioning** at interface matters (Cr enrichment in HCP)  
        ✓ When the **total HCP fraction should change** with time (nucleation + growth)  
        
        ### Numerical Stability Tips
        
        ```
        Δt ≲ min[ 0.01·(Δx)⁴/(M_c·κ_c),  0.01·(Δx)²/(L_struct·W) ]
        ```
        
        Recommendations:
        1. Ensure ξ/Δx ≥ 3 for proper interface resolution
        2. L_struct is typically 10-100× larger than M_c (structural relaxation is faster)
        3. Monitor c ∈ [0.01, 0.99] and η ∈ [0, 1] for physical bounds
        4. Check that ⟨c⟩ remains constant (conservation) while ⟨η⟩ evolves
        
        ### Interpreting Results
        
        - **⟨η⟩ increases**: HCP phase grows from seeds (Allen-Cahn driving force)
        - **⟨c⟩ constant**: Global Co conservation (Cahn-Hilliard)
        - **c^ε < c^γ inside interface**: Cr enrichment in HCP (KKS partitioning)
        - **Energy decay**: Total F decreases monotonically (thermodynamic consistency)
        
        ### References
        
        1. Kim, S.G., Kim, W.T., Suzuki, T. (1999). *Phys. Rev. E* **60**, 7186. [KKS model]
        2. Moelans, N., Blanpain, B., Wollants, P. (2008). *Phys. Rev. B* **78**, 024113. [Interpolation]
        3. Steinbach, I. (2013). *Model. Simul. Mater. Sci. Eng.* **21**, 015005. [Review]
        4. Yamanaka, K. et al. (2018). *Dent. Mater. J.* **37**, 1. [Mediloy microstructure]
        5. Provatas, N., Elder, K. (2010). *Phase-Field Methods in Materials Science*. [Textbook]
        """)
    
    # =============================================================================
    # Auto-run Feature
    # =============================================================================
    st.sidebar.divider()
    
    with st.sidebar.expander("🔄 Auto-run Settings"):
        auto_run = st.checkbox("Enable auto-run", value=False)
        auto_speed = st.slider("Speed (steps/second)", 1, 500, 50)
        auto_max_steps = st.number_input("Max steps (0 = unlimited)", 0, 200000, 0)
        
        if auto_run:
            stop_auto = st.button("⏹️ Stop Auto-run", type="secondary")
            
            if not stop_auto:
                steps_this_frame = min(
                    auto_speed, 
                    auto_max_steps - sim.step if auto_max_steps > 0 else auto_speed
                )
                
                if steps_this_frame > 0 and (auto_max_steps == 0 or sim.step < auto_max_steps):
                    with st.spinner(f"Auto-running {steps_this_frame} steps..."):
                        sim.run_steps(steps_this_frame)
                    st.rerun()
                elif auto_max_steps > 0 and sim.step >= auto_max_steps:
                    st.success(f"✓ Reached max steps: {sim.step:,}")
    
    # Footer
    st.markdown("---")
    st.caption(
        "Mediloy γ→ε KKS Phase Transformation | c: Cahn-Hilliard (conserved) | "
        "η: Allen-Cahn (non-conserved) | KKS interface: μ^γ=μ^ε | "
        f"T = {sim.T_celsius}°C | Pseudo-binary Co-M<sub>y</sub> (c₀ = 0.61)"
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    print("⚙️ Starting Mediloy KKS Phase Transformation Simulator...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Numba: JIT compilation enabled")
    print(f"   Streamlit: launching interactive app")
    print(f"   Temperature: 950°C (1223.15 K)")
    print(f"   Model: Cahn-Hilliard (c) + Allen-Cahn (η) + KKS interface")
    print(f"   Initial condition: FCC matrix + random HCP seeds")
    
    main()
