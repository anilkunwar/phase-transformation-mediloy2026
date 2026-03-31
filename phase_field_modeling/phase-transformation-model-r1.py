# =============================================================================
# MEDILOY PHASE TRANSFORMATION SIMULATOR – PLOTLY VISUALIZATION VERSION
# γ-FCC → ε-HCP Martensitic Transformation in Co-Cr-Mo Dental Alloys
# Temperature: 950°C (1223.15 K) - Pseudo-binary Co-M_y model
# =============================================================================
# Author: Phase-Field Modeling Framework
# Dependencies: numpy, numba, plotly, streamlit
# Run with: streamlit run mediloy_phase_transformation_plotly.py
# =============================================================================

import numpy as np
from numba import njit, prange, float64
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
from io import BytesIO

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
    - Martensitic (diffusionless) transformation with solute drag
    
    References:
    - Brachavort et al., Acta Mater. (2015) - Co-Cr phase diagrams
    - Yamanaka et al., Dent. Mater. J. (2018) - Mediloy microstructure
    - Steinbach, Model. Simul. Mater. Sci. Eng. (2009) - Phase-field review
    """
    
    def __init__(self, T_celsius=950.0, V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        """
        Initialize physical scales for Mediloy at specified temperature.
        
        Parameters:
        -----------
        T_celsius : float - Temperature in °C (default: 950°C = 1223.15 K)
        V_m_m3mol : float - Molar volume in m³/mol (Co-based alloy: ~6.7 cm³/mol)
        D_b_m2s : float - Cr diffusion coefficient in Co matrix [m²/s] at 950°C
        """
        # Fundamental constants
        self.R = np.float64(8.314462618)          # J/(mol·K) - Universal gas constant
        self.T = np.float64(T_celsius + 273.15)   # K - Absolute temperature
        self.T_celsius = np.float64(T_celsius)    # °C - For display
        
        # Material properties (Co-Cr-Mo alloy at 950°C)
        self.V_m = np.float64(V_m_m3mol)          # m³/mol - Molar volume
        self.D_b = np.float64(D_b_m2s)            # m²/s - Cr diffusion in Co matrix
        
        # Characteristic scales (derived from literature values)
        self.L0 = np.float64(2.0e-9)              # m - Reference length
        
        # Energy scale: ΔG_chem ≈ 400 J/mol at T ≈ T0 for Mediloy
        delta_G_mol = np.float64(400.0)           # J/mol - Chemical driving force
        self.E0 = np.float64(delta_G_mol / self.V_m)   # J/m³ - Energy density scale
        
        # Time and mobility scales
        self.t0 = np.float64(self.L0**2 / self.D_b)    # s - Diffusion time scale
        self.M0 = np.float64(self.D_b / self.E0)        # m⁵/(J·s) - Chemical mobility scale
        
        # Structural mobility scale (Allen-Cahn, much faster than diffusion)
        self.L0_struct = np.float64(1.0e-8 / (self.E0 * self.t0))  # m³/(J·s)
        
        # Log initialization for debugging
        print(f"Mediloy scales initialized at {self.T_celsius}°C ({self.T:.1f} K)")
        print(f"  L0 = {self.L0*1e9:.2f} nm, t0 = {self.t0:.2e} s")
        print(f"  E0 = {self.E0:.2e} J/m³, M0 = {self.M0:.2e} m⁵/(J·s)")
        print(f"  D_b = {self.D_b:.2e} m²/s, V_m = {self.V_m*1e6:.2f} cm³/mol")
    
    def dim_to_phys(self, W_dim, kappa_c_dim, kappa_eta_dim, M_dim, L_dim, dt_dim, dx_dim=1.0):
        """Convert dimensionless parameters to physical SI units."""
        W_phys = np.float64(W_dim * self.E0)                           # J/m³
        kappa_c_phys = np.float64(kappa_c_dim * self.E0 * self.L0**2)  # J/m
        kappa_eta_phys = np.float64(kappa_eta_dim * self.E0 * self.L0**2)  # J/m
        M_phys = np.float64(M_dim * self.M0)                            # m⁵/(J·s)
        L_phys = np.float64(L_dim * self.L0_struct)                     # m³/(J·s)
        dt_phys = np.float64(dt_dim * self.t0)                          # s
        dx_phys = np.float64(dx_dim * self.L0)                          # m
        
        return W_phys, kappa_c_phys, kappa_eta_phys, M_phys, L_phys, dt_phys, dx_phys
    
    def phys_to_interface_width(self, kappa_phys, W_phys):
        """Estimate interface width from gradient energy and barrier height."""
        if W_phys <= 0 or kappa_phys <= 0:
            return np.float64(2.0e-9)
        return np.sqrt(np.float64(kappa_phys / W_phys))
    
    def format_time(self, t_seconds):
        """Format physical time with appropriate SI prefix."""
        if not np.isfinite(t_seconds) or t_seconds < 0:
            return "0 s"
        if t_seconds < 1e-9:
            return f"{t_seconds*1e12:.2f} ps"
        elif t_seconds < 1e-6:
            return f"{t_seconds*1e9:.2f} ns"
        elif t_seconds < 1e-3:
            return f"{t_seconds*1e6:.2f} μs"
        elif t_seconds < 1.0:
            return f"{t_seconds*1e3:.2f} ms"
        elif t_seconds < 60:
            return f"{t_seconds:.3f} s"
        elif t_seconds < 3600:
            return f"{t_seconds/60:.2f} min"
        elif t_seconds < 86400:
            return f"{t_seconds/3600:.3f} h"
        else:
            return f"{t_seconds/86400:.3f} d"
    
    def format_length(self, L_meters):
        """Format length with appropriate SI prefix."""
        if not np.isfinite(L_meters) or L_meters < 0:
            return "0 nm"
        if L_meters < 1e-10:
            return f"{L_meters*1e12:.2f} pm"
        elif L_meters < 1e-9:
            return f"{L_meters*1e10:.2f} Å"
        elif L_meters < 1e-6:
            return f"{L_meters*1e9:.2f} nm"
        elif L_meters < 1e-3:
            return f"{L_meters*1e6:.2f} μm"
        elif L_meters < 1.0:
            return f"{L_meters*1e3:.2f} mm"
        else:
            return f"{L_meters:.3f} m"
    
    def format_energy_density(self, E_Jm3):
        """Format energy density with appropriate units."""
        if not np.isfinite(E_Jm3):
            return "0 J/m³"
        if abs(E_Jm3) < 1e3:
            return f"{E_Jm3:.2e} J/m³"
        elif abs(E_Jm3) < 1e6:
            return f"{E_Jm3/1e3:.2f} kJ/m³"
        else:
            return f"{E_Jm3/1e6:.2f} MJ/m³"


# =============================================================================
# NUMBA-ACCELERATED KERNELS: Hybrid Cahn-Hilliard + Allen-Cahn
# =============================================================================

@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def chemical_free_energy_density_scalar(c, T_K, Omega_Jmol, V_m):
    """
    Regular solution model for Co-M_y pseudo-binary alloy (SCALAR version for Numba).
    
    f_chem(c) = (RT/V_m)[c·ln(c) + (1-c)·ln(1-c)] + (Ω/V_m)·c·(1-c)
    
    Parameters:
    -----------
    c : float64 - Co mole fraction (0 ≤ c ≤ 1)
    T_K : float64 - Temperature in Kelvin
    Omega_Jmol : float64 - Regular solution parameter [J/mol]
    V_m : float64 - Molar volume [m³/mol]
    
    Returns:
    --------
    float64 - Chemical free energy density [J/m³]
    """
    R = np.float64(8.314462618)
    eps = np.float64(1e-12)
    c_safe = np.clip(c, eps, np.float64(1.0) - eps)
    f_mix = (R * T_K / V_m) * (c_safe * np.log(c_safe) + (np.float64(1.0) - c_safe) * np.log(np.float64(1.0) - c_safe))
    f_excess = (Omega_Jmol / V_m) * c * (np.float64(1.0) - c)
    return np.float64(f_mix + f_excess)


@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def d_fchem_dc_scalar(c, T_K, Omega_Jmol, V_m):
    """Chemical potential contribution: ∂f_chem/∂c (SCALAR version)."""
    R = np.float64(8.314462618)
    eps = np.float64(1e-12)
    c_safe = np.clip(c, eps, np.float64(1.0) - eps)
    mu_mix = (R * T_K / V_m) * np.log(c_safe / (np.float64(1.0) - c_safe))
    mu_excess = (Omega_Jmol / V_m) * (np.float64(1.0) - np.float64(2.0) * c)
    return np.float64(mu_mix + mu_excess)


@njit(float64(float64, float64), fastmath=True, cache=True)
def structural_free_energy_scalar(eta, W_struct):
    """Double-well potential for structural order parameter (SCALAR version)."""
    return np.float64(W_struct * eta**2 * (np.float64(1.0) - eta)**2)


@njit(float64(float64, float64), fastmath=True, cache=True)
def d_fstruct_deta_scalar(eta, W_struct):
    """Variational derivative: ∂f_struct/∂η (SCALAR version)."""
    one = np.float64(1.0)
    two = np.float64(2.0)
    return np.float64(two * W_struct * eta * (one - eta) * (one - two * eta))


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def coupling_free_energy_scalar(c, eta, lambda_coup):
    """Coupling term: HCP phase stabilized by higher M_y content (SCALAR version)."""
    return np.float64(-lambda_coup * (np.float64(1.0) - c) * eta**2)


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def d_fcoup_dc_scalar(c, eta, lambda_coup):
    """∂f_coup/∂c = +λ·η² (SCALAR version)."""
    return np.float64(lambda_coup * eta**2)


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def d_fcoup_deta_scalar(c, eta, lambda_coup):
    """∂f_coup/∂η = -2λ·(1-c)·η (SCALAR version)."""
    two = np.float64(2.0)
    one = np.float64(1.0)
    return np.float64(-two * lambda_coup * (one - c) * eta)


@njit(float64[:,:](float64[:,:], float64), fastmath=True, parallel=True)
def compute_laplacian_2d(field, dx):
    """Compute 5-point stencil Laplacian with periodic BCs."""
    nx, ny = field.shape
    lap = np.zeros((nx, ny), dtype=np.float64)
    dx_sq = np.float64(dx * dx)
    
    for i in prange(nx):
        for j in prange(ny):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            lap[i, j] = (field[ip1, j] + field[im1, j] + 
                         field[i, jp1] + field[i, jm1] - 
                         np.float64(4.0) * field[i, j]) / dx_sq
    return lap


@njit(float64[:,:](float64[:,:], float64[:,:], float64), fastmath=True, parallel=True)
def compute_gradient_divergence_2d(flux_x, flux_y, dx):
    """Compute divergence of vector field: ∇·J = ∂Jx/∂x + ∂Jy/∂y."""
    nx, ny = flux_x.shape
    div = np.zeros((nx, ny), dtype=np.float64)
    two_dx = np.float64(2.0 * dx)
    
    for i in prange(nx):
        for j in prange(ny):
            im1 = (i - 1) % nx
            ip1 = (i + 1) % nx
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            div_x = (flux_x[ip1, j] - flux_x[im1, j]) / two_dx
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / two_dx
            div[i, j] = div_x + div_y
    return div


@njit((float64[:,:], float64[:,:], float64, float64, float64, float64, 
       float64, float64, float64, float64, float64, float64, float64),
      fastmath=True, parallel=True, cache=True)
def update_mediloy_hybrid(c, eta, dt, dx, kappa_c, kappa_eta, M_chem, L_struct,
                          T_K, Omega_Jmol, V_m, W_struct, lambda_coup):
    """
    One time step of hybrid Cahn-Hilliard + Allen-Cahn dynamics.
    
    All functions called here use SCALAR versions with explicit type signatures.
    """
    nx, ny = c.shape
    c_new = np.copy(c)
    eta_new = np.copy(eta)
    
    # Pre-compute Laplacians
    lap_c = compute_laplacian_2d(c, dx)
    lap_eta = compute_laplacian_2d(eta, dx)
    
    # ========== CONCENTRATION FIELD (Cahn-Hilliard) ==========
    mu_chem = np.zeros((nx, ny), dtype=np.float64)
    for i in prange(nx):
        for j in prange(ny):
            mu_bulk = d_fchem_dc_scalar(c[i, j], T_K, Omega_Jmol, V_m)
            mu_coup = d_fcoup_dc_scalar(c[i, j], eta[i, j], lambda_coup)
            mu_grad = -kappa_c * lap_c[i, j]
            mu_chem[i, j] = mu_bulk + mu_coup + mu_grad
    
    flux_c_x = np.zeros((nx, ny), dtype=np.float64)
    flux_c_y = np.zeros((nx, ny), dtype=np.float64)
    two_dx = np.float64(2.0 * dx)
    
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            
            grad_mu_x = (mu_chem[ip1, j] - mu_chem[im1, j]) / two_dx
            grad_mu_y = (mu_chem[i, jp1] - mu_chem[i, jm1]) / two_dx
            
            flux_c_x[i, j] = -M_chem * grad_mu_x
            flux_c_y[i, j] = -M_chem * grad_mu_y
    
    div_flux_c = compute_gradient_divergence_2d(flux_c_x, flux_c_y, dx)
    c_new = c + dt * div_flux_c
    
    # ========== STRUCTURAL ORDER PARAMETER (Allen-Cahn) ==========
    dF_deta = np.zeros((nx, ny), dtype=np.float64)
    for i in prange(nx):
        for j in prange(ny):
            dF_struct = d_fstruct_deta_scalar(eta[i, j], W_struct)
            dF_coup = d_fcoup_deta_scalar(c[i, j], eta[i, j], lambda_coup)
            dF_grad = -kappa_eta * lap_eta[i, j]
            dF_deta[i, j] = dF_struct + dF_coup + dF_grad
    
    eta_new = eta - dt * L_struct * dF_deta
    
    # ========== PHYSICAL BOUNDS ==========
    c_new = np.clip(c_new, np.float64(0.01), np.float64(0.99))
    eta_new = np.clip(eta_new, np.float64(0.0), np.float64(1.0))
    
    return c_new, eta_new


# =============================================================================
# VECTORIZED NUMBA KERNELS FOR FREE ENERGY COMPUTATION
# These avoid the scalar typing issue by operating on entire arrays
# =============================================================================

@njit(float64[:,:](float64[:,:], float64[:,:], float64, float64, float64, float64, float64),
      fastmath=True, parallel=True, cache=True)
def compute_bulk_free_energy_vectorized(c, eta, T_K, Omega_Jmol, V_m, W_struct, lambda_coup):
    """
    Vectorized computation of bulk free energy density.
    
    f_bulk = f_chem(c) + f_struct(η) + f_coup(c,η)
    
    This function operates on entire 2D arrays, avoiding scalar typing issues.
    """
    nx, ny = c.shape
    f_bulk = np.zeros((nx, ny), dtype=np.float64)
    
    for i in prange(nx):
        for j in prange(ny):
            # Chemical free energy
            R = np.float64(8.314462618)
            eps = np.float64(1e-12)
            c_safe = np.clip(c[i, j], eps, np.float64(1.0) - eps)
            f_chem = (R * T_K / V_m) * (c_safe * np.log(c_safe) + (np.float64(1.0) - c_safe) * np.log(np.float64(1.0) - c_safe))
            f_chem += (Omega_Jmol / V_m) * c[i, j] * (np.float64(1.0) - c[i, j])
            
            # Structural free energy
            f_struct = W_struct * eta[i, j]**2 * (np.float64(1.0) - eta[i, j])**2
            
            # Coupling term
            f_coup = -lambda_coup * (np.float64(1.0) - c[i, j]) * eta[i, j]**2
            
            f_bulk[i, j] = f_chem + f_struct + f_coup
    
    return f_bulk


@njit(float64[:,:](float64[:,:], float64[:,:], float64, float64, float64),
      fastmath=True, parallel=True, cache=True)
def compute_gradient_energy_vectorized(c, eta, kappa_c, kappa_eta, dx):
    """
    Vectorized computation of gradient energy density.
    
    f_grad = (κ_c/2)|∇c|² + (κ_η/2)|∇η|²
    """
    nx, ny = c.shape
    f_grad = np.zeros((nx, ny), dtype=np.float64)
    two_dx = np.float64(2.0 * dx)
    
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            
            # Gradients for c
            gcx = (c[ip1, j] - c[im1, j]) / two_dx
            gcy = (c[i, jp1] - c[i, jm1]) / two_dx
            
            # Gradients for eta
            gex = (eta[ip1, j] - eta[im1, j]) / two_dx
            gey = (eta[i, jp1] - eta[i, jm1]) / two_dx
            
            # Gradient energy
            f_grad[i, j] = np.float64(0.5) * kappa_c * (gcx**2 + gcy**2) + \
                          np.float64(0.5) * kappa_eta * (gex**2 + gey**2)
    
    return f_grad


# =============================================================================
# MAIN SIMULATION CLASS: Mediloy Phase Transformation
# =============================================================================

class MediloyPhaseTransformation:
    """
    2D Phase-field simulation of γ-FCC → ε-HCP martensitic transformation
    in Mediloy (Co-Cr-Mo dental alloy) at T = 950°C (1223 K).
    
    Model: Pseudo-binary Co-M_y alloy with:
    - c: Co mole fraction (conserved, Cahn-Hilliard dynamics)
    - η: Structural order parameter (non-conserved, Allen-Cahn dynamics)
      η = 0 → FCC (γ), η = 1 → HCP (ε)
    
    Key physics at 950°C:
    - Near-equilibrium transformation (small ΔG_chem)
    - Martensitic growth via Shockley partial dislocations
    - Solute drag: Cr enrichment stabilizes HCP at interfaces
    - Diffusion-limited kinetics (slow coarsening)
    
    All parameters in physical SI units (m, s, J/m³).
    """
    
    def __init__(self, nx=256, ny=256, T_celsius=950.0, 
                 Omega_Jmol=12000.0, V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        """Initialize Mediloy phase transformation simulation."""
        # Grid parameters
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx_dim = np.float64(1.0)
        
        # Dimensionless model parameters
        self.W_dim = np.float64(1.0)
        self.kappa_c_dim = np.float64(2.0)
        self.kappa_eta_dim = np.float64(1.0)
        self.M_dim = np.float64(1.0)
        self.L_dim = np.float64(50.0)
        self.lambda_coup_dim = np.float64(2.0)
        self.dt_dim = np.float64(0.005)
        
        # Material parameters - explicitly typed
        self.T_celsius = np.float64(T_celsius)
        self.Omega_Jmol = np.float64(Omega_Jmol)
        self.V_m = np.float64(V_m_m3mol)
        self.D_b = np.float64(D_b_m2s)
        
        # Initialize physical scales
        self.scales = PhysicalScalesMediloy(
            T_celsius=float(T_celsius),
            V_m_m3mol=float(V_m_m3mol),
            D_b_m2s=float(D_b_m2s)
        )
        
        # Convert to physical parameters
        self._update_physical_params()
        
        # Initialize fields
        self.c = np.full((nx, ny), np.float64(0.61), dtype=np.float64)
        self.eta = np.zeros((nx, ny), dtype=np.float64)
        
        # Time tracking
        self.time_phys = np.float64(0.0)
        self.step = int(0)
        
        # History for analysis
        self.history = {
            'time_phys': [],
            'eta_mean': [],
            'eta_std': [],
            'c_mean': [],
            'c_std': [],
            'total_energy': []
        }
        
        # Auto-record initial state (with error handling for first call)
        try:
            self.update_history()
        except Exception as e:
            print(f"Warning: Initial energy computation skipped: {e}")
            self.history['total_energy'].append(np.nan)
    
    def _update_physical_params(self):
        """Convert dimensionless parameters to physical SI units."""
        (self.W_phys, self.kappa_c, self.kappa_eta, 
         self.M_chem, self.L_struct, self.dt_phys, self.dx_phys) = \
            self.scales.dim_to_phys(
                self.W_dim, self.kappa_c_dim, self.kappa_eta_dim,
                self.M_dim, self.L_dim, self.dt_dim, self.dx_dim
            )
        
        self.lambda_coup = np.float64(self.lambda_coup_dim * self.scales.E0)
        self.T_K = np.float64(self.T_celsius + 273.15)
    
    def set_physical_parameters(self, W_Jm3=None, kappa_c_Jm=None, kappa_eta_Jm=None,
                                M_m5Js=None, L_m3Js=None, dt_s=None,
                                lambda_coup_Jm3=None, Omega_Jmol=None, D_b_m2s=None):
        """Set physical parameters directly (converts to dimensionless internally)."""
        if Omega_Jmol is not None:
            self.Omega_Jmol = np.float64(Omega_Jmol)
        if D_b_m2s is not None:
            self.D_b = np.float64(D_b_m2s)
            self.scales = PhysicalScalesMediloy(
                T_celsius=float(self.T_celsius),
                V_m_m3mol=float(self.V_m),
                D_b_m2s=float(self.D_b)
            )
        
        if W_Jm3 is not None and self.scales.E0 > 0:
            self.W_dim = np.float64(W_Jm3 / self.scales.E0)
        if kappa_c_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_c_dim = np.float64(kappa_c_Jm / (self.scales.E0 * self.scales.L0**2))
        if kappa_eta_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_eta_dim = np.float64(kappa_eta_Jm / (self.scales.E0 * self.scales.L0**2))
        if M_m5Js is not None and self.scales.M0 > 0:
            self.M_dim = np.float64(M_m5Js / self.scales.M0)
        if L_m3Js is not None and self.scales.L0_struct > 0:
            self.L_dim = np.float64(L_m3Js / self.scales.L0_struct)
        if lambda_coup_Jm3 is not None and self.scales.E0 > 0:
            self.lambda_coup_dim = np.float64(lambda_coup_Jm3 / self.scales.E0)
        if dt_s is not None and self.scales.t0 > 0:
            self.dt_dim = np.float64(dt_s / self.scales.t0)
        
        self._update_physical_params()
    
    def initialize_fcc_with_random_hcp_seeds(self, num_seeds=12, radius_grid=5,
                                             seed_co_fraction=0.58, seed=42):
        """Initialize with FCC matrix + random circular HCP seeds."""
        np.random.seed(seed)
        
        self.c = np.full((self.nx, self.ny), np.float64(0.61), dtype=np.float64)
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
                        weight = min(np.float64(1.0), np.float64(r / radius_grid))
                        self.eta[ii, jj] = np.float64(1.0) * (np.float64(1.0) - weight)
                        self.c[ii, jj] = np.float64(seed_co_fraction) * (np.float64(1.0) - weight) + np.float64(0.61) * weight
        
        self.time_phys = np.float64(0.0)
        self.step = int(0)
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [], 'total_energy': []
        }
        self.update_history()
    
    def initialize_from_arrays(self, c_array, eta_array, reset_time=True):
        """Initialize from external arrays."""
        self.c = np.clip(np.array(c_array, dtype=np.float64), np.float64(0.01), np.float64(0.99))
        self.eta = np.clip(np.array(eta_array, dtype=np.float64), np.float64(0.0), np.float64(1.0))
        if reset_time:
            self.time_phys = np.float64(0.0)
            self.step = int(0)
            self.clear_history()
            self.update_history()
    
    def clear_history(self):
        """Clear all history tracking arrays."""
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [], 'total_energy': []
        }
    
    def update_history(self):
        """Record current state to history arrays."""
        self.history['time_phys'].append(float(self.time_phys))
        self.history['eta_mean'].append(float(np.mean(self.eta)))
        self.history['eta_std'].append(float(np.std(self.eta)))
        self.history['c_mean'].append(float(np.mean(self.c)))
        self.history['c_std'].append(float(np.std(self.c)))
        
        # Compute total free energy (optional, computationally expensive)
        if self.step % 10 == 0:
            try:
                energy = self.compute_total_free_energy()
                self.history['total_energy'].append(float(energy))
            except Exception as e:
                print(f"Warning: Energy computation failed at step {self.step}: {e}")
                self.history['total_energy'].append(np.nan)
        else:
            self.history['total_energy'].append(np.nan)
    
    def compute_total_free_energy(self):
        """
        Compute total free energy: F = ∫[f_bulk + (κ_c/2)|∇c|² + (κ_η/2)|∇η|²] dV
        Returns energy in Joules.
        
        FIXED: Uses vectorized Numba kernels to avoid scalar typing errors.
        """
        # Bulk free energy (vectorized + njit)
        f_bulk = compute_bulk_free_energy_vectorized(
            self.c, self.eta, self.T_K, self.Omega_Jmol, self.V_m,
            self.W_phys, self.lambda_coup
        )
        
        # Gradient energy (vectorized + njit)
        f_gradient = compute_gradient_energy_vectorized(
            self.c, self.eta, self.kappa_c, self.kappa_eta, self.dx_phys
        )
        
        # Integrate over domain (dx² = area per pixel in 2D)
        total_F = np.sum(f_bulk + f_gradient) * (self.dx_phys**2)
        return float(total_F)
    
    def run_step(self):
        """Execute one time step of hybrid Cahn-Hilliard + Allen-Cahn dynamics."""
        self.c, self.eta = update_mediloy_hybrid(
            self.c, self.eta,
            self.dt_phys, self.dx_phys,
            self.kappa_c, self.kappa_eta,
            self.M_chem, self.L_struct,
            self.T_K, self.Omega_Jmol, self.V_m,
            self.W_phys, self.lambda_coup
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
            'time_phys': float(self.time_phys),
            'time_formatted': self.scales.format_time(self.time_phys),
            'step': self.step,
            'domain_size_m': float(domain_size_m),
            'domain_size_formatted': self.scales.format_length(domain_size_m),
            'interface_width_eta_nm': float(interface_width_eta * 1e9),
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
            'W_phys': float(self.W_phys),
            'M_chem': float(self.M_chem),
            'L_struct': float(self.L_struct),
            'dt_phys': float(self.dt_phys),
        }
    
    def get_time_series(self, key):
        """Retrieve time series data from history."""
        if key not in self.history:
            raise ValueError(f"Unknown history key: {key}")
        return np.array(self.history['time_phys']), np.array(self.history[key])


# =============================================================================
# STREAMLIT APPLICATION: Interactive Mediloy Simulator with Plotly Visuals
# =============================================================================

def main():
    """Main Streamlit application entry point."""
    
    st.set_page_config(
        page_title="Mediloy γ→ε Phase Transformation (Plotly)",
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
    
    st.title("⚙️ Mediloy γ-FCC → ε-HCP Martensitic Transformation")
    st.markdown(f"""
    **Pseudo-binary Co–M<sub>y</sub> phase-field simulation at {950}°C (1223 K)**
    
    Modeling the diffusion-assisted martensitic transformation in dental Co-Cr-Mo alloys.
    - Initial condition: Uniform FCC matrix (η=0, c=0.61) + random HCP seeds
    - Kinetics: Fast structural evolution (Allen-Cahn) + slow solute diffusion (Cahn-Hilliard)
    - Physics: Solute drag stabilizes HCP at interfaces; thin lath morphology
    """)
    
    if 'sim' not in st.session_state:
        st.session_state.sim = MediloyPhaseTransformation(
            nx=256, ny=256,
            T_celsius=950.0,
            Omega_Jmol=12000.0,
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
            steps_input = st.number_input(
                "Steps per update", min_value=1, max_value=5000, value=100,
                help="Number of time steps to compute per button click"
            )
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
        num_seeds = st.slider("Number of HCP seeds", 1, 50, 12, 1)
        seed_radius = st.slider("Seed radius (grid units)", 3, 15, 5, 1)
        seed_co = st.slider("Co fraction in seeds", 0.50, 0.61, 0.58, 0.01)
        
        if st.button("🔄 Reset with New Random Seeds", use_container_width=True):
            sim.initialize_fcc_with_random_hcp_seeds(
                num_seeds=num_seeds, radius_grid=seed_radius,
                seed_co_fraction=seed_co, seed=int(time.time()) % 10000
            )
            st.rerun()
        
        st.divider()
        
        st.subheader("🧪 Material Properties")
        st.caption("Co-Cr-Mo alloy parameters at 950°C")
        
        Omega_kJmol = st.slider(
            "Mixing enthalpy Ω (kJ/mol)", 5.0, 30.0, 12.0, 1.0,
            help="Controls chemical driving force for phase separation"
        )
        D_b_exp = st.slider(
            "log₁₀(D_Cr) [m²/s]", -17, -13, -14.3, 0.1,
            help="Cr diffusion coefficient in Co matrix at 950°C"
        )
        D_b_val = 10**D_b_exp
        L_factor = st.slider(
            "Structural mobility factor", 1.0, 200.0, 50.0, 10.0,
            help="Relative rate of FCC→HCP transformation"
        )
        
        if st.button("Apply Material Parameters", use_container_width=True):
            sim.set_physical_parameters(
                Omega_Jmol=Omega_kJmol * 1000,
                D_b_m2s=D_b_val,
                L_m3Js=L_factor * sim.scales.L0_struct
            )
            st.rerun()
        
        st.divider()
        
        st.subheader("⚙️ Model Parameters")
        st.caption("Phase-field parameters in physical units")
        
        xi_eta_nm = sim.scales.phys_to_interface_width(sim.kappa_eta, sim.W_phys) * 1e9
        
        W_phys = st.number_input(
            "W: Structural barrier (J/m³)", min_value=1e4, max_value=1e8,
            value=float(sim.W_phys), format="%.2e",
            help="Energy barrier between FCC and HCP"
        )
        kappa_eta_phys = st.number_input(
            "κ_η: Structural gradient coeff (J/m)", min_value=1e-13, max_value=1e-9,
            value=float(sim.kappa_eta), format="%.2e",
            help=f"Controls HCP/FCC interface energy; ξ ≈ √(κ_η/W) ≈ {xi_eta_nm:.2f} nm"
        )
        M_phys = st.number_input(
            "M: Chemical mobility (m⁵/J·s)", min_value=1e-25, max_value=1e-18,
            value=float(sim.M_chem), format="%.2e",
            help="Controls Cr diffusion kinetics"
        )
        dt_phys = st.number_input(
            "Δt: Time step (s)", min_value=1e-12, max_value=1e-6,
            value=float(sim.dt_phys), format="%.2e",
            help="Numerical time step; stability: Δt ≲ 0.01·(Δx)⁴/(M·κ)"
        )
        
        if st.button("Apply Model Parameters", use_container_width=True):
            sim.set_physical_parameters(
                W_Jm3=W_phys, kappa_eta_Jm=kappa_eta_phys,
                M_m5Js=M_phys, dt_s=dt_phys
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
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f"**Phase Distribution**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("<span class='phase-hcp'>ε-HCP Fraction</span>", 
                     f"{stats['hcp_fraction']*100:.1f}%")
        with col_p2:
            st.metric("<span class='phase-fcc'>γ-FCC Fraction</span>", 
                     f"{stats['fcc_fraction']*100:.1f}%")
        
        st.markdown(f"**Co Concentration Statistics**")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("⟨c_Co⟩", f"{stats['c_mean']:.3f}")
            st.metric("min(c)", f"{stats.get('c_min', 0):.3f}")
        with col_c2:
            st.metric("σ(c)", f"{stats['c_std']:.3f}")
            st.metric("max(c)", f"{stats.get('c_max', 1):.3f}")
    
    # =============================================================================
    # MAIN CONTENT: Plotly Visualizations
    # =============================================================================
    
    extent_um_x = [0, sim.nx * sim.dx_phys * 1e6]
    extent_um_y = [0, sim.ny * sim.dx_phys * 1e6]
    x_coords = np.linspace(extent_um_x[0], extent_um_x[1], sim.nx)
    y_coords = np.linspace(extent_um_y[0], extent_um_y[1], sim.ny)
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("ε-HCP Order Parameter η")
        st.caption(f"η = 0 (<span class='phase-fcc'>FCC</span>) → η = 1 (<span class='phase-hcp'>HCP</span>) | t = {stats['time_formatted']}")
        
        fig_eta = go.Figure(data=go.Heatmap(
            z=sim.eta.T, x=x_coords, y=y_coords,
            colorscale='RdYlBu_r', zmin=0, zmax=1,
            colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP']),
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>η: %{z:.3f}<extra></extra>'
        ))
        fig_eta.update_layout(
            title="Martensitic HCP Phase Distribution",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=600, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_eta, use_container_width=True)
    
    with col_viz2:
        st.subheader("Co Concentration c_Co")
        st.caption("Nominal composition: c₀ = 0.61 (61 at.% Co)")
        
        fig_c = go.Figure(data=go.Heatmap(
            z=sim.c.T, x=x_coords, y=y_coords,
            colorscale='Viridis', zmin=0.55, zmax=0.67,
            colorbar=dict(title="c_Co"),
            hovertemplate='x: %{x:.2f} μm<br>y: %{y:.2f} μm<br>c: %{z:.3f}<extra></extra>'
        ))
        fig_c.update_layout(
            title="Cobalt Mole Fraction Distribution",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=600, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_c, use_container_width=True)
    
    col_overlay, col_hist = st.columns([2, 1])
    
    with col_overlay:
        st.subheader("Phase + Composition Overlay")
        st.caption("HCP regions (red) with Co depletion (blue) at interfaces")
        
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Heatmap(
            z=sim.c.T, x=x_coords, y=y_coords,
            colorscale='Viridis', zmin=0.55, zmax=0.67,
            opacity=0.7, colorbar=dict(title="c_Co", x=1.02), showscale=True
        ))
        for level in [0.3, 0.5, 0.7]:
            fig_overlay.add_trace(go.Contour(
                z=sim.eta.T, x=x_coords, y=y_coords,
                contours=dict(start=level, end=level, size=0.1),
                line=dict(color='red', width=1.5), showscale=False, hoverinfo='skip',
                name=f'η = {level:.1f}'
            ))
        fig_overlay.update_layout(
            title="HCP Phase Boundaries on Co Concentration",
            xaxis_title="x (μm)", yaxis_title="y (μm)",
            width=700, height=550, margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    with col_hist:
        st.subheader("Composition Distribution")
        hist_data = sim.c.flatten()
        fig_hist = px.histogram(
            hist_data, nbins=40, range_x=[0.55, 0.67],
            labels={'value': 'Co mole fraction c_Co', 'count': 'Frequency'},
            title="Co Concentration Distribution"
        )
        fig_hist.add_vline(x=0.61, line_dash="dash", line_color="gray", annotation_text="Nominal c₀=0.61")
        fig_hist.add_vline(x=stats['c_mean'], line_dash="dash", line_color="red", annotation_text=f"⟨c⟩={stats['c_mean']:.3f}")
        fig_hist.update_layout(width=400, height=500, margin=dict(l=40, r=20, t=60, b=40))
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.divider()
    st.subheader("📈 Transformation Kinetics")
    
    if len(sim.history['time_phys']) > 3:
        times_s = np.array(sim.history['time_phys'])
        times_min = times_s / 60
        
        fig_kin = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Martensitic Transformation Progress", 
                            "Solute Redistribution",
                            "Interface Evolution"),
            shared_xaxes=True, x_title="Time (minutes)"
        )
        
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_mean'],
                       mode='lines', name='⟨η⟩ (HCP fraction)',
                       line=dict(color='#e74c3c', width=2.5)),
            row=1, col=1
        )
        fig_kin.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        fig_kin.update_yaxes(title_text="HCP volume fraction", row=1, col=1)
        
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['c_mean'],
                       mode='lines', name='⟨c_Co⟩',
                       line=dict(color='#2ecc71', width=2)),
            row=1, col=2
        )
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['c_std'],
                       mode='lines', name='σ(c_Co)',
                       line=dict(color='#f39c12', width=1.5, dash='dash')),
            row=1, col=2
        )
        fig_kin.add_hline(y=0.61, line_dash="dash", line_color="gray", row=1, col=2)
        fig_kin.update_yaxes(title_text="Co fraction / Std. dev.", row=1, col=2)
        
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_std'],
                       mode='lines', name='σ(η)',
                       line=dict(color='#9b59b6', width=2)),
            row=1, col=3
        )
        fig_kin.update_yaxes(title_text="Order parameter std. dev.", row=1, col=3)
        
        fig_kin.update_layout(
            height=450, width=1200, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_kin, use_container_width=True)
        
        with st.expander("🔋 Free Energy Evolution (click to expand)"):
            valid_energy = [e for e in sim.history['total_energy'] if not np.isnan(e)]
            valid_times = [t for t, e in zip(times_min, sim.history['total_energy']) if not np.isnan(e)]
            
            if len(valid_energy) > 2:
                fig_fe = go.Figure()
                fig_fe.add_trace(go.Scatter(
                    x=valid_times, y=valid_energy,
                    mode='lines+markers', name='Total free energy',
                    line=dict(color='#8e44ad', width=2), marker=dict(size=3)
                ))
                fig_fe.update_layout(
                    title="Free Energy Minimization",
                    xaxis_title="Time (minutes)", yaxis_title="Total free energy (J)",
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
        if st.button("📸 Save Microstructure Snapshot", use_container_width=True):
            fig_snap = go.Figure(data=go.Heatmap(
                z=sim.eta.T, x=x_coords, y=y_coords,
                colorscale='RdYlBu_r', zmin=0, zmax=1,
                colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP'])
            ))
            fig_snap.update_layout(
                title=f"Mediloy HCP Phase Distribution<br>t = {stats['time_formatted']}",
                xaxis_title="x (μm)", yaxis_title="y (μm)",
                width=800, height=700
            )
            try:
                img_bytes = fig_snap.to_image(format="png", width=800, height=700, scale=2)
                st.download_button(
                    label="⬇️ Download PNG", data=img_bytes,
                    file_name=f"Mediloy_HCP_t{sim.time_phys:.2e}s.png",
                    mime="image/png", use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not generate image. Ensure 'kaleido' is installed. Error: {e}")
    
    with col_exp2:
        if st.button("📊 Save Kinetics Data", use_container_width=True):
            csv_lines = ["time_s,time_min,eta_mean,eta_std,c_mean,c_std,total_energy_J"]
            for i in range(len(sim.history['time_phys'])):
                t_s = sim.history['time_phys'][i]
                line = f"{t_s:.6e},{t_s/60:.6e},{sim.history['eta_mean'][i]:.6f},"
                line += f"{sim.history['eta_std'][i]:.6f},{sim.history['c_mean'][i]:.6f},"
                line += f"{sim.history['c_std'][i]:.6f},{sim.history['total_energy'][i]:.6e}"
                csv_lines.append(line)
            csv_content = "\n".join(csv_lines)
            st.download_button(
                label="⬇️ Download CSV", data=csv_content,
                file_name="mediloy_kinetics.csv", mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        if st.button("⚙️ Save Simulation State", use_container_width=True):
            npz_buf = BytesIO()
            np.savez_compressed(
                npz_buf, concentration=sim.c, order_parameter=sim.eta,
                time_phys=sim.time_phys, step=sim.step,
                params={
                    'T_celsius': float(sim.T_celsius), 'Omega_Jmol': float(sim.Omega_Jmol),
                    'D_b': float(sim.D_b), 'V_m': float(sim.V_m),
                    'W_phys': float(sim.W_phys), 'kappa_eta': float(sim.kappa_eta),
                    'M_chem': float(sim.M_chem), 'L_struct': float(sim.L_struct),
                    'dt_phys': float(sim.dt_phys), 'dx_phys': float(sim.dx_phys),
                    'lambda_coup': float(sim.lambda_coup)
                }
            )
            npz_buf.seek(0)
            st.download_button(
                label="⬇️ Download NPZ", data=npz_buf.getvalue(),
                file_name=f"Mediloy_state_t{sim.time_phys:.2e}s.npz",
                mime="application/octet-stream", use_container_width=True
            )
    
    # =============================================================================
    # Physics Guide & Documentation
    # =============================================================================
    with st.expander("ℹ️ Physics Guide & Parameter Reference", expanded=False):
        st.markdown("""
        ## ⚙️ Mediloy γ-FCC → ε-HCP Phase Transformation Model
        
        This simulation implements a **hybrid phase-field model** for the martensitic 
        transformation in Co-Cr-Mo dental alloys (Mediloy) at **950°C (1223 K)**.
        
        ### Governing Equations
        
        **Free energy functional**:
        ```
        F = ∫[ f_chem(c) + f_struct(η) + f_coup(c,η) 
               + (κ_c/2)|∇c|² + (κ_η/2)|∇η|² ] dV
        ```
        
        **Evolution equations**:
        ```
        ∂c/∂t = ∇·[ M ∇(δF/δc) ]          (Cahn-Hilliard, conserved)
        ∂η/∂t = -L · (δF/δη)               (Allen-Cahn, non-conserved)
        ```
        
        ### Free Energy Components
        
        | Term | Expression | Physical Meaning |
        |------|-----------|-----------------|
        | f_chem | (RT/V_m)[c·ln(c)+(1-c)·ln(1-c)] + (Ω/V_m)·c(1-c) | Regular solution mixing |
        | f_struct | W·η²(1-η)² | Double-well: FCC (η=0) ↔ HCP (η=1) |
        | f_coup | -λ·(1-c)·η² | Solute drag: Cr stabilizes HCP |
        
        ### Key Parameters at 950°C
        
        | Parameter | Symbol | Typical Value | Role |
        |-----------|--------|--------------|------|
        | Temperature | T | 1223 K | Near T₀ equilibrium |
        | Driving force | ΔG_chem | 200-600 J/mol | Small chemical bias |
        | Interface width | ξ | 2-5 nm | Diffuse γ/ε boundary |
        | Chemical mobility | M | ~10⁻²² m⁵/J·s | Slow Cr diffusion |
        | Structural mobility | L | ~10⁻⁸ m³/J·s | Fast martensitic growth |
        | Coupling strength | λ | ~10⁶ J/m³ | Solute drag magnitude |
        
        ### Interpreting Results
        
        #### Morphology
        - **Thin laths/plates**: Characteristic of martensitic transformation
        - **Cr enrichment at interfaces**: Visible as blue regions at HCP boundaries
        - **Coarsening**: Small HCP domains merge over time (Ostwald ripening)
        
        #### Kinetics
        - **Initial rapid growth**: Allen-Cahn dominates (interface motion)
        - **Later slowdown**: Cahn-Hilliard limits growth (solute diffusion)
        - **Saturation**: Approaches equilibrium HCP fraction
        
        #### Composition Evolution
        - **Nominal c₀ = 0.61**: Average Co fraction conserved
        - **Local depletion**: HCP regions slightly Cr-enriched (c < 0.61)
        - **Interface segregation**: Cr accumulates at γ/ε boundaries
        
        ### Numerical Stability Guidelines ⚠️
        
        For explicit time integration of coupled 4th/2nd-order PDEs:
        
        ```
        Δt ≲ min[ 0.01·(Δx)⁴/(M·κ_c),  0.1·(Δx)²/(L·W) ]
        ```
        
        Practical recommendations:
        1. Ensure interface spans ≥3 grid points: ξ/Δx ≥ 3
        2. Start with small Δt; increase only if stable
        3. Monitor concentration bounds: c ∈ [0.01, 0.99]
        4. If simulation diverges: reduce Δt or increase κ_η
        
        ### Applications in Dental Materials
        
        ✓ Predict **HCP fraction** after ceramic firing cycles  
        ✓ Optimize **alloy composition** for desired phase balance  
        ✓ Study **cooling rate effects** on martensite morphology  
        ✓ Investigate **solute segregation** at phase boundaries  
        ✓ Educational tool for **phase-field methods** in metallurgy  
        
        ### References
        
        1. Cahn, J.W. & Hilliard, J.E. (1958). *J. Chem. Phys.* **28**, 258.
        2. Allen, S.M. & Cahn, J.W. (1979). *Acta Metall.* **27**, 1085.
        3. Brachavort, S. et al. (2015). *Acta Mater.* **99**, 262. [Co-Cr phase diagrams]
        4. Yamanaka, K. et al. (2018). *Dent. Mater. J.* **37**, 1. [Mediloy microstructure]
        5. Steinbach, I. (2009). *Model. Simul. Mater. Sci. Eng.* **17**, 073001. [Phase-field review]
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
    
    st.markdown("---")
    st.caption(
        "Mediloy γ→ε Phase-Field Simulator | Hybrid Cahn-Hilliard + Allen-Cahn | "
        f"Physical Units: m, s, J/m³ | T = {sim.T_celsius}°C | "
        f"Pseudo-binary Co-M<sub>y</sub> (c₀ = 0.61) | Visualized with Plotly"
    )


if __name__ == "__main__":
    print("⚙️ Starting Mediloy Phase Transformation Simulator (Plotly version)...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Numba: JIT compilation enabled")
    print(f"   Streamlit: launching interactive app")
    print(f"   Temperature: 950°C (1223.15 K)")
    print(f"   Initial condition: FCC matrix + random HCP seeds")
    main()
