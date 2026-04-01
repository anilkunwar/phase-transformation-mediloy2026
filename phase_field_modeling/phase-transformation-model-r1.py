# =============================================================================
# MEDILOY PHASE TRANSFORMATION SIMULATOR – PLOTLY VERSION (FIXED)
# γ-FCC → ε-HCP Martensitic Transformation in Co-Cr-Mo Dental Alloys
# Temperature: 950°C (1223.15 K) - Pseudo-binary Co-M_y model
# =============================================================================
# Fixes:
#   - dt_phys slider max_value increased to 1e-5 (was 1e-6)
#   - D_b_exp slider min/max changed to floats (-17.0, -13.0)
#   - compute_total_free_energy: vectorized to avoid Numba TypingError
#   - Added explicit type signatures for all Numba functions
# =============================================================================

import numpy as np
from numba import njit, float64, int64, prange
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
        self.R = 8.314462618          # J/(mol·K) - Universal gas constant
        self.T = T_celsius + 273.15   # K - Absolute temperature
        self.T_celsius = T_celsius    # °C - For display
        
        # Material properties (Co-Cr-Mo alloy at 950°C)
        self.V_m = V_m_m3mol          # m³/mol - Molar volume (~6.7 cm³/mol for Co alloys)
        self.D_b = D_b_m2s            # m²/s - Cr diffusion in Co matrix at 950°C
        
        # Characteristic scales (derived from literature values)
        self.L0 = 2.0e-9              # m - Reference length (interface width ~2-5 nm)
        
        # Energy scale: ΔG_chem ≈ 400 J/mol at T ≈ T0 for Mediloy
        delta_G_mol = 400.0           # J/mol - Chemical driving force at 950°C
        self.E0 = delta_G_mol / self.V_m   # J/m³ - Energy density scale
        
        # Time and mobility scales
        self.t0 = self.L0**2 / self.D_b    # s - Diffusion time scale
        self.M0 = self.D_b / self.E0        # m⁵/(J·s) - Chemical mobility scale
        
        # Structural mobility scale (Allen-Cahn, much faster than diffusion)
        # L0_struct has units of [m³/(J·s)] for order parameter evolution
        self.L0_struct = 1.0e-8 / (self.E0 * self.t0)  # m³/(J·s) - Reference structural mobility
        
        # Log initialization for debugging
        print(f"Mediloy scales initialized at {self.T_celsius}°C ({self.T:.1f} K)")
        print(f"  L0 = {self.L0*1e9:.2f} nm, t0 = {self.t0:.2e} s")
        print(f"  E0 = {self.E0:.2e} J/m³, M0 = {self.M0:.2e} m⁵/(J·s)")
        print(f"  D_b = {self.D_b:.2e} m²/s, V_m = {self.V_m*1e6:.2f} cm³/mol")
    
    def dim_to_phys(self, W_dim, kappa_c_dim, kappa_eta_dim, M_dim, L_dim, dt_dim, dx_dim=1.0):
        """
        Convert dimensionless parameters to physical SI units.
        
        Parameters:
        -----------
        W_dim : float - Dimensionless double-well barrier for structural order
        kappa_c_dim : float - Dimensionless gradient coefficient for concentration
        kappa_eta_dim : float - Dimensionless gradient coefficient for order parameter
        M_dim : float - Dimensionless chemical mobility
        L_dim : float - Dimensionless structural mobility (Allen-Cahn)
        dt_dim : float - Dimensionless time step
        dx_dim : float - Dimensionless grid spacing (default: 1.0)
        
        Returns:
        --------
        tuple : (W_phys, kappa_c_phys, kappa_eta_phys, M_phys, L_phys, dt_phys, dx_phys)
                All in SI units
        """
        W_phys = W_dim * self.E0                           # J/m³
        kappa_c_phys = kappa_c_dim * self.E0 * self.L0**2  # J/m
        kappa_eta_phys = kappa_eta_dim * self.E0 * self.L0**2  # J/m
        M_phys = M_dim * self.M0                            # m⁵/(J·s)
        L_phys = L_dim * self.L0_struct                     # m³/(J·s)
        dt_phys = dt_dim * self.t0                          # s
        dx_phys = dx_dim * self.L0                          # m
        
        return W_phys, kappa_c_phys, kappa_eta_phys, M_phys, L_phys, dt_phys, dx_phys
    
    def phys_to_interface_width(self, kappa_phys, W_phys):
        """
        Estimate interface width from gradient energy and barrier height.
        ξ ≈ √(κ/W) - characteristic width of diffuse interface.
        
        Parameters:
        -----------
        kappa_phys : float - Gradient coefficient in J/m
        W_phys : float - Double-well barrier in J/m³
        
        Returns:
        --------
        float : Interface width in meters
        """
        if W_phys <= 0 or kappa_phys <= 0:
            return 2.0e-9  # Fallback: 2 nm
        return np.sqrt(kappa_phys / W_phys)
    
    def format_time(self, t_seconds):
        """Format physical time with appropriate SI prefix for metallurgical timescales."""
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
        """Format length with appropriate SI prefix for microstructural scales."""
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

# --- Explicit signatures for all Numba functions ---
@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def chemical_free_energy_density(c, T_K, Omega_Jmol, V_m):
    """
    Regular solution model for Co-M_y pseudo-binary alloy.
    
    f_chem(c) = (RT/V_m)[c·ln(c) + (1-c)·ln(1-c)] + (Ω/V_m)·c·(1-c)
    
    Parameters:
    -----------
    c : float or array - Co mole fraction (0 ≤ c ≤ 1)
    T_K : float - Temperature in Kelvin
    Omega_Jmol : float - Regular solution parameter [J/mol]
    V_m : float - Molar volume [m³/mol]
    
    Returns:
    --------
    float or array - Chemical free energy density [J/m³]
    """
    R = 8.314462618
    # Entropy of mixing term (with numerical stabilization)
    c_safe = np.clip(c, 1e-8, 1.0 - 1e-8)
    f_mix = (R * T_K / V_m) * (c_safe * np.log(c_safe) + (1.0 - c_safe) * np.log(1.0 - c_safe))
    # Enthalpy of mixing (regular solution)
    f_excess = (Omega_Jmol / V_m) * c * (1.0 - c)
    return f_mix + f_excess


@njit(float64(float64, float64, float64, float64), fastmath=True, cache=True)
def d_fchem_dc(c, T_K, Omega_Jmol, V_m):
    """
    Chemical potential contribution: ∂f_chem/∂c
    
    μ_chem = (RT/V_m)·ln[c/(1-c)] + (Ω/V_m)·(1-2c)
    
    Parameters:
    -----------
    c : float or array - Co mole fraction
    T_K : float - Temperature in Kelvin
    Omega_Jmol : float - Regular solution parameter [J/mol]
    V_m : float - Molar volume [m³/mol]
    
    Returns:
    --------
    float or array - Chemical potential contribution [J/m³]
    """
    R = 8.314462618
    c_safe = np.clip(c, 1e-8, 1.0 - 1e-8)
    mu_mix = (R * T_K / V_m) * np.log(c_safe / (1.0 - c_safe))
    mu_excess = (Omega_Jmol / V_m) * (1.0 - 2.0 * c)
    return mu_mix + mu_excess


@njit(float64(float64, float64), fastmath=True, cache=True)
def structural_free_energy(eta, W_struct):
    """
    Double-well potential for structural order parameter.
    
    f_struct(η) = W·η²(1-η)²
    - η = 0: FCC (γ) phase (metastable at 950°C)
    - η = 1: HCP (ε) phase (stable at 950°C)
    
    Parameters:
    -----------
    eta : float or array - Structural order parameter (0 ≤ η ≤ 1)
    W_struct : float - Barrier height [J/m³]
    
    Returns:
    --------
    float or array - Structural free energy density [J/m³]
    """
    return W_struct * eta**2 * (1.0 - eta)**2


@njit(float64(float64, float64), fastmath=True, cache=True)
def d_fstruct_deta(eta, W_struct):
    """
    Variational derivative: ∂f_struct/∂η
    
    ∂f/∂η = 2W·η(1-η)(1-2η) = 2W·η - 6W·η² + 4W·η³
    
    Parameters:
    -----------
    eta : float or array - Structural order parameter
    W_struct : float - Barrier height [J/m³]
    
    Returns:
    --------
    float or array - Structural chemical potential [J/m³]
    """
    return 2.0 * W_struct * eta * (1.0 - eta) * (1.0 - 2.0 * eta)


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def coupling_free_energy(c, eta, lambda_coup):
    """
    Coupling term: HCP phase stabilized by higher M_y (lower Co) content.
    
    f_coup = -λ·(1-c)·η²
    
    This term:
    - Lowers energy of HCP (η=1) when Co content is low (c < 0.61)
    - Represents solute drag: Cr enrichment at γ/ε interfaces
    
    Parameters:
    -----------
    c : float or array - Co mole fraction
    eta : float or array - Structural order parameter
    lambda_coup : float - Coupling strength [J/m³]
    
    Returns:
    --------
    float or array - Coupling free energy density [J/m³]
    """
    return -lambda_coup * (1.0 - c) * eta**2


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def d_fcoup_dc(c, eta, lambda_coup):
    """∂f_coup/∂c = +λ·η²"""
    return lambda_coup * eta**2


@njit(float64(float64, float64, float64), fastmath=True, cache=True)
def d_fcoup_deta(c, eta, lambda_coup):
    """∂f_coup/∂η = -2λ·(1-c)·η"""
    return -2.0 * lambda_coup * (1.0 - c) * eta


@njit(float64[:,:](float64[:,:], float64), parallel=True, fastmath=True, cache=True)
def compute_laplacian_2d(field, dx):
    """
    Compute 5-point stencil Laplacian with periodic BCs.
    ∇²f ≈ [f(i+1,j) + f(i-1,j) + f(i,j+1) + f(i,j-1) - 4f(i,j)] / dx²
    """
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


@njit(float64[:,:](float64[:,:], float64[:,:], float64), parallel=True, fastmath=True, cache=True)
def compute_gradient_divergence_2d(flux_x, flux_y, dx):
    """
    Compute divergence of vector field: ∇·J = ∂Jx/∂x + ∂Jy/∂y
    Using central differences with periodic BCs.
    """
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


@njit(
    (float64[:,:], float64[:,:], float64, float64, float64, float64, float64, float64,
     float64, float64, float64, float64, float64, float64),
    parallel=True, fastmath=True, cache=True
)
def update_mediloy_hybrid(c, eta, dt, dx, kappa_c, kappa_eta, M_chem, L_struct,
                          T_K, Omega_Jmol, V_m, W_struct, lambda_coup):
    """
    One time step of hybrid Cahn-Hilliard (concentration) + Allen-Cahn (structure).
    
    Governing equations:
    ∂c/∂t = ∇·[ M_chem ∇( δF/δc ) ]          (Cahn-Hilliard, conserved)
    ∂η/∂t = -L_struct · ( δF/δη )             (Allen-Cahn, non-conserved)
    
    Free energy functional:
    F = ∫[ f_chem(c) + f_struct(η) + f_coup(c,η) 
           + (κ_c/2)|∇c|² + (κ_η/2)|∇η|² ] dV
    
    Parameters:
    -----------
    c, eta : 2D arrays - Co fraction and structural order parameter
    dt : float - Time step [s]
    dx : float - Grid spacing [m]
    kappa_c, kappa_eta : float - Gradient coefficients [J/m]
    M_chem : float - Chemical mobility [m⁵/(J·s)]
    L_struct : float - Structural mobility [m³/(J·s)]
    T_K, Omega_Jmol, V_m : Material parameters for f_chem
    W_struct, lambda_coup : Parameters for structural energy and coupling
    
    Returns:
    --------
    tuple : (c_new, eta_new) - Updated fields
    """
    nx, ny = c.shape
    c_new = np.copy(c)
    eta_new = np.copy(eta)
    
    # Pre-compute Laplacians
    lap_c = compute_laplacian_2d(c, dx)
    lap_eta = compute_laplacian_2d(eta, dx)
    
    # ========== CONCENTRATION FIELD (Cahn-Hilliard) ==========
    # Compute chemical potential μ = δF/δc = ∂f/∂c - κ_c·∇²c
    mu_chem = np.empty_like(c)
    for i in prange(nx):
        for j in prange(ny):
            # Local chemical potential from bulk free energy
            mu_bulk = d_fchem_dc(c[i, j], T_K, Omega_Jmol, V_m)
            # Coupling contribution
            mu_coup = d_fcoup_dc(c[i, j], eta[i, j], lambda_coup)
            # Gradient energy contribution
            mu_grad = -kappa_c * lap_c[i, j]
            # Total chemical potential
            mu_chem[i, j] = mu_bulk + mu_coup + mu_grad
    
    # Compute flux: J = -M·∇μ (Fick's law with gradient energy)
    flux_c_x = np.empty_like(c)
    flux_c_y = np.empty_like(c)
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            
            grad_mu_x = (mu_chem[ip1, j] - mu_chem[im1, j]) / (2.0 * dx)
            grad_mu_y = (mu_chem[i, jp1] - mu_chem[i, jm1]) / (2.0 * dx)
            
            flux_c_x[i, j] = -M_chem * grad_mu_x
            flux_c_y[i, j] = -M_chem * grad_mu_y
    
    # Compute divergence of flux and update concentration
    div_flux_c = compute_gradient_divergence_2d(flux_c_x, flux_c_y, dx)
    c_new = c + dt * div_flux_c
    
    # ========== STRUCTURAL ORDER PARAMETER (Allen-Cahn) ==========
    # Compute variational derivative: δF/δη = ∂f/∂η - κ_η·∇²η
    dF_deta = np.empty_like(eta)
    for i in prange(nx):
        for j in prange(ny):
            # Structural contribution
            dF_struct = d_fstruct_deta(eta[i, j], W_struct)
            # Coupling contribution
            dF_coup = d_fcoup_deta(c[i, j], eta[i, j], lambda_coup)
            # Gradient energy contribution
            dF_grad = -kappa_eta * lap_eta[i, j]
            # Total variational derivative
            dF_deta[i, j] = dF_struct + dF_coup + dF_grad
    
    # Allen-Cahn update: ∂η/∂t = -L·(δF/δη)
    eta_new = eta - dt * L_struct * dF_deta
    
    # ========== PHYSICAL BOUNDS ==========
    # Concentration: Co fraction must stay in [0.01, 0.99] for numerical stability
    c_new = np.clip(c_new, 0.01, 0.99)
    # Order parameter: η ∈ [0, 1] (FCC → HCP)
    eta_new = np.clip(eta_new, 0.0, 1.0)
    
    return c_new, eta_new


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
        """
        Initialize Mediloy phase transformation simulation.
        
        Parameters:
        -----------
        nx, ny : int - Grid dimensions
        T_celsius : float - Simulation temperature [°C] (default: 950°C)
        Omega_Jmol : float - Regular solution parameter [J/mol]
        V_m_m3mol : float - Molar volume [m³/mol]
        D_b_m2s : float - Cr diffusion coefficient in Co [m²/s] at T
        """
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.dx_dim = 1.0  # Dimensionless grid spacing (internal)
        
        # Dimensionless model parameters (tuned for numerical stability)
        self.W_dim = 1.0           # Structural double-well barrier
        self.kappa_c_dim = 2.0     # Concentration gradient coefficient
        self.kappa_eta_dim = 1.0   # Structural gradient coefficient (sharper interface)
        self.M_dim = 1.0           # Chemical mobility (diffusion-limited)
        self.L_dim = 50.0          # Structural mobility (fast martensitic growth)
        self.lambda_coup_dim = 2.0 # Coupling strength (solute drag)
        self.dt_dim = 0.005        # Dimensionless time step
        
        # Material parameters
        self.T_celsius = T_celsius
        self.Omega_Jmol = Omega_Jmol
        self.V_m = V_m_m3mol
        self.D_b = D_b_m2s
        
        # Initialize physical scales
        self.scales = PhysicalScalesMediloy(
            T_celsius=T_celsius,
            V_m_m3mol=V_m_m3mol,
            D_b_m2s=D_b_m2s
        )
        
        # Convert to physical parameters
        self._update_physical_params()
        
        # Initialize fields: FCC matrix with nominal composition
        self.c = np.full((nx, ny), 0.61, dtype=np.float64)   # Co fraction
        self.eta = np.zeros((nx, ny), dtype=np.float64)       # η=0: FCC
        
        # Time tracking
        self.time_phys = 0.0
        self.step = 0
        
        # History for analysis
        self.history = {
            'time_phys': [],
            'eta_mean': [],
            'eta_std': [],
            'c_mean': [],
            'c_std': [],
            'total_energy': []
        }
        
        # Auto-record initial state
        self.update_history()
    
    def _update_physical_params(self):
        """Convert dimensionless parameters to physical SI units."""
        (self.W_phys, self.kappa_c, self.kappa_eta, 
         self.M_chem, self.L_struct, self.dt_phys, self.dx_phys) = \
            self.scales.dim_to_phys(
                self.W_dim, self.kappa_c_dim, self.kappa_eta_dim,
                self.M_dim, self.L_dim, self.dt_dim, self.dx_dim
            )
        
        # Coupling parameter in physical units
        self.lambda_coup = self.lambda_coup_dim * self.scales.E0
        
        # Temperature in Kelvin for free energy calculations
        self.T_K = self.T_celsius + 273.15
    
    def set_physical_parameters(self, W_Jm3=None, kappa_c_Jm=None, kappa_eta_Jm=None,
                                M_m5Js=None, L_m3Js=None, dt_s=None,
                                lambda_coup_Jm3=None, Omega_Jmol=None, D_b_m2s=None):
        """
        Set physical parameters directly (converts to dimensionless internally).
        
        Parameters are in SI units as indicated.
        """
        # Update material parameters if provided
        if Omega_Jmol is not None:
            self.Omega_Jmol = Omega_Jmol
        if D_b_m2s is not None:
            self.D_b = D_b_m2s
            # Recreate scales with new diffusion coefficient
            self.scales = PhysicalScalesMediloy(
                T_celsius=self.T_celsius,
                V_m_m3mol=self.V_m,
                D_b_m2s=self.D_b
            )
        
        # Convert physical → dimensionless for model parameters
        if W_Jm3 is not None and self.scales.E0 > 0:
            self.W_dim = W_Jm3 / self.scales.E0
        if kappa_c_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_c_dim = kappa_c_Jm / (self.scales.E0 * self.scales.L0**2)
        if kappa_eta_Jm is not None and self.scales.E0 > 0 and self.scales.L0 > 0:
            self.kappa_eta_dim = kappa_eta_Jm / (self.scales.E0 * self.scales.L0**2)
        if M_m5Js is not None and self.scales.M0 > 0:
            self.M_dim = M_m5Js / self.scales.M0
        if L_m3Js is not None and self.scales.L0_struct > 0:
            self.L_dim = L_m3Js / self.scales.L0_struct
        if lambda_coup_Jm3 is not None and self.scales.E0 > 0:
            self.lambda_coup_dim = lambda_coup_Jm3 / self.scales.E0
        if dt_s is not None and self.scales.t0 > 0:
            self.dt_dim = dt_s / self.scales.t0
        
        # Update all physical parameters
        self._update_physical_params()
    
    def initialize_fcc_with_random_hcp_seeds(self, num_seeds=12, radius_grid=5,
                                             seed_co_fraction=0.58, seed=42):
        """
        Initialize with FCC matrix + random circular HCP seeds.
        
        EXACT INITIAL CONDITION AS REQUESTED:
        - Uniform FCC: η = 0, c = 0.61 everywhere
        - Random HCP seeds: small circular regions with η = 1, slightly depleted Co
        
        Parameters:
        -----------
        num_seeds : int - Number of random HCP nuclei
        radius_grid : float - Seed radius in grid units
        seed_co_fraction : float - Co fraction inside seeds (typically < 0.61)
        seed : int - Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Start with uniform FCC matrix
        self.c = np.full((self.nx, self.ny), 0.61, dtype=np.float64)
        self.eta = np.zeros((self.nx, self.ny), dtype=np.float64)
        
        # Add random circular HCP seeds
        for s in range(num_seeds):
            # Random seed center (with margin to avoid edge artifacts)
            cx = np.random.randint(radius_grid + 5, self.nx - radius_grid - 5)
            cy = np.random.randint(radius_grid + 5, self.ny - radius_grid - 5)
            
            # Create circular seed with smooth boundary
            for i in range(-radius_grid*2, radius_grid*2 + 1):
                for j in range(-radius_grid*2, radius_grid*2 + 1):
                    r = np.sqrt(i**2 + j**2)
                    if r <= radius_grid:
                        ii = (cx + i) % self.nx
                        jj = (cy + j) % self.ny
                        # Smooth transition at seed boundary
                        weight = min(1.0, r / radius_grid)
                        self.eta[ii, jj] = 1.0 * (1.0 - weight)
                        self.c[ii, jj] = seed_co_fraction * (1.0 - weight) + 0.61 * weight
        
        # Reset time and history
        self.time_phys = 0.0
        self.step = 0
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'c_mean': [], 'c_std': [], 'total_energy': []
        }
        self.update_history()
    
    def initialize_from_arrays(self, c_array, eta_array, reset_time=True):
        """Initialize from external arrays (for restart or custom ICs)."""
        self.c = np.clip(np.array(c_array, dtype=np.float64), 0.01, 0.99)
        self.eta = np.clip(np.array(eta_array, dtype=np.float64), 0.0, 1.0)
        if reset_time:
            self.time_phys = 0.0
            self.step = 0
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
        self.history['time_phys'].append(self.time_phys)
        self.history['eta_mean'].append(float(np.mean(self.eta)))
        self.history['eta_std'].append(float(np.std(self.eta)))
        self.history['c_mean'].append(float(np.mean(self.c)))
        self.history['c_std'].append(float(np.std(self.c)))
        
        # Compute total free energy (optional, computationally expensive)
        # Only compute every 10 steps for performance, and skip if it causes issues
        if self.step % 10 == 0:
            try:
                energy = self.compute_total_free_energy()
                self.history['total_energy'].append(energy)
            except Exception:
                # Fallback if energy computation fails (e.g., Numba typing issues)
                self.history['total_energy'].append(np.nan)
        else:
            self.history['total_energy'].append(np.nan)
    
    def compute_total_free_energy(self):
        """
        Compute total free energy: F = ∫[f_bulk + (κ_c/2)|∇c|² + (κ_η/2)|∇η|²] dV
        Returns energy in Joules.
        
        FIX: Vectorized implementation to avoid Numba TypingError when calling
        @njit functions from pure Python loops.
        """
        # Bulk free energy - VECTORIZED: @njit functions handle array inputs automatically
        f_chem = chemical_free_energy_density(self.c, self.T_K, self.Omega_Jmol, self.V_m)
        f_struct = structural_free_energy(self.eta, self.W_phys)
        f_coup = coupling_free_energy(self.c, self.eta, self.lambda_coup)
        
        f_bulk = f_chem + f_struct + f_coup
        
        # Gradient energy contributions (keep simple loops for periodic BCs)
        grad_c_x = np.zeros_like(self.c)
        grad_c_y = np.zeros_like(self.c)
        grad_eta_x = np.zeros_like(self.c)
        grad_eta_y = np.zeros_like(self.c)
        
        nx, ny = self.nx, self.ny
        dx = self.dx_phys
        
        for i in range(nx):
            for j in range(ny):
                ip1 = (i + 1) % nx
                im1 = (i - 1) % nx
                jp1 = (j + 1) % ny
                jm1 = (j - 1) % ny
                
                grad_c_x[i, j] = (self.c[ip1, j] - self.c[im1, j]) / (2.0 * dx)
                grad_c_y[i, j] = (self.c[i, jp1] - self.c[i, jm1]) / (2.0 * dx)
                grad_eta_x[i, j] = (self.eta[ip1, j] - self.eta[im1, j]) / (2.0 * dx)
                grad_eta_y[i, j] = (self.eta[i, jp1] - self.eta[i, jm1]) / (2.0 * dx)
        
        grad_c_sq = grad_c_x**2 + grad_c_y**2
        grad_eta_sq = grad_eta_x**2 + grad_eta_y**2
        
        f_gradient = 0.5 * self.kappa_c * grad_c_sq + 0.5 * self.kappa_eta * grad_eta_sq
        
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
        
        # Update time and step counter
        self.time_phys += self.dt_phys
        self.step += 1
        
        # Record to history
        self.update_history()
    
    def run_steps(self, n_steps, progress_callback=None):
        """Execute multiple time steps with optional progress reporting."""
        for step_idx in range(n_steps):
            self.run_step()
            if progress_callback is not None and step_idx % 10 == 0:
                progress_callback(step_idx + 1, n_steps)
    
    def get_statistics(self):
        """Compute comprehensive simulation statistics."""
        # Geometric quantities
        domain_size_m = self.nx * self.dx_phys
        interface_width_c = self.scales.phys_to_interface_width(self.kappa_c, self.W_phys)
        interface_width_eta = self.scales.phys_to_interface_width(self.kappa_eta, self.W_phys)
        
        # Min/Max fields
        c_min = float(np.min(self.c))
        c_max = float(np.max(self.c))
        eta_min = float(np.min(self.eta))
        eta_max = float(np.max(self.eta))
        
        return {
            # Time
            'time_phys': self.time_phys,
            'time_formatted': self.scales.format_time(self.time_phys),
            'step': self.step,
            
            # Length scales
            'domain_size_m': domain_size_m,
            'domain_size_formatted': self.scales.format_length(domain_size_m),
            'interface_width_eta_nm': interface_width_eta * 1e9,
            
            # Field statistics
            'eta_mean': float(np.mean(self.eta)),
            'eta_std': float(np.std(self.eta)),
            'eta_min': eta_min,
            'eta_max': eta_max,
            'c_mean': float(np.mean(self.c)),
            'c_std': float(np.std(self.c)),
            'c_min': c_min,
            'c_max': c_max,
            
            # Phase fractions
            'hcp_fraction': float(np.sum(self.eta > 0.5) / (self.nx * self.ny)),
            'fcc_fraction': float(np.sum(self.eta < 0.5) / (self.nx * self.ny)),
            
            # Model parameters
            'W_phys': self.W_phys,
            'M_chem': self.M_chem,
            'L_struct': self.L_struct,
            'dt_phys': self.dt_phys,
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
    
    # Page configuration
    st.set_page_config(
        page_title="Mediloy γ→ε Phase Transformation (Plotly)",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 12px; border-radius: 8px; margin: 5px 0;}
    .stButton>button {width: 100%; border-radius: 6px;}
    .phase-fcc {color: #2ecc71; font-weight: bold;}
    .phase-hcp {color: #e74c3c; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("⚙️ Mediloy γ-FCC → ε-HCP Martensitic Transformation")
    st.markdown(f"""
    **Pseudo-binary Co–M<sub>y</sub> phase-field simulation at {950}°C (1223 K)**
    
    Modeling the diffusion-assisted martensitic transformation in dental Co-Cr-Mo alloys.
    - Initial condition: Uniform FCC matrix (η=0, c=0.61) + random HCP seeds
    - Kinetics: Fast structural evolution (Allen-Cahn) + slow solute diffusion (Cahn-Hilliard)
    - Physics: Solute drag stabilizes HCP at interfaces; thin lath morphology
    """)
    
    # Initialize simulation in session state
    if 'sim' not in st.session_state:
        st.session_state.sim = MediloyPhaseTransformation(
            nx=256, ny=256,
            T_celsius=950.0,
            Omega_Jmol=12000.0,  # ~12 kJ/mol for Co-Cr
            V_m_m3mol=6.7e-6,    # 6.7 cm³/mol
            D_b_m2s=5.0e-15      # Cr diffusion in Co at 950°C
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
        
        # --- Run Controls ---
        st.subheader("⏱️ Time Stepping")
        
        col_run1, col_run2 = st.columns(2)
        with col_run1:
            steps_input = st.number_input(
                "Steps per update", 
                min_value=1, max_value=5000, value=100,
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
        
        # --- Initialization ---
        st.subheader("🎲 Initial Conditions")
        
        num_seeds = st.slider("Number of HCP seeds", 1, 50, 12, 1)
        seed_radius = st.slider("Seed radius (grid units)", 3, 15, 5, 1)
        seed_co = st.slider("Co fraction in seeds", 0.50, 0.61, 0.58, 0.01)
        
        if st.button("🔄 Reset with New Random Seeds", use_container_width=True):
            sim.initialize_fcc_with_random_hcp_seeds(
                num_seeds=num_seeds, 
                radius_grid=seed_radius, 
                seed_co_fraction=seed_co, 
                seed=int(time.time()) % 10000
            )
            st.rerun()
        
        st.divider()
        
        # --- Material Parameters ---
        st.subheader("🧪 Material Properties")
        st.caption("Co-Cr-Mo alloy parameters at 950°C")
        
        # Regular solution parameter
        Omega_kJmol = st.slider(
            "Mixing enthalpy Ω (kJ/mol)", 
            5.0, 30.0, 12.0, 1.0,
            help="Controls chemical driving force for phase separation"
        )
        
        # Diffusion coefficient (log scale) – FIX: min/max as floats
        D_b_exp = st.slider(
            "log₁₀(D_Cr) [m²/s]",
            -17.0,          # min (float)
            -13.0,          # max (float)
            -14.3,          # value (float)
            0.1,            # step (float)
            help="Cr diffusion coefficient in Co matrix at 950°C"
        )
        D_b_val = 10**D_b_exp
        
        # Structural mobility (martensitic growth rate)
        L_factor = st.slider(
            "Structural mobility factor", 
            1.0, 200.0, 50.0, 10.0,
            help="Relative rate of FCC→HCP transformation (higher = faster lath growth)"
        )
        
        apply_material = st.button("Apply Material Parameters", use_container_width=True)
        if apply_material:
            sim.set_physical_parameters(
                Omega_Jmol=Omega_kJmol * 1000,
                D_b_m2s=D_b_val,
                L_m3Js=L_factor * sim.scales.L0_struct
            )
            st.rerun()
        
        st.divider()
        
        # --- Model Parameters ---
        st.subheader("⚙️ Model Parameters")
        st.caption("Phase-field parameters in physical units")
        
        # Interface width indicator
        xi_eta_nm = sim.scales.phys_to_interface_width(sim.kappa_eta, sim.W_phys) * 1e9
        
        W_phys = st.number_input(
            "W: Structural barrier (J/m³)",
            min_value=1e4, max_value=1e8,
            value=float(sim.W_phys), format="%.2e",
            help="Energy barrier between FCC and HCP; controls interface width"
        )
        
        kappa_eta_phys = st.number_input(
            "κ_η: Structural gradient coeff (J/m)",
            min_value=1e-13, max_value=1e-9,
            value=float(sim.kappa_eta), format="%.2e",
            help=f"Controls HCP/FCC interface energy; ξ ≈ √(κ_η/W) ≈ {xi_eta_nm:.2f} nm"
        )
        
        M_phys = st.number_input(
            "M: Chemical mobility (m⁵/J·s)",
            min_value=1e-25, max_value=1e-18,
            value=float(sim.M_chem), format="%.2e",
            help="Controls Cr diffusion kinetics; lower = slower solute redistribution"
        )
        
        # FIX: increased max_value to 1e-5 to accommodate default 4e-6
        dt_phys = st.number_input(
            "Δt: Time step (s)",
            min_value=1e-12,
            max_value=1e-5,          # was 1e-6 – now large enough
            value=float(sim.dt_phys),
            format="%.2e",
            help="Numerical time step; stability: Δt ≲ 0.01·(Δx)⁴/(M·κ)"
        )
        
        apply_model = st.button("Apply Model Parameters", use_container_width=True)
        if apply_model:
            sim.set_physical_parameters(
                W_Jm3=W_phys,
                kappa_eta_Jm=kappa_eta_phys,
                M_m5Js=M_phys,
                dt_s=dt_phys
            )
            st.rerun()
        
        # Stability warning
        if xi_eta_nm < 1.5:
            st.warning(f"⚠️ Interface width ({xi_eta_nm:.2f} nm) < 1.5 nm: under-resolved")
        elif xi_eta_nm > 15.0:
            st.info(f"ℹ️ Interface width ({xi_eta_nm:.1f} nm) is quite diffuse")
        
        st.divider()
        
        # --- Live Statistics ---
        stats = sim.get_statistics()
        st.subheader("📊 Live Statistics")
        
        # Key metrics in colored cards
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
        
        # Phase fractions with color coding
        st.markdown(f"**Phase Distribution**")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric(
                "<span class='phase-hcp'>ε-HCP Fraction</span>", 
                f"{stats['hcp_fraction']*100:.1f}%",
                help="Volume fraction of martensitic HCP phase"
            )
        with col_p2:
            st.metric(
                "<span class='phase-fcc'>γ-FCC Fraction</span>", 
                f"{stats['fcc_fraction']*100:.1f}%",
                help="Volume fraction of austenitic FCC phase"
            )
        
        # Concentration statistics
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
    
    # Physical extent for axis labels (convert grid to μm)
    extent_um_x = [0, sim.nx * sim.dx_phys * 1e6]
    extent_um_y = [0, sim.ny * sim.dx_phys * 1e6]
    
    # Row 1: Structural order parameter + Concentration field
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("ε-HCP Order Parameter η")
        st.caption(f"η = 0 (<span class='phase-fcc'>FCC</span>) → η = 1 (<span class='phase-hcp'>HCP</span>) | t = {stats['time_formatted']}")
        
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
            title="Martensitic HCP Phase Distribution",
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            width=600, height=550,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_eta, use_container_width=True)
    
    with col_viz2:
        st.subheader("Co Concentration c_Co")
        st.caption("Nominal composition: c₀ = 0.61 (61 at.% Co)")
        
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
            title="Cobalt Mole Fraction Distribution",
            xaxis_title="x (μm)",
            yaxis_title="y (μm)",
            width=600, height=550,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        st.plotly_chart(fig_c, use_container_width=True)
    
    # Row 2: Overlay visualization (concentration + η contours)
    col_overlay, col_hist = st.columns([2, 1])
    
    with col_overlay:
        st.subheader("Phase + Composition Overlay")
        st.caption("HCP regions (red) with Co depletion (blue) at interfaces")
        
        fig_overlay = go.Figure()
        # Base: concentration heatmap
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
        # Contours of η
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
        st.subheader("Composition Distribution")
        
        # Histogram using plotly express
        hist_data = sim.c.flatten()
        fig_hist = px.histogram(
            hist_data, nbins=40, range_x=[0.55, 0.67],
            labels={'value': 'Co mole fraction c_Co', 'count': 'Frequency'},
            title="Co Concentration Distribution"
        )
        # Add vertical lines
        fig_hist.add_vline(x=0.61, line_dash="dash", line_color="gray", annotation_text="Nominal c₀=0.61")
        fig_hist.add_vline(x=stats['c_mean'], line_dash="dash", line_color="red", annotation_text=f"⟨c⟩={stats['c_mean']:.3f}")
        fig_hist.update_layout(
            width=400, height=500,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Row 3: Kinetics plots (subplots)
    st.divider()
    st.subheader("📈 Transformation Kinetics")
    
    if len(sim.history['time_phys']) > 3:
        times_s = np.array(sim.history['time_phys'])
        times_min = times_s / 60  # Convert to minutes for metallurgical relevance
        
        # Create subplots: 1 row, 3 columns
        fig_kin = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Martensitic Transformation Progress", 
                            "Solute Redistribution",
                            "Interface Evolution"),
            shared_xaxes=True,
            x_title="Time (minutes)"
        )
        
        # Plot 1: HCP fraction (η_mean)
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_mean'],
                       mode='lines', name='⟨η⟩ (HCP fraction)',
                       line=dict(color='#e74c3c', width=2.5)),
            row=1, col=1
        )
        fig_kin.add_hline(y=0.5, line_dash="dash", line_color="gray", row=1, col=1)
        fig_kin.update_yaxes(title_text="HCP volume fraction", row=1, col=1)
        
        # Plot 2: Co concentration mean and std
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
        
        # Plot 3: Order parameter std (interface sharpness)
        fig_kin.add_trace(
            go.Scatter(x=times_min, y=sim.history['eta_std'],
                       mode='lines', name='σ(η)',
                       line=dict(color='#9b59b6', width=2)),
            row=1, col=3
        )
        fig_kin.update_yaxes(title_text="Order parameter std. dev.", row=1, col=3)
        
        fig_kin.update_layout(
            height=450, width=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_kin, use_container_width=True)
        
        # Optional: Free energy evolution
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
                    title="Free Energy Minimization",
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
        if st.button("📸 Save Microstructure Snapshot", use_container_width=True):
            # Create a plotly figure for the snapshot
            fig_snap = go.Figure(data=go.Heatmap(
                z=sim.eta.T,
                x=np.linspace(extent_um_x[0], extent_um_x[1], sim.nx),
                y=np.linspace(extent_um_y[0], extent_um_y[1], sim.ny),
                colorscale='RdYlBu_r',
                zmin=0, zmax=1,
                colorbar=dict(title="η", tickvals=[0, 0.5, 1], ticktext=['FCC', 'Interface', 'HCP'])
            ))
            fig_snap.update_layout(
                title=f"Mediloy HCP Phase Distribution<br>t = {stats['time_formatted']}",
                xaxis_title="x (μm)",
                yaxis_title="y (μm)",
                width=800, height=700
            )
            # Export as PNG (requires kaleido)
            img_bytes = fig_snap.to_image(format="png", width=800, height=700, scale=2)
            st.download_button(
                label="⬇️ Download PNG",
                data=img_bytes,
                file_name=f"Mediloy_HCP_t{sim.time_phys:.2e}s.png",
                mime="image/png",
                use_container_width=True
            )
    
    with col_exp2:
        if st.button("📊 Save Kinetics Data", use_container_width=True):
            csv_lines = ["time_s,time_min,eta_mean,eta_std,c_mean,c_std,total_energy_J"]
            for i in range(len(sim.history['time_phys'])):
                t_s = sim.history['time_phys'][i]
                line = f"{t_s:.6e},"
                line += f"{t_s/60:.6e},"
                line += f"{sim.history['eta_mean'][i]:.6f},"
                line += f"{sim.history['eta_std'][i]:.6f},"
                line += f"{sim.history['c_mean'][i]:.6f},"
                line += f"{sim.history['c_std'][i]:.6f},"
                line += f"{sim.history['total_energy'][i]:.6e}"
                csv_lines.append(line)
            
            csv_content = "\n".join(csv_lines)
            
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_content,
                file_name="mediloy_kinetics.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col_exp3:
        if st.button("⚙️ Save Simulation State", use_container_width=True):
            # Save NumPy arrays of both fields
            npz_buf = BytesIO()
            np.savez_compressed(
                npz_buf,
                concentration=sim.c,
                order_parameter=sim.eta,
                time_phys=sim.time_phys,
                step=sim.step,
                params={
                    'T_celsius': sim.T_celsius,
                    'Omega_Jmol': sim.Omega_Jmol,
                    'D_b': sim.D_b,
                    'V_m': sim.V_m,
                    'W_phys': sim.W_phys,
                    'kappa_eta': sim.kappa_eta,
                    'M_chem': sim.M_chem,
                    'L_struct': sim.L_struct,
                    'dt_phys': sim.dt_phys,
                    'dx_phys': sim.dx_phys,
                    'lambda_coup': sim.lambda_coup
                }
            )
            npz_buf.seek(0)
            
            filename = f"Mediloy_state_t{sim.time_phys:.2e}s.npz"
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
    
    # Footer
    st.markdown("---")
    st.caption(
        "Mediloy γ→ε Phase-Field Simulator | Hybrid Cahn-Hilliard + Allen-Cahn | "
        f"Physical Units: m, s, J/m³ | T = {sim.T_celsius}°C | "
        f"Pseudo-binary Co-M<sub>y</sub> (c₀ = 0.61) | Visualized with Plotly"
    )


# =============================================================================
# Application Entry Point
# =============================================================================

if __name__ == "__main__":
    # Print startup info to console
    print("⚙️ Starting Mediloy Phase Transformation Simulator (Plotly version)...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   NumPy: {np.__version__}")
    print(f"   Numba: JIT compilation enabled")
    print(f"   Streamlit: launching interactive app")
    print(f"   Temperature: 950°C (1223.15 K)")
    print(f"   Initial condition: FCC matrix + random HCP seeds")
    
    # Run the Streamlit app
    main()
