# =============================================================================
# MEDILOY γ-FCC → ε-HCP PHASE DECOMPOSITION SIMULATOR (CAHN-HILLIARD FOR η)
# Adapted from LiFePO₄ Cahn-Hilliard framework + Mediloy physical scales & free energy
# η now evolved as a CONSERVED order parameter via full Cahn-Hilliard equation
# (as requested: "representation of order parameter with Cahn Hilliard equation")
# Properties taken directly from Mediloy (Co-Cr-Mo dental alloy at 950°C)
# =============================================================================
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import time
import sys

# =============================================================================
# PHYSICAL SCALES – MEDILOY (Co-Cr-Mo at 950°C) – reused verbatim from original
# =============================================================================
class PhysicalScalesMediloy:
    """
    Physical unit conversion for Mediloy phase-field simulations.
    T = 950°C (1223.15 K), pseudo-binary Co-M_y alloy.
    ΔG_chem ≈ 400 J/mol → small driving force near T0.
    """
    def __init__(self, T_celsius=950.0, V_m_m3mol=6.7e-6, D_b_m2s=5.0e-15):
        self.R = 8.314462618
        self.T = T_celsius + 273.15
        self.T_celsius = T_celsius
        self.V_m = V_m_m3mol
        self.D_b = D_b_m2s
        self.L0 = 2.0e-9  # m – interface width scale
        delta_G_mol = 400.0  # J/mol
        self.E0 = delta_G_mol / self.V_m  # J/m³
        self.t0 = self.L0**2 / self.D_b  # s
        self.M0 = self.D_b / self.E0  # m⁵/(J·s)
        print(f"Mediloy scales initialized at {self.T_celsius}°C – E0={self.E0:.2e} J/m³")

    def dim_to_phys(self, W_dim, kappa_dim, M_dim, dt_dim, dx_dim=1.0):
        W_phys = W_dim * self.E0
        kappa_phys = kappa_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        dx_phys = dx_dim * self.L0
        return W_phys, kappa_phys, M_phys, dt_phys, dx_phys

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
# NUMBA KERNELS – Cahn-Hilliard for structural order parameter η (Mediloy style)
# =============================================================================
@njit(fastmath=True, cache=True)
def structural_free_energy(eta, W_struct):
    """f_struct(η) = W·η²(1-η)² – double-well for FCC (0) ↔ HCP (1)"""
    return W_struct * eta**2 * (1.0 - eta)**2

@njit(fastmath=True, cache=True)
def d_fstruct_deta(eta, W_struct):
    """∂f_struct/∂η = 2W·η(1-η)(1-2η)"""
    return 2.0 * W_struct * eta * (1.0 - eta) * (1.0 - 2.0 * eta)

@njit(fastmath=True, parallel=True)
def compute_laplacian(field, dx):
    nx, ny = field.shape
    lap = np.zeros_like(field)
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

@njit(fastmath=True, parallel=True)
def compute_gradient_x(field, dx):
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
    return grad_x

@njit(fastmath=True, parallel=True)
def compute_gradient_y(field, dx):
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
    return grad_y

@njit(fastmath=True, parallel=True)
def update_eta_cahn_hilliard(eta, dt, dx, kappa, M, W_struct):
    """
    Full Cahn-Hilliard evolution for conserved order parameter η (HCP fraction).
    ∂η/∂t = ∇·[ M ∇(δF/δη) ] with δF/δη = ∂f_struct/∂η - κ ∇²η
    (Learned from LiFePO₄ kernel but using Mediloy structural free energy)
    """
    nx, ny = eta.shape

    # Laplacian
    lap_eta = compute_laplacian(eta, dx)

    # Local chemical potential from double-well
    mu_local = d_fstruct_deta(eta, W_struct)

    # Full variational derivative (including gradient penalty)
    mu = mu_local - kappa * lap_eta

    # Gradients of μ
    mu_x = compute_gradient_x(mu, dx)
    mu_y = compute_gradient_y(mu, dx)

    # Flux J = -M ∇μ
    flux_x = -M * mu_x
    flux_y = -M * mu_y

    # Divergence of flux
    div_flux = np.zeros_like(eta)
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            div_x = (flux_x[ip1, j] - flux_x[im1, j]) / (2.0 * dx)
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
            div_flux[i, j] = div_x + div_y

    # Forward Euler + physical clipping
    eta_new = eta + dt * div_flux
    eta_new = np.clip(eta_new, 0.0, 1.0)
    return eta_new

# =============================================================================
# MAIN SIMULATION CLASS – η as conserved order parameter (Cahn-Hilliard)
# =============================================================================
class MediloyCHPhaseDecomposition:
    """
    2D Cahn-Hilliard simulation of γ-FCC ↔ ε-HCP phase decomposition
    in Mediloy (Co-Cr-Mo) at 950°C. η is now a CONSERVED order parameter.
    """
    def __init__(self, nx=256, ny=256, dx_dim=1.0, dt_dim=0.005,
                 T_celsius=950.0, V_m=6.7e-6, D_b=5.0e-15):
        self.nx = nx
        self.ny = ny
        self.dx_dim = dx_dim
        self.dt_dim = dt_dim

        # Dimensionless model parameters (tuned for Mediloy)
        self.W_dim = 1.0
        self.kappa_dim = 2.0
        self.M_dim = 1.0

        # Initialize scales with Mediloy properties
        self.scales = PhysicalScalesMediloy(T_celsius=T_celsius, V_m_m3mol=V_m, D_b_m2s=D_b)

        # Convert to physical
        self._update_physical_params()

        # Fields
        self.eta = np.zeros((nx, ny), dtype=np.float64)  # η = 0 → FCC, η = 1 → HCP

        # History
        self.time_phys = 0.0
        self.step = 0
        self.history = {
            'time_phys': [], 'eta_mean': [], 'eta_std': [],
            'hcp_fraction': [], 'fcc_fraction': [], 'energy': []
        }
        self.update_history()

    def _update_physical_params(self):
        (self.W_phys, self.kappa_phys, self.M_phys,
         self.dt_phys, self.dx_phys) = self.scales.dim_to_phys(
            self.W_dim, self.kappa_dim, self.M_dim, self.dt_dim
        )

    def set_physical_parameters(self, W_Jm3=None, kappa_Jm=None, M_m5Js=None, dt_s=None):
        if W_Jm3 is not None:
            self.W_dim = W_Jm3 / self.scales.E0
        if kappa_Jm is not None:
            self.kappa_dim = kappa_Jm / (self.scales.E0 * self.scales.L0**2)
        if M_m5Js is not None:
            self.M_dim = M_m5Js / self.scales.M0
        if dt_s is not None:
            self.dt_dim = dt_s / self.scales.t0
        self._update_physical_params()

    def initialize_random(self, eta0=0.0, noise_amplitude=0.02, seed=42):
        np.random.seed(seed)
        noise = noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.eta = np.clip(eta0 + noise, 0.0, 1.0)
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def initialize_hcp_seeds(self, num_seeds=12, radius_grid=8, seed=42):
        np.random.seed(seed)
        self.eta = np.zeros((self.nx, self.ny), dtype=np.float64)
        for _ in range(num_seeds):
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
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()

    def clear_history(self):
        self.history = {'time_phys': [], 'eta_mean': [], 'eta_std': [],
                        'hcp_fraction': [], 'fcc_fraction': [], 'energy': []}
        self.update_history()

    def update_history(self):
        self.history['time_phys'].append(self.time_phys)
        self.history['eta_mean'].append(float(np.mean(self.eta)))
        self.history['eta_std'].append(float(np.std(self.eta)))
        self.history['hcp_fraction'].append(float(np.sum(self.eta > 0.5) / (self.nx * self.ny)))
        self.history['fcc_fraction'].append(float(np.sum(self.eta < 0.5) / (self.nx * self.ny)))
        # Bulk energy only (gradient added in total_energy if needed)
        f_bulk = structural_free_energy(self.eta, self.W_phys)
        self.history['energy'].append(float(np.mean(f_bulk)))

    def run_step(self):
        self.eta = update_eta_cahn_hilliard(
            self.eta, self.dt_phys, self.dx_phys,
            self.kappa_phys, self.M_phys, self.W_phys
        )
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()

    def run_steps(self, n_steps):
        for _ in range(n_steps):
            self.run_step()

    def get_statistics(self):
        domain_size_m = self.nx * self.dx_phys
        interface_width_nm = self.scales.phys_to_interface_width(self.kappa_phys, self.W_phys) * 1e9
        return {
            'time_formatted': self.scales.format_time(self.time_phys),
            'step': self.step,
            'domain_size_formatted': self.scales.format_length(domain_size_m),
            'interface_width_nm': interface_width_nm,
            'eta_mean': float(np.mean(self.eta)),
            'eta_std': float(np.std(self.eta)),
            'hcp_fraction': float(np.sum(self.eta > 0.5) / (self.nx * self.ny)),
            'fcc_fraction': float(np.sum(self.eta < 0.5) / (self.nx * self.ny)),
            'W_phys': self.W_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
        }

# =============================================================================
# STREAMLIT APP
# =============================================================================
def main():
    st.set_page_config(page_title="Mediloy γ→ε (Cahn-Hilliard η)", page_icon="⚙️", layout="wide")
    st.title("⚙️ Mediloy γ-FCC → ε-HCP Phase Decomposition (Cahn-Hilliard for η)")
    st.markdown("""
    **Conserved structural order parameter η evolved with full Cahn-Hilliard equation**  
    (Learned from LiFePO₄ code structure + Mediloy physical scales & double-well)  
    η = 0 → γ-FCC | η = 1 → ε-HCP | T = 950°C, pseudo-binary Co-M_y
    """)

    if 'sim' not in st.session_state:
        st.session_state.sim = MediloyCHPhaseDecomposition(nx=256, ny=256)
        st.session_state.sim.initialize_random(eta0=0.0, noise_amplitude=0.02, seed=42)

    sim = st.session_state.sim

    # Sidebar controls (adapted from both codes)
    with st.sidebar:
        st.header("🎛️ Control Panel")
        steps_input = st.number_input("Steps per update", 1, 5000, 100)
        if st.button("▶️ Run", type="primary"):
            with st.spinner(f"Running {steps_input} steps..."):
                sim.run_steps(steps_input)
            st.rerun()

        if st.button("⏭️ Single Step"):
            sim.run_step()
            st.rerun()

        st.divider()
        st.subheader("🎲 Initial Conditions")
        if st.button("🔄 Random FCC matrix + noise"):
            sim.initialize_random(eta0=0.0, noise_amplitude=0.02, seed=int(time.time()))
            st.rerun()
        if st.button("🌱 Random HCP seeds in FCC"):
            sim.initialize_hcp_seeds(num_seeds=12, radius_grid=8, seed=int(time.time()))
            st.rerun()

        st.divider()
        st.subheader("⚙️ Model Parameters (Mediloy)")
        W_phys = st.number_input("W (J/m³)", 1e4, 1e8, float(sim.W_phys), format="%.2e")
        kappa_phys = st.number_input("κ (J/m)", 1e-13, 1e-9, float(sim.kappa_phys), format="%.2e")
        dt_phys = st.number_input("Δt (s)", 1e-12, 1e-5, float(sim.dt_phys), format="%.2e")
        if st.button("Apply"):
            sim.set_physical_parameters(W_Jm3=W_phys, kappa_Jm=kappa_phys, dt_s=dt_phys)
            st.rerun()

        stats = sim.get_statistics()
        st.subheader("📊 Live Stats")
        st.metric("Physical Time", stats['time_formatted'])
        st.metric("HCP Fraction", f"{stats['hcp_fraction']*100:.1f}%")
        st.metric("Interface Width", f"{stats['interface_width_nm']:.2f} nm")

    # Main visualizations
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("ε-HCP Order Parameter η")
        fig, ax = plt.subplots(figsize=(8, 7))
        extent_um = [0, sim.nx * sim.dx_phys * 1e6]
        im = ax.imshow(sim.eta.T, cmap='RdYlBu_r', origin='lower',
                       vmin=0, vmax=1, extent=extent_um,
                       interpolation='bilinear')
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        plt.colorbar(im, ax=ax, label="η (0=FCC → 1=HCP)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("η Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(sim.eta.flatten(), bins=50, range=(0,1), color="#e74c3c", alpha=0.8)
        ax.axvline(0.5, color="gray", ls="--", label="FCC/HCP boundary")
        ax.set_xlabel("η")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # Kinetics
    st.subheader("📈 Transformation Kinetics")
    if len(sim.history['time_phys']) > 2:
        times_min = np.array(sim.history['time_phys']) / 60
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(times_min, sim.history['eta_mean'], color="#e74c3c", lw=2)
        axes[0].set_title("⟨η⟩ (HCP fraction)")
        axes[1].plot(times_min, sim.history['eta_std'], color="#9b59b6", lw=2)
        axes[1].set_title("σ(η)")
        axes[2].plot(times_min, np.array(sim.history['hcp_fraction'])*100, color="#e74c3c", lw=2, label="HCP")
        axes[2].plot(times_min, np.array(sim.history['fcc_fraction'])*100, color="#2ecc71", lw=2, label="FCC")
        axes[2].set_title("Phase Fractions (%)")
        for ax in axes:
            ax.set_xlabel("Time (min)")
            ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Export
    st.divider()
    if st.button("📸 Save η Snapshot"):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(sim.eta.T, cmap='RdYlBu_r', origin='lower', vmin=0, vmax=1)
        ax.set_title(f"Mediloy η – t = {stats['time_formatted']}")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        plt.close(fig)
        buf.seek(0)
        st.download_button("⬇️ Download PNG", buf.getvalue(), f"mediloy_eta_{sim.time_phys:.2e}s.png", "image/png")

# =============================================================================
if __name__ == "__main__":
    main()
