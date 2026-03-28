import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from MOC import moc_solve


# -----------------------------------------------------------------------------
# Shared physical setup (kept identical to MOC.py)
# -----------------------------------------------------------------------------
L = 300.0                    # Pipe length [m]
Hr = 70.0                    # Upstream reservoir head [m]
N = 25                       # Number of pipe reaches
NS = N + 1                   # Number of nodes
wall_thickness = 0.001651    # Pipe wall thickness [m]
D = 0.00635 - 2.0 * wall_thickness  # Inner diameter [m]
K = 2.1e9                    # Bulk modulus [Pa]
rho = 1000.0                 # Fluid density [kg/m^3]
E = 2.1e11                   # Young's modulus [Pa]
g = 9.806                    # Gravitational acceleration [m/s^2]
f = 0.018                    # Darcy-Weisbach friction factor [-]
V0 = 0.1                     # Initial steady velocity [m/s]
t_max = 20.0                 # Total simulation time [s]
area = np.pi * D**2 / 4.0    # Pipe cross-sectional area [m^2]
a = np.sqrt(K / rho / (1.0 + K * D / (E * wall_thickness)))  # Wave speed [m/s]
dx = L / N                   # Spatial step [m]
dt = dx / a                  # Time step aligned with MOC
x = np.arange(NS) * dx       # Node coordinates [m]

# Lattice scaling used only for the characteristic-LBM variables
lambda_h = a * a / g
hr_lattice = Hr / lambda_h


# -----------------------------------------------------------------------------
# Reference MOC solver
# -----------------------------------------------------------------------------

def steady_initial_head_profile() -> np.ndarray:
    head0 = Hr - f * x * V0**2 / (2.0 * g * D)
    head0[0] = Hr
    return head0


def simulate_moc_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = moc_solve(N=25)
    return ref["time"], ref["H"], ref["Q"] / ref["A"]


def to_lattice_head(head_phys: np.ndarray) -> np.ndarray:
    return head_phys / lambda_h



def from_lattice_head(head_lat: np.ndarray) -> np.ndarray:
    return head_lat * lambda_h



def to_lattice_velocity(vel_phys: np.ndarray) -> np.ndarray:
    return vel_phys / a



def from_lattice_velocity(vel_lat: np.ndarray) -> np.ndarray:
    return vel_lat * a



def reconstruct_populations(v_lat: np.ndarray, h_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f_plus = 0.5 * (v_lat + h_lat)
    f_minus = 0.5 * (v_lat - h_lat)
    return f_plus, f_minus



def apply_wave_streaming(
    f_plus: np.ndarray,
    f_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    f_plus_new = np.empty_like(f_plus)
    f_minus_new = np.empty_like(f_minus)

    f_plus_new[1:] = f_plus[:-1]
    f_minus_new[:-1] = f_minus[1:]

    return f_plus_new, f_minus_new



def apply_moc_like_boundaries(
    f_plus: np.ndarray,
    f_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    f_plus[0] = hr_lattice + f_minus[0]

    f_minus[-1] = -f_plus[-1]

    return f_plus, f_minus



def apply_friction_source(v_lat: np.ndarray) -> np.ndarray:

    v_phys = from_lattice_velocity(v_lat)
    v_new = v_phys.copy()

    # Apply friction only to interior nodes.
    v_new[1:-1] -= dt * f * v_phys[1:-1] * np.abs(v_phys[1:-1]) / (2.0 * D)

    # Enforce the closed-valve condition exactly after the source update.
    v_new[-1] = 0.0

    return to_lattice_velocity(v_new)



def simulate_lbm_tuned() -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n_steps = int(np.ceil(t_max / dt)) + 1
    time_hist = np.zeros(n_steps, dtype=float)
    v_lat_hist = np.zeros((n_steps, NS), dtype=float)
    h_lat_hist = np.zeros((n_steps, NS), dtype=float)

    h_lat_hist[0, :] = to_lattice_head(steady_initial_head_profile())
    v_lat_hist[0, :] = to_lattice_velocity(np.full(NS, V0, dtype=float))

    last = 0
    for n in range(n_steps - 1):

        f_plus, f_minus = reconstruct_populations(v_lat_hist[n, :], h_lat_hist[n, :])

        f_plus, f_minus = apply_wave_streaming(f_plus, f_minus)

        f_plus, f_minus = apply_moc_like_boundaries(f_plus, f_minus)

  
        v_wave = f_plus + f_minus
        h_wave = f_plus - f_minus

        v_next = apply_friction_source(v_wave)
        h_next = h_wave.copy()

        h_next[0] = hr_lattice
        v_next[-1] = 0.0

        time_hist[n + 1] = time_hist[n] + dt
        v_lat_hist[n + 1, :] = v_next
        h_lat_hist[n + 1, :] = h_next
        last = n + 1

        if time_hist[n + 1] >= t_max:
            break

    return (
        time_hist[: last + 1],
        from_lattice_head(h_lat_hist[: last + 1, :]),
        from_lattice_velocity(v_lat_hist[: last + 1, :]),
    )


# -----------------------------------------------------------------------------
# Comparison helpers
# -----------------------------------------------------------------------------

def relative_l2_error(reference: np.ndarray, numerical: np.ndarray) -> float:
    """Return the relative L2 error between two signals."""
    denom = np.linalg.norm(reference)
    if denom == 0.0:
        return np.linalg.norm(numerical - reference)
    return np.linalg.norm(numerical - reference) / denom



def main() -> None:
    start = time.perf_counter()

    time_moc, head_moc, vel_moc = simulate_moc_reference()
    time_lbm, head_lbm, vel_lbm = simulate_lbm_tuned()

    n_common = min(len(time_moc), len(time_lbm))
    time_cmp = time_moc[:n_common]
    valve_head_moc = head_moc[:n_common, -1]
    valve_head_lbm = head_lbm[:n_common, -1]

    valve_rel_l2 = relative_l2_error(valve_head_moc, valve_head_lbm)
    valve_max_abs = np.max(np.abs(valve_head_lbm - valve_head_moc))

    output_dir = Path("png")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "lbm_vs_moc_tuned_valve_head.png"

    plt.figure(figsize=(8.5, 5.0))
    plt.plot(time_cmp, valve_head_moc, label="MOC valve head", linewidth=2.0)
    plt.plot(time_cmp, valve_head_lbm, "--", label="Tuned LBM valve head", linewidth=1.8)
    plt.plot(time_cmp, np.full_like(time_cmp, Hr), ":", label="Reservoir head", linewidth=1.5)
    plt.title("Valve Head: Tuned LBM vs MOC")
    plt.xlabel("Time (s)")
    plt.ylabel("Head (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    elapsed = time.perf_counter() - start

    print(f"Aligned grid nodes            : {NS}")
    print(f"Aligned dx                    : {dx:.6f} m")
    print(f"Aligned dt                    : {dt:.6e} s")
    print(f"Wave speed                    : {a:.6f} m/s")
    print(f"Valve relative L2 error       : {valve_rel_l2:.6e}")
    print(f"Valve maximum absolute error  : {valve_max_abs:.6e} m")
    print(f"Figure saved to               : {figure_path}")
    print(f"Elapsed time                  : {elapsed:.6f} s")


if __name__ == "__main__":
    main()