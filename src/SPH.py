import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Kernel functions
# -----------------------------------------------------------------------------
def cubic_b_spline_w(r: float, h: float) -> float:
    """Return the 1D cubic B-spline kernel value W(r, h)."""
    q = abs(r) / h
    alpha = 1.0 / h
    if 0.0 <= q < 1.0:
        return alpha * (2.0 / 3.0 - q * q + 0.5 * q**3)
    if 1.0 <= q < 2.0:
        return alpha * ((2.0 - q) ** 3) / 6.0
    return 0.0


def cubic_b_spline_grad(r: float, h: float) -> float:
    """Return dW/dr for the 1D cubic B-spline kernel.

    The MATLAB draft used abs(r) before differentiating, which removed the
    sign and made the gradient incorrect. This implementation keeps the sign.
    """
    if r == 0.0:
        return 0.0
    sign = 1.0 if r > 0.0 else -1.0
    q = abs(r) / h
    alpha = 1.0 / h
    if 0.0 <= q < 1.0:
        dwdq = -2.0 * q + 1.5 * q**2
        return sign * alpha * dwdq / h
    if 1.0 <= q < 2.0:
        dwdq = -0.5 * (2.0 - q) ** 2
        return sign * alpha * dwdq / h
    return 0.0


# -----------------------------------------------------------------------------
# Boundary handling
# -----------------------------------------------------------------------------
def apply_ghost_particles(head: np.ndarray, vel: np.ndarray, hr: float) -> None:
    """Apply ghost-particle values in place.

    Layout:
        0, 1           : upstream ghost particles
        2 ... N + 1    : physical particles
        N + 2, N + 3   : downstream ghost particles

    Upstream boundary: constant reservoir head.
    Downstream boundary: closed valve represented by velocity anti-symmetry and
    head mirror symmetry.
    """
    n = len(head)
    last_real = n - 3
    second_last_real = n - 4

    # Upstream reservoir.
    head[0] = hr
    head[1] = hr
    vel[0] = vel[2]
    vel[1] = vel[2]

    # Downstream closed end.
    head[n - 2] = head[last_real]
    head[n - 1] = head[second_last_real]
    vel[n - 2] = -vel[last_real]
    vel[n - 1] = -vel[second_last_real]


# -----------------------------------------------------------------------------
# Main solver
# -----------------------------------------------------------------------------
def simulate_sph_water_hammer(save_dir: str = "/mnt/data"):
    """Run an SPH-style water-hammer simulation.

    This Python version completes the unfinished MATLAB draft and fixes several
    important issues:

    1. Added the missing initial flow velocity.
    2. Corrected the kernel gradient sign.
    3. Replaced the unstable moving-particle update with a fixed particle
       lattice, which is much more robust for this 1D pipe-transient problem.
    4. Corrected the friction term so it always opposes the flow.
    5. Completed the boundary reconstruction for the valve head curve.
    6. Converted all comments to English.
    """
    # -------------------------------------------------------------------------
    # Physical parameters
    # -------------------------------------------------------------------------
    L = 300.0                         # Pipe length (m)
    Hr = 70.0                         # Upstream reservoir head (m)
    N = 10                            # Number of physical particles
    NS = N + 4                        # Two ghost particles on each side
    e = 0.001651                      # Wall thickness (m)
    D_outer = 0.00635                 # Outer diameter (m)
    K = 2.1e9                         # Bulk modulus (Pa)
    rho_ref = 1000.0                  # Reference density (kg/m^3)
    E = 2.1e11                        # Young's modulus (Pa)
    g = 9.806                         # Gravity acceleration (m/s^2)
    f = 0.018                         # Darcy-Weisbach friction factor
    u0 = 0.1                          # Initial steady velocity (m/s)

    # -------------------------------------------------------------------------
    # Numerical parameters
    # -------------------------------------------------------------------------
    CFL = 0.2                         # Conservative time-step factor
    t_max = 20.0                      # Total simulation time (s)

    # -------------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------------
    D = D_outer - 2.0 * e
    if D <= 0.0:
        raise ValueError("The inner diameter must be positive.")

    dx = L / N
    a = math.sqrt(K / rho_ref / (1.0 + K * D / (E * e)))
    dt = CFL * dx / a
    n_steps = int(math.ceil(t_max / dt))

    # Particle locations: two ghosts at each side.
    x = np.arange(-1.5 * dx, L + 1.5 * dx + 0.5 * dx, dx, dtype=float)
    if len(x) != NS:
        raise RuntimeError("Unexpected particle count. Please check the lattice setup.")

    # The smoothing length is kept close to the original draft.
    h = 1.3 * dx

    # Use a 1D particle volume weight instead of the inconsistent density-area
    # expression from the MATLAB draft.
    weight = dx

    # Storage arrays.
    time = np.arange(n_steps + 1, dtype=float) * dt
    time[-1] = min(time[-1], t_max)
    head = np.zeros((n_steps + 1, NS), dtype=float)
    vel = np.zeros((n_steps + 1, NS), dtype=float)

    # Output arrays in the same spirit as the MATLAB script:
    # [upstream boundary, N interior particles, downstream valve].
    head_nodes = np.zeros((n_steps + 1, N + 2), dtype=float)
    vel_nodes = np.zeros((n_steps + 1, N + 2), dtype=float)

    real_slice = slice(2, N + 2)
    x_real = x[real_slice]

    # -------------------------------------------------------------------------
    # Initial condition: steady flow before the valve is closed
    # -------------------------------------------------------------------------
    head_loss_total = f * L / D * u0 * u0 / (2.0 * g)
    head_gradient = head_loss_total / L

    head[0, real_slice] = Hr - head_gradient * x_real
    vel[0, real_slice] = u0
    apply_ghost_particles(head[0], vel[0], Hr)

    head_nodes[0, 0] = Hr
    head_nodes[0, 1:-1] = head[0, real_slice]
    head_nodes[0, -1] = Hr - head_loss_total
    vel_nodes[0, 0] = u0
    vel_nodes[0, 1:-1] = vel[0, real_slice]
    vel_nodes[0, -1] = u0

    neighbor_offsets = (-2, -1, 1, 2)
    diffusion_coeff = 0.01 * a

    # -------------------------------------------------------------------------
    # Time marching
    # -------------------------------------------------------------------------
    for n in range(n_steps):
        apply_ghost_particles(head[n], vel[n], Hr)

        for i in range(2, N + 2):
            xi = x[i]
            hi = head[n, i]
            vi = vel[n, i]

            div_v = 0.0
            grad_h = 0.0
            vel_diffusion = 0.0

            for off in neighbor_offsets:
                j = i + off
                if j < 0 or j >= NS:
                    continue
                r = xi - x[j]
                if abs(r) >= 2.0 * h + 1e-12:
                    continue

                grad_w = cubic_b_spline_grad(r, h)
                div_v += weight * (vel[n, j] - vi) * grad_w
                grad_h += weight * (head[n, j] - hi) * grad_w
                vel_diffusion += weight * (vel[n, j] - vi) * abs(grad_w)

            # Linearized acoustic SPH update.
            dHdt = -(a * a / g) * div_v
            dVdt = -g * grad_h + diffusion_coeff * vel_diffusion

            # Friction must always remove momentum.
            dVdt += -(f / (2.0 * D)) * vi * abs(vi)

            head[n + 1, i] = max(1e-8, head[n, i] + dt * dHdt)
            vel[n + 1, i] = vel[n, i] + dt * dVdt

        apply_ghost_particles(head[n + 1], vel[n + 1], Hr)

        # Build nodal outputs.
        head_nodes[n + 1, 0] = Hr
        head_nodes[n + 1, 1:-1] = head[n + 1, real_slice]
        vel_nodes[n + 1, 1:-1] = vel[n + 1, real_slice]

        # Upstream boundary relation.
        vel_nodes[n + 1, 0] = vel_nodes[n + 1, 1] + (vel_nodes[n, 0] - vel_nodes[n, 1]) * g / a

        # Instantaneous valve closure at the downstream end.
        vel_nodes[n + 1, -1] = 0.0
        head_nodes[n + 1, -1] = head_nodes[n + 1, -2] + (a / g) * vel_nodes[n + 1, -2]

    # -------------------------------------------------------------------------
    # Save the figure
    # -------------------------------------------------------------------------
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    figure_path = save_dir_path / "sph_valve_head.png"

    plt.figure(figsize=(8, 4.5))
    plt.plot(time, head_nodes[:, -1])
    plt.title("SPH water hammer: valve head curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Head (m)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    return {
        "time": time,
        "head_nodes": head_nodes,
        "vel_nodes": vel_nodes,
        "particle_head": head,
        "particle_velocity": vel,
        "x": x,
        "figure_path": str(figure_path),
        "wave_speed": a,
        "dt": dt,
        "n_steps": n_steps,
    }


if __name__ == "__main__":
    result = simulate_sph_water_hammer()
    print("SPH water hammer simulation finished.")
    print(f"Wave speed: {result['wave_speed']:.6f} m/s")
    print(f"Time step:  {result['dt']:.6f} s")
    print(f"Steps:      {result['n_steps']}")
    print(f"Figure:     {result['figure_path']}")