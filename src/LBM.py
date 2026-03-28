import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def simulate_lbm_instant_valve_closure():
    """
    Lattice-Boltzmann-style simulation of a 1D water-hammer problem
    with an upstream constant-head reservoir and a downstream valve
    that closes instantaneously at t = 0.

    Returns
    -------
    dict
        Simulation results and metadata.
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Physical parameters  (consistent with FVM.py)
    # ------------------------------------------------------------------
    L = 300.0              # Pipe length [m]
    Hr = 70.0              # Upstream reservoir head [m]
    wall_thickness = 0.001651   # Pipe wall thickness [m]
    D = 0.00635 - 2.0 * wall_thickness   # Inner pipe diameter [m]
    bulk_modulus = 2.1e9   # Bulk modulus of water [Pa]
    density = 1000.0       # Fluid density [kg/m^3]
    young_modulus = 2.1e11 # Young's modulus of pipe wall [Pa]
    g = 9.806              # Gravitational acceleration [m/s^2]
    f = 0.018              # Darcy-Weisbach friction factor [-]
    V0 = 0.1               # Initial flow velocity [m/s]
    t_max = 20.0           # Total simulation time [s]

    # Wave speed derived from material properties
    a = np.sqrt(bulk_modulus / density / (1.0 + bulk_modulus * D / (young_modulus * wall_thickness)))

    # ------------------------------------------------------------------
    # Spatial and temporal discretization
    # ------------------------------------------------------------------
    nodes = 200
    dx = L / nodes
    dt = dx / a
    timesteps = int(np.ceil(t_max / dt))

    # ------------------------------------------------------------------
    # Lattice / scaling parameters
    # ------------------------------------------------------------------
    lambda_x = dx
    lambda_t = dt
    lambda_u = a
    lambda_h = lambda_u * a / g    # Converts lattice head to physical head
    lambda_f = 1.0 / (lambda_t * lambda_u)

    ff = f / lambda_f              # Dimensionless friction factor used in the source term
    hr_lattice = Hr / lambda_h     # Dimensionless upstream reservoir head
    d_lattice = D / lambda_x       # Dimensionless diameter

    # Single-relaxation-time collision parameter.
    # The original code used omega = dt / 1 with dt = 1, which is just omega = 1.
    omega = 1.0

    # ------------------------------------------------------------------
    # Initial conditions in lattice units
    # ------------------------------------------------------------------
    f0 = np.zeros(nodes, dtype=float)
    f1 = np.zeros(nodes, dtype=float)
    f2 = np.zeros(nodes, dtype=float)

    vv = np.zeros((timesteps + 2, nodes), dtype=float)  # Lattice velocity history
    hh = np.zeros((timesteps + 2, nodes), dtype=float)  # Lattice head history
    time_hist = np.zeros(timesteps + 2, dtype=float)

    vv[0, :] = V0 / lambda_u
    hh[0, :] = hr_lattice

    # ------------------------------------------------------------------
    # Main time-stepping loop
    # ------------------------------------------------------------------
    for n in range(timesteps):
        # Equilibrium distributions.
        feq0 = np.zeros(nodes, dtype=float)
        feq1 = vv[n, :] + 0.5 * hh[n, :]
        feq2 = vv[n, :] - 0.5 * hh[n, :]

        # Friction source term. Use u * |u| so that friction always opposes the flow.
        R = ff * vv[n, :] * np.abs(vv[n, :]) / (4.0 * d_lattice)

        # Collision step.
        f0 = (1.0 - omega) * f0 + omega * feq0
        f1 = (1.0 - omega) * f1 + omega * feq1 - R
        f2 = (1.0 - omega) * f2 + omega * feq2 - R

        # Streaming step. Copy the pre-streaming values explicitly to avoid
        # in-place overwrite issues when translating from MATLAB to NumPy.
        f1_old = f1.copy()
        f2_old = f2.copy()
        f1[1:] = f1_old[:-1]
        f2[:-1] = f2_old[1:]

        # Boundary conditions.
        # Upstream: constant reservoir head H = Hr.
        f1[0] = hr_lattice + f2[0]

        # Downstream: instant valve closure, i.e. velocity v = 0.
        f2[-1] = -f1[-1] - f0[-1]

        # Reconstruct macroscopic variables.
        vv[n + 1, :] = f0 + f1 + f2
        hh[n + 1, :] = f1 - f2
        time_hist[n + 1] = time_hist[n] + dt

    # Convert back to physical units.
    h = hh * lambda_h
    v = vv * lambda_u

    elapsed = time.perf_counter() - t0

    return {
        'time': time_hist,
        'head': h,
        'velocity': v,
        'head_lattice': hh,
        'velocity_lattice': vv,
        'dx': dx,
        'dt': dt,
        'nodes': nodes,
        'timesteps': timesteps,
        'elapsed_seconds': elapsed,
    }


if __name__ == '__main__':
    results = simulate_lbm_instant_valve_closure()

    time_hist = results['time']
    h = results['head']

    # Plot the head at the valve node.
    plt.figure(figsize=(8, 4.8))
    plt.plot(time_hist, h[:, -1])
    plt.title('LBM valve head history')
    plt.xlabel('Time [s]')
    plt.ylabel('Head [m]')
    plt.tight_layout()

    output_path = Path("png") / "lbm_valve_head.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"dx = {results['dx']:.6f} m")
    print(f"dt = {results['dt']:.6f} s")
    print(f"nodes = {results['nodes']}")
    print(f"timesteps = {results['timesteps']}")
    print(f"elapsed = {results['elapsed_seconds']:.6f} s")
    print(f"figure saved to: {output_path}")