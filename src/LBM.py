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
    # Physical parameters
    # ------------------------------------------------------------------
    L = 500.0          # Pipe length [m]
    f = 0.02           # Darcy-Weisbach friction factor [-]
    a = 1000.0         # Pressure wave speed [m/s]
    Hr = 400.0         # Upstream reservoir head [m]
    V0 = 3.25996       # Initial flow velocity [m/s]
    D = 2.0            # Pipe diameter [m]
    g = 9.81           # Gravity acceleration [m/s^2]

    # ------------------------------------------------------------------
    # Spatial and temporal discretization
    # ------------------------------------------------------------------
    nodes = 200
    timesteps = 400

    dx = L / nodes                 # Physical cell length [m]
    dt = dx / a                    # Physical time step [s], consistent with lattice mapping

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

    output_path = Path(__file__).with_name('lbm_valve_head.png')
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"dx = {results['dx']:.6f} m")
    print(f"dt = {results['dt']:.6f} s")
    print(f"nodes = {results['nodes']}")
    print(f"timesteps = {results['timesteps']}")
    print(f"elapsed = {results['elapsed_seconds']:.6f} s")
    print(f"figure saved to: {output_path}")