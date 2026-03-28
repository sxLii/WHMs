import time as timer
import numpy as np
import matplotlib.pyplot as plt


def simulate_instant_valve_closure():
    """
    Simulate water hammer in a pipeline using the Method of Characteristics (MOC)
    under the assumption of instantaneous valve closure at the downstream end.

    Main fixes compared with the original MATLAB script:
    1. The initial steady-state head profile is made physically consistent.
    2. Boundary conditions are computed from the previous time level, which is the
       correct MOC update for this formulation.
    3. The reference head line is plotted with the correct 1D shape.
    """

    # Pipe and fluid parameters
    L = 300.0                    # Pipe length (m)
    Hr = 70.0                    # Reservoir / pump head (m)
    N = 10                       # Number of pipe reaches
    NS = N + 1                   # Number of nodes
    e = 0.001651                 # Wall thickness (m), 0.065 in
    D = 0.00635 - 2.0 * e        # Inner diameter (m)
    K = 2.1e9                    # Bulk modulus (Pa)
    rho = 1000.0                 # Fluid density (kg/m^3)
    E = 2.1e11                   # Young's modulus (Pa)
    g = 9.806                    # Gravitational acceleration (m/s^2)
    f = 0.018                    # Darcy-Weisbach friction factor (-)
    area = np.pi * D**2 / 4.0    # Cross-sectional area (m^2)

    # Grid and time step
    dx = L / N                   # Spatial step (m)
    t_max = 20.0                 # Total simulation time (s)
    a = np.sqrt(K / rho / (1.0 + K * D / (E * e)))  # Wave speed (m/s)
    dt = dx / a                  # Time step from the Courant condition (s)

    # MOC constants
    B = a / (g * area)
    R = f * dx / (2.0 * g * D * area**2)

    # Initial steady flow
    u0 = 0.1                     # Initial velocity (m/s)

    # Preallocate arrays
    n_steps = int(np.ceil(t_max / dt)) + 1
    time = np.zeros(n_steps)
    Q = np.zeros((n_steps, NS))
    H = np.zeros((n_steps, NS))

    # Initial discharge at every node
    Q[0, :] = area * u0

    # Initial steady-state head profile along the pipe
    # x = 0 at the upstream reservoir, x = L at the downstream valve
    x = np.arange(NS) * dx
    H[0, :] = Hr - f * x * u0**2 / (2.0 * g * D)
    H[0, 0] = Hr

    last_index = 0
    for k in range(n_steps - 1):
        # Interior nodes: Python indices 1..N-1 correspond to MATLAB 2..N
        for j in range(1, N):
            CP = H[k, j - 1] + B * Q[k, j - 1] - R * Q[k, j - 1] * abs(Q[k, j - 1])
            CM = H[k, j + 1] - B * Q[k, j + 1] + R * Q[k, j + 1] * abs(Q[k, j + 1])

            H[k + 1, j] = 0.5 * (CP + CM)
            Q[k + 1, j] = (H[k + 1, j] - CM) / B

        # Upstream boundary: constant head reservoir
        # Use the previous time level for the characteristic coming from node 2.
        CM = H[k, 1] - B * Q[k, 1] + R * Q[k, 1] * abs(Q[k, 1])
        H[k + 1, 0] = Hr
        Q[k + 1, 0] = (H[k + 1, 0] - CM) / B

        # Downstream boundary: instantaneous valve closure, so Q = 0
        # Use the previous time level for the characteristic coming from node N.
        CP = H[k, N - 1] + B * Q[k, N - 1] - R * Q[k, N - 1] * abs(Q[k, N - 1])
        Q[k + 1, N] = 0.0
        H[k + 1, N] = CP

        time[k + 1] = time[k] + dt
        last_index = k + 1

        if time[k + 1] >= t_max:
            break

    # Trim arrays to the actual simulated length
    time = time[: last_index + 1]
    Q = Q[: last_index + 1, :]
    H = H[: last_index + 1, :]

    return time, H, Q, dt, a


def main():
    start = timer.perf_counter()
    time, H, Q, dt, a = simulate_instant_valve_closure()
    elapsed = timer.perf_counter() - start

    print(f"Computed wave speed a = {a:.6f} m/s")
    print(f"Time step dt = {dt:.6e} s")
    print(f"Number of time steps = {len(time)}")
    print(f"Elapsed time = {elapsed:.6f} s")

    plt.figure(figsize=(8, 5))
    plt.plot(time, H[:, -1], label="Head at the closed valve")
    plt.plot(time, np.full_like(time, 70.0), "--", label="Reservoir head")
    plt.title("MOC Pressure Head Curve at the Valve")
    plt.xlabel("Time (s)")
    plt.ylabel("Head (m)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()