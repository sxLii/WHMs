import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def physical_flux(state: np.ndarray, average_velocity: float, g: float, c: float) -> np.ndarray:
    """
    Compute the physical flux for the 1D water-hammer system.

    Parameters
    ----------
    state : np.ndarray
        State vector [head, velocity].
    average_velocity : float
        Arithmetic mean of the left and right velocities.
    g : float
        Gravitational acceleration.
    c : float
        Wave speed.

    Returns
    -------
    np.ndarray
        Flux vector.
    """
    head, velocity = state
    return np.array(
        [
            average_velocity * head + (c**2 / g) * velocity,
            g * head + average_velocity * velocity,
        ],
        dtype=float,
    )



def godunov_flux(right_state: np.ndarray, left_state: np.ndarray, g: float, c: float) -> np.ndarray:
    """
    Compute the Godunov-type numerical flux.

    Parameters
    ----------
    right_state : np.ndarray
        State on the right side of the interface [head, velocity].
    left_state : np.ndarray
        State on the left side of the interface [head, velocity].
    g : float
        Gravitational acceleration.
    c : float
        Wave speed.

    Returns
    -------
    np.ndarray
        Numerical flux vector.
    """
    delta_head = right_state[0] - left_state[0]
    delta_velocity = right_state[1] - left_state[1]
    average_velocity = 0.5 * (right_state[1] + left_state[1])

    c_minus = (
        0.5
        * (delta_head - (c / g) * delta_velocity)
        * abs(average_velocity - c)
        * np.array([1.0, -g / c], dtype=float)
    )
    c_plus = (
        0.5
        * (delta_head + (c / g) * delta_velocity)
        * abs(average_velocity + c)
        * np.array([1.0, g / c], dtype=float)
    )

    return 0.5 * (
        physical_flux(right_state, average_velocity, g, c)
        + physical_flux(left_state, average_velocity, g, c)
    ) - 0.5 * (c_plus + c_minus)



def source_term(state: np.ndarray, g: float, f: float, diameter: float) -> np.ndarray:
    """
    Compute the source term.

    The friction term is written as u * |u| so that friction always opposes
    the flow direction.

    Parameters
    ----------
    state : np.ndarray
        State vector [head, velocity].
    g : float
        Gravitational acceleration.
    f : float
        Darcy-Weisbach friction factor.
    diameter : float
        Inner pipe diameter.

    Returns
    -------
    np.ndarray
        Source vector.
    """
    velocity = state[1]
    return np.array([0.0, -g * f * velocity * abs(velocity) / (2.0 * diameter)], dtype=float)



def run_simulation() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate instantaneous valve closure using a finite-volume scheme.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        time, head, velocity histories.
    """
    # Physical parameters
    pipe_length = 300.0
    reservoir_head = 70.0
    num_cells = 25
    num_nodes = num_cells + 1
    wall_thickness = 0.001651
    diameter = 0.00635 - 2.0 * wall_thickness
    bulk_modulus = 2.1e9
    density = 1000.0
    young_modulus = 2.1e11
    g = 9.806
    friction_factor = 0.018

    # Mesh and time settings
    cfl = 0.5
    dx = pipe_length / num_cells
    t_max = 20.0
    wave_speed = np.sqrt(bulk_modulus / density / (1.0 + bulk_modulus * diameter / (young_modulus * wall_thickness)))
    dt = cfl * dx / wave_speed
    initial_velocity = 0.1

    x = np.linspace(0.0, pipe_length, num_nodes)

    # The initial head should decrease linearly along the pipe according to the
    # steady Darcy-Weisbach head-loss relation.
    initial_head = reservoir_head - friction_factor * x * initial_velocity**2 / (2.0 * g * diameter)
    initial_head[0] = reservoir_head

    # Preallocate history arrays.
    max_steps = int(np.ceil(t_max / dt)) + 1
    time_hist = np.zeros(max_steps, dtype=float)
    head_hist = np.zeros((max_steps, num_nodes), dtype=float)
    vel_hist = np.zeros((max_steps, num_nodes), dtype=float)

    head_hist[0, :] = initial_head
    vel_hist[0, :] = initial_velocity

    step = 0
    t = 0.0

    while t < t_max - 1e-14:
        dt_step = min(dt, t_max - t)
        lam = dt_step / dx

        head_old = head_hist[step, :]
        vel_old = vel_hist[step, :]
        head_new = head_old.copy()
        vel_new = vel_old.copy()

        # Update interior nodes.
        for j in range(1, num_cells):
            state_center = np.array([head_old[j], vel_old[j]], dtype=float)
            state_left = np.array([head_old[j - 1], vel_old[j - 1]], dtype=float)
            state_right = np.array([head_old[j + 1], vel_old[j + 1]], dtype=float)

            updated_state = (
                state_center
                + dt_step * source_term(state_center, g, friction_factor, diameter)
                - lam
                * (
                    godunov_flux(state_right, state_center, g, wave_speed)
                    - godunov_flux(state_center, state_left, g, wave_speed)
                )
            )

            head_new[j] = updated_state[0]
            vel_new[j] = updated_state[1]

        # Upstream boundary: fixed reservoir head.
        head_new[0] = reservoir_head
        vel_new[0] = vel_new[1] + (g / wave_speed) * (head_new[0] - head_new[1])

        # Downstream boundary: instantaneous valve closure.
        vel_new[-1] = 0.0
        head_new[-1] = head_new[-2] + (wave_speed / g) * (vel_new[-2] - vel_new[-1])

        step += 1
        t += dt_step
        time_hist[step] = t
        head_hist[step, :] = head_new
        vel_hist[step, :] = vel_new

    return time_hist[: step + 1], head_hist[: step + 1, :], vel_hist[: step + 1, :]



def main() -> None:
    start = time.perf_counter()
    time_hist, head_hist, vel_hist = run_simulation()
    elapsed = time.perf_counter() - start

    output_dir = Path("/mnt/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "fvm_valve_head.png"

    plt.figure(figsize=(8, 4.8))
    plt.plot(time_hist, head_hist[:, -1], label="Valve head")
    plt.title("FVM Pressure-Head Curve at the Valve")
    plt.xlabel("Time (s)")
    plt.ylabel("Head (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Simulation finished in {elapsed:.3f} s")
    print(f"Number of saved time steps: {len(time_hist)}")
    print(f"Final valve head: {head_hist[-1, -1]:.6f} m")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()