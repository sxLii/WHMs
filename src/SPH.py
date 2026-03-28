
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from MOC import moc_solve


def cubic_b_spline_w(r: float, h: float) -> float:
    """Return the 1D cubic B-spline kernel value."""
    q = abs(r) / h
    alpha = 1.0 / h
    if 0.0 <= q < 1.0:
        return alpha * (2.0 / 3.0 - q * q + 0.5 * q**3)
    if 1.0 <= q < 2.0:
        return alpha * ((2.0 - q) ** 3) / 6.0
    return 0.0


def interpolate_shifted_characteristic(
    x_target: float,
    x_particles: np.ndarray,
    values: np.ndarray,
    shift: float,
    smoothing_length: float,
) -> float:

    weights = np.array(
        [cubic_b_spline_w(x_target - (xp + shift), smoothing_length) for xp in x_particles],
        dtype=float,
    )
    weight_sum = np.sum(weights)

    if weight_sum <= 0.0:
        nearest = np.argmin(np.abs(x_target - (x_particles + shift)))
        return float(values[nearest])

    return float(np.dot(weights, values) / weight_sum)


def run_moc_reference() -> dict:
    """Run the MOC reference solution used for alignment."""
    return moc_solve(N=25)


def run_sph_aligned_with_moc() -> dict:
    ref = run_moc_reference()

    time = ref["time"]
    x = ref["x"]
    N = ref["N"]
    NS = ref["NS"]
    dx = ref["dx"]
    Hr = ref["Hr"]
    B = ref["B"]
    R = ref["R"]

    n_steps = len(time)

    H = np.zeros((n_steps, NS), dtype=float)
    Q = np.zeros((n_steps, NS), dtype=float)

    # Match the MOC initial condition exactly.
    H[0, :] = ref["H"][0, :]
    Q[0, :] = ref["Q"][0, :]

    smoothing_length = 0.5 * dx

    for n in range(n_steps - 1):
        c_plus = H[n, :] + B * Q[n, :] - R * Q[n, :] * np.abs(Q[n, :])
        c_minus = H[n, :] - B * Q[n, :] + R * Q[n, :] * np.abs(Q[n, :])

        # Interior nodes.
        for j in range(1, N):
            cp = interpolate_shifted_characteristic(
                x_target=x[j],
                x_particles=x,
                values=c_plus,
                shift=dx,
                smoothing_length=smoothing_length,
            )
            cm = interpolate_shifted_characteristic(
                x_target=x[j],
                x_particles=x,
                values=c_minus,
                shift=-dx,
                smoothing_length=smoothing_length,
            )

            H[n + 1, j] = 0.5 * (cp + cm)
            Q[n + 1, j] = (H[n + 1, j] - cm) / B

        # Use the same MOC-consistent boundary semantics.
        cm_up = interpolate_shifted_characteristic(
            x_target=x[0],
            x_particles=x,
            values=c_minus,
            shift=-dx,
            smoothing_length=smoothing_length,
        )
        H[n + 1, 0] = Hr
        Q[n + 1, 0] = (H[n + 1, 0] - cm_up) / B

        cp_down = interpolate_shifted_characteristic(
            x_target=x[-1],
            x_particles=x,
            values=c_plus,
            shift=dx,
            smoothing_length=smoothing_length,
        )
        Q[n + 1, -1] = 0.0
        H[n + 1, -1] = cp_down

    valve_head_error = H[:, -1] - ref["H"][:, -1]
    relative_l2 = np.linalg.norm(valve_head_error) / max(np.linalg.norm(ref["H"][:, -1]), 1e-14)
    max_abs_error = np.max(np.abs(valve_head_error))

    return {
        "time": time,
        "x": x,
        "H": H,
        "Q": Q,
        "moc_H": ref["H"],
        "moc_Q": ref["Q"],
        "dx": dx,
        "dt": ref["dt"],
        "a": ref["a"],
        "relative_l2_valve_head": float(relative_l2),
        "max_abs_valve_head_error": float(max_abs_error),
        "smoothing_length": smoothing_length,
    }


def main() -> None:
    results = run_sph_aligned_with_moc()

    output_dir = Path("png")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "sph_vs_moc_valve_head.png"

    plt.figure(figsize=(8, 4.8))
    plt.plot(results["time"], results["moc_H"][:, -1], label="MOC")
    plt.plot(results["time"], results["H"][:, -1], "--", label="SPH aligned")
    plt.title("SPH vs MOC: Valve Head History")
    plt.xlabel("Time (s)")
    plt.ylabel("Head (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Wave speed a = {results['a']:.6f} m/s")
    print(f"Time step dt = {results['dt']:.6e} s")
    print(f"Smoothing length = {results['smoothing_length']:.6f} m")
    print(f"Relative L2 error at valve = {results['relative_l2_valve_head']:.6e}")
    print(f"Max absolute valve-head error = {results['max_abs_valve_head_error']:.6e} m")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()