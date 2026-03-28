
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MOC import moc_solve


def moc_reference():
    ref = moc_solve(N=25)
    meta = {k: ref[k] for k in ["L", "Hr", "N", "NS", "e", "D", "K", "rho", "E", "g", "f", "A", "dx", "t_max", "a", "dt", "B", "R", "u0"]}
    return ref["time"], ref["H"], ref["Q"], meta


def flux(U, c_hq, c_qh):
    H = U[0]
    Q = U[1]
    return np.array([c_hq * Q, c_qh * H], dtype=float)


def exact_linear_flux(U_left, U_right, a, c_hq, c_qh):
    return 0.5 * (flux(U_left, c_hq, c_qh) + flux(U_right, c_hq, c_qh)) - 0.5 * a * (U_right - U_left)


def friction_source(U, f, D, A):
    H, Q = U
    return np.array([0.0, -f * Q * abs(Q) / (2.0 * D * A)], dtype=float)


def run_fvm_aligned():
    time_ref, H_ref, Q_ref, meta = moc_reference()
    L = meta["L"]
    Hr = meta["Hr"]
    N = meta["N"]
    NS = meta["NS"]
    D = meta["D"]
    g = meta["g"]
    f = meta["f"]
    A = meta["A"]
    dx = meta["dx"]
    t_max = meta["t_max"]
    a = meta["a"]
    dt = meta["dt"]
    u0 = meta["u0"]

    c_hq = a**2 / (g * A)
    c_qh = g * A

    max_steps = int(np.ceil(t_max / dt)) + 1
    time_hist = np.zeros(max_steps)
    H = np.zeros((max_steps, NS))
    Q = np.zeros((max_steps, NS))

    x = np.arange(NS) * dx
    H[0, :] = Hr - f * x * u0**2 / (2.0 * g * D)
    H[0, 0] = Hr
    Q[0, :] = A * u0

    last = 0
    for n in range(max_steps - 1):
        H_old = H[n].copy()
        Q_old = Q[n].copy()
        U_old = np.vstack([H_old, Q_old])

        U_new = U_old.copy()

        for j in range(1, N):
            U_jm1 = U_old[:, j - 1]
            U_j = U_old[:, j]
            U_jp1 = U_old[:, j + 1]

            F_l = exact_linear_flux(U_jm1, U_j, a, c_hq, c_qh)
            F_r = exact_linear_flux(U_j, U_jp1, a, c_hq, c_qh)

            U_star = U_j - (dt / dx) * (F_r - F_l)
            U_new[:, j] = U_star + dt * friction_source(U_star, f, D, A)

        B = a / (g * A)
        R = f * dx / (2.0 * g * D * A**2)

        CM = H_old[1] - B * Q_old[1] + R * Q_old[1] * abs(Q_old[1])
        U_new[0, 0] = Hr
        U_new[1, 0] = (Hr - CM) / B

        CP = H_old[N - 1] + B * Q_old[N - 1] - R * Q_old[N - 1] * abs(Q_old[N - 1])
        U_new[1, N] = 0.0
        U_new[0, N] = CP

        H[n + 1] = U_new[0]
        Q[n + 1] = U_new[1]
        time_hist[n + 1] = time_hist[n] + dt
        last = n + 1
        if time_hist[n + 1] >= t_max:
            break

    meta_out = dict(meta)
    meta_out.update({"c_hq": c_hq, "c_qh": c_qh})
    return time_hist[: last + 1], H[: last + 1], Q[: last + 1], meta_out, time_ref, H_ref, Q_ref


def main():
    start = time.perf_counter()
    time_fvm, H_fvm, Q_fvm, meta, time_moc, H_moc, Q_moc = run_fvm_aligned()
    elapsed = time.perf_counter() - start

    common_time = time_moc if len(time_moc) <= len(time_fvm) else time_fvm
    valve_moc = np.interp(common_time, time_moc, H_moc[:, -1])
    valve_fvm = np.interp(common_time, time_fvm, H_fvm[:, -1])

    rel_l2 = np.linalg.norm(valve_fvm - valve_moc) / np.linalg.norm(valve_moc)
    max_abs = np.max(np.abs(valve_fvm - valve_moc))

    output_dir = Path("png")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "fvm_vs_moc_valve_head.png"

    plt.figure(figsize=(8, 4.8))
    plt.plot(time_moc, H_moc[:, -1], label="MOC valve head", linewidth=2.0)
    plt.plot(time_fvm, H_fvm[:, -1], "--", label="Aligned FVM valve head", linewidth=1.8)
    plt.title("FVM Aligned with MOC: Valve Head Comparison")
    plt.xlabel("Time [s]")
    plt.ylabel("Head [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Wave speed a = {meta['a']:.6f} m/s")
    print(f"dx = {meta['dx']:.6f} m")
    print(f"dt = {meta['dt']:.6e} s")
    print(f"Saved time steps (FVM) = {len(time_fvm)}")
    print(f"Saved time steps (MOC) = {len(time_moc)}")
    print(f"Valve-head relative L2 error = {rel_l2:.6e}")
    print(f"Valve-head max absolute error = {max_abs:.6e} m")
    print(f"Elapsed time = {elapsed:.6f} s")
    print(f"Figure saved to: {figure_path}")


if __name__ == "__main__":
    main()