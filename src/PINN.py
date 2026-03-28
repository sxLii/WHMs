import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PhysParams:
    g: float = 9.81
    D: float = 1.81
    friction: float = 0.012
    wave_speed: float = 1000.0

    @property
    def area(self) -> float:
        return math.pi * self.D * self.D / 4.0


@dataclass
class NormStats:
    h_min: float
    h_max: float
    q_min: float
    q_max: float
    x_max: float
    t_max: float

    @staticmethod
    def _safe_span(vmin: float, vmax: float) -> float:
        span = vmax - vmin
        return span if abs(span) > 1e-12 else 1.0

    def norm_h(self, h: np.ndarray) -> np.ndarray:
        return 2.0 * (h - self.h_min) / self._safe_span(self.h_min, self.h_max) - 1.0

    def denorm_h(self, h_hat: torch.Tensor) -> torch.Tensor:
        return (h_hat + 1.0) * self._safe_span(self.h_min, self.h_max) / 2.0 + self.h_min

    def denorm_q(self, q_hat: torch.Tensor) -> torch.Tensor:
        return (q_hat + 1.0) * self._safe_span(self.q_min, self.q_max) / 2.0 + self.q_min

    def norm_x(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x / max(self.x_max, 1e-12) - 1.0

    def norm_t(self, t: torch.Tensor) -> torch.Tensor:
        return 2.0 * t / max(self.t_max, 1e-12) - 1.0


class PINN(nn.Module):
    def __init__(self, hidden_width: int = 20, hidden_depth: int = 8):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(hidden_depth):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.Tanh())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, stats: NormStats) -> torch.Tensor:
        xt = torch.cat([stats.norm_x(x), stats.norm_t(t)], dim=1)
        return self.net(xt)


class WaterHammerPINN:
    def __init__(self, model: PINN, stats: NormStats, phys: PhysParams):
        self.model = model
        self.stats = stats
        self.phys = phys
        self.mse = nn.MSELoss()

    def predict_hq(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x, t, self.stats)
        h = self.stats.denorm_h(out[:, 0:1])
        q = self.stats.denorm_q(out[:, 1:2])
        return h, q

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, q = self.predict_hq(x, t)

        h_x = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
        h_t = torch.autograd.grad(h.sum(), t, create_graph=True)[0]
        q_x = torch.autograd.grad(q.sum(), x, create_graph=True)[0]
        q_t = torch.autograd.grad(q.sum(), t, create_graph=True)[0]

        A = self.phys.area
        g = self.phys.g
        D = self.phys.D
        f = self.phys.friction
        a = self.phys.wave_speed

        f_res = A * q_t + q * q_x + g * A * A * h_x + f * q * torch.abs(q) / (2.0 * D)
        g_res = A * h_t + q * h_x + (a * a / g) * q_x
        return f_res, g_res

    def data_loss(self, x: torch.Tensor, t: torch.Tensor, h_obs_norm: torch.Tensor) -> torch.Tensor:
        pred = self.model(x, t, self.stats)
        return self.mse(pred[:, 0:1], h_obs_norm)

    def physics_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        f_res, g_res = self.pde_residual(x, t)
        zeros = torch.zeros_like(f_res)
        return self.mse(f_res, zeros) + self.mse(g_res, zeros)


def load_case(csv_path: str | Path, reaches: int = 51, downsample: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, NormStats]:
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[::downsample, :].reset_index(drop=True)

    t_space = df["T"].to_numpy(dtype=np.float32)
    length = 500.0
    x_space = np.linspace(0.0, length, reaches, dtype=np.float32)

    p_all = np.stack([df.iloc[:, 2 * i + 2].to_numpy(dtype=np.float32) for i in range(reaches)], axis=0)
    q_all = np.stack([df.iloc[:, 2 * i + 3].to_numpy(dtype=np.float32) for i in range(reaches)], axis=0)

    x_all = np.tile(x_space[:, None], (1, len(t_space)))
    t_all = np.tile(t_space[None, :], (reaches, 1))

    stats = NormStats(
        h_min=float(p_all.min()),
        h_max=float(p_all.max()),
        q_min=float(q_all.min()),
        q_max=float(q_all.max()),
        x_max=float(length),
        t_max=float(t_space.max()),
    )
    return x_all, t_all, p_all, q_all, stats


def build_observation_set(
    x_all: np.ndarray,
    t_all: np.ndarray,
    h_all: np.ndarray,
    stats: NormStats,
    node_idx: list[int],
    tlen: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x_all[node_idx, :tlen].reshape(-1, 1)
    t = t_all[node_idx, :tlen].reshape(-1, 1)
    h = h_all[node_idx, :tlen].reshape(-1, 1)
    h_norm = stats.norm_h(h.astype(np.float32))

    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE, requires_grad=True)
    t_t = torch.tensor(t, dtype=torch.float32, device=DEVICE, requires_grad=True)
    h_t = torch.tensor(h_norm, dtype=torch.float32, device=DEVICE)
    return x_t, t_t, h_t


def build_test_set(
    x_all: np.ndarray,
    t_all: np.ndarray,
    h_all: np.ndarray,
    q_all: np.ndarray,
    node_idx: int,
    tlen: int,
) -> dict[str, np.ndarray | torch.Tensor]:
    x = x_all[node_idx, :tlen].reshape(-1, 1)
    t = t_all[node_idx, :tlen].reshape(-1, 1)
    h = h_all[node_idx, :tlen].reshape(-1, 1)
    q = q_all[node_idx, :tlen].reshape(-1, 1)

    return {
        "x": x,
        "t": t,
        "h": h,
        "q": q,
        "x_tensor": torch.tensor(x, dtype=torch.float32, device=DEVICE, requires_grad=True),
        "t_tensor": torch.tensor(t, dtype=torch.float32, device=DEVICE, requires_grad=True),
    }


def build_collocation_set(
    x_all: np.ndarray,
    t_all: np.ndarray,
    reaches: int,
    tlen: int,
    n_collo: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = np.linspace(0, reaches - 1, n_collo, dtype=int)
    x = x_all[idx, :tlen].reshape(-1, 1)
    t = t_all[idx, :tlen].reshape(-1, 1)
    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE, requires_grad=True)
    t_t = torch.tensor(t, dtype=torch.float32, device=DEVICE, requires_grad=True)
    return x_t, t_t


def train(
    solver: WaterHammerPINN,
    x_obs: torch.Tensor,
    t_obs: torch.Tensor,
    h_obs_norm: torch.Tensor,
    x_collo: torch.Tensor,
    t_collo: torch.Tensor,
    epochs: int = 20000,
    lr: float = 1e-3,
    physics_weight: float = 0.01,
    log_every: int = 500,
) -> dict[str, list[float]]:
    optimizer = torch.optim.Adam(solver.model.parameters(), lr=lr)
    history = {"total": [], "data": [], "physics": []}

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        loss_data = solver.data_loss(x_obs, t_obs, h_obs_norm)
        loss_phys = solver.physics_loss(x_collo, t_collo)
        loss = loss_data + physics_weight * loss_phys

        loss.backward()
        optimizer.step()

        history["total"].append(loss.item())
        history["data"].append(loss_data.item())
        history["physics"].append((physics_weight * loss_phys).item())

        if epoch % log_every == 0 or epoch == 1:
            print(
                f"epoch={epoch:6d} | total={loss.item():.6e} | "
                f"data={loss_data.item():.6e} | physics={(physics_weight * loss_phys).item():.6e}"
            )

    return history


def plot_results(
    solver: WaterHammerPINN,
    test_data: dict[str, np.ndarray | torch.Tensor],
    history: dict[str, list[float]],
    out_prefix: str = "pinn_water_hammer",
) -> None:
    with torch.no_grad():
        pred_norm = solver.model(test_data["x_tensor"], test_data["t_tensor"], solver.stats)
        pred_h = solver.stats.denorm_h(pred_norm[:, 0:1]).cpu().numpy()

    true_h = np.asarray(test_data["h"])
    time = np.asarray(test_data["t"])

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time, true_h, label="Observed head")
    plt.plot(time, pred_h, label="Predicted head")
    plt.xlabel("Time")
    plt.ylabel("Head")
    plt.legend()
    plt.title("PINN prediction")

    plt.subplot(1, 2, 2)
    plt.plot(history["total"], label="Total loss")
    plt.plot(history["data"], label="Data loss")
    plt.plot(history["physics"], label="Physics loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training history")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=200)
    plt.show()


def main() -> None:
    csv_path = Path("../data/case0.csv")
    reaches = 51
    tlen = 199
    obs_nodes = [5, 20, 50]
    test_node = 40
    n_collo = 20

    x_all, t_all, h_all, q_all, stats = load_case(csv_path, reaches=reaches, downsample=5)

    model = PINN(hidden_width=20, hidden_depth=8).to(DEVICE)
    solver = WaterHammerPINN(model=model, stats=stats, phys=PhysParams())

    x_obs, t_obs, h_obs_norm = build_observation_set(x_all, t_all, h_all, stats, obs_nodes, tlen)
    x_collo, t_collo = build_collocation_set(x_all, t_all, reaches, tlen, n_collo)
    test_data = build_test_set(x_all, t_all, h_all, q_all, test_node, tlen)

    history = train(
        solver,
        x_obs,
        t_obs,
        h_obs_norm,
        x_collo,
        t_collo,
        epochs=20000,
        lr=1e-3,
        physics_weight=0.01,
        log_every=500,
    )

    plot_results(solver, test_data, history)


if __name__ == "__main__":
    main()