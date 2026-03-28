# WHMs — Water Hammer Methods

Numerical simulation of water hammer (instantaneous valve closure) using five different methods.
All solvers share the same physical setup: pipe length L=300 m, upstream reservoir head Hr=70 m,
wave speed derived from elastic pipe/fluid properties, initial steady velocity u0=0.1 m/s.

## Methods

| File | Method | Status | Nodes |
|------|--------|--------|-------|
| [src/MOC.py](src/MOC.py) | Method of Characteristics | works | N=100 |
| [src/FVM.py](src/FVM.py) | Finite Volume Method | works | N=25 |
| [src/LBM.py](src/LBM.py) | Lattice Boltzmann Method | numerical oscillation | N=25 |
| [src/SPH.py](src/SPH.py) | Smoothed Particle Hydrodynamics | works (aligned with MOC) | N=25 |
| [src/PINN.py](src/PINN.py) | Physics-Informed Neural Network | PyTorch works | — |

## Method Details

### [MOC.py](src/MOC.py) — Method of Characteristics
Standard MOC solver with N=100 reaches (101 nodes). Computes characteristic variables
CP/CM at each interior node, with fixed-head upstream and closed-valve downstream boundary
conditions. Outputs valve head vs time to `png/MOC_valve_head.png`.

### [FVM.py](src/FVM.py) — Finite Volume Method
Godunov-type FVM using the exact Roe/upwind flux for the linearised water-hammer system
`[H, Q]`. Aligned with the MOC grid (N=25) and time step (dt = dx/a). Uses operator
splitting for the Darcy-Weisbach friction source. Includes an embedded MOC reference solver
for error comparison. Outputs `png/fvm_valve_head.png`.

### [LBM.py](src/LBM.py) — Lattice Boltzmann Method
Characteristic-aligned LBM where the two populations correspond to half-characteristic
variables `f± = 0.5*(v ± h)`. Streaming propagates right/left-going wave families exactly
one grid step per time step. Friction is applied by operator splitting on the velocity
field. Boundary conditions mirror MOC wave relations. Shows numerical oscillation vs MOC.
Outputs `png/lbm_vs_moc_tuned_valve_head.png`.

### [SPH.py](src/SPH.py) — Smoothed Particle Hydrodynamics
SPH-style characteristic solver: carries the two MOC characteristic quantities on fixed
particles and reconstructs them after one travel step using a cubic B-spline kernel
(smoothing length = 0.5·dx). Boundary conditions follow MOC wave semantics. Outputs
`png/sph_vs_moc_valve_head.png`.

### [PINN.py](src/PINN.py) — Physics-Informed Neural Network
PyTorch PINN (8 hidden layers × 20 neurons, tanh activation) trained on head observations
from `data/case0.csv` (51 nodes). Loss = data MSE + weighted PDE residual from the
water-hammer equations. Inputs are normalised (x, t); outputs are denormalised (H, Q).
Requires `data/case0.csv` with alternating pressure/flow columns (N1=H₀, N2=Q₀, …).

## Benchmark Results

All runs on the same machine (wave speed a = 1435.94 m/s).

| Method | Nodes | dt (s) | Steps | Valve L2 error | Valve max abs error | Runtime |
|--------|-------|--------|-------|---------------|---------------------|---------|
| MOC | 101 (N=100) | 2.089e-03 | 9574 | — (reference) | — | 0.878 s |
| FVM | 26 (N=25) | 8.357e-03 | 2395 | 1.853e-04 | 3.60e-02 m | 0.513 s |
| LBM | 26 (N=25) | 8.357e-03 | — | 1.968e-03 | 1.75e-01 m | 0.231 s |
| SPH | 26 (N=25) | 8.357e-03 | — | 8.29e-15 | 1.75e-12 m | — |

> SPH near-zero error is expected: with smoothing length = 0.5·dx the kernel collapses to
> the nearest-neighbour interpolation, making SPH identical to MOC in this configuration.

## Reference Data

`data/case0.csv` — ground-truth simulation FOR **PINN**:
- `No, T, N1, N2, …, N102` where N(2i-1) = pressure head and N(2i) = flow at node i,
for i = 1…51. Time range: 0.01 s to 11.0 s (1100 rows).
- It is from one paper's repository [Water Research](https://github.com/EthanYe/PINN-for-Pipeline-Systems).
