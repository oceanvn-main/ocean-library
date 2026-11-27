# Lorenz Chaotic Butterfly Attractor

## Overview

`exp_lorenz_butterfly` visualizes the **Edward Lorenz (1963) chaotic strange attractor** in 3D using the Ocean Library visualization engine. It demonstrates three core ideas from nonlinear dynamics:

1. **Numerical integration** of the Lorenz ODE system using a hand-rolled 4th-order Runge-Kutta (RK4) solver — no `ode45` black-box.
2. **Phase portrait** of the strange attractor colored by time to reveal the characteristic double-wing butterfly shape.
3. **Sensitivity to initial conditions** (the Butterfly Effect): two trajectories separated by epsilon = 1e-8 diverge exponentially, with an estimated Lyapunov exponent.

## The Lorenz System

The Lorenz equations model thermal convection in the atmosphere:

```
dx/dt = sigma * (y - x)
dy/dt = x * (rho - z) - y
dz/dt = x * y - beta * z
```

**Canonical parameters** (Lorenz 1963, chaotic regime):

| Parameter | Symbol | Value  | Physical meaning             |
|-----------|--------|--------|------------------------------|
| sigma     | sigma  | 10     | Prandtl number               |
| rho       | rho    | 28     | Rayleigh number (buoyancy)   |
| beta      | beta   | 8/3    | Geometric aspect ratio       |

The system transitions to chaos when `rho > rho_crit` (Hopf bifurcation), where:

```
rho_crit = sigma * (sigma + beta + 3) / (sigma - beta - 1)  approx  24.74
```

## Usage

```matlab
% Run from the workspace root (Ocean Library)
run('Experiments/Scripts/Simulation/exp_lorenz_butterfly.m');
```

No inputs are required. All parameters are set internally.

## Outputs

| File | Description |
|------|-------------|
| `Results/lorenz_butterfly_3d.png` | 3D phase portrait, color-coded by time (parula colormap) |
| `Results/lorenz_butterfly_effect.png` | 2-panel: dual trajectory + exponential divergence log-plot |

Console output includes:
- Parameter validation and chaos threshold check
- Integration ranges and bounded attractor verification
- Estimated Lyapunov Maximum Lyapunov Exponent (MLE) vs. theoretical ~0.905 nats/s

## Examples

```matlab
% Customize parameters before running
sigma = 10; rho = 28; beta = 8/3;   % Lorenz 1963 (default)

% Steady-state spirals (non-chaotic, rho < rho_crit)
sigma = 10; rho = 15; beta = 8/3;

% Near onset of chaos
sigma = 10; rho = 24; beta = 8/3;
```

## Design Decisions

- **RK4 from scratch**: Follows Ocean Library research philosophy — explicit steps, not hidden ode45.
- **Colored segments**: Trajectory is split into 200-step segments and drawn with per-segment colors interpolated from `parula`, avoiding a slow `scatter3` over 10 000 points.
- **Lyapunov estimate**: Fits `log(||delta_x(t)||) = lambda * t + c` over the first 10% of the trajectory (before saturation at attractor scale ~30 units). Reported alongside the theoretical MLE.
- **Bounded attractor check**: Verifies `max_radius < 500` to catch integration failures.

## See Also

- `exp_euler_gimbal_lock.m` — 3D rotation visualization using the same OceanViz Dark theme
- `Experiments/Scripts/Simulation/` — Other simulation-driven experiments
- Lorenz, E. N. (1963). *Deterministic nonperiodic flow.* Journal of Atmospheric Sciences, 20(2), 130–141.
