# Installation Guide

## Prerequisites

- [Julia](https://julialang.org/downloads/) **1.9 or later** (tested on 1.9, 1.10, and 1.11)
- Git

## Step 1: Clone the repository

```bash
git clone https://github.com/ans036/tumor-immune-ude.git
cd tumor-immune-ude
```

## Step 2: Install dependencies

The project includes a `Project.toml` in `src/`. To install all dependencies into a reproducible environment:

```bash
julia --project=src -e 'using Pkg; Pkg.instantiate()'
```

This installs all required packages, including:

| Package | Purpose |
|---|---|
| `DifferentialEquations.jl` | ODE solvers (Tsit5, etc.) |
| `Flux.jl` | Neural network layers and training |
| `SciMLSensitivity.jl` | Adjoint-based gradient computation |
| `Optimization.jl` | Unified optimization interface |
| `OptimizationOptimJL.jl` | L-BFGS refinement |
| `OptimizationOptimisers.jl` | AdamW optimizer |
| `CSV.jl` / `DataFrames.jl` | Data loading |
| `Plots.jl` | Visualization |

Alternatively, the test scripts bootstrap a temporary environment and install packages automatically — no manual setup needed (see Step 3).

## Step 3: Verify installation

Run the test suite to confirm everything works:

```bash
# Phase 1 (Hybrid UDE) tests
julia --color=yes test/runtests.jl

# Phase 2 (Structured-Parameter UDE) tests
julia --color=yes test/pure_mechanistic_tests.jl
```

Both should complete without errors and produce result plots in the `results/` directory.

## Step 4: Run the pipelines

### Phase 1: Hybrid UDE (Dual-Network)

```julia
julia> include("src/JuliaconSubmission.jl")
julia> main(time_file="Data/tumor_time_to_event_data.csv",
            immune_file="Data/tumor_volume_vs_Im_cells_rate.csv",
            save_plots=true)
```

### Phase 2: Structured-Parameter UDE

```julia
julia> include("src/FutureWorks.jl")
julia> main(time_file="Data/tumor_time_to_event_data.csv",
            immune_rate_file="Data/tumor_volume_vs_Im_cells_rate.csv",
            save_plots=true)
```

Results (training curves, trajectories, counterfactuals, component dynamics, global fits) are saved to timestamped subdirectories under `results/`.

## Troubleshooting

**Package precompilation is slow on first run:**
This is normal. Julia precompiles all dependencies on first use. Subsequent runs will be much faster.

**Out-of-memory errors:**
Training uses ~4 GB RAM. If running on a constrained system, close other applications or reduce the number of random restarts in the source code (`N_RESTARTS`).

**Plot backend issues:**
If plots fail to save, ensure the GR backend is available:
```julia
julia> using Plots; gr()
```

## System requirements

- **RAM:** 4 GB minimum, 8 GB recommended
- **CPU:** Any modern processor (Intel i5/i7, AMD Ryzen, Apple Silicon)
- **OS:** Linux, macOS, or Windows
- **Disk:** ~1 GB for Julia depot and dependencies
- **Training time:** 15–30 minutes per cross-validation fold on standard hardware
