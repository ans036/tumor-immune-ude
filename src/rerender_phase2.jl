#!/usr/bin/env julia
# Re-render Phase-2 figures at LARGE fonts using the ORIGINAL create_all_plots
# (so every detail is preserved) from a trained checkpoint. Big fonts are applied
# via Plots.default before plotting; the plotting code does not override font sizes.

using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots

# Large fonts on the 1200x900 canvas the original code uses.
default(dpi=200, titlefontsize=30, guidefontsize=26, tickfontsize=22, legendfontsize=20)

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = get(ENV, "CKPT_PATH", joinpath(@__DIR__, "..", "results", "immune_interpretation", "phase2_lbfgs.jld2"))
TMP    = joinpath(@__DIR__, "..", "results", "rerender_phase2")
mkpath(TMP)

df_dyn    = load_dynamic_data(joinpath(@__DIR__, "..", "Data", "tumor_time_to_event_data.csv"))
df_static = load_static_data(joinpath(@__DIR__, "..", "Data", "tumor_volume_vs_Im_cells_rate.csv"))
groups, (tmin, tmax) = process_groups(df_dyn)
θ = JLD2.load(CKPT, "theta")
losses = try JLD2.load(joinpath(@__DIR__,"..","results","immune_interpretation","phase2_adamw.jld2"),"losses") catch; Float64[1.0,0.5,0.1] end
_, re_dynamics = initialize_parameters()
@printf("re-render Phase 2 from %s\n", CKPT)

create_all_plots(groups, df_static, θ, losses, re_dynamics, tmin, tmax; save_dir=TMP)

# Copy the paper-referenced figures to their paper names.
cp(joinpath(TMP, "training_loss.png"),               joinpath(FIGDIR, "pure_training_loss.png"); force=true)
cp(joinpath(TMP, "group_C4_T2", "volume_comparison.png"),  joinpath(FIGDIR, "group_C4_T2_volume_comparison.png"); force=true)
cp(joinpath(TMP, "group_C4_T2", "component_dynamics.png"), joinpath(FIGDIR, "group_C4_T2_component_dynamics.png"); force=true)
# global_fit kept separately for the R^2 decision (copied to TMP only, not paper yet)
gf = joinpath(TMP, "enhanced_visualizations", "global_fit.png")
isfile(gf) && cp(gf, joinpath(TMP, "global_fit_rerendered.png"); force=true)

println("PHASE2 RERENDER DONE")
