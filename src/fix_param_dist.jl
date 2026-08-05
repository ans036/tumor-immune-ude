#!/usr/bin/env julia
# Re-plot ONLY hybrid_param_distributions.png with a non-garbled layout:
# fewer x-ticks, extra inter-panel spacing, taller canvas. Uses the saved
# Phase-1 checkpoint (no retraining).
using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "JuliaconSubmission.jl"))
using Plots
default(dpi=200, titlefontsize=26, guidefontsize=22, tickfontsize=18, legendfontsize=16)

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = joinpath(@__DIR__, "..", "results", "phase1_hybrid.jld2")

Random.seed!(42)
df,_,_ = load_and_merge_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"),
                             joinpath(@__DIR__,"..","Data","tumor_volume_vs_Im_cells_rate.csv"))
groups,(tmin,tmax) = process_groups(df)
_, _, _, netsz = smart_parameter_initialization(groups; seed=42)
θ = JLD2.load(CKPT, "theta")
_, _, log_r, log_K, log_α = unpack_parameters(θ, netsz, length(groups))
r = clamp.(exp.(log_r), MIN_GROWTH_RATE, MAX_GROWTH_RATE)
K = exp.(log_K)
α = clamp.(exp.(log_α), 0.01, 1.0)
@printf("r[%.3f,%.3f] K[%.2f,%.2f] a[%.3f,%.3f]\n", minimum(r),maximum(r),minimum(K),maximum(K),minimum(α),maximum(α))

m = 14Plots.mm
h1 = histogram(r, bins=6, color=:steelblue, legend=false, title="Growth rate r",
               xlabel="r (day⁻¹)", ylabel="count", xticks=[0.08,0.12,0.16],
               left_margin=m, bottom_margin=m, top_margin=6Plots.mm, right_margin=6Plots.mm)
h2 = histogram(K, bins=6, color=:seagreen, legend=false, title="Carrying capacity K",
               xlabel="K (norm.)", xticks=[4,6,8,10],
               left_margin=m, bottom_margin=m, top_margin=6Plots.mm, right_margin=6Plots.mm)
h3 = histogram(α, bins=6, color=:indianred, legend=false, title="Immune strength α",
               xlabel="α", xticks=[0.1,0.2,0.3,0.4],
               left_margin=m, bottom_margin=m, top_margin=6Plots.mm, right_margin=6Plots.mm)
p = plot(h1, h2, h3, layout=(1,3), size=(1750,720))
savefig(p, joinpath(FIGDIR, "hybrid_param_distributions.png"))
println("PARAM DIST FIX DONE")
