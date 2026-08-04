#!/usr/bin/env julia
# Re-render parameter_trajectories_C4T5.png at big fonts: learned r(t), K(t),
# c_kill(t), h_sat(t) for the representative C4-T5 group.
using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots
default(dpi=200, titlefontsize=26, guidefontsize=22, tickfontsize=18, legendfontsize=16,
        left_margin=10Plots.mm, bottom_margin=8Plots.mm, top_margin=4Plots.mm, right_margin=4Plots.mm, lw=4)

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = get(ENV,"CKPT_PATH", joinpath(@__DIR__,"..","results","immune_interpretation","phase2_lbfgs.jld2"))
df = load_dynamic_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"))
groups,(tmin,tmax) = process_groups(df)
θ = JLD2.load(CKPT,"theta"); _,re = initialize_parameters(); net = re(θ)
gi = findfirst(g->g.id==("C4","T5"),groups); g = groups[gi]
tp,vp = predict_group(g,gi,θ,groups,re,tmin,tmax;dense_time_points=120)

rt=Float64[]; Kt=Float64[]; ct=Float64[]; ht=Float64[]
for (t,V) in zip(tp,vp)
    Vp=max(V,1e-8); tn=(t-tmin)/(tmax-tmin+eps()); I=g.immune_interp(t)
    o=net([Vp,tn,I])
    push!(rt, MIN_GROWTH_RATE+(MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(o[1][1]))
    push!(Kt, max(g.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(o[1][2])),Vp*1.01)*VOL_SCALE)
    push!(ct, MIN_IMMUNE_KILLING+(MAX_IMMUNE_KILLING-MIN_IMMUNE_KILLING)*sigmoid(o[2][1]))
    push!(ht, (MIN_HALF_SATURATION+(MAX_HALF_SATURATION-MIN_HALF_SATURATION)*sigmoid(o[2][2]))*VOL_SCALE)
end
p = plot(layout=(2,2), size=(1600,1150), legend=false)
plot!(p[1], tp, rt, title="Growth Rate",            xlabel="Time (days)", ylabel="r(t) (day⁻¹)",  color=:steelblue)
plot!(p[2], tp, Kt, title="Carrying Capacity",       xlabel="Time (days)", ylabel="K(t) (mm³)",     color=:darkorange)
plot!(p[3], tp, ct, title="Immune Killing Rate",     xlabel="Time (days)", ylabel="c_kill(t) (day⁻¹)", color=:seagreen)
plot!(p[4], tp, ht, title="Half-Saturation Constant",xlabel="Time (days)", ylabel="h_sat(t) (mm³)",  color=:crimson)
savefig(p, joinpath(FIGDIR, "parameter_trajectories_C4T5.png"))
println("PARAM TRAJ DONE")
