#!/usr/bin/env julia
# Re-render the two C4-T2 figures with big fonts, the ORIGINAL details
# (High Immune Activity markers, original tab20 colors), and legends placed so
# they do NOT overlap the curves (outer legend for the component plot).

using JLD2, Statistics, Printf, ColorSchemes
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots
default(dpi=200, titlefontsize=30, guidefontsize=26, tickfontsize=22, legendfontsize=20)
C = ColorSchemes.tab20.colors

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = get(ENV, "CKPT_PATH", joinpath(@__DIR__, "..", "results", "immune_interpretation", "phase2_lbfgs.jld2"))
df = load_dynamic_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"))
groups,(tmin,tmax) = process_groups(df)
θ = JLD2.load(CKPT,"theta"); _,re = initialize_parameters(); net = re(θ)
gi = findfirst(g->g.id==("C4","T2"),groups); g = groups[gi]

tp,vp   = predict_group(g,gi,θ,groups,re,tmin,tmax;dense_time_points=120)
tn2,vn2 = predict_without_immune(g,gi,θ,groups,re,tmin,tmax;dense_time_points=120)

# ---- Tumor dynamics (legend top-left sits in the empty low-volume region) ----
pv = plot(title="Tumor Dynamics: C4 | T2", xlabel="Time (days)", ylabel="Volume (mm³)",
          legend=:topleft, size=(1200,900), left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=5Plots.mm, right_margin=6Plots.mm)
plot!(pv, tn2, vn2 .* VOL_SCALE, label="Predicted (no immune)", color=C[3], lw=4, ls=:dash)
plot!(pv, tp,  vp  .* VOL_SCALE, label="Predicted (with immune)", color=C[2], lw=4)
scatter!(pv, g.times, g.volumes .* VOL_SCALE, label="Observed", color=C[1], ms=11)
savefig(pv, joinpath(FIGDIR, "group_C4_T2_volume_comparison.png"))

# ---- Component dynamics (OUTER legend => guaranteed no overlap) ----
gm=Float64[]; kl=Float64[]
for (t,V) in zip(tp,vp)
    Vp=max(V,1e-8); tn=(t-tmin)/(tmax-tmin+eps()); I=g.immune_interp(t)
    o=net([Vp,tn,I])
    r=MIN_GROWTH_RATE+(MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(o[1][1])
    K=max(g.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(o[1][2])),Vp*1.01)
    c=MIN_IMMUNE_KILLING+(MAX_IMMUNE_KILLING-MIN_IMMUNE_KILLING)*sigmoid(o[2][1])
    h=MIN_HALF_SATURATION+(MAX_HALF_SATURATION-MIN_HALF_SATURATION)*sigmoid(o[2][2])
    push!(gm, r*Vp*log(K/Vp)); push!(kl, I>1e-3 ? (c*Vp*I)/(h+Vp) : 0.0)
end
imm = g.immune_interp.(tp); mask = imm .> 0.1*maximum(imm)
pc = plot(title="Component Dynamics: C4 | T2", xlabel="Time (days)", ylabel="Rate of change dV/dt (norm.)",
          legend=:outertopright, size=(1550,900), left_margin=12Plots.mm, bottom_margin=10Plots.mm,
          top_margin=5Plots.mm, right_margin=6Plots.mm)
plot!(pc, tp, gm,        label="Gompertz Growth", color=C[1], lw=4)
plot!(pc, tp, -kl,       label="Immune Killing",  color=C[2], lw=4)
plot!(pc, tp, gm .- kl,  label="Net Growth",      color=C[6], lw=3, ls=:dash)
hline!(pc, [0], color=:black, ls=:dot, alpha=0.6, lw=2, label="Zero Growth")
any(mask) && scatter!(pc, tp[mask], gm[mask], color=:red, ms=7, alpha=0.4, label="High Immune Activity")
savefig(pc, joinpath(FIGDIR, "group_C4_T2_component_dynamics.png"))

println("C4T2 LEGEND FIX DONE")
