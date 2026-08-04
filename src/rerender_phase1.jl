#!/usr/bin/env julia
# Re-render the four Phase-1 (hybrid UDE) paper figures at LARGE fonts:
#   hybrid_training_loss.png, immune_network_surface.png,
#   correction_network_surface.png, hybrid_param_distributions.png
# The committed Phase-1 create_all_plots does not emit the network surfaces, so we
# train a bounded Phase-1 model (seed 42) and plot the learned networks/params here.

using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "JuliaconSubmission.jl"))
using Plots
using Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity

default(dpi=200, size=(1100,850), titlefontsize=28, guidefontsize=24, tickfontsize=20, legendfontsize=18,
        left_margin=8Plots.mm, bottom_margin=7Plots.mm, top_margin=4Plots.mm, right_margin=5Plots.mm, lw=3)

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = joinpath(@__DIR__, "..", "results", "phase1_hybrid.jld2")

Random.seed!(42)
df, vmean, vstd = load_and_merge_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"),
                                      joinpath(@__DIR__,"..","Data","tumor_volume_vs_Im_cells_rate.csv"))
groups, (tmin, tmax) = process_groups(df)
@printf("Phase1: %d groups\n", length(groups))

θ0, re_imm, re_corr, netsz = smart_parameter_initialization(groups; seed=42)

# ---- bounded training (AdamW cap 400 + short L-BFGS) with flushed progress ----
prog = open(joinpath(@__DIR__,"..","results","phase1_progress.txt"), "w")
logp(m) = (println(prog, m); flush(prog); println(m))
local θ, losses
if isfile(CKPT)
    logp("loading existing phase1 checkpoint")
    θ = JLD2.load(CKPT, "theta"); losses = JLD2.load(CKPT, "losses")
else
    losses = Float64[]; best = Ref(Inf); bestθ = Ref(copy(θ0)); noimp = Ref(0)
    lf = (p,_) -> compute_enhanced_loss(p, groups, re_imm, re_corr, netsz, tmin, tmax; training_phase=1)
    cb = function (s, l)
        push!(losses, l)
        if l < best[]; best[]=l; bestθ[]=copy(s.u); noimp[]=0; else; noimp[]+=1; end
        length(losses) % 25 == 0 && logp(@sprintf("Phase1 AdamW iter %d: loss=%.5f best=%.5f", length(losses), l, best[]))
        return noimp[] >= 120 || length(losses) >= 400
    end
    of = OptimizationFunction(lf, AutoZygote())
    logp("Phase1 training (AdamW cap 400)...")
    try; solve(OptimizationProblem(of, θ0), Optimisers.AdamW(5e-4); callback=cb, maxiters=400); catch e; logp("adamw stopped: $e"); end
    θ = bestθ[]
    try
        logp("Phase1 L-BFGS (cap 50)...")
        r = solve(OptimizationProblem(of, θ), OptimizationOptimJL.LBFGS(); maxiters=50)
        r.objective < best[] && (θ = r.u; best[] = r.objective)
    catch e; logp("lbfgs stopped: $e"); end
    logp(@sprintf("Phase1 done, best loss=%.5f", best[]))
    JLD2.save(CKPT, "theta", θ, "losses", losses)
end
close(prog)

θ_imm, θ_corr, log_r, log_K, log_α = unpack_parameters(θ, netsz, length(groups))
immune_net = re_imm(θ_imm); corr_net = re_corr(θ_corr)
r_vals = clamp.(exp.(log_r), MIN_GROWTH_RATE, MAX_GROWTH_RATE)
K_vals = exp.(log_K); α_vals = clamp.(exp.(log_α), 0.01, 1.0)

# ---- 1. training loss ----
pl = plot(1:length(losses), max.(losses,1e-8), xlabel="Iteration", ylabel="Loss",
          title="Phase 1 Training Loss", yscale=:log10, legend=false, color=:darkblue, lw=3)
savefig(pl, joinpath(FIGDIR, "hybrid_training_loss.png"))

# ---- 2. Immune Response Network surface over (V_norm, immune fraction) ----
Vg = range(0.0, 1.0, length=45)      # normalized volume V/VOL_SCALE
Ig = range(0.0, 1.0, length=45)      # immune fraction [0,1]
Zimm = [immune_net([V, I])[1] for V in Vg, I in Ig]
ps1 = surface(Ig, Vg, Zimm, xlabel="Immune fraction", ylabel="Norm. volume", zlabel="IRN output",
              title="Immune Response Network", color=:viridis, camera=(40,30))
savefig(ps1, joinpath(FIGDIR, "immune_network_surface.png"))

# ---- 3. Time-Aware Correction Network surface over (V_norm, t_norm) ----
Tg = range(0.0, 1.0, length=45)
Zcor = [corr_net([V, t])[1] for V in Vg, t in Tg]
ps2 = surface(Tg, Vg, Zcor, xlabel="Norm. time", ylabel="Norm. volume", zlabel="TACN output",
              title="Time-Aware Correction Network", color=:plasma, camera=(40,30))
savefig(ps2, joinpath(FIGDIR, "correction_network_surface.png"))

# ---- 4. learned parameter distributions ----
pp = plot(layout=(1,3), size=(1500,520), legend=false)
histogram!(pp[1], r_vals, bins=min(8,length(r_vals)), color=:steelblue, title="Growth rate r", xlabel="r (day⁻¹)", ylabel="count")
histogram!(pp[2], K_vals, bins=min(8,length(K_vals)), color=:seagreen,  title="Carrying capacity K", xlabel="K (norm.)")
histogram!(pp[3], α_vals, bins=min(8,length(α_vals)), color=:indianred, title="Immune strength α", xlabel="α")
savefig(pp, joinpath(FIGDIR, "hybrid_param_distributions.png"))

@printf("r range [%.3f,%.3f], alpha range [%.3f,%.3f]\n", minimum(r_vals),maximum(r_vals),minimum(α_vals),maximum(α_vals))
println("PHASE1 RERENDER DONE")
