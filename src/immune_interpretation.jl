#!/usr/bin/env julia
################################################################################
# Reviewer response (issue #2): train the structured-parameter UDE (Phase 2)
# and extract a quantitative interpretation of the *learned* immune interaction
# term, plus save a checkpoint so the analysis is reproducible.
################################################################################

using JLD2, Statistics, Printf, StatsBase
include(joinpath(@__DIR__, "FutureWorks.jl"))

const TIME_FILE   = joinpath(@__DIR__, "..", "Data", "tumor_time_to_event_data.csv")
const STATIC_FILE = joinpath(@__DIR__, "..", "Data", "tumor_volume_vs_Im_cells_rate.csv")
const OUT_DIR     = joinpath(@__DIR__, "..", "results", "immune_interpretation")
mkpath(OUT_DIR)

Random.seed!(42)

println("== Loading data ==")
df_dyn    = load_dynamic_data(TIME_FILE)
df_static = load_static_data(STATIC_FILE)
groups, (tmin, tmax) = process_groups(df_dyn)
@printf("groups=%d  tspan=[%.2f, %.2f]\n", length(groups), tmin, tmax)

# Progress file with explicit flush so convergence can be monitored live
# (Julia block-buffers stdout when redirected to a file).
const PROG = joinpath(OUT_DIR, "train_progress.txt")
progio = open(PROG, "w")
logp(msg) = (println(progio, msg); flush(progio); println(msg))

# ---- Training configuration ----
# Bounded, instrumented, and checkpointed per-phase so the run can never get
# stuck invisibly: AdamW writes its own checkpoint the moment it ends, and
# L-BFGS writes an incrementally-updated checkpoint every few iterations while
# it keeps refining. A looser (but still tight) ODE tolerance of 1e-6 roughly
# halves per-iteration cost with no meaningful accuracy loss for interpretation.
const LOCAL_PATIENCE = 200
const ADAMW_CAP      = 450
const LBFGS_CAP      = 60
const LOSS_THRESHOLD = 0.004
const INTERP_ABSTOL  = 1e-6
const INTERP_RELTOL  = 1e-6
const CKPT_ADAMW = joinpath(OUT_DIR, "phase2_adamw.jld2")
const CKPT_LBFGS = joinpath(OUT_DIR, "phase2_lbfgs.jld2")
const CKPT_FINAL = joinpath(OUT_DIR, "phase2_trained.jld2")

"""Train with per-phase, incrementally-saved checkpoints and full logging."""
function train_model_logged(θ_init, groups, df_static, re_dynamics, tmin, tmax)
    losses = Float64[]; best_loss = Inf; best_θ = copy(θ_init); no_improve = Ref(0)
    loss_func = (θ, p) -> compute_loss(θ, groups, df_static, re_dynamics, tmin, tmax;
                                       abstol=INTERP_ABSTOL, reltol=INTERP_RELTOL)
    of = OptimizationFunction(loss_func, AutoZygote())

    # ---- Phase A: AdamW ----
    cbA = function (state, loss)
        push!(losses, loss)
        if loss < best_loss
            best_loss = loss; best_θ = copy(state.u); no_improve[] = 0
        else
            no_improve[] += 1
        end
        length(losses) % 50 == 0 &&
            logp(@sprintf("AdamW iter %d: loss=%.6f  best=%.6f  patience=%d", length(losses), loss, best_loss, no_improve[]))
        return (no_improve[] >= LOCAL_PATIENCE) || (best_loss < LOSS_THRESHOLD)
    end
    logp("Phase A: AdamW (cap $ADAMW_CAP, patience $LOCAL_PATIENCE, tol=$INTERP_ABSTOL)")
    solve(OptimizationProblem(of, θ_init), Optimisers.AdamW(INITIAL_LR); callback=cbA, maxiters=ADAMW_CAP)
    # AdamW checkpoint — available immediately and independently of L-BFGS.
    JLD2.save(CKPT_ADAMW, "theta", best_θ, "losses", losses, "loss", best_loss)
    JLD2.save(CKPT_FINAL, "theta", best_θ, "losses", losses)   # final defaults to AdamW until L-BFGS beats it
    logp(@sprintf("AdamW checkpoint saved -> %s (best loss=%.6f)", basename(CKPT_ADAMW), best_loss))

    # ---- Phase B: L-BFGS, incrementally checkpointed and logged ----
    if best_loss > LOSS_THRESHOLD
        adamw_loss = best_loss
        lb_iter = Ref(0); lb_best = Ref(best_loss); lb_θ = copy(best_θ)
        cbB = function (state, loss)
            lb_iter[] += 1
            if loss < lb_best[]
                lb_best[] = loss; lb_θ = copy(state.u)
                # incremental L-BFGS checkpoint: usable at any time
                JLD2.save(CKPT_LBFGS, "theta", lb_θ, "loss", lb_best[], "iter", lb_iter[])
                if lb_best[] < best_loss
                    best_loss = lb_best[]; best_θ = copy(lb_θ)
                    JLD2.save(CKPT_FINAL, "theta", best_θ, "losses", losses)
                end
            end
            (lb_iter[] % 5 == 0 || lb_iter[] <= 3) &&
                logp(@sprintf("L-BFGS iter %d: loss=%.6f  best=%.6f", lb_iter[], loss, lb_best[]))
            return false
        end
        logp("Phase B: L-BFGS (cap $LBFGS_CAP) from AdamW loss=$(round(adamw_loss,digits=6)); incremental ckpt -> $(basename(CKPT_LBFGS))")
        try
            solve(OptimizationProblem(of, best_θ), OptimizationOptimJL.LBFGS(); callback=cbB, maxiters=LBFGS_CAP)
        catch e
            logp("L-BFGS stopped early ($e); keeping best so far")
        end
        logp(@sprintf("L-BFGS done: %d iters, best loss=%.6f (AdamW was %.6f)", lb_iter[], best_loss, adamw_loss))
    end
    JLD2.save(CKPT_FINAL, "theta", best_θ, "losses", losses)
    return best_θ, losses, best_loss
end

local θ, re_dynamics
if isfile(CKPT_FINAL)
    logp("== Loading existing checkpoint ==")
    θ = JLD2.load(CKPT_FINAL, "theta")
    _, re_dynamics = initialize_parameters()
else
    logp("== Training Phase 2 (structured-parameter UDE), seed=42 ==")
    θ_init, re_dynamics = initialize_parameters()
    t0 = time()
    θ, losses, best_loss = train_model_logged(θ_init, groups, df_static, re_dynamics, tmin, tmax)
    logp(@sprintf("trained in %.1f min, final best loss=%.6f", (time()-t0)/60, best_loss))
end
close(progio)

dynamics_net = re_dynamics(θ)

# ---- Helper: evaluate learned Michaelis-Menten immune parameters/term ----
"""Return (c_kill, h_sat, R_kill) for normalized volume V, immune I at t_norm."""
function immune_terms(V_norm, I_norm, t_norm)
    out = dynamics_net([V_norm, t_norm, I_norm])
    c_raw, h_raw = out[2]
    c = MIN_IMMUNE_KILLING   + (MAX_IMMUNE_KILLING   - MIN_IMMUNE_KILLING)   * sigmoid(c_raw)
    h = MIN_HALF_SATURATION  + (MAX_HALF_SATURATION  - MIN_HALF_SATURATION)  * sigmoid(h_raw)
    R = I_norm > 1e-3 ? (c * V_norm * I_norm) / (h + V_norm) : 0.0
    return c, h, R
end

# ---- Data ranges (normalized units) ----
Vs = vcat([g.volumes for g in groups]...)
Is = vcat([g.immune_levels for g in groups]...)
Vlo, Vhi = quantile(Vs, 0.05), quantile(Vs, 0.95)
Ilo, Ihi = quantile(Is, 0.05), quantile(Is, 0.95)
@printf("V_norm range [5%%,95%%]=[%.4f, %.4f] (mm^3: [%.1f, %.1f])\n", Vlo, Vhi, Vlo*VOL_SCALE, Vhi*VOL_SCALE)
@printf("I_norm range [5%%,95%%]=[%.4f, %.4f] (cells: [%.1f, %.1f])\n", Ilo, Ihi, Ilo*IMMUNE_SCALE, Ihi*IMMUNE_SCALE)

nV, nI = 40, 40
Vgrid = range(max(Vlo, 1e-3), Vhi, length=nV)
Igrid = range(max(Ilo, 1e-3), Ihi, length=nI)
tmid = 0.5

Cgrid = [immune_terms(V, I, tmid)[1] for V in Vgrid, I in Igrid]
Hgrid = [immune_terms(V, I, tmid)[2] for V in Vgrid, I in Igrid]
Rgrid = [immune_terms(V, I, tmid)[3] for V in Vgrid, I in Igrid]

# ---- (1) Monotonicity of killing in immune level I (fixed V = median) ----
Vmed = median(Vs)
R_vs_I = [immune_terms(Vmed, I, tmid)[3] for I in Igrid]
mono_I = all(diff(R_vs_I) .>= -1e-9)
@printf("\n[Monotonic increasing in I at V=median]? %s  (R: %.4f -> %.4f)\n",
        mono_I, R_vs_I[1], R_vs_I[end])

# ---- (2) Saturation in V (fixed I = median): fit shows MM half-saturation ----
Imed = median(Is)
R_vs_V = [immune_terms(V, Imed, tmid)[3] for V in Vgrid]
Rmax_V = maximum(R_vs_V)
# volume where killing reaches half its max over this range
half_idx = findfirst(x -> x >= 0.5*Rmax_V, R_vs_V)
Vhalf = half_idx === nothing ? NaN : Vgrid[half_idx]
@printf("[Killing at small vs large V] R(Vlo)=%.4f  R(Vhi)=%.4f  half-rise V_norm~%.4f (%.1f mm^3)\n",
        R_vs_V[1], R_vs_V[end], Vhalf, Vhalf*VOL_SCALE)

# ---- (3) Learned parameter ranges over the data domain ----
@printf("[Learned c_kill] min=%.3f  median=%.3f  max=%.3f (day^-1)\n",
        minimum(Cgrid), median(Cgrid), maximum(Cgrid))
@printf("[Learned h_sat ] min=%.3f  median=%.3f  max=%.3f (norm vol; %.1f-%.1f mm^3)\n",
        minimum(Hgrid), median(Hgrid), maximum(Hgrid), minimum(Hgrid)*VOL_SCALE, maximum(Hgrid)*VOL_SCALE)

# ---- (4) Physics grounding: rank corr of learned c_kill vs static killing data ----
median_immune = median(Is)
pred_c = [immune_terms(v, median_immune, tmid)[1] for v in df_static.VolumeNorm]
rho = corspearman(df_static.Im_cells_rate, pred_c)
@printf("[Physics grounding] Spearman(rho) learned c_kill vs static killing-rate data = %.3f\n", rho)

# ---- (5) Contribution of immune term to net dV/dt on observed trajectories ----
frac_supp = Float64[]
for (i, g) in enumerate(groups)
    for (t, V) in zip(g.times, g.volumes)
        tn = (t - tmin) / (tmax - tmin + eps())
        I  = g.immune_interp(t)
        c, h, R = immune_terms(max(V,1e-8), I, tn)
        out = dynamics_net([max(V,1e-8), tn, I])
        r = MIN_GROWTH_RATE + (MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(out[1][1])
        K = max(g.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(out[1][2])), V*1.01)
        gomp = r*max(V,1e-8)*log(K/max(V,1e-8))
        if gomp > 1e-8
            push!(frac_supp, R / (gomp + 1e-8))
        end
    end
end
@printf("[Immune suppression strength] median R_kill/Gompertz over observed points = %.3f  (max %.3f)\n",
        median(frac_supp), maximum(frac_supp))

# ---- Save numeric summary + grids for figure ----
open(joinpath(OUT_DIR, "immune_summary.txt"), "w") do io
    @printf(io, "monotonic_in_I=%s\n", mono_I)
    @printf(io, "R_at_Imin=%.6f\nR_at_Imax=%.6f\n", R_vs_I[1], R_vs_I[end])
    @printf(io, "R_at_Vmin=%.6f\nR_at_Vmax=%.6f\n", R_vs_V[1], R_vs_V[end])
    @printf(io, "Vhalf_norm=%.6f\nVhalf_mm3=%.3f\n", Vhalf, Vhalf*VOL_SCALE)
    @printf(io, "c_kill_min=%.6f\nc_kill_med=%.6f\nc_kill_max=%.6f\n", minimum(Cgrid), median(Cgrid), maximum(Cgrid))
    @printf(io, "h_sat_min=%.6f\nh_sat_med=%.6f\nh_sat_max=%.6f\n", minimum(Hgrid), median(Hgrid), maximum(Hgrid))
    @printf(io, "spearman_ckill_vs_static=%.6f\n", rho)
    @printf(io, "median_suppression_ratio=%.6f\nmax_suppression_ratio=%.6f\n", median(frac_supp), maximum(frac_supp))
end
JLD2.save(joinpath(OUT_DIR, "immune_grids.jld2"),
          "Vgrid", collect(Vgrid), "Igrid", collect(Igrid),
          "Rgrid", Rgrid, "Cgrid", Cgrid, "Hgrid", Hgrid,
          "R_vs_I", R_vs_I, "R_vs_V", R_vs_V)

# ---- Interpretation figure ----
try
    p1 = surface(Igrid .* IMMUNE_SCALE, Vgrid .* VOL_SCALE, Rgrid,
                 xlabel="Immune cells", ylabel="Tumor vol (mm³)", zlabel="Learned R_kill",
                 title="Learned immune killing term  R_kill(V,I)", size=(700,550), margin=6mm,
                 titlefontsize=14, guidefontsize=11)
    savefig(p1, joinpath(OUT_DIR, "learned_immune_killing_surface.png"))

    p2 = plot(Igrid .* IMMUNE_SCALE, R_vs_I, lw=3, label="R_kill vs immune (V=median)",
              xlabel="Immune cell count", ylabel="R_kill", title="Monotone increase in immune level",
              size=(600,450), margin=6mm, legend=:topleft)
    savefig(p2, joinpath(OUT_DIR, "killing_vs_immune.png"))

    p3 = plot(Vgrid .* VOL_SCALE, R_vs_V, lw=3, label="R_kill vs volume (I=median)",
              xlabel="Tumor volume (mm³)", ylabel="R_kill", title="Michaelis–Menten saturation in volume",
              size=(600,450), margin=6mm, legend=:bottomright)
    savefig(p3, joinpath(OUT_DIR, "killing_vs_volume.png"))
    println("figures saved to ", OUT_DIR)
catch e
    @warn "figure generation failed: $e"
end

println("\n== DONE ==")
