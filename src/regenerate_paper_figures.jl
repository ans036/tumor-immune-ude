#!/usr/bin/env julia
################################################################################
# Regenerate ONLY the low-readability paper figures at 300 DPI with enlarged
# fonts (reviewer issue #2: "enlarge the fonts in all figures"). We deliberately
# rebuild only the half-column-width offenders:
#   forecast_short, forecast_long, forecast_comparison, r2_summary, rmse_summary
# Forecasts come from the trained checkpoint; the CV summary bars use the paper's
# reported Table-2 values so nothing contradicts the text. global_fit and the
# training-loss curves are left as the originals.
################################################################################

using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots

# Large fonts on a modest canvas => big text even at 0.49\linewidth.
default(dpi=300, size=(600,450), framestyle=:box, grid=true,
        titlefontsize=16, guidefontsize=15, tickfontsize=13, legendfontsize=12,
        left_margin=6mm, bottom_margin=6mm, top_margin=3mm, right_margin=3mm, lw=3)

const FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
const CKPT   = get(ENV, "CKPT_PATH", joinpath(@__DIR__, "..", "results", "immune_interpretation", "phase2_lbfgs.jld2"))

df_dyn = load_dynamic_data(joinpath(@__DIR__, "..", "Data", "tumor_time_to_event_data.csv"))
groups, (tmin, tmax) = process_groups(df_dyn)
θ = JLD2.load(CKPT, "theta")
_, re_dynamics = initialize_parameters()
@printf("regen from %s\n", CKPT)

# ---------- forecasts for C4-T5 ----------
gi = findfirst(g -> g.id == ("C4","T5"), groups); g5 = groups[gi]
function integrate_to(tend; immune=true)
    function rhs(u,p,t)
        net = re_dynamics(p); V = max(u[1],1e-8)
        tn = (t - tmin)/(tmax - tmin + eps())
        I = immune ? g5.immune_interp(t) : 0.0
        o = net([V, tn, I])
        r = MIN_GROWTH_RATE + (MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(o[1][1])
        K = max(g5.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(o[1][2])), V*1.01)
        gomp = r*V*log(K/V)
        c = MIN_IMMUNE_KILLING+(MAX_IMMUNE_KILLING-MIN_IMMUNE_KILLING)*sigmoid(o[2][1])
        h = MIN_HALF_SATURATION+(MAX_HALF_SATURATION-MIN_HALF_SATURATION)*sigmoid(o[2][2])
        kill = (immune && I>1e-3) ? (c*V*I)/(h+V) : 0.0
        return [max(gomp-kill, -0.1*V)]
    end
    ts = range(g5.tspan[1], tend, length=200)
    sol = solve(ODEProblem(rhs, g5.u0, (g5.tspan[1], tend), θ), Tsit5(); saveat=ts, dense=false, abstol=1e-6, reltol=1e-6)
    return sol.t, [max(u[1],0.0)*VOL_SCALE for u in sol.u]
end

t_hist = g5.tspan[2]
for (tag, extra, ttl) in (("short",14.0,"Short-term Forecast: C4-T5"), ("long",90.0,"Long-term Forecast: C4-T5"))
    tend = t_hist + extra
    th, vh = integrate_to(t_hist;  immune=true)
    ti, vi = integrate_to(tend;    immune=true)
    tn, vn = integrate_to(tend;    immune=false)
    p = plot(title=ttl, xlabel="Time (days)", ylabel="Volume (mm³)", legend=:topleft)
    scatter!(p, g5.times, g5.volumes .* VOL_SCALE, label="Observed", color=:navy, ms=7)
    plot!(p, th, vh, label="Model (history)", color=:orange)
    plot!(p, ti[ti .>= t_hist], vi[ti .>= t_hist], label="Forecast (with immune)", color=:seagreen, ls=:dash)
    plot!(p, tn[tn .>= t_hist], vn[tn .>= t_hist], label="Forecast (no immune)", color=:orchid, ls=:dashdot)
    savefig(p, joinpath(FIGDIR, "forecast_$(tag).png"))
end

tc, vc = integrate_to(t_hist; immune=true)
pc = plot(title="Forecast Agreement: C4-T5", xlabel="Time (days)", ylabel="Volume (mm³)", legend=:topleft)
scatter!(pc, g5.times, g5.volumes .* VOL_SCALE, label="Observed", color=:navy, ms=7)
plot!(pc, tc, vc, label="Predicted", color=:orange)
savefig(pc, joinpath(FIGDIR, "forecast_comparison.png"))

# ---------- CV summary bars (values from paper Table 2, so consistent with text) ----------
labels = ["C2-T1","C2-T2","C2-T3","C2-T4","C4-T5"]
r2s    = [0.909, 0.915, 0.947, 0.958, 0.998]
rmses  = [30.5, 17.2, 8.7, 0.9, 16.5]
pr = bar(labels, r2s, title="Cross-Validation R² by Fold", ylabel="R²", legend=false,
         color=:steelblue, ylims=(0.8,1.0), rotation=25)
hline!(pr, [mean(r2s)], color=:red, ls=:dash)
savefig(pr, joinpath(FIGDIR, "r2_summary.png"))
pm = bar(labels, rmses, title="Cross-Validation RMSE by Fold", ylabel="RMSE (mm³)", legend=false,
         color=:indianred, rotation=25)
hline!(pm, [mean(rmses)], color=:navy, ls=:dash)
savefig(pm, joinpath(FIGDIR, "rmse_summary.png"))

println("REGEN DONE (forecasts + CV summaries) -> ", FIGDIR)
