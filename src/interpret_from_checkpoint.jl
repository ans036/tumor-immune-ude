#!/usr/bin/env julia
# Standalone: load a trained Phase-2 checkpoint and produce the immune-interaction
# interpretation (stats + grids + global R2) without retraining. Runs in parallel
# with an ongoing training process by reading a *stable* checkpoint file.

using JLD2, Statistics, Printf, StatsBase
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots

CKPT = get(ENV, "CKPT_PATH", joinpath(@__DIR__, "..", "results", "immune_interpretation", "phase2_adamw.jld2"))
OUT  = get(ENV, "OUT_DIR",  joinpath(@__DIR__, "..", "results", "immune_interpretation", "adamw"))
mkpath(OUT)

df_dyn    = load_dynamic_data(joinpath(@__DIR__, "..", "Data", "tumor_time_to_event_data.csv"))
df_static = load_static_data(joinpath(@__DIR__, "..", "Data", "tumor_volume_vs_Im_cells_rate.csv"))
groups, (tmin, tmax) = process_groups(df_dyn)

θ = JLD2.load(CKPT, "theta")
_, re_dynamics = initialize_parameters()
dynamics_net = re_dynamics(θ)
@printf("Loaded checkpoint %s\n", CKPT)

# ---- global R2 (consistency with paper's reported 0.991) ----
allobs, allpred = Float64[], Float64[]
for (i,g) in enumerate(groups)
    t_pred, v = predict_group(g, i, θ, groups, re_dynamics, tmin, tmax; dense_time_points=100)
    any(isnan,v) && continue
    itp = linear_interpolation(t_pred, v; extrapolation_bc=Line())
    append!(allobs, g.volumes .* VOL_SCALE); append!(allpred, itp.(g.times) .* VOL_SCALE)
end
R2global = 1 - sum((allpred .- allobs).^2)/sum((allobs .- mean(allobs)).^2)
@printf("GLOBAL R2 = %.4f\n", R2global)

# ---- learned immune Michaelis-Menten term ----
function immune_terms(V, I, tn)
    o = dynamics_net([V, tn, I]); c_raw,h_raw = o[2]
    c = MIN_IMMUNE_KILLING  + (MAX_IMMUNE_KILLING - MIN_IMMUNE_KILLING) *sigmoid(c_raw)
    h = MIN_HALF_SATURATION + (MAX_HALF_SATURATION- MIN_HALF_SATURATION)*sigmoid(h_raw)
    R = I > 1e-3 ? (c*V*I)/(h+V) : 0.0
    return c,h,R
end

Vs = vcat([g.volumes for g in groups]...); Is = vcat([g.immune_levels for g in groups]...)
Vlo,Vhi = quantile(Vs,0.05), quantile(Vs,0.95); Ilo,Ihi = quantile(Is,0.05), quantile(Is,0.95)
Vgrid = range(max(Vlo,1e-3),Vhi,length=40); Igrid = range(max(Ilo,1e-3),Ihi,length=40); tmid=0.5
Cgrid=[immune_terms(V,I,tmid)[1] for V in Vgrid, I in Igrid]
Hgrid=[immune_terms(V,I,tmid)[2] for V in Vgrid, I in Igrid]
Rgrid=[immune_terms(V,I,tmid)[3] for V in Vgrid, I in Igrid]

Vmed=median(Vs); Imed=median(Is)
R_vs_I=[immune_terms(Vmed,I,tmid)[3] for I in Igrid]
R_vs_V=[immune_terms(V,Imed,tmid)[3] for V in Vgrid]
mono_I = all(diff(R_vs_I) .>= -1e-9)
Rmax_V = maximum(R_vs_V); half_idx = findfirst(x->x>=0.5*Rmax_V, R_vs_V)
Vhalf = half_idx===nothing ? NaN : Vgrid[half_idx]

median_immune = median(Is)
pred_c = [immune_terms(v, median_immune, tmid)[1] for v in df_static.VolumeNorm]
rho = corspearman(df_static.Im_cells_rate, pred_c)

frac_supp = Float64[]
for g in groups, (t,V) in zip(g.times, g.volumes)
    tn=(t-tmin)/(tmax-tmin+eps()); I=g.immune_interp(t)
    _,_,R = immune_terms(max(V,1e-8),I,tn)
    o=dynamics_net([max(V,1e-8),tn,I])
    r=MIN_GROWTH_RATE+(MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(o[1][1])
    K=max(g.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(o[1][2])),V*1.01)
    gomp=r*max(V,1e-8)*log(K/max(V,1e-8))
    gomp>1e-8 && push!(frac_supp, R/(gomp+1e-8))
end

open(joinpath(OUT,"immune_summary.txt"),"w") do io
    @printf(io,"checkpoint=%s\n", CKPT)
    @printf(io,"global_R2=%.6f\n", R2global)
    @printf(io,"monotonic_in_I=%s\nR_at_Imin=%.6f\nR_at_Imax=%.6f\n", mono_I, R_vs_I[1], R_vs_I[end])
    @printf(io,"R_at_Vmin=%.6f\nR_at_Vmax=%.6f\nVhalf_norm=%.6f\nVhalf_mm3=%.3f\n", R_vs_V[1],R_vs_V[end],Vhalf,Vhalf*VOL_SCALE)
    @printf(io,"c_kill_min=%.4f\nc_kill_med=%.4f\nc_kill_max=%.4f\n", minimum(Cgrid),median(Cgrid),maximum(Cgrid))
    @printf(io,"h_sat_min=%.4f\nh_sat_med=%.4f\nh_sat_max=%.4f\n", minimum(Hgrid),median(Hgrid),maximum(Hgrid))
    @printf(io,"h_sat_mm3_min=%.1f\nh_sat_mm3_max=%.1f\n", minimum(Hgrid)*VOL_SCALE, maximum(Hgrid)*VOL_SCALE)
    @printf(io,"spearman_ckill_vs_static=%.4f\n", rho)
    @printf(io,"median_suppression_ratio=%.4f\nmax_suppression_ratio=%.4f\n", median(frac_supp),maximum(frac_supp))
end
JLD2.save(joinpath(OUT,"immune_grids.jld2"), "Vgrid",collect(Vgrid),"Igrid",collect(Igrid),
          "Rgrid",Rgrid,"Cgrid",Cgrid,"Hgrid",Hgrid,"R_vs_I",R_vs_I,"R_vs_V",R_vs_V)

println("\n===== IMMUNE INTERPRETATION SUMMARY (from ", basename(CKPT), ") =====")
print(read(joinpath(OUT,"immune_summary.txt"), String))
println("===== DONE =====")
