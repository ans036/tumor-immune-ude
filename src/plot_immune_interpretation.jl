#!/usr/bin/env julia
# Intuitive, publication-quality visualisations of the LEARNED immune interaction
# term of the trained Phase-2 UDE. Reads a trained checkpoint and produces:
#   (a) 3D surface of the learned killing rate R_kill(V, I)
#   (b) grounding scatter: learned c_kill vs independent static killing-rate data
#   (c) mechanistic time-decomposition (growth vs immune killing) for C4-T5
#   (d) context-dependent parameter heatmaps c(V,I), h(V,I)
# plus a combined 2x2 figure for the paper.

using JLD2, Statistics, Printf, StatsBase
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots

default(dpi=300, framestyle=:box, titlefontsize=15, guidefontsize=13,
        tickfontsize=11, legendfontsize=10, grid=true, lw=3,
        left_margin=6mm, bottom_margin=6mm, top_margin=3mm, right_margin=4mm)

CKPT   = get(ENV, "CKPT_PATH", joinpath(@__DIR__,"..","results","immune_interpretation","phase2_trained.jld2"))
FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
XDIR   = joinpath(@__DIR__, "..", "results", "immune_interpretation", "figures")
mkpath(XDIR)

df_dyn    = load_dynamic_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"))
df_static = load_static_data(joinpath(@__DIR__,"..","Data","tumor_volume_vs_Im_cells_rate.csv"))
groups,(tmin,tmax) = process_groups(df_dyn)
θ = JLD2.load(CKPT,"theta"); _, re = initialize_parameters(); net = re(θ)
@printf("plotting from %s\n", CKPT)

immune_terms(V,I,tn) = begin
    o = net([V,tn,I]); cr,hr = o[2]
    c = MIN_IMMUNE_KILLING  + (MAX_IMMUNE_KILLING - MIN_IMMUNE_KILLING) *sigmoid(cr)
    h = MIN_HALF_SATURATION + (MAX_HALF_SATURATION- MIN_HALF_SATURATION)*sigmoid(hr)
    (c, h, I>1e-3 ? (c*V*I)/(h+V) : 0.0)
end

Vs = vcat([g.volumes for g in groups]...); Is = vcat([g.immune_levels for g in groups]...)
Vlo,Vhi = quantile(Vs,0.02), quantile(Vs,0.98); Ilo,Ihi = quantile(Is,0.02), quantile(Is,0.98)
Vg = range(max(Vlo,1e-3),Vhi,length=45); Ig = range(max(Ilo,1e-3),Ihi,length=45); tmid=0.5
Vmm = collect(Vg).*VOL_SCALE; Icell = collect(Ig).*IMMUNE_SCALE
R = [immune_terms(V,I,tmid)[3] for V in Vg, I in Ig]
C = [immune_terms(V,I,tmid)[1] for V in Vg, I in Ig]
H = [immune_terms(V,I,tmid)[2].*VOL_SCALE for V in Vg, I in Ig]

# (a) 3D surface of learned killing rate
pa = surface(Icell, Vmm, R, xlabel="Immune cells", ylabel="Tumor vol (mm³)", zlabel="R_kill (day⁻¹)",
             title="(a) Learned immune killing rate  R_kill(V,I)", color=:viridis, size=(720,560), camera=(35,30))
savefig(pa, joinpath(XDIR,"immune_surface.png"))

# (a2) heatmap version (often clearer in print)
ph = heatmap(Icell, Vmm, R, xlabel="Immune cell count", ylabel="Tumor volume (mm³)",
             title="(a) Learned killing rate R_kill(V,I)", color=:viridis, size=(560,460), colorbar_title="  R_kill")
savefig(ph, joinpath(XDIR,"immune_heatmap.png"))

# (b) grounding scatter: learned c_kill vs measured static killing rate
med_I = median(Is)
cpred = [immune_terms(v, med_I, tmid)[1] for v in df_static.VolumeNorm]
rho = corspearman(df_static.Im_cells_rate, cpred)
pb = scatter(df_static.Im_cells_rate, cpred, xlabel="Measured killing rate (static data)",
             ylabel="Learned c_kill (day⁻¹)", title="(b) Grounding in independent data",
             label="39 static measurements", color=:darkorange, ms=6, alpha=0.8, size=(560,460), legend=:bottomright)
annotate!(pb, minimum(df_static.Im_cells_rate)+0.02*(maximum(df_static.Im_cells_rate)-minimum(df_static.Im_cells_rate)),
          maximum(cpred), text(@sprintf("Spearman ρ = %.2f", rho), :left, 13, :navy))
savefig(pb, joinpath(XDIR,"immune_grounding.png"))

# (c) mechanistic time-decomposition for C4-T5
gi = findfirst(g->g.id==("C4","T5"), groups); g5 = groups[gi]
t_pred, v_pred = predict_group(g5, gi, θ, groups, re, tmin, tmax; dense_time_points=120)
gomp = Float64[]; kill = Float64[]
for (t,V) in zip(t_pred, v_pred)
    Vp=max(V,1e-8); tn=(t-tmin)/(tmax-tmin+eps()); I=g5.immune_interp(t)
    o=net([Vp,tn,I]); r=MIN_GROWTH_RATE+(MAX_GROWTH_RATE-MIN_GROWTH_RATE)*sigmoid(o[1][1])
    K=max(g5.max_vol*(MIN_CARRYING_CAPACITY_FACTOR+(MAX_CARRYING_CAPACITY_FACTOR-MIN_CARRYING_CAPACITY_FACTOR)*sigmoid(o[1][2])),Vp*1.01)
    push!(gomp, (r*Vp*log(K/Vp))*VOL_SCALE)
    push!(kill, immune_terms(Vp,I,tn)[3]*VOL_SCALE)
end
pc = plot(t_pred, gomp, label="Gompertz growth (+)", color=:seagreen, fillrange=0, fillalpha=0.25,
          xlabel="Time (days)", ylabel="dV/dt (mm³/day)", title="(c) Mechanistic decomposition  (C4-T5)",
          size=(560,460), legend=:topright)
plot!(pc, t_pred, -kill, label="Immune killing (−)", color=:crimson, fillrange=0, fillalpha=0.25)
plot!(pc, t_pred, gomp .- kill, label="Net dV/dt", color=:navy, ls=:dash, lw=3)
hline!(pc, [0], color=:black, ls=:dot, lw=1, label="")
savefig(pc, joinpath(XDIR,"immune_decomposition.png"))

# (d) context-dependent parameter heatmaps
pd1 = heatmap(Icell, Vmm, C, xlabel="Immune cell count", ylabel="Tumor volume (mm³)",
              title="(d) Learned killing coeff. c(V,I)", color=:magma, size=(560,460), colorbar_title="  c (day⁻¹)")
savefig(pd1, joinpath(XDIR,"immune_c_field.png"))
pd2 = heatmap(Icell, Vmm, H, xlabel="Immune cell count", ylabel="Tumor volume (mm³)",
              title="Learned half-saturation h(V,I)", color=:cividis, size=(560,460), colorbar_title="  h (mm³)")
savefig(pd2, joinpath(XDIR,"immune_h_field.png"))

# (e) learned immune parameters over time, overlaid on the growth trajectory (C4-T5)
ck_t=Float64[]; h_t=Float64[]; R_t=Float64[]
for (t,V) in zip(t_pred, v_pred)
    Vp=max(V,1e-8); tn=(t-tmin)/(tmax-tmin+eps()); I=g5.immune_interp(t)
    c,h,Rk = immune_terms(Vp,I,tn); push!(ck_t,c); push!(h_t,h*VOL_SCALE); push!(R_t,Rk*VOL_SCALE)
end
nrm(x) = (x .- minimum(x)) ./ (maximum(x)-minimum(x)+eps())
pe = plot(t_pred, v_pred.*VOL_SCALE, label="Tumor volume V(t)", color=:black, lw=4, fillrange=0, fillalpha=0.08,
          xlabel="Time (days)", ylabel="Tumor volume (mm³)", title="(e) Immune parameters vs growth trajectory",
          size=(600,460), legend=:topleft)
scatter!(pe, g5.times, g5.volumes.*VOL_SCALE, color=:black, ms=5, label="Observed")
pe2 = twinx(pe); plot!(pe2, ylabel="normalized parameter (min–max)", legend=:right, ylims=(-0.05,1.15))
plot!(pe2, t_pred, nrm(ck_t), color=:crimson,  lw=3,           label="c_kill(t)")
plot!(pe2, t_pred, nrm(R_t),  color=:orange,   lw=3, ls=:dot,  label="R_kill(t)")
plot!(pe2, t_pred, nrm(h_t),  color=:seagreen, lw=3, ls=:dash, label="h_sat(t)")
vline!(pe2, [t_pred[argmax(ck_t)]], color=:crimson, alpha=0.35, lw=2, label="")
vline!(pe2, [t_pred[argmax(R_t)]],  color=:orange,  alpha=0.35, lw=2, label="")
savefig(pe, joinpath(XDIR,"immune_time_overlay.png"))
@printf("timing (C4-T5): c_kill peaks day %.1f, R_kill peaks day %.1f, h_sat peaks day %.1f\n",
        t_pred[argmax(ck_t)], t_pred[argmax(R_t)], t_pred[argmax(h_t)])

# combined 3x2 paper figure: (a) heatmap, (b) grounding, (c) decomposition, (d) c-field, (d) h-field, (e) time overlay
combined = plot(ph, pb, pc, pd1, pd2, pe, layout=(3,2), size=(1200,1400))
savefig(combined, joinpath(FIGDIR,"learned_immune_interaction.png"))
savefig(combined, joinpath(XDIR,"learned_immune_interaction.png"))

@printf("figures written. grounding rho=%.3f, R_kill range [%.4f,%.4f], c med=%.2f, h med(mm3)=%.0f\n",
        rho, minimum(R), maximum(R), median(C), median(H))
println("PLOT DONE")
