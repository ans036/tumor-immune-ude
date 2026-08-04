#!/usr/bin/env julia
# Genuine attempt to retrain Phase-2 to a higher global R^2 (target ~0.99) with the
# ORIGINAL tolerance (1e-8) and more iterations. If R^2 >= 0.985 we re-render
# global_fit at big fonts; otherwise we keep the original (per user instruction).

using JLD2, Statistics, Printf
include(joinpath(@__DIR__, "FutureWorks.jl"))
using Plots, Optimization, OptimizationOptimisers, OptimizationOptimJL, SciMLSensitivity
default(dpi=200, size=(1200,900), titlefontsize=30, guidefontsize=26, tickfontsize=22, legendfontsize=20)

FIGDIR = joinpath(@__DIR__, "..", "paper", "figures")
CKPT   = joinpath(@__DIR__, "..", "results", "phase2_hiR2.jld2")
PROG   = joinpath(@__DIR__, "..", "results", "globalfit_progress.txt")

Random.seed!(42)
df   = load_dynamic_data(joinpath(@__DIR__,"..","Data","tumor_time_to_event_data.csv"))
dfs  = load_static_data(joinpath(@__DIR__,"..","Data","tumor_volume_vs_Im_cells_rate.csv"))
groups,(tmin,tmax) = process_groups(df)

prog = open(PROG,"w"); logp(m)=(println(prog,m);flush(prog);println(m))

θ, re = initialize_parameters()
if isfile(CKPT)
    θ = JLD2.load(CKPT,"theta")
else
    losses=Float64[]; best=Ref(Inf); bestθ=Ref(copy(θ)); noimp=Ref(0)
    lf=(p,_)->compute_loss(p,groups,dfs,re,tmin,tmax; abstol=1e-8, reltol=1e-8)
    cb=function(s,l)
        push!(losses,l)
        if l<best[]; best[]=l; bestθ[]=copy(s.u); noimp[]=0; else; noimp[]+=1; end
        length(losses)%50==0 && logp(@sprintf("AdamW %d: loss=%.6f best=%.6f", length(losses), l, best[]))
        return noimp[]>=300 || length(losses)>=1200
    end
    of=OptimizationFunction(lf,AutoZygote())
    logp("AdamW (cap 1200, tol 1e-8)...")
    try solve(OptimizationProblem(of,θ),Optimisers.AdamW(1e-3);callback=cb,maxiters=1200) catch e; logp("adamw: $e") end
    JLD2.save(CKPT,"theta",bestθ[],"losses",losses); θ=bestθ[]
    logp("L-BFGS (cap 300)...")
    lbi=Ref(0); lbb=Ref(best[])
    cb2=function(s,l); lbi[]+=1; if l<lbb[]; lbb[]=l; bestθ[]=copy(s.u); JLD2.save(CKPT,"theta",bestθ[],"losses",losses); end; lbi[]%10==0&&logp(@sprintf("LBFGS %d: loss=%.6f",lbi[],lbb[])); return false; end
    try solve(OptimizationProblem(of,bestθ[]),OptimizationOptimJL.LBFGS();callback=cb2,maxiters=300) catch e; logp("lbfgs: $e") end
    θ=bestθ[]; JLD2.save(CKPT,"theta",θ,"losses",losses)
end

# global R^2
obs=Float64[]; pred=Float64[]
for (i,g) in enumerate(groups)
    tp,v=predict_group(g,i,θ,groups,re,tmin,tmax;dense_time_points=100)
    any(isnan,v)&&continue
    itp=linear_interpolation(tp,v;extrapolation_bc=Line())
    append!(obs,g.volumes.*VOL_SCALE); append!(pred,itp.(g.times).*VOL_SCALE)
end
R2=1-sum((pred.-obs).^2)/sum((obs.-mean(obs)).^2)
rmse=sqrt(mean((pred.-obs).^2)); mae=mean(abs.(pred.-obs))
logp(@sprintf("ATTEMPT RESULT: global R2 = %.4f (rmse=%.1f mae=%.1f)", R2, rmse, mae))

if R2 >= 0.985
    mx=max(maximum(obs),maximum(pred))
    p=scatter(obs,pred,xlabel="Observed Volume (mm³)",ylabel="Predicted Volume (mm³)",title="Global Model Fit",
              label="Groups",color=:steelblue,ms=7,alpha=0.7,legend=:bottomright,
              left_margin=8Plots.mm,bottom_margin=7Plots.mm)
    plot!(p,[0,mx],[0,mx],color=:red,ls=:dash,lw=3,label="Perfect fit")
    annotate!(p,0.05mx,0.9mx,text(@sprintf("R² = %.3f",R2),:left,24,:navy))
    savefig(p,joinpath(FIGDIR,"global_fit.png"))
    logp("R2>=0.985 -> global_fit re-rendered at big fonts")
else
    logp("R2<0.985 -> KEEPING ORIGINAL global_fit")
end
close(prog)
println("GLOBALFIT ATTEMPT DONE")
