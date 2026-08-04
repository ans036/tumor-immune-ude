#!/usr/bin/env julia
# Exact parameter-count extraction for both UDE variants.
# Reuses the *actual* network-construction code from the pipelines so the
# reported counts are authoritative (not hand-derived).

using Flux
using Flux: Chain, Dense, destructure, f64, selu, swish, tanh, sigmoid, LayerNorm, Parallel
using Functors

# ---- Phase 2 (structured-parameter UDE) network, verbatim from FutureWorks.jl ----
struct ResBlock{F}
    dense1::Dense
    dense2::Dense
    activation::F
end
@functor ResBlock
ResBlock(dim::Int; activation=selu) = ResBlock(Dense(dim, dim), Dense(dim, dim), activation)
(b::ResBlock)(x) = b.activation.(x .+ b.dense2(b.activation.(b.dense1(x))))

function create_dynamics_network()
    Chain(
        Dense(3, 32, selu),
        ResBlock(32, activation=selu),
        ResBlock(32, activation=selu),
        Parallel(
            tuple,
            Chain(Dense(32, 16, selu), Dense(16, 2)),
            Chain(Dense(32, 16, selu), Dense(16, 2))
        )
    ) |> f64
end

# ---- Phase 1 (dual-network hybrid UDE) networks, verbatim from JuliaconSubmission.jl ----
struct EnhancedResBlock{F}
    dense1::Dense
    dense2::Dense
    norm1::LayerNorm
    norm2::LayerNorm
    activation::F
    dropout_rate::Float64
end
@functor EnhancedResBlock
EnhancedResBlock(dim::Int; activation=swish, dropout_rate=0.1) =
    EnhancedResBlock(Dense(dim, dim), Dense(dim, dim), LayerNorm(dim), LayerNorm(dim), activation, dropout_rate)

create_enhanced_immune_network() = Chain(
    Dense(2, 32, swish), EnhancedResBlock(32), EnhancedResBlock(32),
    Dense(32, 16, swish), Dense(16, 1, sigmoid)) |> f64

create_enhanced_correction_network() = Chain(
    Dense(2, 32, swish), EnhancedResBlock(32), EnhancedResBlock(32),
    Dense(32, 16, swish), Dense(16, 1, tanh)) |> f64

nparams(nn) = length(Flux.destructure(nn)[1])

# ---- Phase 2 ----
dyn = create_dynamics_network()
p2 = nparams(dyn)
println("PHASE2_dynamics_network = ", p2)

# ---- Phase 1 ----
imm = create_enhanced_immune_network()
cor = create_enhanced_correction_network()
p1_imm = nparams(imm)
p1_cor = nparams(cor)
println("PHASE1_immune_network = ", p1_imm)
println("PHASE1_correction_network = ", p1_cor)
println("PHASE1_networks_total = ", p1_imm + p1_cor)

# Phase 1 also has 3 per-group scalar params (log r, log K, log alpha).
# There are 13 groups in the full dataset (from results dirs: C2-T1..T4? etc).
# Report both the network-only and the per-group additive component.
for ngroups in (12, 13)
    println("PHASE1_total_with_$(ngroups)_groups = ", p1_imm + p1_cor + 3*ngroups)
end

# Phase 2 shares one network across all groups (context enters via NN inputs),
# so its parameter count is independent of the number of groups.
println("PHASE2_total = ", p2, "  (group-independent; context via NN inputs)")
