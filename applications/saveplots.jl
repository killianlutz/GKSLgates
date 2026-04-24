using Pkg
Pkg.activate("./")
include("../src/modules/QGateDescent.jl")
using .QGateDescent
using Random: default_rng, seed!
import .QGateDescent as qgd

using JLD2: @load
using CairoMakie
CairoMakie.activate!()

begin
    save_dir = "./applications/sims/"
    models = readdir(save_dir)[begin+1:end]
end

foreach(models) do model
    save_dir = "./scripts/applications/sims/"*model*"/"
    filename = save_dir*model*".jld2"
    @load filename var sap qs Tη

    ###
    rng = default_rng()
    seed!(rng, 4837908543)
    X = 100*randn(rng, typeof(var.T), 3, 5);
    ϱ0 = pauli_span(X);
    
    ### plots
    fns = save_dir.*[
            "descent.pdf",
        ]
    figs = [
        showresults(var, sap),
        ]

    if qs.dim == 2 # bloch ball
        push!(fns, save_dir*"noctrl_vs_ctrl.png")
        push!(figs, showbloch_orbits(var, sap, qs, ϱ0; with_ctrl=true))
    end

    foreach(zip(fns, figs)) do (fn, fig)
        CairoMakie.save(fn, fig)
    end
end