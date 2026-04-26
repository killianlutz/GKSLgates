using Pkg
Pkg.activate("./")
include("../src/QGateDescent.jl")
using .QGateDescent
using Random: default_rng, seed!
import .QGateDescent as qgd

using JLD2: @load
using DataFrames: DataFrame
using CSV
using CairoMakie

begin
    save_dir = "./applications/sims/"
    models = readdir(save_dir)[begin+1:end]
end

foreach(models) do model
    save_dir = "./applications/sims/"*model*"/"
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

    # compare estimates to numerical results
    if qs.dim == 2
        nexp_Hgen = nothing # use generic qubit case
    else
        nexp_Hgen = 3 # the gate we chose is known to be H_3-generated
    end
    bounds = aprioribounds!(var, qs; nexp_Hgen);
    J_val, T_val = verifybounds(var, sap, bounds; dtmax=1e-5*var.T); # validate with independent solver
    results = DataFrame(
        J_upper=bounds.Jup, 
        J_validated=J_val, 
        J_lower=bounds.Jlow, 
        T_upper=bounds.Tup, 
        T_validated=T_val, 
        T_lower=bounds.Tlow
        )


    CSV.write(save_dir*"results.csv", results)
    foreach(zip(fns, figs)) do (fn, fig)
        CairoMakie.save(fn, fig)
    end
end