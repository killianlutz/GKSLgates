using Pkg
Pkg.activate("./")
include("../src/QGateDescent.jl")
using .QGateDescent
using Random: default_rng, seed!
import .QGateDescent as qgd

using JLD2: @load

begin
    save_dir = "./applications/sims/"
    models = readdir(save_dir)[begin+1:end]
    models = Dict(i => model for (i, model) in enumerate(models))
end

begin
    model = models[2]

    rng = default_rng()
    seed!(rng, 4837908543)
    save_dir = "./applications/sims/"*model*"/"
    filename = save_dir*model*".jld2"
    @load filename var sap qs Tη

    X = 100*randn(rng, typeof(var.T), 3, 5);
    ϱ0 = qgd.pauli_span(X);
    dtmax = 1e-5*var.T
end;

### optimization results
τ = qgd.decoherence(qs) # T2
# bounds = aprioribounds!(var, qs);
bounds = aprioribounds!(var, qs; nexp_Hgen=nothing);
verifybounds(var, sap, bounds; dtmax)
accuracy_metrics(var, sap)
fig = qgd.showresults(var, sap)

### qualitative validation
# controlled
fig = showbloch_orbits(var, sap, ϱ0; with_ctrl=false, single_figure=true)
# uncontrolled
fig = showbloch_orbits(qs, ϱ0; tspan=(0.0, 5.0))
# both
fig = qgd.showbloch_orbits(var, sap, qs, ϱ0; with_ctrl=true)

### quantitative validation
R0 = first(state(sap))
results = validate(var, sap, R0; interpolate=:constant, kwargs=(; dtmax));

1 - fidelity(qgd.finalstate(sap), var.vecQ)
1 - fidelity(results.sol.u[end], var.vecQ)

mse(qgd.finalstate(sap), var.vecQ)
mse(results.sol.u[end], var.vecQ)