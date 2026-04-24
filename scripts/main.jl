using Pkg
Pkg.activate("./")

# written in Julia 1.9.4
include("../src/QGateDescent.jl")
using .QGateDescent

using Random: default_rng, seed!
using SparseArrays: SparseMatrixCSC
using LinearAlgebra: I
import .QGateDescent as qgd

include("./parameters.jl")

######## SYSTEM DEFINITION & OPTIMIZATION VARIABLES
qs = QuantumSystem{F,C,M}(qsp..., free_hamiltonian, control_hamiltonians, collapse_operators, damping_rates); 
var = VariationalProblem{F,V,C,M}(control_guess, time_horizon_guess, varp...; RNG=rng);

######## SOLVE PROBLEM BY HOMOTOPY/CONTINUATION OVER THE PENALTY PARAMETER η
η = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] 
sap, Tη = homotopy_descent!(var, qs, η; tolerances, verbose=true)

######## COMPUTE THEORETICAL A PRIORI ESTIMATES AND COMPARE RESULTS
bounds = aprioribounds!(var, qs); # only suited to reproduce Figures
verifybounds(var, sap, bounds)
rescale!(var) # (T, u) -> (ϵT, u/ϵ) until constraint is satured
sap = gradient_descent!(var, qs; keep_optimizing=true, verbose=true); # keep_optimizing -> continues optimization
verifybounds(var, sap, bounds) # a priori estimates

#### POST PROCESS
metrics = accuracy_metrics(var, sap)
fig = showresults(var, sap)

#### VALIDATION OF THE RESULTS OVER A FINER GRID (INDEPENDENT ODE SOLVER)
x0 = Matrix{C}(I(qs.dim^2))
results = validate(var, sap, x0; interpolate=:constant);

# SHOULD BE APPROXIMATELY THE SAME
fidelity(qgd.finalstate(sap), var.vecQ)
fidelity(results.sol.u[end], var.vecQ)

mse(qgd.finalstate(sap), var.vecQ)
mse(results.sol.u[end], var.vecQ)

sum(abs2, qgd.finalstate(sap) - var.vecQ)/2
J_validation = sum(abs2, results.sol.u[end] - var.vecQ)/2

#### GEOMETRIC INTERPRETATION IN BLOCH BALL (SINGLE QUBIT ONLY !)
# controlled
fig = showbloch_orbits(var, sap, ϱ0; with_ctrl=false, single_figure=true)

# uncontrolled
tspan = (0.0, 5.0)
fig = showbloch_orbits(qs, ϱ0; tspan)

# both
fig = showbloch_orbits(var, sap, qs, ϱ0; with_ctrl=true)


#### SAVE RESULTS FOR LATER USE
qgd.save((var, sap), "./scripts/sims/results.jld2")