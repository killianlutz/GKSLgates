using Pkg
Pkg.activate("./")
# written in Julia 1.9.4
include("../src/modules/QGateDescent.jl")
using .QGateDescent
using Random: default_rng, seed!
using JLD2: @save
import .QGateDescent as qgd

include("./models/qudits.jl")
save_dir = "./applications/sims/"

# qubit
dim = 2
gate = QFT
noises = [
    dephasing
    ]
models = [
    "qft_dephasing"
    ]

foreach(noises, models) do noise, model
    include("./initialize.jl")

    model = "d$(dim)_"*model
    qs = noise(dim ; F, C, M)
    target_gate = gate(dim, C)

    ###
    target!(var, target_gate)
    bounds = aprioribounds!(var, qs);

    ### homotopy
    η = reverse(range(0.0, 0.9, 10))
    sap, Tη = homotopy_descent!(var, qs, η; tolerances, verbose=true);
    verifybounds(var, sap, bounds)
    rescale!(var)
    for _ in 1:10
        sap = gradient_descent!(var, qs; tolerances, keep_optimizing=true, verbose=true);
        Tη[end] = var.T
    end
    verifybounds(var, sap, bounds)

    ## save
    dirn = save_dir*model*"/"
    isdir(dirn) ? nothing : mkdir(dirn)
    filename = dirn*model*".jld2"
    @save filename var sap qs Tη
end


# qutrit
dim = 3
gate = (dim, C) -> begin
    ctrlH = vcat([qgd.controlhamiltonian(dim, k, C) for k in 1:dim-1]...) 
    ctrlH = Matrix{C}.(ctrlH)

    A1 = exp(-im*ctrlH[1])
    A2 = exp(-im*0.25π*ctrlH[4])
    A3 = exp(im*0.5π*ctrlH[2])
    A3*A2*A1
end
noise = dephasing
model = "nexp3_dephasing"

begin    
    include("./initialize.jl")

    model = "d$(dim)_"*model
    qs = noise(dim ; F, C, M)
    target_gate = gate(dim, C)

    ###
    target!(var, target_gate)
    bounds = aprioribounds!(var, qs; nexp_Hgen=3);

    ### homotopy
    η = reverse(range(0.0, 0.9, 10))
    sap, Tη = homotopy_descent!(var, qs, η; tolerances, verbose=true);
    verifybounds(var, sap, bounds)
    rescale!(var)
    for _ in 1:10
        sap = gradient_descent!(var, qs; tolerances, keep_optimizing=true, verbose=true);
        Tη[end] = var.T
    end
    verifybounds(var, sap, bounds)

    ## save
    dirn = save_dir*model*"/"
    isdir(dirn) ? nothing : mkdir(dirn)
    filename = dirn*model*".jld2"
    @save filename var sap qs Tη
end