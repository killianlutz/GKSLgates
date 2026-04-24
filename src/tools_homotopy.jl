function homotopy_descent!(var::VariationalProblem, qs::QuantumSystem, penalties::AbstractArray{<:Real}; 
    ngrads::AbstractVector{<:Int}=fill(var.gradp.ngrad, size(penalties)), 
    grmaxsteps::AbstractVector{<:Real}=fill(var.gradp.grmaxstep, size(penalties)), 
    verbose=false,
    tolerances=nothing
    )

    nη = length(penalties)
    gate_times = [var.T for _ in 1:nη]
    sap = nothing

    for i in 1:nη 
        η, ngrad, step = penalties[i], ngrads[i], grmaxsteps[i]
        keep_optimizing = i == 1 ? false : true

        @show η
        grad_parameters!(var; ngrad, grmaxstep=step)
        penalty!(var, η)

        sap = gradient_descent!(var, qs; tolerances, keep_optimizing, verbose);
        gate_times[i] = var.T
    end

    return (; sap, gate_times)
end