begin
    # Seeding
    rng = default_rng()
    seed!(rng, 2768342643894)
    # Data type choices
    F, C = Float64, ComplexF64 # number types
    V, M = Vector{F}, Matrix{C} # array types
end;

begin
    control_guess = [0.01*ones(F, 2*(dim - 1)) for _ in 1:200]
    time_horizon_guess = 1.0
    control_bounds = (-1.0, 1.0)    

    ubounds = control_bounds
    Q = zeros(C, dim, dim)
    nt = length(control_guess)
    ngrad = 50_000
    maxStepSizeGR = 100.0
    maxIterGR = 200
    abstol = convert(F, 1e-6)
    reltol = convert(F, 1e-6)

    varp = (; ubounds, Q, η=0, nt, ngrad, maxStepSizeGR, maxIterGR)                             
    tolerances = (; abstol, reltol)
    var = VariationalProblem{F,V,C,M}(control_guess, time_horizon_guess, varp...; RNG=rng)
end;