begin
    # RNG seed
    rng = default_rng()
    seed!(rng, 2768342643894)
    # Data types
    F, C = Float64, ComplexF64 # number types
    V, M = Vector{F}, SparseMatrixCSC{C} # array types

    # Useful matrices for plots
    X = F(100.0)*randn(rng, F, 3, 10)
    ϱ0 = pauli_span(X)
    pauli = build_pauli(C, M)
end;

### QUBIT TARGET GATE AND GKSL GENERATOR
begin
    qudit_dim = 2
    free_hamiltonian = QGateDescent.random_hermitian(qudit_dim, C; rng)
    control_hamiltonians = [pauli.σ1, pauli.σ2] 
    collapse_operators = [QGateDescent.dissipator(qudit_dim, C; rng, pnz=1.0) for _ in 1:3]
    damping_rates = ones(length(collapse_operators))
    target_gate = QGateDescent.gate1b_hadamard(C, M)
    # target_gate = QGateDescent.random_gate(qudit_dim, C; rng)
end;

### INITIAL CONTROL GUESS AND OPTIMIZATION HYPER-PARAMETERS
begin
    ncontrols = length(control_hamiltonians)
    control_guess = [ones(F, ncontrols) for _ in 1:100] # number time points determined from length control_guess
    time_horizon_guess = 1.0
    control_bounds = (-400.0, 400.0)    

    qsp = (; nqudits=1, levels=[qudit_dim], dim=qudit_dim, ncontrols)
    varp = (; ubounds=control_bounds, Q=target_gate, η=0.0, nt=length(control_guess), ngrad=1_000, maxStepSizeGR=100.0, maxIterGR=400)                             
    tolerances = (; abstol=F(1e-10), reltol=F(1e-10))
end;