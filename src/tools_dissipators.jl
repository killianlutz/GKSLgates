# NB: Possible improvement of gate storage and kronecker products with LuxurySparse.jl package!

" Particular h_k term appearing in Environment/Dissipative operator L_D "
function raise_lower_operators(dimhilbert::Integer, C::DataType)
    d = dimhilbert
    v = ones(C, d - 1) / sqrt(d - 1)       # normalize
    Jminus = spdiagm(-1 => v)              # lowering operator J_−
    Jplus  = spdiagm(+1 => v)              # raising  operator J_+
    return (Jplus, Jminus)
end

" Random term h_k (with unit euclidean norm) in decomposition of Environment/Dissipative operator L_D "
function dissipator(dimhilbert::Integer, T::DataType; rng::Union{<:AbstractRNG,Nothing}=nothing, pnz::Real=0.01)
    # pnz = probability non zero coeff, i.e. mean number of non zero coeffs
    d = dimhilbert
    isnothing(rng) ? A = sprand(T, d, d, pnz) : A = sprand(rng, T, d, d, pnz)
    A / (norm(A) + eps(real(T)))
end

" Random hermitian matrix "
function random_hermitian(dimhilbert::Integer, C::DataType; rng::Union{<:AbstractRNG,Nothing}=nothing)
    RNG = isnothing(rng) ? Random.default_rng(234879084) : rng
    x = randn(RNG, C, dimhilbert, dimhilbert)
    return (x + x')/2
end

" Random unitary matrix using exponentiation of -iH with H self-adjoint "
function random_gate(dimhilbert::Integer, C::DataType; rng::Union{<:AbstractRNG,Nothing}=nothing)
    exp(-im * random_hermitian(dimhilbert, C; rng))
end

" Random H_0 "
function freehamiltonian(dimhilbert::Integer, C::DataType; rng::Union{<:AbstractRNG,Nothing}=nothing)
    d = dimhilbert
    isnothing(rng) ? ω = collect(C, 1:d) : ω = rand(rng, C, d) .* d
    spdiagm(0 => ω / norm(ω))
end

" Control H_j's "
function controlhamiltonian(dimhilbert::Integer, level::Integer, T::DataType)
    d, k = dimhilbert, level
    # Control hamiltonian correpsonding to ↑↓ transition from level k -> k+1, 1 ≤ k ≤ d-1
    Xk_kp1 = sparse([k, k+1], [k+1, k], [one(T), one(T)], d, d)
    Yk_kp1 = sparse([k, k+1], [k+1, k], [-im * one(T), im * one(T)], d, d)
    return [Xk_kp1, Yk_kp1]
end

" Stochastic Gate Synthesis problem: returns both the QuantumSystem and VariationalProblem "
function random_model(dim, guess, control_bounds; rng=Random.default_rng(2347689232))
    C = complex(eltype(guess.time_horizon))

    free_hamiltonian = QGateDescent.random_hermitian(dim, C; rng)
    control_hamiltonians = vcat([QGateDescent.controlhamiltonian(dim, k, C) for k in 1:dim-1]...)
    collapse_operators = [QGateDescent.dissipator(dim, C, rng=rng, pnz=1/dim) for _ in 1:3] 
    damping_rates = [1.0 for _ in collapse_operators]
    target_gate = exp(im*random_hermitian(dim, C; rng))

    qsp = (; nqudits=1, levels=[dim], dim=dim, ncontrols=2*(dim - 1))
    varp = (; ubounds=control_bounds, Q=target_gate, η=0.01, nt=length(guess.control), ngrad=10_000, maxStepSizeGR=1.0, maxIterGR=200)                             

    qs = QuantumSystem{F,C,M}(qsp..., free_hamiltonian, control_hamiltonians, collapse_operators, damping_rates)
    var = VariationalProblem{F,V,C,M}(guess.control, guess.time_horizon, varp...; RNG=rng)

    return (qs, var)
end