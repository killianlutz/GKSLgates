# NB: Possible improvement of gate storage and kronecker products with LuxurySparse.jl package!

" Given matrix Q, returns matrix associated to A ↦ QAQ* "
function vecgate(U::AbstractArray{<:Number})
    # maps a quantum gate U acting on qudits to the corresponding 
    # gate vecU = (conj(U) ⊗ U) acting on density operators ϱ
    kron(conj(U), U)
end

" Computes a special unitary gate of dimension dimhilbert using the exponential of -iH where H hermitian traceless "
function randomgate(dimhilbert::Int, M::Union{DataType,UnionAll}; rng::Union{TaskLocalRNG,Nothing}=nothing)
    d = dimhilbert; C = eltype(M)
    # random matrix
    isnothing(rng) ? H = randn(C, d, d) : H = randn(rng, C, d, d)
    # hermitian
    H = (H + H') / 2
    # traceless
    H = H - dot(H, I)/d * I
    # gate
    Q = exp(-im * H)

    return convert(M, Q)
end

function gate1b_not(C::DataType, M::Union{DataType,UnionAll})
    return M([zero(C) 1; 1 0])
end

function gate1b_hadamard(C::DataType, M::Union{DataType,UnionAll})
    return M([one(C) 1; 1 -1] / C(sqrt(2)))
end

function gate1b_phaseshift(ϕ::Real, C::DataType, M::Union{DataType,UnionAll})
    return M([one(C) 0; 0 exp(im * C(ϕ))])
end

function gate1b_rx(θ::Real, C::DataType, M::Union{DataType,UnionAll})
    t = C(0.5 * θ)
    return M(
        [cos(t) -im*sin(t);
         -im*sin(t) cos(t)]
         )
end

function gate1b_ry(θ::Real,  C::DataType, M::Union{DataType,UnionAll})
    t = C(0.5 * θ)
    return M(
        [cos(t) -sin(t);
         sin(t) cos(t)]
         )
end

function gate1b_rz(θ::Real,  C::DataType, M::Union{DataType,UnionAll})
    t = C(0.5 * θ)
    return M([exp(-im * t) 0; 0 exp(im * t)])
end

function gate2b_cu(U::AbstractArray{<:Number})
    # Controlled U gate : 4x4 gate with U unitary 2x2
    G = 0*similar(U, 4, 4)
    # identity on first qbit
    G[1,1] = one(eltype(U))
    G[2,2] = one(eltype(U))
    G[3:4,3:4] .= U

    return G
end

function gate2b_swap(C::DataType, M::Union{DataType,UnionAll})
    Q = gate1b_not(C, M)
    G = 0*similar(Q, 4, 4)
    # identity on first qbit
    G[1,1] = one(C)
    G[4,4] = one(C)
    G[2:3,2:3] .= Q

    return G
end