" Stores Lindbladian matrix corresponding to L_0 + L_D (free and environment term) and control Lindbladians L_Cj "
struct Lindbladians{M<:AbstractMatrix{<:Number}}
    # matrix I ⊗ (-iH - \sum_k γk hk† hk) + (iH - \sum_k γk hk† hk)^T ⊗ I + \sum_k 2 *γk hk†^T ⊗ hk of the vector GKSL ode
    Lfree::M # includes dissipative terms
    # each control matrix I ⊗ (-iHk) + (iH)^T ⊗ I associated to the column-major reshaped GKSL ode
    Lcontrol::Vector{M}
end

" Qudits quantum system details and GKSL equation parameters "
@with_kw struct QuantumSystem{F<:AbstractFloat,C<:Complex,M<:AbstractMatrix{C}}
    nqudits::Int64
    levels::Vector{Int64}  # [2, 2] for two 2-level qudits
    dim::Int64             # composite system dimension
    ncontrols::Int64       # number of control hamiltonians
    H::M                   # free environment renormalized hamiltonian
    Hc::Vector{M}          # control hamiltonians
    h::Vector{M}           # transfer operators h_k
    γ::Vector{F}           # non negative reals
end

" Builds Lindbladians, i.e. matrices (super-operators) associated to GKSL equation parameters "
function lindbladians(qs::QuantumSystem{F,C,M}) where {F<:AbstractFloat,C<:Complex,M<:AbstractMatrix{C}}
    dim2 = qs.dim^2
    Lf  = oftype(qs.H, spzeros(C, dim2, dim2)) # free component of GKSL generator
    tmp = zero(qs.H)
    eye = one(qs.H)

    for (hk, γk) in zip(qs.h, qs.γ)
        tmp .+= -γk * hk' * hk             # \sum_k    γk hk† * hk
        Lf .+= kron(2 * γk * conj(hk), hk) # \sum_k 2 *γk hk†^T ⊗ hk
    end
    Lf .+= kron(eye, -im * qs.H + tmp) .+ kron(transpose(im * qs.H) + transpose(tmp), eye)

    f(M) = kron(eye, -im * M) .+ kron(transpose(im * M), eye)
    Lc = map(f, qs.Hc) # controlled component of GKSL generator

    Lindbladians{M}(Lf, Lc)
end

function adjoint_lindbladians(L::Lindbladians{M}) where M<:AbstractMatrix{<:Number}
    aLf = M(adjoint(L.Lfree))
    aLc = M.(adjoint.(L.Lcontrol))
    Lindbladians{M}(aLf, aLc)
end

function isunital(qs::QuantumSystem, L::Lindbladians{M}) where M<:AbstractMatrix{<:Number}
    # checking unitality of L amounts to that of dissipative part L_D
    l, H = L.Lfree, qs.H
    x = one(H) # identity matrix
    y = reshape(l * vec(x), size(H))
    return (isapprox(zero(H), y), y)  
end

function decoherence(qs::QuantumSystem)
    L = lindbladians(qs)
    free_lindbladian = L.Lfree
    λ = abs(tr(free_lindbladian))
    τ = isapprox(λ, 0) ? oftype(λ, Inf) : 1/λ

    return τ
end