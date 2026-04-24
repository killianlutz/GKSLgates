using LinearAlgebra: diagm

function qudit_relaxation(d, C)
    v = ones(C, d - 1)
    diagm(+1 => v)
end

function qudit_excitation(d, C)
    v = ones(C, d - 1)
    diagm(-1 => v)
end

function qudit_dephasing(d, C)
    j = convert(C, (d - 1)/2)
    v = [j - k for k in 0:d-1] ### from -d/2 → + 1 -> d/2-1 (e.g. d=2 corresponds to σ_z/2)
    diagm(v)
end

function pauliZ(d, C)
    ω = convert(C, exp(im*2π/d))
    P = zeros(C, d, d)
    P[1, 1] = ω
    for k in 1:d-1
        P[k + 1, k + 1] = ω*P[k, k]
    end
    return P
end

function QFT(d, C)
    ω = convert(C, exp(im*2π/d))
    Ω = ones(C, d)
    for i in 1:d-1
        Ω[i+1] = ω*Ω[i]
    end

    F = ones(C, d, d)
    for j in 1:d-1
        F[:, j+1] .= F[:, j] .* Ω
    end
    return F/sqrt(d)
end

function quDit(d, collapse_operators, damping_rates; F=Float64, C=Complex{F}, M=Matrix{C})
    free_hamiltonian = pauliZ(d, C)
    control_hamiltonians = vcat([QGateDescent.controlhamiltonian(d, k, C) for k in 1:d-1]...) 

    ncontrols = length(control_hamiltonians)
    qsp = (; nqudits=1, levels=[d], dim=d, ncontrols)

    return QuantumSystem{F,C,M}(qsp..., free_hamiltonian, control_hamiltonians, collapse_operators, damping_rates)
end

function dephrelax(d; F=Float64, C=ComplexF64, M=Matrix{C})
    collapse_operators = [qudit_relaxation(d, C), qudit_dephasing(d, C)]
    damping_rates = [2, 1]

    return quDit(d, collapse_operators, damping_rates; F, C, M)
end

function dephasing(d; F=Float64, C=ComplexF64, M=Matrix{C})
    collapse_operators = [qudit_dephasing(d, C)]
    damping_rates = [1]

    return quDit(d, collapse_operators, damping_rates; F, C, M)

end

function excitrelax(d; F=Float64, C=ComplexF64, M=Matrix{C})
    collapse_operators = [qudit_relaxation(d, C), qudit_excitation(d, C)]
    damping_rates = [2, 1]

    return quDit(d, collapse_operators, damping_rates; F, C, M)
end