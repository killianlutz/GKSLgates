using LinearAlgebra: tr

function aprioribounds!(var, qs; nexp_Hgen=nothing)
    dim = qs.dim
    L = QGateDescent.lindbladians(qs)
    drift = L.Lfree

    if dim == 2
        m = 2*(dim - 1)
        Ωmin = 48*sqrt(2)*π
        δtimesΩ = 12*π*sqrt(2)
        offset = 4/10
        controls = L.Lcontrol
    else
        m = 2*(dim - 1) + 1
        ν = νfun(dim; nexp_Hgen) # to be updated for general control Hamiltonians
        Ωmin = 2π*ν*dim^3
        δtimesΩ = 2π*ν*dim
        offset = 1/dim
        controls = [L.Lcontrol; [4*one(drift)/dim]]
    end

    p = (; dim, m, Ωmin, δtimesΩ, offset, controls, drift)

    return aprioribounds(var, p)
end

function aprioribounds(var, p)
    # data
    drift_norm = sqrt(sum(abs2, p.drift))
    control_norm = sqrt(sum(sum(abs2, x) for x in p.controls))
    C2 = max(control_norm, drift_norm)
    τ = real(-1/tr(p.drift))

    # check assumption control to drift ratio Ω = M/|D|
    Ω = check_assumption!(var, p.Ωmin, drift_norm)
    M = Ω*drift_norm
    δ = p.δtimesΩ/Ω

    # bounds
    dist_to_identity = sqrt(sum(abs2, var.vecQ - one(var.vecQ)))
    ε = dist_to_identity - p.offset

    return estimates(p.dim, p.m, M, δ, ε, C2, τ)
end

function estimates(d, m, M, δ, ε, C2, τ)
    mε = min(ε, sqrt(ε))
    ε_tilde = ε/(1 + mε*exp(mε))

    # LOWER bound T0
    num = ε_tilde
    den = C2*(1 + sqrt(m)*M)
    Tlow = num/den
    # UPPER bound T0
    Tup = -τ*log(1 - ψ(δ, d))
    # LOWER bound J0
    num = δ^2 * (1 - exp(-Tlow/τ))^2
    den = 2 * ψ(δ, d)^2
    Jlow = num/den;
    # UPPER bound J0
    Jup = δ^2 / 2;

    return (; Jup, Jlow, Tup, Tlow)
end

function νfun(d; nexp_Hgen=nothing)
    if isnothing(nexp_Hgen)
        δ = log(2)/4
        c = 2 - sqrt(2)
        rhat = sin(π/(2d))/d
        z0 = 2δ/3
        z1 = sqrt(2)*rhat/192
        z2 = 6*sqrt(c)*rhat/(δ + δ/c)
        
        r = z0*min(1, z1*min(1, z2))
        q = ceil(6π*sqrt(d)/r)
        l = 3*d - 2 + d*(d - 1)*(2*d - 1)/3
        ν = q*l
    else
        ν = nexp_Hgen
    end

    return ν
end

function check_assumption!(var, Ωmin, drift_norm)
    Ω = last(var.ubounds)/drift_norm

    if Ωmin > Ω
        @show Ωmin ≤ Ω
        println("maximal control amplitude updated")
        u_min = Ωmin*drift_norm
        var.ubounds = (-u_min, u_min)
    end

    Ω = last(var.ubounds)/drift_norm
    @show Ωmin ≤ Ω
    return Ω
end

function verifybounds(var, sap, bounds; dtmax=1e-3)
    Jup, Jlow, Tup, Tlow = bounds
    x0 = first(QGateDescent.state(sap))
    results = QGateDescent.validate(var, sap, x0; interpolate=:constant, kwargs=(; dtmax));
    
    J = sum(abs2, results.sol.u[end] - var.vecQ)/2
    T = var.T

    isJ = (Jlow < J) * (J <= Jup)
    isT = (Tlow < T) * (T <= Tup)

    @show Jup J Jlow
    @show Tup T Tlow 
    return (; isJ, isT)
end

function ψ(x, d)
    x*(d + x)^(d^2 -1) / d^(d^2 - 2)
end