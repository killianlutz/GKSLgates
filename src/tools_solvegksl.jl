" Vector-field defining GKSL ode in the autonomous case (time-independent), where the parameters 
p = L::AbstractMatrix s.t. ̇x(t) = L x(t) "
function autonomousgksl!(dx, x, p, t)
    mul!(dx, p, x)
    nothing
end

" Vector-field defining GKSL ode in the NON-autonomous case, where the parameters 
p = (L::Lindbladians, z=(c!, u)::Tuple{Function,AbstractVector}) with signature c!(y, t) "
function gksl!(dx, x, p, t)
    L, (c!, u) = p
    A, B = L.Lfree, L.Lcontrol

    c!(u, t) # control vector (time-dependent part)
    mul!(dx, A, x)
    for i in eachindex(B, u)
        mul!(dx, B[i], x, u[i], 1)
    end
    nothing
end

" Solves Cauchy prob. for GKSL equation starting from initial data x0 (density operators must be vectorized) over tspan and applying eventually a control c!(y, t) 
which internally creates, updates in-place y and uses y to compute the vector-field at time t "
function simulate_gksl(qs::QuantumSystem, x0::AbstractVecOrMat, tspan::NTuple{2,Any}; ctrl::Union{Function,Nothing}=nothing, solver=RK4(), kwargs=(;))
    L = lindbladians(qs)
    simulate_gksl(L, x0, tspan; ctrl, solver, kwargs)
end

function simulate_gksl(L::Lindbladians{M}, x0::AbstractVecOrMat, tspan::NTuple{2,Any}; ctrl::Union{Function,Nothing}=nothing, solver=RK4(), kwargs=(;)) where {C<:Complex,M<:AbstractMatrix{C}}
    if isnothing(ctrl) # autonomous and control-less GKSL
        p = L.Lfree #+ sum(L.Lcontrol)
        prob = ODEProblem(autonomousgksl!, x0, tspan, p)
    else
        u = zeros(real(C), length(L.Lcontrol))
        p = (L, (ctrl, u))
        prob = ODEProblem(gksl!, x0, tspan, p)
    end
    
    sol = solve(prob, solver; kwargs...)
    return (sol, prob, L)
end