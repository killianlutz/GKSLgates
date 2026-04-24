#################################################
# terminal cost
#################################################

" Terminal cost g(R(1)) in objective function J "
function g!(y::AbstractArray{T}, x::AbstractArray{T}, q::AbstractArray{T}) where {T<:Number}
    y .= x .- q # temporary array
    sum(abs2, y) / 2
end

" Gradient of terminal cost g(R(1)) "
function ∇g!(y::AbstractArray{T}, x::AbstractArray{T}, q::AbstractArray{T}) where {T<:Number}
    y .= x .- q
end

#################################################
# structs
#################################################

" Mesh data to step forward RK4 integrator "
@with_kw struct RK4Parameters{F<:AbstractFloat}
    t::Vector{F} # uniform Δt grid of (0,1)
    nt::Int64 # length(t)
    Δt::F
    Δto2::F
    Δto6::F

    function RK4Parameters{F}(t) where F<:AbstractFloat
        nt = length(t)
        Δt = t[begin + 1] - t[begin]
        Δto2 = oftype(Δt, 0.5) * Δt
        Δto6 = oftype(Δt, 1/6) * Δt
        new{F}(t, nt, Δt, Δto2, Δto6)
    end
end

" Line search and gradient parameters "
@with_kw mutable struct GradientParameters{F<:AbstractFloat}
    ngrad::Int64                         # number gradient steps
    grmaxstep::F                      # step size line search upper bound
    grmaxiter::Int64                        # step size line search max number iterations
    objvalues::Vector{F}            # values of J(T,u)
    rng::AbstractRNG
end

" State and Adjoint ODE problems information for in-place efficient computations "
@with_kw mutable struct CauchyProblem{F<:AbstractFloat,V<:Vector{F},C<:Complex,M<:AbstractMatrix{C},S<:AbstractMatrix{C}}
    f::Function   # f(dy, y, p, t)
    y0::S          # initial datum
    y::Vector{S}   # pre-allocated state evolution
    odep::Tuple{M,Vector{M},Vector{Vector{C}},C,Int64} # p in f(dy, y, p, t)
    rkp::RK4Parameters{F} # time-stepping constants and grids
end

" Optimization problem parameters and approximate solution "
@with_kw mutable struct VariationalProblem{F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    u::Vector{V}             # control vector s.t. u[k,t] * Lc[k] = kth control at time t
    T::F                                # time horizon
    ∇uJ::Vector{V}           # objective gradient wrt u
    ∇TJ::F                              # objective gradient wrt T
    Q::Matrix{C}          # logic gate to be approximated
    vecQ::Matrix{C}       # gate acting on density operators (Q')^T ⊗ Q 
    η::F                                # time horizon penalty ∈ [0,1]
    nt::Int64                   # number nodes in [0,1] time mesh
    ubounds::Tuple{S,S} where S<:F         # bounds (a,b) s.t. component-wise a ≤ u(t) ≤ b
    gradp::GradientParameters{F}

    function VariationalProblem{F,V,C,M}(u, T, ubounds, Q, η, nt, ngrad, grmaxstep, grmaxiter; RNG=nothing) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
        vecQ = vecgate(Q)
        rng = isnothing(RNG) ? Random.default_rng(2748902402) : RNG
        gradp = GradientParameters{F}(ngrad, grmaxstep, grmaxiter, Vector{F}(undef, ngrad), rng)
        new{F,V,C,M}(deepcopy(u), T, deepcopy(u), one(F), Q, vecQ, η, nt, ubounds, gradp)
    end
end

" PMP state and adjoint pair CauchyProblems "
@with_kw mutable struct StateAdjointPair{F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    L::Lindbladians{M} # vectorized GKSL ode matrices (uncontrolled and controlled parts)
    aL::Lindbladians{M} # store adjoints for performance purposes
    scp::CauchyProblem{F,V,C,M} # state Cauchy problem
    acp::CauchyProblem{F,V,C,M} # adjoint Cauchy problem

    function StateAdjointPair{F,V,C,M}(var::VariationalProblem, qs::QuantumSystem) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
        L = lindbladians(qs)
        aL = adjoint_lindbladians(L)
        new{F,V,C,M}(L, aL, state_adjoint_cauchyprobs(var, L, aL)...)
    end
end

#################################################
# functions associated to structs
#################################################

" Sets up state and adjoint cauchy problems (vector-fields, intial/terminal conditions and mesh parameters) "
function state_adjoint_cauchyprobs(var::VariationalProblem{F,V,C,M}, L::Lindbladians{M}, aL::Lindbladians{M}) where {F<:AbstractFloat,V<:Vector{F},C<:Complex,M<:AbstractMatrix{C}}
    S = Matrix{C}
    # vector fields
    issparse(L.Lfree) ? state_vectorfield = sparse_stateode_rhs! : state_vectorfield = dense_stateode_rhs!
    issparse(L.Lfree) ? adjoint_vectorfield = sparse_adjointode_rhs! : adjoint_vectorfield = dense_adjointode_rhs!
    # initial state and terminal adjoint
    s0 = S(one(C)I, size(L.Lfree))
    a1 = similar(s0)
    # solver parameters
    rkp   = RK4Parameters{F}(collect(LinRange(zero(F), one(F), var.nt)))
    odep  = ( L.Lfree,  L.Lcontrol, var.u, var.T, var.nt)
    aodep = (aL.Lfree, aL.Lcontrol, var.u, var.T, var.nt)
    # unknown function pre-allocs
    state   = [S(undef, size(L.Lfree)) for _ in 1:var.nt]
    adjoint = [S(undef, size(L.Lfree)) for _ in 1:var.nt]
    # problem definition
    sprob = CauchyProblem{F,V,C,M,S}(state_vectorfield, s0, state  , odep, rkp)
    aprob = CauchyProblem{F,V,C,M,S}(adjoint_vectorfield, a1, adjoint, aodep, rkp)

    return (sprob, aprob)
end

" Projection onto admissible set: gate time T component"
function gatetime_projection(t::Real)
    return max(t, zero(t)) # projection onto admissible time-horizon R_+
end

" Projections onto admissible set: control u component"
function control_projection(x::Union{<:AbstractArray{<:Real},<:Real}, bounds::Tuple{T,T}) where T<:Real
    return min.(max.(x, bounds[1]), bounds[2]) # projection onto [bounds[1], bounds[2]] ⊂ R
end

function control_projection!(x::Union{<:AbstractArray{<:Real},<:Real}, bounds::Tuple{T,T}) where T<:Real
    for i in eachindex(x)
        x[i] = min.(max.(x[i], bounds[1]), bounds[2])
    end
    nothing
end

" (Sparse) Vector field defining the state resolvant R(t) Cauchy problem "
function sparse_stateode_rhs!(dx, x, p, t)
    Lf, Lc, u, T, nt = p         # p = [L, Lc, u, T, nt]
    idx = current_time(nt, t)  # index current time (BACKWARD solve)
    mul!(dx, Lf, x, T, 0)     # computes in-place dx = T * Lfree * x
    for i in eachindex(Lc)
        mul!(dx, Lc[i], x, T * u[idx][i], 1)
    end
    nothing
end

" (Dense) Vector field defining the state resolvant R(t) Cauchy problem "
function dense_stateode_rhs!(dx, x, p, t)
    Lf, Lc, u, T, nt = p                                # p = [L, Lc, u, T, nt]
    C = eltype(Lf)
    idx = current_time(nt, t) # index current time : 1 + integer part of (t / dt)
    BLAS.gemm!('N', 'N', complex(T), Lf, x, zero(C), dx)             # computes in-place dx = T * Lfree * x
    for (Lck, uk) in zip(Lc, u[idx])
        BLAS.gemm!('N', 'N', complex(T * uk), Lck, x, one(C), dx) # computes in-place dx += T * uk * vecHk * x
    end
    nothing
end

" (Sparse) Vector field defining the adjoint state A(t) Cauchy problem "
function sparse_adjointode_rhs!(dx, x, p, t)
    aLf, aLc, u, T, nt = p                                # p = [L, Lc, u, T, nt]
    idx = nt - current_time(nt, t) + 1 # index current time (BACKWARD solve)
    mul!(dx, aLf, x, T, 0)     # computes in-place dx = T * Lfree† * x (pre-computed adjoint)
    for i in eachindex(aLc)
        mul!(dx, aLc[i], x, T * u[idx][i], 1) # computes in-place dx += T * uk * Lck† * x
    end
    nothing
end

" (Dense) Vector field defining the adjoint state A(t) Cauchy problem "
function dense_adjointode_rhs!(dx, x, p, t)
    aLf, aLc, u, T, nt = p                                # p = [L, Lc, u, T, nt]
    C = eltype(aLf)
    idx = nt - current_time(nt, t) + 1 # index current time (BACKWARD solve)
    BLAS.gemm!('N', 'N', complex(T), aLf, x, zero(C), dx)           # computes in-place dx = T * Lfree† * x
    for (aLck, uk) in zip(aLc, u[idx])
        BLAS.gemm!('N', 'N', complex(T * uk), aLck, x, one(C), dx) # computes in-place dx += T * uk * Lck† * x
    end
    nothing
end

" RK4 integrator over whole time interval "
function rungekutta4!(f::Function, u::Vector{S}, u0::S, p::Tuple, rkp::RK4Parameters, tmps::NTuple{5,S}) where S<:AbstractMatrix{<:Number}
    x0, x1, x2, x3, x4 = tmps # pre-allocs
    t, Δts, nt = solverparameters(rkp)
    Δt, Δto2, Δto6 = Δts
    u[1] .= u0

    for i in 1:nt-1     # assumes u and t have same lengths, f(du, u, p, t)
        f(x1, u[i], p, t[i]       )
        x0 .= u[i] .+ Δto2 .* x1
        f(x2,   x0, p, t[i] + Δto2)
        x0 .= u[i] .+ Δto2 .* x2
        f(x3,   x0, p, t[i] + Δto2)
        x0 .= u[i] .+ Δt   .* x3
        f(x4,   x0, p, t[i] + Δt  )
        u[i+1] .= u[i] .+ Δto6 .* (x1 .+ 2 .* (x2 .+ x3) .+ x4)
    end
    return (t, u)
end

" Solve CauchyProblem in-place over whole time-interval using pre-allocated arrays "
function solve!(cp::CauchyProblem{F,V,C,M,S}, tmps::NTuple{5,S}) where {F<:AbstractFloat,V<:Vector{F},C<:Complex,M<:AbstractMatrix{C},S<:AbstractArray{C}}
    rungekutta4!(cp.f, cp.y, cp.y0, cp.odep, cp.rkp, tmps)
end

" Allocating version "
function solve!(cp::CauchyProblem{F,V,C,M,S}) where {F<:AbstractFloat,V<:Vector{F},C<:Complex,M<:AbstractMatrix{C},S<:AbstractArray{C}}
    # pre-allocs
    PA0::S = similar(cp.y0)
    PA1::S = similar(cp.y0)
    PA2::S = similar(cp.y0)
    PA3::S = similar(cp.y0)
    PA4::S = similar(cp.y0)
    rungekutta4!(cp.f, cp.y, cp.y0, cp.odep, cp.rkp, (PA0, PA1, PA2, PA3, PA4))
end

" Computes the index i ∈ [1, n] s.t. i/n = floor(t) + 1 "
function current_time(n_nodes, time)
    min(floor(eltype(n_nodes), n_nodes * time) + 1, n_nodes)
end

#################################################
# gradient descent auxiliary functions
#################################################
" Evaluates objective function J with in-place intermediate computations and compute its gradient wrt. (T,u) "
function objective_value!(y::AbstractArray{C}, r::AbstractArray{C}, q::AbstractArray{C}, T::F, η::F) where {F<:AbstractFloat,C<:Complex}
    η * T + (1 - η) * g!(y, r, q)
end

function objective_value!(var::VariationalProblem, sap::StateAdjointPair, iter::Integer, tmp::AbstractArray{<:Number})
    set_objvalues!(var, iter, objective_value!(tmp, last(sap.scp.y), var.vecQ, var.T, var.η))
end

" Evaluates gradient ∇J of objective function wrt. (T,u) with in-place intermediate computations "
function objective_gradient!(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair{F,V,C,M}) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    dotp = similar(first(var.u))
    var.∇TJ = zero(var.∇TJ)
    α = (1 - var.η) * var.T
    β = (1 - var.η) / var.nt # divide by nt because Riemann sum
    # loop over time
    for i in eachindex(var.∇uJ)
        fill!(var.∇uJ[i], 0)
        # sum the dot products A(t)[:,j]* ⋅ (L R(t)[:,j]) for j in axes(R, 2)
        var.∇TJ += real( dot(sap.acp.y[i], sap.L.Lfree, sap.scp.y[i]) )
        # same but looping over control components
        map!(L -> real( dot(sap.acp.y[i], L, sap.scp.y[i]) ), dotp, sap.L.Lcontrol)
        axpy!(α, dotp, var.∇uJ[i]) # var.∇uJ[i] .+= α .* dotp
        # the following amounts to var.∇TJ += sum(var.u[i] .* dotp), but no allocations
        map!(*, dotp, var.u[i], dotp) # dotp .*= var.u[i]
        var.∇TJ += sum(dotp)
    end
    var.∇TJ *= β
    var.∇TJ += var.η
    # nothing
    
    # test dropout
    n_controls = length(var.∇uJ[begin]) 
    components = 1:n_controls
    dkeep = rand(var.gradp.rng, components, rand(var.gradp.rng, components.-1))
    foreach(var.∇uJ) do x
        x[dkeep] .= 0
    end
    # unitgradient!(var)

    nothing
end

" Threaded computation of ∇J. Efficient for large number of time nodes in mesh of interval [0, 1] "
function threaded_objective_gradient!(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair{F,V,C,M}) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    dotp = [similar(first(var.u)) for _ in 1:Threads.nthreads()]
    tmp = zeros(F, Threads.nthreads())

    α = (1 - var.η) * var.T
    β = (1 - var.η) / var.nt # divide by nt because Riemann sum
    # loop over time
    Threads.@threads for i in eachindex(var.∇uJ)
        fill!(var.∇uJ[i], 0)
        # sum the dot products A(t)[:,j]* ⋅ (L R(t)[:,j]) for j in axes(R, 2)
        tmp[Threads.threadid()] += real( dot(sap.acp.y[i], sap.L.Lfree, sap.scp.y[i]) )
        # same but looping over control components
        map!(L -> real( dot(sap.acp.y[i], L, sap.scp.y[i]) ), dotp[Threads.threadid()], sap.L.Lcontrol)
        axpy!(α, dotp[Threads.threadid()], var.∇uJ[i]) # var.∇uJ[i] .+= α .* dotp
        # the following amounts to var.∇TJ += sum(var.u[i] .* dotp), but no allocations
        map!(*, dotp[Threads.threadid()], var.u[i], dotp[Threads.threadid()]) # dotp .*= var.u[i]
        tmp[Threads.threadid()] += sum(dotp[Threads.threadid()])
    end
    var.∇TJ = sum(tmp) * β + var.η
    nothing
end

" Updates CauchyProblem initial/terminal conditions (state/adjoint) + solve + evaluate objective function gradient ∇J (Non-allocating) "
function objective_gradient!(var::VariationalProblem, sap::StateAdjointPair, tmps::NTuple{5,S}) where S<:AbstractArray{<:Number}
    # run state (forward problem)
    update_odeparams!(var, sap)
    solve!(sap.scp, tmps)
    # update in-place terminal condition of adjoint problem
    ∇g!(sap.acp.y0, last(sap.scp.y), var.vecQ)
    # run adjoint (backward problem)
    solve!(sap.acp, tmps)
    reverse!(sap.acp.y) # reverse time in-place
    # compute in-place gradients of objective function
    objective_gradient!(var, sap)
    nothing
end

" (Allocating version) Updates CauchyProblem initial/terminal conditions (state/adjoint) + solve + evaluate objective function gradient ∇J (Non-allocating) "
function _objective_gradient!(var::VariationalProblem, sap::StateAdjointPair)
    # for convenient calls to solve the GKSL equation and its adjoint
    PA0 = similar(sap.scp.y0)
    PA1 = similar(PA0)
    PA2 = similar(PA1)
    PA3 = similar(PA2)
    PA4 = similar(PA3)
    objective_gradient!(var, sap, (PA0, PA1, PA2, PA3, PA4))
end

" Performs projected gradient step in-place with step size ξ "
function gradient_descent_step!(var::VariationalProblem, minusξ::Real, cdir::String)
    if cdir === "T"                  # step T-direction only
        var.T = gatetime_projection(var.T + minusξ * var.∇TJ)
    elseif cdir === "u"              # step u-direction only
        Threads.@threads for i in eachindex(var.u)
            axpy!(minusξ, var.∇uJ[i], var.u[i])
            control_projection!(var.u[i], var.ubounds)
        end
    elseif cdir === "both"           # step both ways (T,u)
        var.T = gatetime_projection(var.T + minusξ * var.∇TJ)
        Threads.@threads for i in eachindex(var.u)
            axpy!(minusξ, var.∇uJ[i], var.u[i])
            control_projection!(var.u[i], var.ubounds)
        end
    end
    nothing
end

" All arrays needed for non-allocating gradient descent iterations "
function preallocate(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    Tξ::F          = zero(F)                      # for line searches
    uξ::Vector{V}  = [zeros(F, size(v)) for v in var.u]  # for line searches
    PA::Matrix{C}  = similar(last(sap.scp.y))     # for terminal condition computation at t=1
    PA0::Matrix{C} = similar(PA)                  # for RK4 ...
    PA1::Matrix{C} = similar(PA)
    PA2::Matrix{C} = similar(PA)
    PA3::Matrix{C} = similar(PA)
    PA4::Matrix{C} = similar(PA)

    temparrays = (PA0, PA1, PA2, PA3, PA4)
    preallocs = (Tξ, uξ, PA)

    return (preallocs, temparrays)
end

" Projected Gradient Descent with Golden Section Line Search algorithm (mutating) "
function gradient_descent!(var::VariationalProblem{F,V,C,M}, qs::QuantumSystem{F,C,M}; tolerances=nothing, keep_optimizing=false, verbose=false, with_plots=false) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    # state and adjoint Cauchy problems
    sap = StateAdjointPair{F,V,C,M}(var, qs)
    preallocations = preallocate(var, sap)
    update_directions = ["both", "T", "u"]

    # initial step and cost
    loop_iterator = keep_optimizing ? extend_objvalues!(var) : make_initial_iteration!(var, sap, preallocations, first(update_directions), verbose)

    ## REPEAT
    for iter in loop_iterator
        make_iteration!(var, sap, preallocations, update_directions, iter)
        # print and plot information about the iteration
        if mod(iter, 100) == 0
            verbose ? showinfo(var, sap, iter; with_plots) : nothing
            istolerances(var, tolerances, loop_iterator, iter) ? break : nothing
        end
    end
    
    # compute state evolution corresponding to the last (T,u) estimate (necessary because in-place computations on sap.scp)
    objective_gradient!(var, sap, last(preallocations))
    verbose ? showinfo(var, sap, length(get_objvalues(var))) : nothing
    
    return sap
end

" Performs one full iteration of gradient descent, including intermediate calculations (cost, state, adjoint, gradient) " 
function make_iteration!(var::VariationalProblem, sap::StateAdjointPair, preallocations, update_directions, iter)
    preallocs, temparrays = preallocations
    current_direction = rand(var.gradp.rng, update_directions) # update_directions[mod(iter, 3) + 1] # ∈ [(T,u), T, u]

    objective_gradient!(var, sap, temparrays) # update state and adjoint ODEs with new (T, u) and compute ∇J(T, u)
    ξopt = stepsize_linesearch!(var, sap, current_direction, iter, preallocs, temparrays) # perform line search and computes next cost J(T,u)
    gradient_descent_step!(var, -ξopt, current_direction) # step forward
end
" Performs FIRST iterations of gradient descent just to evluate cost associated to cost of initial guess for (T,u) "
function make_initial_iteration!(var::VariationalProblem, sap::StateAdjointPair, preallocations, cdir, verbose)
    preallocs, temparrays = preallocations

    objective_gradient!(var, sap, temparrays)   # computes non trivial final state
    objective_value!(var, sap, 1, last(preallocs)) # computes initial cost J(T, u)
    ξopt = stepsize_linesearch!(var, sap, cdir, 2, preallocs, temparrays)
    gradient_descent_step!(var, -ξopt, cdir) # step forward

    verbose ? showinfo(var, sap, 1) : nothing
    loop_iterator = 3:var.gradp.ngrad

    return loop_iterator
end

" Golden section line search for optimizing step size in gradient descent algorithm "
function golden_section_search(f::Function, interval::NTuple{2,<:Real}, abstol::Real, max_iterations::Integer; reference_value::Real = Inf)
    a, b = interval
    γ = oftype(a, (1 + sqrt(5)) / 2) # golden number

    c = b - (γ - 1) * (b - a)
    d = a + (γ - 1) * (b - a)
    left_value = f(c)
    right_value = f(d)

    count = 1
    while (count <= max_iterations) && (b - a > abstol)
        if left_value <= right_value
            b = d; d = c;
            c = b - (γ - 1) * (b - a)
            # only 1 evaluation of f instead of 2
            right_value = left_value
            left_value = f(c)
        elseif left_value > right_value
            a = c; c = d;
            d = a + (γ - 1) * (b - a)

            left_value = right_value
            right_value = f(d)
        end
        count += 1
    end

    # estimated minimizer 
    midpoint = (a + b) / 2
    midpoint_value = f(midpoint)
    if reference_value > midpoint_value
        # println("GS ----- steps: $(count) // minimizer estimate: $(midpoint) // Δf = $(reference_value-midpoint_value)")
        return (midpoint, midpoint_value)
    else
        # The line search failed to reduce the reference_value of f
        # println("GS FAILED ----- steps: $(count) // last estimate: $(midpoint)")
        return (eps(a), f(eps(a)))
    end
end

function golden_section_search(fls::Function, var::VariationalProblem{F,V,C,M}, ξmax::Real, iter::Integer) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    abstol = oftype(ξmax, 1f-4)
    search_interval = (zero(ξmax), ξmax)
    reference_value = get_objvalues(var, iter - 1)
    golden_section_search(fls, search_interval, abstol, var.gradp.grmaxiter; reference_value)
end

" For the line search with minimal allocations: returns cost for a given choice of step-size "
function try_stepsize!(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair{F,V,C,M}, minusξ::F, cdir::String, preallocs::Tuple{F,Vector{V},S}, tmps::NTuple{5,S}) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C},S<:AbstractMatrix{C}}
    Tξ, uξ, PA = preallocs
    if cdir === "T"                  # step T-direction only
        Tξ = gatetime_projection(var.T + minusξ * var.∇TJ)
        Threads.@threads for i in eachindex(var.u)
            axpby!(1, var.u[i], 0, uξ[i])
        end
    elseif cdir === "u"              # step u-direction only
        Tξ = var.T
        Threads.@threads for i in eachindex(var.u)
            axpby!(1, var.u[i], 0, uξ[i])
            axpy!(minusξ, var.∇uJ[i], uξ[i])
            control_projection!(uξ[i], var.ubounds)
        end
    elseif cdir === "both"           # step both ways (T,u)
        Tξ  = gatetime_projection(var.T + minusξ * var.∇TJ)
        Threads.@threads for i in eachindex(var.u)
            axpby!(1, var.u[i], 0, uξ[i])
            axpy!(minusξ, var.∇uJ[i], uξ[i])
            control_projection!(uξ[i], var.ubounds)
        end
    end
    # compute final state corresponding to candidate (Tξ, uξ)
    sap.scp.odep = (sap.L.Lfree, sap.L.Lcontrol, uξ, Tξ, var.nt)
    solve!(sap.scp, tmps)
    # return the associated cost
    objective_value!(PA, last(sap.scp.y), var.vecQ, Tξ, var.η)
end

" Performs line-search for step-size with update dependant on choice of direction btw. (T, u), T and u "
function stepsize_linesearch!(var::VariationalProblem, sap::StateAdjointPair, cdir::String, iter::Integer, preallocs, tmps)
    ξmax = var.gradp.grmaxstep    
    ξopt, val = golden_section_search(var, ξmax, iter) do ξ
        try_stepsize!(var, sap, -ξ, cdir, preallocs, tmps)
    end
    # if cost did not decrease, second chance with smaller initial step
    if isless(get_objvalues(var, iter-1), val)
        ξmax = ξmax / 100
        ξopt, val = golden_section_search(var, ξmax, iter) do ξ
            try_stepsize!(var, sap, -ξ, cdir, preallocs, tmps)
        end
    end
    set_objvalues!(var, iter, val)

    return ξopt
end

#################################################
# plots
#################################################

" Plot optimal control estimate and objective function value against number of gradient iterations "
function showresults(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair{F,V,C,M}; fig=Figure(), unitctrl=true) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    t = sap.scp.rkp.t
    z = zeros(F, var.nt)
    J = get_objvalues(var) .+ eps(F)
    n_iter = length(J)
    n_controls = length(first(var.u))

    control = reduce(hcat, var.u)
    control = unitctrl ? control/last(var.ubounds) : control
    gradient = reduce(hcat, var.∇uJ)

    u_max = maximum(abs, control)
    ∇uJ_max = maximum(abs, gradient)
    ∇uJ_max_tick = round(Int, log10(∇uJ_max))

    gradient = gradient/∇uJ_max

    colors = n_controls == 2 ? [:blue] : cgrad([:blue, :red], n_controls÷2, categorical=true)
    linestyles = [:solid, :dashdot]
    xlabels = [L"\log \, i", L"t / T", L"t / T"]
    ylabels = [L"\log \, J^{(i)}", L"u/M", L"10^{%$(-∇uJ_max_tick)} \cdot ∇J_0"]
    # titles  = [L"\mathrm{objective}", L"\mathrm{control}", L"\mathrm{gradient}"]

    axs = [Axis(fig[1, j], 
        aspect=AxisAspect(1), 
        xlabel=xlabels[j], 
        title=ylabels[j]) for j in 1:3]

    ylims!(axs[2], (-1.1*u_max, 1.1*u_max))
    ylims!(axs[3], (-1.1, 1.1))

    lines!(axs[1], log10.(1:n_iter), log10.(J); linewidth=2, color=:black)
    lines!(axs[2], t, z, linestyle=:dot, color=:black, alpha=0.5)
    lines!(axs[3], t, z, linestyle=:dot, color=:black, alpha=0.5)
    if mod(n_controls, 2) == 0
        for i in 1:n_controls÷2
            lines!(axs[2], t, view(control, 2*(i - 1) + 1, :); linewidth=2, linestyle=linestyles[1], color=colors[i])
            lines!(axs[2], t, view(control, 2*i, :); linewidth=2, linestyle=linestyles[2], color=colors[i])
            lines!(axs[3], t, view(gradient, 2*(i - 1) + 1, :); linewidth=2, linestyle=linestyles[1], color=colors[i])
            lines!(axs[3], t, view(gradient, 2*i, :); linewidth=2, linestyle=linestyles[2], color=colors[i])
        end
    else
        for i in 1:n_controls
            lines!(axs[2], t, view(control, i, :); linewidth=2, color=colors[i])
            lines!(axs[3], t, view(gradient, i, :); linewidth=2, color=colors[i])
        end
    end

    return fig
end

function showinfo(var::VariationalProblem, sap::StateAdjointPair, iter::Integer; with_plots::Bool=false)
    J = get_objvalues(var, iter)
    l = length(get_objvalues(var))
    n = var.gradp.ngrad
    r = (iter - (l - n)) / n # percentage

    @printf "iter : %.2f || Loss = %.6e || Gate time T = %.6e \n" r J var.T

    if with_plots
        display(showresults(var, sap))
        sleep(0.01)
    end
end

function accuracy_metrics(var::VariationalProblem{F,V,C,M}, sap::StateAdjointPair{F,V,C,M}) where {F<:AbstractFloat,V<:AbstractVector{F},C<:Complex,M<:AbstractMatrix{C}}
    R1 = finalstate(sap)

    d2target = sqrt(sum(abs2, var.vecQ - R1))
    dotp = dot(var.vecQ, R1) / (norm(R1) * norm(var.vecQ))
    fidelity = real(dotp)
    d2unitary = sqrt(max(0, real(dot(R1, R1)) - 2 * sum(svdvals(R1)) + size(R1, 1)))  # size(R1, 1) = d^2

    return ("T" => gatetime(var), "d2Q" => d2target, "d2unitary" => d2unitary, "dotQR1" => dotp, "fidelity" => fidelity)
end

#################################################
# saving results
#################################################

" Save optimization data and results using JLD2 "
function save(m::Tuple{<:VariationalProblem,<:StateAdjointPair}, filename::String)
    var, sap = m
    saved_var = var
    saved_sap = sap
    saved_uopt = var.u
    saved_Topt = var.T
    saved_R1 = finalstate(sap)

    saved_d2targ = sqrt(sum(abs2, saved_R1 .- var.vecQ))
    saved_fidelity = real(dot(var.vecQ, saved_R1)) / (norm(saved_R1) * norm(var.vecQ))
    saved_OCPloss = last(get_objvalues(var))

    @save filename saved_var saved_sap saved_uopt saved_Topt saved_R1 saved_d2targ saved_fidelity saved_OCPloss
end

#################################################
# Querying and updating attributes of structs
#################################################
function get_projectors(var::VariationalProblem)
    Tproj(x) = gatetime_projection(x)
    Uproj(x) = control_projection!(x, var.ubounds)
    (Tproj, Uproj)
end

function control(var::VariationalProblem)
    var.u
end
function gatetime(var::VariationalProblem)
    var.T
end
function state(sap::StateAdjointPair)
    sap.scp.y
end
function finalstate(sap::StateAdjointPair)
    last(sap.scp.y)
end
function adjointstate(sap::StateAdjointPair)
    sap.acp.y
end
function get_objvalues(var::VariationalProblem)
    var.gradp.objvalues
end
function get_objvalues(var::VariationalProblem, idx::Integer)
    var.gradp.objvalues[idx]
end
function set_objvalues!(var::VariationalProblem, idx::Integer, val::Real)
    var.gradp.objvalues[idx] = val
end
function extend_objvalues!(var::VariationalProblem)
    val = get_objvalues(var)
    l = length(val)
    n = var.gradp.ngrad
    loop_iterator = (l + 1):(l + n)

    var.gradp.objvalues = vcat(val, similar(val, n))

    return loop_iterator
end
function grad_parameters!(var::VariationalProblem; ngrad=nothing, grmaxiter=nothing, grmaxstep=nothing)
    isnothing(ngrad) ? nothing : var.gradp.ngrad = ngrad
    isnothing(grmaxiter) ? nothing : var.gradp.grmaxiter = grmaxiter
    isnothing(grmaxstep) ? nothing : var.gradp.grmaxstep = grmaxstep
    nothing
end
function penalty!(var::VariationalProblem, η)
    var.η = η
end

function solverparameters(rkp::RK4Parameters)
    (rkp.t, (rkp.Δt, rkp.Δto2, rkp.Δto6), rkp.nt)
end
function update_odeparams!(var::VariationalProblem, sap::StateAdjointPair)
    sap.scp.odep = ( sap.L.Lfree,  sap.L.Lcontrol, var.u, var.T, var.nt) # state ode
    sap.acp.odep = (sap.aL.Lfree, sap.aL.Lcontrol, var.u, var.T, var.nt) # adjoint ode -> adjoint lindbladians
end

function _istolerances(var::VariationalProblem, tolerances, loop_iterator, iter)
    abstol, reltol = tolerances
    J = var.gradp.objvalues

    iter0 = first(loop_iterator) - 1
    currJ = J[iter]
    meanΔJ = running_mean(J, iter0, iter)
    # pastJ = J[iter0]
    # ΔJ = abs(currJ - pastJ)
    isrel = meanΔJ <= reltol*currJ
    isabs = meanΔJ <= abstol
    return isrel*isabs
end

function running_mean(J, iter0, iter; window=50)
    if iter < iter0 + window
        return Inf
    else
        currJ = J[iter]
        i = iter - window
        return sum(pastJ - currJ for pastJ in view(J, i:iter-1))/window
    end
end

function istolerances(var::VariationalProblem, tolerances, loop_iterator, iter)
    if isnothing(tolerances)
        return false
    else
        if _istolerances(var, tolerances, loop_iterator, iter)
            var.gradp.objvalues = var.gradp.objvalues[begin:iter]
            n_iter = iter - first(loop_iterator) + 1
            println("reached atol/rtol in $(n_iter) iter.")

            return true
        else
            return false
        end
    end
end

function rescale!(var::VariationalProblem)
    u_norm = mapreduce(max, var.u) do x 
        maximum(abs, x)
    end
    ratio_to_saturation = u_norm/last(var.ubounds)
    var.u ./= ratio_to_saturation
    var.T *= ratio_to_saturation
    
    return ratio_to_saturation
end

function target!(var::VariationalProblem, target_gate::AbstractMatrix)
    var.Q = target_gate
    var.vecQ = vecgate(target_gate)
end

function energy(u::AbstractVector{<:AbstractVector})
    sum(sum(abs2, x) for x in u)
end

function unitgradient!(var::VariationalProblem)
    magnitude = sqrt(energy(var.∇uJ) + var.∇TJ^2)
    if !isapprox(magnitude, 0)
        foreach(var.∇uJ) do x
            x ./= magnitude
        end
        var.∇TJ /= magnitude
    end
    return magnitude
end