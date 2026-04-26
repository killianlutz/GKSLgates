" Qubit Pauli matrices (euclidean norm normalized or not) "
function build_pauli(C::DataType, M::Union{DataType,UnionAll}; normalized=false)
    normalized ? α = C(sqrt(2)) : α = one(C)
    σ0 = M([one(C) 0; 0 1] / α)
    σ1 = M([zero(C) 1; 1 0] / α)
    σ2 = M([zero(C) -im; im 0] / α)
    σ3 = M([one(C) 0; 0 -1] / α)
    return (; σ0, σ1, σ2, σ3)
end

# linear combination pauli matrices
function pauli_span(x, pauli)
    α = norm(x)
    y = isapprox(α, 0) ? x :  x/α
    H = zero(first(pauli))
    for (a, b) in zip(y, pauli)
        H .+= a .* b
    end
    return H
end

function pauli_span(x::AbstractVector{<:Number})
    σ1 = [0   1;  1  0] ./ sqrt(2)
    σ2 = [0 -im; im  0] ./ sqrt(2)
    σ3 = [1   0;  0 -1] ./ sqrt(2)

    C = Complex{eltype(x)}
    M = Matrix{C}
    pauli = map([σ1, σ2, σ3]) do σ; M(σ); end
    pauli_span(x, pauli)
end

function pauli_span(x::AbstractMatrix{<:Number})
    map(pauli_span, eachcol(x))
end

###
function sphereplot!(axis; radius::Real=1)
    S = Sphere(zero(Point3), radius)
    basis = map(1:3) do i
        Point3([j == i ? radius : zero(radius) for j in 1:3]...)
    end

    # mesh!(axis, S, color=:purple, transparency=true, alpha=1/10)
    wireframe!(axis, S, color=:purple, transparency=true, alpha=1/20)
    arrows3d!(axis, zero.(basis), basis, shaftradius=0.01, tipradius=0.05, tiplength=0.1, alpha=0.8)
    return axis
end

function normalized_pauli()
    return [
        [0   1/sqrt(2); 1/sqrt(2)  0], 
        [0 -im/sqrt(2); im/sqrt(2) 0],
        [1/sqrt(2)   0; 0 -1/sqrt(2)]
    ]
end

function to_blochball(density_operator::AbstractMatrix, npauli::AbstractVector)
    map(npauli) do P
        real(dot(P, reshape(density_operator, 2, 2)))
    end |> Point3
end

function to_blochball(density_operators::AbstractVecOrMat{T}) where T<:AbstractMatrix{<:Number}
    npauli = convert.(T, normalized_pauli())
    map(density_operators) do density_operator
        to_blochball(density_operator, npauli)
    end
end

function to_blochball(initial_density_operator::AbstractVecOrMat, npauli::AbstractVector, resolvant::AbstractVector{T}) where T<:AbstractMatrix
    orbit = map(resolvant) do R
        to_blochball(reshape(R * vec(initial_density_operator), 2, 2), npauli)
    end

    return orbit
end

function to_blochball(control::AbstractVector{T}) where T<:AbstractVector{<:Number}
    u_max = 2*mapreduce(norm, max, control)

    n_controls = length(first(control))
    if n_controls == 3
        f = u -> Point3(u[1]/u_max, u[2]/u_max, u[3]/u_max)
    else
        f = u -> Point3(first(u)/u_max, last(u)/u_max, 0)
    end
    angular_velocity = map(f, control)

    return angular_velocity
end


function showbloch_orbits(bloch_vector_orbits::AbstractVecOrMat{T}; fig=Figure(), single_figure=false) where T<:AbstractVector{<:Point3}
    sz = size(bloch_vector_orbits)
    if single_figure
        I, J = (1:1, 1:1)
    else
        I, J = length(sz) > 1 ? (1:first(sz), 1:last(sz)) : (1:1, 1:last(sz))
    end

    axs = [Axis3(
        fig[i, j], 
        aspect=:equal, 
        azimuth=π/4, 
        xlabel=L"X", 
        ylabel=L"Y", 
        zlabel=L"Z",
        xzpanelcolor = (:gray, 0.1), 
        yzpanelcolor = (:gray, 0.1),
        xypanelcolor = (:gray, 0.1),
        ) for i in I, j in J]

    foreach(axs) do axis
        hidedecorations!(axis, label=false)
        sphereplot!(axis; radius=1)
    end
    
    if single_figure
        orbits = vec(bloch_vector_orbits)
        colors = cgrad(:viridis, length(orbits); categorical=true)
        foreach(orbits, colors) do orbit, color  
            lines!(first(axs), orbit; color, linewidth=5, transparency=false)
            scatter!(first(axs), first(orbit), marker=:circle, color=:black, markersize=20)
        end
    else
        foreach(axs, bloch_vector_orbits) do axis, orbit
            scatter!(axis, orbit; color=1:length(orbit), colormap=:viridis, markersize=10)
        end
    end

    return fig, axs
end

function showbloch_orbits(bloch_vector_orbits::AbstractVecOrMat{V}, target_states::AbstractVecOrMat{V}; fig=Figure(), single_figure=false) where {T<:Point3, V<:AbstractVector{T}}
    fig, axs = showbloch_orbits(bloch_vector_orbits; fig, single_figure)
    if single_figure
        colors = cgrad(:viridis, length(bloch_vector_orbits); categorical=true)
        foreach(target_states, colors) do orbits_endpoints, color
            scatter!(first(axs), last(orbits_endpoints), marker=:cross, color=color, markersize=25)
        end
    else
        foreach(axs, target_states) do axis, orbits_endpoints
            scatter!(axis, last(orbits_endpoints), marker=:cross, color=:red, markersize=20)
        end
    end

    return fig, axs
end

function showbloch_orbits(bloch_vector_orbits::AbstractVecOrMat{V}, target_states::AbstractVecOrMat{V}, control::AbstractVector{T}; fig=Figure(), single_figure=false) where {T<:Point3, V<:AbstractVector{T}}
    fig, axs = showbloch_orbits(bloch_vector_orbits, target_states; fig, single_figure)
    
    origin = [zero(Point3)]
    foreach(axs) do axis
        keep = 1:5:length(control)
        arrows3d!(axis, origin, view(control, keep), color=keep, shaftradius=0.01, tipradius=0.035, tiplength=0.05, colormap=:grays)
    end

    return fig, axs
end

function showbloch_orbits(var::VariationalProblem, sap::StateAdjointPair, initial_density_operators::AbstractVecOrMat{T}; fig=Figure(), with_ctrl=false, single_figure=false) where {T<:AbstractMatrix}
    npauli = convert.(T, normalized_pauli())

    resolvant = state(sap)
    bloch_vector_orbits = map(initial_density_operators) do x
        to_blochball(x, npauli, resolvant)
    end

    target_gate = var.vecQ
    gates = [first(resolvant), target_gate]
    target_states = map(initial_density_operators) do x
        to_blochball(x, npauli, gates)
    end

    if with_ctrl
        angular_velocity = to_blochball(control(var))
        return showbloch_orbits(bloch_vector_orbits, target_states, angular_velocity; fig, single_figure) |> first
    else
        return showbloch_orbits(bloch_vector_orbits, target_states; fig, single_figure) |> first
    end
end

function showbloch_orbits(qs::QuantumSystem{F,C,M}, initial_density_operators::AbstractArray{T}; fig=Figure(), ctrl::Union{Function,Nothing}=nothing, tspan=nothing, solver=Tsit5(), kwargs=(;)) where {F<:AbstractFloat, C<:Complex, M<:AbstractMatrix{C}, T<:AbstractMatrix}
    if isnothing(tspan)
        τ = decoherence(qs)
        tspan = isapprox(τ, 0) ? (zero(τ), one(τ)) : (zero(τ), 10τ)
    end

    x0 = reduce(hcat, vec(ϱ0) for ϱ0 in initial_density_operators)
    ode_sol, _, _ = simulate_gksl(qs, x0, tspan; ctrl, solver, kwargs);
    n_orbits = size(x0, 2)
    nt = length(ode_sol.t)

    npauli = convert.(M, normalized_pauli())
    orbits = stack(ode_sol.u, dims=3)
    orbits = map(1:n_orbits) do i
        map(1:nt) do t
            ϱ = reshape(orbits[:, i, t], 2, 2)
            to_blochball(ϱ, npauli)
        end
    end
    return first(showbloch_orbits(orbits; fig, single_figure=true))
end

function showbloch_orbits(var::VariationalProblem, sap::StateAdjointPair, qs::QuantumSystem, initial_density_operators::AbstractVecOrMat{T}; fig=Figure(size=(1_500, 1_000)), ctrl::Union{Function,Nothing}=nothing, tspan=nothing, solver=Tsit5(), kwargs=(;), with_ctrl=false) where {T<:AbstractMatrix}
    showbloch_orbits(qs, initial_density_operators; fig=fig[1, 1], ctrl, tspan, solver, kwargs)
    showbloch_orbits(var, sap, initial_density_operators; fig=fig[1, 2], with_ctrl, single_figure=true)
    return fig
end