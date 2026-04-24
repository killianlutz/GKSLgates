function validate(var::VariationalProblem, sap::StateAdjointPair, x0::AbstractVecOrMat; interpolate=:constant, kwargs=(; ))
    T = QGateDescent.gatetime(var)
    u = QGateDescent.control(var)
    R = QGateDescent.state(sap)

    coarse_dt = T/(length(u) - 1)
    ctrl! = piecewise_to_function(u, coarse_dt; interpolate)

    sol, ode_prob, _ = QGateDescent.simulate_gksl(sap.L, x0, (zero(T), T); ctrl=ctrl!, kwargs);
    x = sol.u
    x_targets = var.vecQ * x0
    x_resolvants = map(sol.t) do s
        current_index = floor(Int, s/coarse_dt) + 1
        R[current_index] * x0
    end

    δx = missfit(x, x_targets, x_resolvants)

    return (; sol, δx, ode_prob)
end

function missfit(x, x_targets, x_resolvants)
    δx_target = map(x) do x_t
        missfit(x_t, x_targets)
    end

    δx_gate = map(x, x_resolvants) do x_t, xr_t
        missfit(x_t, xr_t)
    end

    δx_target, δx_gate = map((δx_target, δx_gate)) do error
        reduce(vcat, error)
    end

    return (; target=δx_target, gate=δx_gate)
end

function missfit(x::AbstractVecOrMat, y::AbstractVecOrMat)
    sqrt.(sum(abs2, x - y, dims=1))
end

function piecewise_to_function(x::AbstractVector, dt::Real; interpolate=:constant)
    constant_f(y, t) = begin
        current_index = min(floor(Int, t/dt) + 1, length(x))

        y .= x[current_index]
    end
    linear_f(y, t) = begin
        current_index = floor(Int, t/dt) + 1
        current_time = current_index * dt
        next_index = current_index + 1

        if next_index > length(x)
            y .= x[current_index] # interval right endpoint
        else
            y .= x[current_index] .+ (t .- current_time) .* (x[next_index] .- x[current_index])/dt # linear interpolation
        end
    end

    if interpolate == :constant
        return constant_f
    elseif interpolate == :linear
        return linear_f
    else
        @error "Unknown interpolation type. Defaults to :constant"
        return constant_f
    end
end

function showerrors(results)
    ylabels = [L"|\hat{R}(t)x_i - Qx_i|", L"|\hat{R}(t)x_i - R(t)x_i|"]
    titles = ["a posteriori wrt. target", "wrt. approximate target"]

    fig = Figure()
    axs = [Axis(
        fig[1, j], 
        aspect=AxisAspect(1), 
        xlabel=L"t", 
        ylabel=ylabels[j], 
        title=titles[j],
        yscale=log10
    ) for j in 1:2]

    foreach(1:2, results.δx) do j, error
        F = eltype(error)
        for δx_i in eachcol(error)
            scatterlines!(axs[j], results.sol.t, δx_i .+ eps(F))
        end
    end

    return fig
end

function mse(x, y)
    sum(abs2, x - y)/size(x, 2)
end

function fidelity(x, y)
    dot(y, x)/(norm(x)*norm(y))
end