export periodic_orbit, OptimizedShooting

using LeastSquaresOptim: optimize, LevenbergMarquardt


@kwdef struct OptimizedShooting <: PeriodicOrbitFinder
    Δt ::Float64 = 1e-6
    p :: Int64 = 2
    optim_kwargs :: NamedTuple = (x_tol=1e-10,)
    abstol :: Float64 = 1e-3
end

function new_rule(rule, T)
    return (u, p, t) -> T*rule(u, p, T*t)
end

function periodic_orbit(ds::CoupledODEs, alg::OptimizedShooting, ig::InitialGuess)
    res = optimize(v -> costfunc(v, ds, alg), [ig.u0..., ig.T], LevenbergMarquardt(); alg.optim_kwargs...)
    if res.ssr <= alg.abstol
        u0 = res.minimizer[1:dimension(ds)]
        T = res.minimizer[end]
        return PeriodicOrbit(ds, u0, T, 0.01; jac=nothing)
    end
    return nothing
end

function periodic_orbits(ds::CoupledODEs, alg::OptimizedShooting, igs::Vector{<:InitialGuess})
    # TODO: annotate `pos` with correct type
    pos = []
    for ig in igs
        res = periodic_orbit(ds, alg, ig)
        if !isnothing(res)
            push!(pos, res)
        end
    end
    return pos
end

function costfunc(v, ds, alg)
    u0 = v[1:dimension(ds)]
    T = v[end]

    f = dynamic_rule(ds)
    nds = CoupledODEs(new_rule(f, T), current_state(ds), current_parameters(ds); diffeq = ds.diffeq)

    len = alg.p * dimension(ds)
    u1s = zeros(len)
    u2s = zeros(len)
    err = zeros(len)

    for i in 0:alg.p-1
        reinit!(nds, u0)
        step!(nds, i*alg.Δt*1)
        u1 = current_state(nds)
        u1s[i*dimension(ds) + 1 : (i+1)*dimension(ds)] .= u1
    end
    for i in 0:alg.p-1
        reinit!(nds, u0)
        step!(nds, 1 + i*alg.Δt*1)
        u2 = current_state(nds)
        u2s[i*dimension(ds) + 1 : (i+1)*dimension(ds)] .= u2
    end
    err = u2s .- u1s
    return err
end