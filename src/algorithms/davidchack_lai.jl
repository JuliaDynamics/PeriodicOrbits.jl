export DavidchackLai, periodic_orbits

using LinearAlgebra: norm

@kwdef struct DavidchackLai
    n::Int64
    m::Int64
    β::Union{Nothing, Function} = nothing
    maxiters::Union{Nothing, Int64} = nothing
    disttol::Float64 = 1e-10
    abstol::Float64 = 1e-8


    function DavidchackLai(n, m, β, maxiters, disttol, abstol)
        if (n < 1)
            throw(ArgumentError("`n` must be a positive integer."))
        end

        if !(1 <= m <= n)
            throw(ArgumentError("`m` must be in [1, `n`=$(n)]"))
        end
        return new(n, m, β, maxiters, disttol, abstol)
    end
end

function periodic_orbits(ds::DeterministicIteratedMap, alg::DavidchackLai, igs::Vector{InitialGuess})
    if isinplace(ds)
        throw(ArgumentError("Algorithms `DavidchackLai` currently supports only out of place systems."))
    end

    type = typeof(current_state(ds))
    fps = storage(type, alg.n)
    igs = [ig.u0 for ig in igs]

    isnothing(alg.β) ? β = n-> 10*1.2^(n) : β = alg.β
    indss, signss = lambdaperms(dimension(ds))
    C_matrices = [cmatrix(inds,signs) for inds in indss, signs in signss]

    initial_detection!(fps, ds, alg, igs, β, C_matrices)
    main_detection!(fps, ds, alg, β, C_matrices)

    return output(fps, type, alg.n)
end

function initial_detection!(fps, ds, alg, igs, β, C_matrices)
    for i in 1:alg.m
        detect_orbits!(fps[i], ds, alg, i, igs, β(i), C_matrices)
    end
end

function main_detection!(fps, ds, alg, β, C_matrices)
    for period = 2:alg.n
        previousfps = fps[period-1]
        currentfps = fps[period]
        nextfps = fps[period+1]
        for (container, seed, order) in [
            (currentfps, previousfps, period), 
            (nextfps, currentfps, period+1), 
            (currentfps, nextfps, period)
            ]
            detect_orbits!(container, ds, alg, order, seed, β(period), C_matrices)
        end
    end
end

function _detect_orbits!(fps, ds, alg, i, igs, C, β)
    for x in igs
        x = x
        for _ in 1:(isnothing(alg.maxiters) ? max(100, 4*β) : alg.maxiters)
            xn = DL_rule(x, β, C, ds, i)
            if converged(ds, xn, i, alg.disttol)
                if previously_detected(fps, xn, alg.abstol) == false
                    completeorbit!(fps, ds, alg, xn, i)
                end
                break
            end
            x = xn
        end
    end
end

function completeorbit!(fps, ds, alg, xn, i)
    traj = trajectory(ds, i, xn)[1]
    for t in traj
        if converged(ds, t, i, alg.disttol)
            storefp!(fps, t, alg.abstol)
        end
    end
end

function converged(ds, xn, n, disttol)
    return norm(g(ds, xn, n)) < disttol
end

function detect_orbits!(fps, ds, alg, i, igs, β, C_matrices)
    for C in C_matrices
        _detect_orbits!(fps, ds, alg, i, igs, C, β)
    end
end

function DL_rule(x, β, C, ds, n)
    Jx = DynamicalSystemsBase.ForwardDiff.jacobian(x0 -> g(ds, x0, n), x)
    gx = g(ds, x, n)
    xn = x + inv(β*norm(gx)*C' - Jx) * gx
    return xn
end

function g(ds, state, n)
    p = current_parameters(ds)
    newst = state
    # not using step!(ds, n) to allow automatic jacobian
    for _ = 1:n
        newst = ds.f(newst, p, 1.0)
    end
    return newst - state
end

function output(fps, type, n)
    len = sum(length.(fps[1:n]))
    po = Vector{PeriodicOrbit{type, Int64}}(undef, len)
    j = 1
    for i in 1:n # not including periodic orbit n+1 because it may be incomplete
        for pp in fps[i]
            po[j] = PeriodicOrbit([pp], i)
            j += 1
        end
    end
    return po
end

function storage(type, n)
    storage = Vector{Set{type}}(undef, n+1)
    for i in 1:n+1
        storage[i] = Set{type}()
    end
    return storage
end