export DavidchackLai, periodic_orbits

using LinearAlgebra: norm

"""
    DavidchackLai(; kwargs...)

Find periodic orbits `fps` of periods `1` to `n+1` for the dynamical system `ds`
using the algorithm propesed by Davidchack & Lai [Davidchack1999](@cite).

## Keyword arguments

* `n::Int64` : Periodic orbits of period up to `n` will be detected. Some (but not all) POs 
   of period `n+1` will be detected. Keyword argument `n` must be a positive integer.
* `m::Int64` : Initial guesses will be used to find POs of period `1` to `m`. These 
   periodic orbits will then be used to detect periodic orbits of periods from `m+1` to 
   `n+1`. Keyword argument `m` must be a positive integer between `1` and `n`.
* `β = nothing`: If it is nothing, then `β(n) = 10*1.2^n`. Otherwise can be a 
   function that takes period `n` and return a number. It is a parameter mentioned
   in the paper[Davidchack1999](@cite).
* `maxiters = nothing`: If it is nothing, then initial condition will be iterated
  `max(100, 4*β(p))` times (where `p` is the period of the periodic orbit)
   before claiming it has not converged. If it is an integer, then it is the maximum 
   amount of iterations an initial condition will be iterated before claiming 
   it has not converged.
* `disttol = 1e-10`: Distance tolerance. If `norm(f^{n}(x)-x) < disttol` 
   where `f^{n}` is the `n`-th iterate of the dynamic rule `f`, then `x` 
   is an `n`-periodic point.
* `abstol = 1e-8`: A detected periodic point isn't stored if it is in `abstol` 
   neighborhood of some previously detected point. Distance is measured by 
   euclidian norm. If you are getting duplicate periodic points, increase this value.

## Description

The algorithm is an extension of Schmelcher & Diakonos[Schmelcher1997](@cite)
implemented as [`SchmelcherDiakonos`](@ref).

The algorithm can detect periodic orbits
by turning fixed points of the original dynamical system `ds` to stable ones, through the 
transformation
```math
\\mathbf{x}_{n+1} = \\mathbf{x}_{n} + 
[\\beta |g(\\mathbf{x}_{n})| C^{T} - J(\\mathbf{x}_{n})]^{-1} g(\\mathbf{x}_{n})
```
where
```math
g(\\mathbf{x}_{n}) = f^{n}(\\mathbf{x}_{n}) - \\mathbf{x}_{n}
```
and
```math
J(\\mathbf{x}_{n}) = \\frac{\\partial g(\\mathbf{x}_{n})}{\\partial \\mathbf{x}_{n}}
```

The main difference between [`SchmelcherDiakonos`](@ref) and 
[`DavidchackLai`](@ref) is that the latter uses periodic points of
previous period as seeds to detect periodic points of the next period.
Additionally, [`SchmelcherDiakonos`](@ref) only detects periodic points of a given period, 
while `davidchacklai` detects periodic points of all periods up to `n`.


## Important note

For low periods `n` circa less than 6, you should select `m = n` otherwise the algorithm 
won't detect periodic orbits correctly. For higher periods, you can select `m` as 6. 
We recommend experimenting with `m` as it may depend on the specific problem. 
Increase `m` in case the orbits are not being detected correctly.

Initial guesses for this algorithm can be selected as a uniform grid of points in the state 
space or subset of a chaotic trajectory.

"""
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
    fps = [type[] for _ in 1:alg.n+1]
    igs = [ig.u0 for ig in igs]

    isnothing(alg.β) ? β = n-> 10*1.2^(n) : β = alg.β
    indss, signss = lambdaperms(dimension(ds))
    C_matrices = [cmatrix(inds,signs) for inds in indss, signs in signss]

    initial_detection!(fps, ds, alg, igs, β, C_matrices)
    main_detection!(fps, ds, alg, β, C_matrices)

    return uniquepos(convert_to_pos(ds, fps, alg.n); atol=alg.abstol)
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
        for (container, seed, period) in [
            (currentfps, previousfps, period), 
            (nextfps, currentfps, period+1), 
            (currentfps, nextfps, period)
            ]
            detect_orbits!(container, ds, alg, period, seed, β(period), C_matrices)
        end
    end
end

function _detect_orbits!(fps, ds, alg, T, igs, C, β)
    for x in igs
        x = x
        for _ in 1:(isnothing(alg.maxiters) ? max(100, 4*β) : alg.maxiters)
            xn = DL_rule(x, β, C, ds, T)
            if norm(g(ds, xn, T)) < alg.disttol
                if !iscontained(xn, fps, alg.abstol)
                    push!(fps, xn)
                    break
                end
            end
            x = xn
        end
    end
end

function detect_orbits!(fps, ds, alg, T, igs, β, C_matrices)
    for C in C_matrices
        _detect_orbits!(fps, ds, alg, T, igs, C, β)
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

function iscontained(x, arr, thresh)
    for y in arr
        if norm(x - y) < thresh
            return true
        end
    end
    return false
end

function convert_to_pos(ds, fps, T)
    len = sum(length.(fps))
    po = Vector{PeriodicOrbit}(undef, len)
    count = 1
    for t in 1:T+1
        for pp in fps[t]
            po[count] = PeriodicOrbit(ds, pp, t, missing)
            count += 1
        end
    end
    return po
end
