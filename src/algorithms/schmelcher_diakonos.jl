export SchmelcherDiakonos, periodic_orbits

using LinearAlgebra: norm


"""
Detect periodic orbits of `ds <: DiscreteTimeDynamicalSystem` using algorithm
proposed by Schmelcher & Diakonos [^Schmelcher1997].

Possible constructors are:

    1. `SchmelcherDiakonos(o::Int64, λs::Vector{Float64}, indss::Vector{Vector{Int64}}, signss::Vector{BitVector}; kwargs...)`
    2. `SchmelcherDiakonos(o::Int64, dim::Int64, λ::Float64=0.001; kwargs...)`

Where arguments are:

* `o` = order of the periodic orbit
* `λs` = vector of λ parameters, see [^Schmelcher1997] for details
* `indss` = vector of vectors of indices for the permutation matrix
* `signss` = vector of vectors of signs for the permutation matrix
* `dim` = dimension of the dynamical system, use `dimension(ds)`

## Keyword arguments

* `maxiters` = maximum amount of iterations an initial guess will be iterated before claiming it has not converged
* `inftol` = if a state reaches `norm(state) ≥ inftol` it is assumed that it has escaped to infinity (and is thus abandoned)
* `disttol` = distance tolerance. If the 2-norm of a previous state with the next one is `≤ disttol` then it has converged to a fixed point
* `roundtol` = nothing. This keyword has been removed in favor of `abstol`
* `abstol` = a detected fixed point isn't stored if it is in `abstol` neighborhood of some previously detected point. Distance is measured by euclidian norm. If you are getting duplicate fixed points, decrease this value

## Description

The algorithm used can detect periodic orbits
by turning fixed points of the original
map `ds` to stable ones, through the transformation
```math
\\mathbf{x}_{n+1} = \\mathbf{x}_n +
\\mathbf{\\Lambda}_k\\left(f^{(o)}(\\mathbf{x}_n) - \\mathbf{x}_n\\right)
```
The index ``k`` counts the various
possible ``\\mathbf{\\Lambda}_k``.

## Performance notes

*All* initial guesses are
evolved for *all* ``\\mathbf{\\Lambda}_k`` which can very quickly lead to
long computation times.

[^Schmelcher1997]:
    P. Schmelcher & F. K. Diakonos, Phys. Rev. Lett. **78**, pp 4733 (1997)
"""
@kwdef struct SchmelcherDiakonos <: PeriodicOrbitFinder
    o::Int64
    λs::Vector{Float64}
    indss::Vector{Vector{Int64}}
    signss::Vector{BitVector}
    maxiters::Int64 = 1000
    disttol::Float64 = 1e-10
    inftol::Float64 = 10.0
    roundtol :: Nothing = nothing
    abstol::Float64 = 1e-8
end

function SchmelcherDiakonos(o::Int64, λs::Vector{Float64}, indss::Vector{Vector{Int64}}, signss::Vector{BitVector}; kwargs...)
    return SchmelcherDiakonos(o=o, λs=λs, indss=indss, signss=signss;kwargs...)
end

function SchmelcherDiakonos(o::Int64, dim::Int64, λ::Float64=0.01; kwargs...)
    inds = randperm(dim)
    signs = rand(Bool, dim)
    return SchmelcherDiakonos(o=o, λs=[λ], indss=[inds], signss=[signs]; kwargs...)
end

function check_parameters(alg::SchmelcherDiakonos)
    if !isnothing(alg.roundtol)
        warn("`roundtol` keyword has been removed in favor of `abstol`")
    end
end


function periodic_orbits(ds::DiscreteTimeDynamicalSystem, alg::SchmelcherDiakonos, igs::Vector{InitialGuess})
    check_parameters(alg)

    type = typeof(current_state(ds))
    POs = Set{type}()
    for λ in alg.λs, inds in alg.indss, sings in alg.signss
        Λ = lambdamatrix(λ, inds, sings)
        _periodicorbits!(POs, ds, alg, igs, Λ)
    end
    po = PeriodicOrbit[PeriodicOrbit{type, Int64}([fp], alg.o) for fp in POs]
    return po
end


function _periodicorbits!(POs, ds, alg, igs, Λ)
    igs = [ig.u0 for ig in igs]
    for st in igs
        reinit!(ds, st)
        prevst = st
        for _ in 1:alg.maxiters
            prevst, st = Sk(ds, prevst, alg.o, Λ)
            norm(st) > alg.inftol && break

            if norm(prevst - st) < alg.disttol
                storefp!(POs, st, alg.abstol)
                break
            end
            prevst = st
        end
    end
end

function Sk(ds, prevst, o, Λ)
    reinit!(ds, prevst)
    step!(ds, o)
    # TODO: For IIP systems optimizations can be done here to not re-allocate vectors...
    return prevst, prevst + Λ*(current_state(ds) .- prevst)
end