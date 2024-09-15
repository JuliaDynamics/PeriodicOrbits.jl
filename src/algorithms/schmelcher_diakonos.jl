export SchmelcherDiakonos, periodic_orbits

using LinearAlgebra: norm


"""
    SchmelcherDiakonos(; kwargs...)

Detect periodic orbits of `ds <: DiscreteTimeDynamicalSystem` using algorithm
proposed by Schmelcher & Diakonos [^Schmelcher1997].

## Keyword arguments

* `o` : order of the periodic orbit
* `λs` : vector of λ parameters, see [^Schmelcher1997] for details
* `indss` : vector of vectors of indices for the permutation matrix
* `signss` : vector of vectors of signs for the permutation matrix
* `maxiters=1000` : maximum amount of iterations an initial guess will be iterated before 
  claiming it has not converged
* `inftol=10.0` : if a state reaches `norm(state) ≥ inftol` it is assumed that it has 
  escaped to infinity (and is thus abandoned)
* `disttol=1e-10` : distance tolerance. If the 2-norm of a previous state with the next one 
  is `≤ disttol` then it has converged to a fixed point

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
    signss::Vector{Vector{Int64}}
    maxiters::Int64 = 1000
    disttol::Float64 = 1e-10
    inftol::Float64 = 10.0
end

function periodic_orbits(ds::DiscreteTimeDynamicalSystem, alg::SchmelcherDiakonos, igs::Vector{InitialGuess})
    type = typeof(current_state(ds))
    POs = type[]
    for λ in alg.λs, inds in alg.indss, sings in alg.signss
        Λ = lambdamatrix(λ, inds, sings)
        _periodic_orbits!(POs, ds, alg, igs, Λ)
    end
    po = PeriodicOrbit[PeriodicOrbit(ds, fp, alg.o) for fp in POs]
    return po
end


function _periodic_orbits!(POs, ds, alg, igs, Λ)
    igs = [ig.u0 for ig in igs]
    for ig in igs
        reinit!(ds, ig)
        previg = ig
        for _ in 1:alg.maxiters
            previg, ig = Sk(ds, previg, alg.o, Λ)
            norm(ig) > alg.inftol && break

            if norm(previg - ig) < alg.disttol
                push!(POs, ig)
                break
            end
            previg = ig
        end
    end
end

function Sk(ds, state, o, Λ)
    reinit!(ds, state)
    step!(ds, o)
    # TODO: For IIP systems optimizations can be done here to not re-allocate vectors...
    return state, state + Λ*(current_state(ds) .- state)
end