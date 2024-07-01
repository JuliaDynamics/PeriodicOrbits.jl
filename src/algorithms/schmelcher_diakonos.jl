export SchmelcherDiakonos, periodic_orbits

using LinearAlgebra: norm


@kwdef struct SchmelcherDiakonos
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

function SchmelcherDiakonos(o::Int64, λs::Vector{Float64}, indss::Vector{Vector{Int64}}, signss::Vector{BitVector})
    return SchmelcherDiakonos(o=o, λs=λs, indss=indss, signss=signss)
end

function SchmelcherDiakonos(o::Int64, dim::Int64, λ::Float64=0.01)
    inds = randperm(dim)
    signs = rand(Bool, dim)
    return SchmelcherDiakonos(o=o, λs=[λ], indss=[inds], signss=[signs])
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