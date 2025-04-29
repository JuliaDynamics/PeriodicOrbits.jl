export InitialGuess,
    PeriodicOrbit,
    PeriodicOrbitFinder,
    isdiscretetime,
    complete_orbit,
    podistance,
    uniquepos,
    poequal,
    periodic_orbit,
    periodic_orbits

import DynamicalSystemsBase
using LinearAlgebra: norm

"""
    InitialGuess

A structure that contains an initial guess for a periodic orbit detection algorithms.

* `u0::AbstractArray{<:Real}` - guess of a point in the periodic orbit
* `T::Union{Real, Nothing}` - guess of period of the orbit. Some algorithms do not require
  the period guess to be given, in which case `T` is set to `nothing`.
"""
struct InitialGuess{U<:AbstractArray{<:Real}, R<:Union{Real, Nothing}}
    u0::U
    T::R
end
InitialGuess(ds::DynamicalSystem, T=nothing) = InitialGuess(current_state(ds), T)
InitialGuess(u0::AbstractArray) = InitialGuess(u0, T)


"""
    PeriodicOrbit

A structure that contains information about a periodic orbit.

* `points::StateSpaceSet` - points of the periodic orbit. This container
  always holds the complete orbit (in the sense of being continuously sampled
  with some sampling `Δt` for continuous time systems).
* `T::Real` - the period of the orbit
* `stable::Union{Bool, Missing}` - local stability of the periodic orbit.
  If the stability is unknown (not computed automatically by an algorithm),
  this is set to `missing`.
"""
struct PeriodicOrbit{D, B, R<:Real, S<:Union{Bool, Missing}}
    points::StateSpaceSet{D, B}
    T::R
    stable::S
end

"""
    PeriodicOrbit(ds::DynamicalSystem, u0, T; Δt, stable)
        T::AbstractFloat, Δt=T/$(default_Δt_partition), stable=missing) → po

Construct a complete [`PeriodicOrbit`](@ref) with period `T` for `ds` by starting from `u0`.

For continuous time systems, the PO formally contains infinitely many points,
but it is approximated by evaluating
a trajectory with step `Δt` which by default it is `T/$(default_Δt_partition)`.
For discrete time systems `Δt` is ignored and `T` must be an integer,
and the orbit has exactly `T` points.

If the stability of the PO is known, you can provide it as a keyword.
"""
function PeriodicOrbit(
        ds::ContinuousTimeDynamicalSystem, u0::AbstractArray{<:Real}, T::Real;
        Δt=T/default_Δt_partition, stable::Union{Bool, Missing}=missing
    )
    return PeriodicOrbit(complete_orbit(ds, u0, T; Δt), T, stable)
end

function PeriodicOrbit(
        ds::DiscreteTimeDynamicalSystem, u0::AbstractArray{<:Real}, T::Integer;
        Δt = nothing, stable::Union{Bool, Missing} = missing
    )
    return PeriodicOrbit(complete_orbit(ds, u0, T; Δt=1), T, stable)
end

"""
    PeriodicOrbitFinder

Supertype for all the periodic orbit detection algorithms.
Each of the concrete subtypes of `PeriodicOrbitFinder`
represents one given algorithm for detecting periodic orbits. This subtype includes
all the necessary metaparameters for the algorithm to work.
"""
abstract type PeriodicOrbitFinder end

"""
    periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder [, ig::InitialGuess]) → PeriodicOrbit

Try to find a single periodic orbit of the dynamical system `ds` using the algorithm `alg`
and optionally given some [`InitialGuess`](@ref) `ig` which defaults to `InitialGuess(ds)`.
Return the result as a [`PeriodicOrbit`](@ref).

Depending on `alg`, it is not guaranteed that a periodic orbit will be found given `ds, ig`.
If one is not found, `nothing` is returned instead.

For more details on the periodic orbit detection algorithm, see the documentation the `alg`.
"""
function periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, ig::InitialGuess = InitialGuess(ds))
    throw(ArugmentError("Not implemented for $(alg)"))
end

"""
    periodic_orbits(ds::DynamicalSystem, alg::PeriodicOrbitFinder [, igs]::Vector{InitialGuess} = InitialGuess(ds)) → Vector{PeriodicOrbit}

Try to find multiple periodic orbits of the dynamical system `ds` using the algorithm `alg`
and optionally given a `Vector` of initial guesses which defaults to `[InitialGuess(ds)]`.
given some initial guesses `igs`.
Return the result as a `Vector` of [`PeriodicOrbit`](@ref).

This function exists because some algorithms optimize/specialize on detecting
multiple periodic orbits.
"""
function periodic_orbits(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{<:InitialGuess} = [InitialGuess(ds)])
    pos = [periodic_orbit(ds, alg, igs[1])]
    for ig in view(igs, 2:length(igs))
        res = periodic_orbit(ds, alg, ig)
        push!(pos, res)
    end
    return pos
end

"""
    isdiscretetime(po::PeriodicOrbit) → true/false

Return `true` if the periodic orbit belongs to a discrete-time dynamical system, `false` if
it belongs to a continuous-time dynamical system.
This simple function only checks whether the period is an integer or not.
"""
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D,B,R}) where {D,B,R<:Integer}
    true
end
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D,B,R}) where {D,B,R<:AbstractFloat}
    false
end


"""
    complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; kwargs...) → StateSpaceSet

Given point `u0` on the periodic orbit with period `T`, compute the remaining points of the
periodic orbit. For POs of discrete-time systems, it means iterating the periodic point
`T - 1` times. For POs of continuous-time systems, it means integrating the system for
duration `po.T - Δt` with stepsize `Δt`.

## Keyword arguments

* `Δt = T/$(default_Δt_partition)`: integration stepsize.
  For discrete-time systems this is ignored.
"""
function complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; Δt::Real=T/default_Δt_partition)
    isdiscretetime(ds) && (Δt = 1)
    traj, _ = trajectory(
        ds,
        T - Δt,
        u0;
        Δt = Δt
    )
    return traj
end


"""
    podistance(po1::PeriodicOrbit, po2::PeriodicOrbit, [, distance]) → Real

Compute the distance between two periodic orbits `po1` and `po2`.
Periodic orbits `po1` and `po2` and the dynamical system `ds` all have to
be either discrete-time or continuous-time.
Distance between the periodic orbits is computed using the given distance function `distance`.
The default distance function is `StrictlyMinimumDistance(true, Euclidean())` which finds the minimal
Euclidean distance between any pair of points where one point belongs to `po1` and the other to `po2`.
For other options of the distance function, see `StateSpaceSets.set_distance`.
Custom distance function can be provided as well.
"""
function podistance(po1, po2, distance=StrictlyMinimumDistance(true, Euclidean()))
    type1 = isdiscretetime(po1)
    type2 = isdiscretetime(po2)
    if type1 == type2
        return set_distance(po1.points, po2.points, distance)
    else
        throw(ArgumentError("Both periodic orbits have to be either discrete-time or continuous-time."))
    end
end


"""
    poequal(po1::PeriodicOrbit, po2::PeriodicOrbit; kwargs...) → true/false

Return `true` if the periodic orbits `po1` and `po2` are approximately equal in period and in location.

## Keyword arguments

* `Tthres=1e-3` : difference in periods of the periodic orbits must be less than this threshold
* `dthres=1e-3` : distance between periodic orbits must be less than this threshold
* `distance` : distance function used to compute the distance between the periodic orbits

Distance between the orbits is computed using `podistance` with `distance`.
"""
function poequal(
    po1::PeriodicOrbit, po2::PeriodicOrbit;
    Tthres=1e-3,
    dthres=1e-3,
    distance=StrictlyMinimumDistance(true, Euclidean())
)
    if abs(po1.T - po2.T) < Tthres
        d = podistance(po1, po2, distance)
        return d < dthres
    else
        return false
    end
end


"""
    uniquepos(pos::Vector{<:PeriodicOrbit}; atol=1e-6) → Vector{PeriodicOrbit}

Return a vector of unique periodic orbits from the vector `pos` of periodic orbits.
By unique we mean that the distance between any two periodic orbits in the vector is
greater than `atol`. To see details about the distance function, see [`podistance`](@ref).

## Keyword arguments

* `atol` : minimal distance between two periodic orbits for them to be considered unique.
"""
function uniquepos(pos::Vector{<:PeriodicOrbit}; atol::Real=1e-6)
    length(pos) == 0 && return pos
    unique_pos = typeof(pos[end])[]
    pos = copy(pos)

    while length(pos) > 0
        push!(unique_pos, pop!(pos))
        filter!(x -> podistance(x, unique_pos[end]) > atol, pos)
    end

    return unique_pos
end