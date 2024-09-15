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
A structure that contains an initial guess for a periodic orbit detection algorithms.

* `u0` - guess of a point in the periodic orbit
* `T` - guess of period of the orbit
"""
struct InitialGuess{U<:AbstractArray{<:Real}, R<:Union{Real, Nothing}}
    u0::U
    T::R
end
InitialGuess(ds::DynamicalSystem, T=nothing) = InitialGuess(current_state(ds), T)


"""
A structure that contains information about a periodic orbit.

* `points::StateSpaceSet` - points in the periodic orbit. This container 
  always holds the whole orbit. Given a point `u` in the periodic orbit, the rest of the 
  orbit is obtained with `complete_orbit`. 
* `T::Real` - the period of the orbit
* `stable::Union{Bool, Nothing}` - local stability of the periodic orbit. Unknown stability 
  is set to `nothing`.

"""
struct PeriodicOrbit{D, B, R<:Real}
    points::StateSpaceSet{D, B}
    T::R
    stable::Union{Bool, Nothing}
end

"""
    PeriodicOrbit(ds::ContinuousTimeDynamicalSystem, u0::AbstractArray{<:Real},
        T::AbstractFloat, Δt=0.1, stable=nothing) → po

Given a point `u0` on the periodic orbit of the dynamical system `ds` and the period `T` 
of the orbit, the remaining points of the orbit are computed and stored in the `points` 
field of the returned `po::PeriodicOrbit`. The orbit which contains infinitely many points
 is approximated by calculating a trajectory with step `Δt`. The trajectory is stored in 
 `po.points`. 
"""
function PeriodicOrbit(ds::ContinuousTimeDynamicalSystem, u0::AbstractArray{<:Real}, 
    T::AbstractFloat, Δt=0.01, stable::Union{Bool, Nothing}=nothing)
    return PeriodicOrbit(complete_orbit(ds, u0, T; Δt=Δt), T, stable)
end

"""
    PeriodicOrbit(ds::DiscreteTimeDynamicalSystem, u0::AbstractArray{<:Real}, 
        T::Integer, stable=nothing) → po

Given a point `u0` on the periodic orbit of the dynamical system `ds` and the period `T` 
of the orbit, the remaining points of the orbit are computed and stored in the `points` 
field of the returned `po::PeriodicOrbit`. The orbit is obtained by iterating the periodic 
point `T-1` times and the points are stored in `po.points`.
"""
function PeriodicOrbit(ds::DiscreteTimeDynamicalSystem, u0::AbstractArray{<:Real}, T::Integer, stable::Union{Bool, Nothing}=nothing)
    discrete_timestep = 1
    return PeriodicOrbit(complete_orbit(ds, u0, T; Δt=discrete_timestep), T, stable)
end

"""
Abstract type `PeriodicOrbitFinder` represents a supertype for all the periodic orbit detection algorithms.
"""
abstract type PeriodicOrbitFinder end

"""
    periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, ig::InitialGuess = InitialGuess(ds)) → PeriodicOrbit

Try to find single periodic orbit of the dynamical system `ds` using the algorithm `alg` given some initial guess `ig`.
For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.
"""
function periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, ig::InitialGuess = InitialGuess(ds))
    result::PeriodicOrbit
    return result
end

"""
    periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = InitialGuess(ds)) → Vector{PeriodicOrbit}

Try to find multiple periodic orbits of the dynamical system `ds` using the algorithm `alg` given some initial guesses `igs`.
For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.
"""
function periodic_orbits(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = [InitialGuess(ds)])
    result::Vector{PeriodicOrbit}
    return result
end

"""
    isdiscretetime(po::PeriodicOrbit) → true/false

Return `true` if the periodic orbit belongs to a discrete-time dynamical system
`false` if it belongs to a continuous-time dynamical system.
"""
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D,B,R}) where {D,B,R<:Integer}
    true
end
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D,B,R}) where {D,B,R<:AbstractFloat}
    false
end


"""
    complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; kwargs...) → StateSpaceSet

Complete the periodic orbit `po` of period `po.T`. For POs of discrete systems, it means iterating 
the periodic point `po.T` times. For POs of continuous-time systems, it means integrating the system for 
`po.T` time units with step `Δt`. For POs of discrete-time systems `Δt` must be equal to `1`. 

## Keyword arguments

* `Δt` : step size for continuous-time systems.
"""
function complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; Δt::Real=1)
    isdiscrete = isdiscretetime(ds)
    isdiscrete &&  Δt ≠ 1 && throw(ArgumentError("Δt must be equal to 1 for discrete-time systems")) 
    traj, _ = trajectory(
        ds,
        isdiscrete ? T - 1 : T,
        u0;
        Δt=Δt
    )
    return traj
end


"""
    podistance(po1::PeriodicOrbit, po2::PeriodicOrbit, [, distance]) → Real

Compute the distance between two periodic orbits `po1` and `po2`. 
Periodic orbits`po1` and `po2` and the dynamical system `ds` all have to 
be either discrete-time or continuous-time.
Distance between the periodic orbits is computed using the given distance function `distance`.
The default distance function is `StrictlyMinimumDistance(true, Euclidean())` which finds the minimal 
Euclidean distance between any pair of points where one point belongs to `po1` and the other to `po2``. 
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

Return `true` if the periodic orbits `po1` and `po2` are equal within the given thresholds.

## Keyword arguments

* `Tthres` : distance between periodic orbits must be less than this threshold
* `dthres` : difference in periods of the periodic orbits must be less than this threshold
* `distance` : distance function used to compute the distance between the periodic orbits

Distance between the orbits is computed using the given distance function `distance`.
The default distance function is `StrictlyMinimumDistance(true, Euclidean())` which finds the minimal 
Euclidean distance between any pair of points where one point belongs to `po1` and the other to `po2``. 
For other options of the distance function, see `StateSpaceSets.set_distance`.
Custom distance function can be provided as well.
"""
function poequal(
    po1::PeriodicOrbit, po2::PeriodicOrbit;
    Tthres=1e-3,
    dthres=1e-3,
    distance=StrictlyMinimumDistance(true, Euclidean())
)
    if abs(po1.T - po2.T) > Tthres
        return false
    end
    d = podistance(po1, po2, distance)
    return d < dthres
end


"""
    uniquepos(pos::Vector{PeriodicOrbit}; atol=1e-6) → Vector{PeriodicOrbit}

Return a vector of unique periodic orbits from the vector `pos` of periodic orbits.
By unique we mean that the distance between any two periodic orbits in the vector is 
greater than `atol`. To see details about the distance function, see `podistance`.

## Keyword arguments

* `atol` : minimal distance between two periodic orbits for them to be considered unique.
"""
function uniquepos(pos::Vector{PeriodicOrbit}; atol::Real=1e-6)
    length(pos) == 0 && return pos
    unique_pos = typeof(pos[end])[]
    pos = copy(pos)

    while length(pos) > 0
        push!(unique_pos, pop!(pos))
        filter!(x -> podistance(x, unique_pos[end]) > atol, pos)
    end

    return unique_pos
end