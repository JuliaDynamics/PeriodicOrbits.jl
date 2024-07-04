export InitialGuess, 
    PeriodicOrbit, 
    PeriodicOrbitFinder,
    isdiscretetime,
    complete_orbit,
    is_complete,
    distance,
    true_period,
    uniquepos,
    stable,
    equal,
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

    * `points::StateSpaceSet` - which is a `StateSpaceSet` of points in the periodic orbit. 
    This container always holds the whole orbit. Given a point `u` in the periodic orbit,
    the rest of the orbit is obtained with `complete_orbit`. 
    * `T::Real` - the period of the orbit

"""
struct PeriodicOrbit{D, B, R<:Real}
    points::StateSpaceSet{D, B}
    T::R
end

"""
    PeriodicOrbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, Δt=1) → po

Given a point `u0` in the periodic orbit of the dynamical system `ds` and the period `T` of the orbit,
the remaining points of the orbit are computed and stored in the `points` field of the returned `PeriodicOrbit`.
In case of continuous dynamical systems, the orbit which contains infinetely many points is approximated by a grid with step 
`Δt` and the points are stored in `po.points`. In case of discrete dynamical systems, the orbit is obtained by iterating the
periodic point `T` times and the points are stored in `po.points`.
"""
function PeriodicOrbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, Δt=1)
    return PeriodicOrbit(complete_orbit(ds, u0, T; Δt=Δt), T)
end


"""
    isdiscretetime(po::PeriodicOrbit) → true/false

Return `true` if the periodic orbit belongs to a discrete dynamical system
`false` if it belongs to a continuous dynamical system.
"""
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D, B, R}) where {D, B, R <: Integer}
    true
end
function DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{D, B, R}) where {D, B, R <: AbstractFloat}
    false
end


"""
    complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; Δt::Real=1) → StateSpaceSet

Complete the periodic orbit `po` of period `po.T`. For POs of discrete systems, it means iterating 
the periodic point `po.T` times. For POs of continuous systems, it means integrating the system for 
`po.T` time units with step `Δt`. For POs of discrete systems `Δt` must be equalt to `1`. 
"""
function complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; Δt::Real=1)
    isdiscrete = isdiscretetime(ds)
    isdiscrete &&  Δt ≠ 1 && throw(ArgumentError("Δt must be equal to 1 for discrete systems")) 
    traj, _ = trajectory(
        ds, 
        isdiscrete ? T-1 : T, 
        u0; 
        Δt=Δt
    )
    return traj
end


"""
    distance(po1::PeriodicOrbit, po2::PeriodicOrbit, distance = StrictlyMinimumDistance(true, Euclidean())) → Real

Computes the distance between two periodic orbits `po1` and `po2`. 
Periodic orbits`po1` and `po2` and the dynamical system `ds` all have to 
be either discrete or continuous.
"""
function distance(po1, po2, distance = StrictlyMinimumDistance(true, Euclidean()))
    type1 = isdiscretetime(po1)
    type2 = isdiscretetime(po2)
    if type1 == type2
        return set_distance(po1.points, po2.points, distance)
    else
        throw(ArgumentError("Both periodic orbits have to be either discrete or continuous."))
    end
end


"""
    equal(po1::PeriodicOrbit, po2::PeriodicOrbit; 
    Tthres=1e-3, dthres=1e-3, dist=StrictlyMinimumDistance(true, Euclidean())) → true/false

Returns `true` if the periodic orbits `po1` and `po2` are equal within the given thresholds.
Distance between the orbits is computed using the given distance function `dist`.
"""
function equal( # better name maybe? isapprox?
        po1::PeriodicOrbit, po2::PeriodicOrbit;
        Tthres = 1e-3,
        dthres = 1e-3,
        dist = StrictlyMinimumDistance(true, Euclidean())
    )
    if abs(po1.T - po2.T) > Tthres
        return false
    end
    d = distance(po1, po2, dist)
    return d < dthres
end

"""
    true_period(ds::DynamicalSystem, po::PeriodicOrbit, atol=1e-4) → po

Computes the true (minimal) period of the periodic orbit `po` of the dynamical system `ds`.
Returns the periodic orbit with the true period.
"""
function true_period(ds::DynamicalSystem, po::PeriodicOrbit, atol=1e-4)
    type1 = isdiscretetime(ds)
    type2 = isdiscretetime(po)
    if type1 == type2
        return _true_period(ds, po, atol)
    else
        throw(ArgumentError("Both the periodic orbit and the dynamical system have to be either discrete or continuous."))
    end
end

function _true_period(ds::DiscreteTimeDynamicalSystem, po::PeriodicOrbit, atol)
    u = po.points[1]
    for n in 1:po.T-1
        po.T % n != 0 && continue
        reinit!(ds, u)
        step!(ds, n)
        if norm(u - current_state(ds)) < atol
            return PeriodicOrbit(ds, u, n, 1)
        end
    end
    return po
end

function _true_period(ds::ContinuousTimeDynamicalSystem, po::PeriodicOrbit, atol)
    # if we encounter an algorithm that would return PO with non-true period, we will implement this function
    return po.T
end


"""
    uniquepos(pos::Vector{PeriodicOrbit}, atol=1e-6) → Vector{PeriodicOrbit}

Returns a vector of unique periodic orbits from the vector `pos` of periodic orbits.
By unique we mean that the distance between any two periodic orbits in the vector is 
greater than `atol`. To see details about the distance function, see `distance`.
"""
function uniquepos(pos::Vector{PeriodicOrbit{D, B, R}}, atol::Real=1e-6) where {D, B, R}
    unique_pos = PeriodicOrbit[]
    
    filter(po -> begin
        if all(unique_po -> distance(unique_po, po) > atol, unique_pos)
            push!(unique_pos, po)
            true  # Keep this element
        else
            false  # Exclude this element
        end
    end, pos)
    
    unique_pos
end


"""
    stable(ds::DynamicalSystem, po::PeriodicOrbit) → true/false

Determine the local stability of the periodic orbit `po`.
"""
function stable(ds::DynamicalSystem, po::PeriodicOrbit; jac=autodiff_jac(ds))
    throw("Function not implemented yet.")
end


"""
Abstract type `PeriodicOrbitFinder` represents a supertype for all the periodic orbit detection algorithms.
"""
abstract type PeriodicOrbitFinder end

"""
    periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = InitialGuess(ds)) → po

Tries to find single periodic orbit of the dynamical system `ds` using the algorithm `alg` given some initial guesses `igs`.
For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.
"""
function periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = InitialGuess(ds))
    result::PeriodicOrbit
    return result
end

"""
    periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = InitialGuess(ds)) → po

Tries to find multiple periodic orbits of the dynamical system `ds` using the algorithm `alg` given some initial guesses `igs`.
For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.
"""
function periodic_orbits(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = [InitialGuess(ds)])
    result::Vector{PeriodicOrbit}
    return result
end