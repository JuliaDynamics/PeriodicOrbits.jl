export InitialGuess, 
    PeriodicOrbit, 
    PeriodicOrbitFinder,
    isdiscretetime,
    complete_orbit!,
    is_complete,
    distance,
    true_period,
    in,
    unique,
    stable,
    periodic_orbit,
    periodic_orbits

import Base: unique, in
import DynamicalSystemsBase

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

    * `points` - which is a vector of points in the periodic orbit. These can be all the 
    points of the periodic orbit or only some of them. To get the complete orbit, see 
    `complete_orbit!` function.
    * `T` - the period of the orbit
"""
struct PeriodicOrbit{U<:AbstractArray{<:Real}, R<:Real}
    points::Vector{U}
    T::R
end


"""
    isdiscretetime(po::PeriodicOrbit) → true/false

Return `true` if period of periodic orbit `po` is a subtype 
of integer, `false` if it is a subtype of `AbstractFloat`.
"""
DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{<:AbstractArray{<:Real}, <:Integer}) = true
DynamicalSystemsBase.isdiscretetime(po::PeriodicOrbit{<:AbstractArray{<:Real}, <:AbstractFloat}) = false


"""
    complete_orbit!(ds::DynamicalSystem, po::PeriodicOrbit; Δt=1)

Complete the periodic orbit `po` of period `po.T`. For POs of discrete systems, it means iterating 
the periodic point `po.T` times. For POs of continuous systems, it means integrating the system for 
`po.T` time units with step `Δt`. For POs of discrete systems `Δt` must be equalt to `1`. 
The periodic orbit `po.points` is modified in place to store `po.T/Δt` points which lie on it.
"""
function complete_orbit!(ds::DynamicalSystem, po::PeriodicOrbit; Δt::Real=1)
    isdiscretetime(ds) &&  Δt ≠ 1 && throw(ArgumentError("Δt must be equal to 1 for discrete systems")) 
    u0 = po.points[1]
    empty!(po.points)
    T = isdiscretetime(ds) ? po.T-1 : po.T
    traj, _ = trajectory(ds, T, u0; Δt=Δt)
    append!(po.points, traj)
end


"""
    is_complete(po::PeriodicOrbit, Δt=0.1) → true/false

Return `true` if the periodic orbit `po` is complete, i.e. the number of points 
in the orbit is equal to the period `po.T` for POs of discrete systems or `po.T/Δt` for 
POs of continuous systems. Otherwise, return `false`.
"""
function is_complete(po::PeriodicOrbit, Δt::Real=1)
    isdiscretetime(po) &&  Δt ≠ 1 && throw(ArgumentError("Δt must be equal to 1 for discrete systems")) 
    length(po.points) == po.T/Δt
end


"""
    distance(ds::DynamicalSystem, po1::PeriodicOrbit, po2::PeriodicOrbit) → distance

Computes the distance between two periodic orbits `po1` and `po2`. 
Periodic orbits`po1` and `po2` and the dynamical system `ds` all have to 
be either discrete or continuous.
"""
function distance(ds::DynamicalSystem, po1::PeriodicOrbit, po2::PeriodicOrbit)
    type1 = isdiscretetime(ds)
    type2 = isdiscretetime(po1)
    type3 = isdiscretetime(po2)
    if type1 == type2 == type3
        return _distance(ds, po1, po2)
    else
        throw(ArgumentError("Both periodic orbits and the dynamical system have to be either discrete or continuous."))
    end
end

function _distance(ds::D, po1::PeriodicOrbit, po2::PeriodicOrbit) where {D<:DiscreteTimeDynamicalSystem}
    is_complete(po1) == false && complete_orbit!(ds, po1; Δt=1)
    _distance(po2.points[1], po1)
end

function _distance(ds::ContinuousTimeDynamicalSystem, po1::PeriodicOrbit, po2::PeriodicOrbit)
    throw("Function not implemented yet.")
end

function _distance(u::AbstractVector{<:Real}, po::PeriodicOrbit)
    # TODO check efficiency
    norm(broadcast(.-, u, po.points), -Inf)
end


"""
    true_period(ds::DynamicalSystem, po::PeriodicOrbit, atol=1e-6) → po

Computes the true (minimal) period of the periodic orbit `po` of the dynamical system `ds`.
Returns the periodic orbit with the true period.
"""
function true_period(ds::DynamicalSystem, po::PeriodicOrbit, atol=1e-6)
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
            return PeriodicOrbit([u], n)
        end
    end
    return po
end

function _true_period(ds::ContinuousTimeDynamicalSystem, po::PeriodicOrbit, atol)
    throw("Function not implemented yet.")
end


"""
    in(u0::AbstractArray{Real}, po::PeriodicOrbit, atol=1e-6) → true/false

Checks whether the point `u0` is in the periodic orbit `po`. Returns `true` if the distance between 
the `u0` and some point in the periodic orbit `po` is less than `atol`, `false` otherwise. This 
function doesn't complete the orbit `po`. Consider using `complete_orbit!` before calling this function.

"""
function Base.in(u0::AbstractArray{Real}, po::PeriodicOrbit, atol=1e-6)
    if isdiscretetime(po)
        return _distance(u0, po) < atol
    else
        # perhaps complete the orbit first and then check the same way as for discrete
        throw("Function not implemented yet.")
    end
end


"""
    unique(ds::DynamicalSystem, pos::Vector{PeriodicOrbit}, atol=1e-6) → Vector{PeriodicOrbit}

Returns a vector of unique periodic orbits from the vector `pos` of periodic orbits.
By unique we mean that the distance between any two periodic orbits in the vector is 
greater than `atol`. To see details about the distance function, see `distance`.
"""
function Base.unique(ds::DynamicalSystem, pos::Vector{PeriodicOrbit}, atol::Real=1e-6)
    if isempty(pos)
        return pos
    end
    
    newvec = Vector{typeof(pos[1])}(undef, length(pos))
    newvec[1] = pos[1]
    unique_count = 1

    for po in pos[2:end]
        if all(npo -> distance(ds, npo, po) > atol, newvec[1:unique_count])
            unique_count += 1
            newvec[unique_count] = po
        end
    end

    resize!(newvec, unique_count)
    return newvec
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