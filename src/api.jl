export InitialGuess, 
    PeriodicOrbit, 
    PeriodicOrbitFinder,
    isdiscretetime,
    complete_orbit,
    podistance,
    uniquepos,
    isstable,
    poequal,
    periodic_orbit,
    periodic_orbits

import DynamicalSystemsBase
using LinearAlgebra: norm, eigvals

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
     always holds the whole orbit. Given a point `u` in the periodic orbit,
    the rest of the orbit is obtained with `complete_orbit`. 
    * `T::Real` - the period of the orbit
    * `stable::Bool` - local stability of the periodic orbit

"""
struct PeriodicOrbit{D, B, R<:Real}
    points::StateSpaceSet{D, B}
    T::R
    stable::Union{Bool, Missing}
end

"""
    PeriodicOrbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, Δt=1; kwargs...) → po

Given a point `u0` in the periodic orbit of the dynamical system `ds` and the period `T` of the orbit,
the remaining points of the orbit are computed and stored in the `points` field of the returned `PeriodicOrbit`.
In case of continuous dynamical systems, the orbit which contains infinetely many points is approximated by a grid with step 
`Δt` and the points are stored in `po.points`. In case of discrete dynamical systems, the orbit is 
obtained by iterating the periodic point `T-1` times and the points are stored in `po.points`.
Local stability of the periodic orbit is determined and stored in the `po.stable` field.
For determining the stability, the Jacobian matrix `jac` is used. The default Jacobian is 
obtained by automatic differentiation.

## Keyword arguments

* `jac` : Jacobian matrix of the dynamical system. Default is obtained by automatic differentiation.

"""
function PeriodicOrbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, Δt=1; jac=autodiff_jac(ds))
    minT = _minimal_period(ds, u0, T) # TODO: allow passing kwargs to _minimal_period
    return PeriodicOrbit(complete_orbit(ds, u0, minT; Δt=Δt), minT, isstable(ds, u0, minT, jac))
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
    complete_orbit(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real; kwargs...) → StateSpaceSet

Complete the periodic orbit `po` of period `po.T`. For POs of discrete systems, it means iterating 
the periodic point `po.T` times. For POs of continuous systems, it means integrating the system for 
`po.T` time units with step `Δt`. For POs of discrete systems `Δt` must be equal to `1`. 

## Keyword arguments

* `Δt` : step size for continuous systems.
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
    podistance(po1::PeriodicOrbit, po2::PeriodicOrbit, [, distance]) → Real

Compute the distance between two periodic orbits `po1` and `po2`. 
Periodic orbits`po1` and `po2` and the dynamical system `ds` all have to 
be either discrete or continuous.
Distance between the periodic orbits is computed using the given distance function `distance`.
The default distance function is `StrictlyMinimumDistance(true, Euclidean())` which finds the minimal 
Euclidean distance between any pair of points where one point belongs to `po1` and the other to `po2``. 
For other options of the distance function, see `StateSpaceSets.set_distance`.
Custom distance function can be provided as well.
"""
function podistance(po1, po2, distance = StrictlyMinimumDistance(true, Euclidean()))
    type1 = isdiscretetime(po1)
    type2 = isdiscretetime(po2)
    if type1 == type2
        return set_distance(po1.points, po2.points, distance)
    else
        throw(ArgumentError("Both periodic orbits have to be either discrete or continuous."))
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
        Tthres = 1e-3,
        dthres = 1e-3,
        distance = StrictlyMinimumDistance(true, Euclidean())
    )
    if abs(po1.T - po2.T) > Tthres
        return false
    end
    d = podistance(po1, po2, distance)
    return d < dthres
end


"""
    uniquepos(pos::Vector{PeriodicOrbit}, atol=1e-6) → Vector{PeriodicOrbit}

Return a vector of unique periodic orbits from the vector `pos` of periodic orbits.
By unique we mean that the distance between any two periodic orbits in the vector is 
greater than `atol`. To see details about the distance function, see `podistance`.
"""
function uniquepos(pos::Vector{PeriodicOrbit{D, B, R}}, atol::Real=1e-6) where {D, B, R}
    length(pos) == 0 && return pos
    unique_pos = typeof(pos[end])[]
    pos = copy(pos)

    while length(pos) > 0
        push!(unique_pos, pop!(pos))
        filter!(x -> podistance(x, unique_pos[end]) > atol, pos)
    end

    return unique_pos
end

function autodiff_jac(ds::DynamicalSystem)
    # TODO: where is this defined? Define if needed
end

"""
    isstable(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, jac) → true/false/missing

Determine the local stability of the point `u0` laying on the periodic orbit with period `T`
using the jacobian `jac`. Returns `true` if the periodic orbit is stable, `false` if it is unstable.

For discrete systems, the stability is determined using eigenvalues of the jacobian of `T`-th 
iterate of the dynamical system `ds` at the point `u0`. If the maximum absolute value of the eigenvalues 
is less than `1`, the periodic orbit is marked as stable.

For continuous systems, the stability check is not implemented yet.

For systems where stability cannot be determined, the function returns `missing`.
"""
function isstable(ds::DynamicalSystem, u0::AbstractArray{<:Real}, T::Real, jac)
    return _isstable(ds, u0, T, jac)
end

function _isstable(ds::DeterministicIteratedMap, u0::AbstractArray{<:Real}, T::Integer, jac)
    # TODO: implement or IIP jacobians
    T < 1 && throw(ArgumentError("Period must be a positive integer."))
    reinit!(ds, u0)
    J = jac(u0, current_parameters(ds), 0.0)

    # this can be derived from chain rule
    for _ in 2:T
        J = jac(current_state(ds), current_parameters(ds), 0.0) * J
        step!(ds, 1)
    end

    eigs = eigvals(Array(J))
    return maximum(abs.(eigs)) < 1
end

function _isstable(ds::PoincareMap, u0::AbstractArray{<:Real}, T::Integer, jac)
    missing
end

function _isstable(ds::ContinuousTimeDynamicalSystem, u0::AbstractArray{<:Real}, T::AbstractFloat, jac)
    @warn "Stability check for continuous systems is not implemented yet. Returning false."
    return false
end