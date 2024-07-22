export minimal_period

using LinearAlgebra: norm, dot

"""
    minimal_period(ds::DynamicalSystem, po::PeriodicOrbit; kw...) → minT_po

Compute the minimal period of the periodic orbit `po` of the dynamical system `ds`.
Return the periodic orbit `minT_po` with the minimal period. 

## Keyword arguments

* `atol = 1e-4` : After stepping the point `u0` for a time `T`, it must return to `atol` neighborhood of itself to be considered periodic.
* `maxiter = 10` : Maximum number of Poincare map iterations. Continuous-time systems only. 
* `stepsize = 1e-4` : Step size for creation of the normal vector. Continuous-time systems only. 

## Description

For discrete systems, a valid period would be any natural multiple of the minimal period. 
Hence, all natural divisors of the period `po.T` are checked as a potential period. 
A point `u0` of the periodic orbit `po` is iterated `n` times and if the distance between the initial point `u0` 
and the final point is less than `atol`, the period of the orbit is `n`.

For continuous systems, a point `u0` of the periodic orbit is integrated for 
a short time `stepsize`. The resulting point `u1` is used to create a normal vector 
`a=(u1-u0)`. A Poincare map is created using this normal vector. Using the Poincare 
map, the hyperplane crossings are checked. Time of the first crossing that 
is within `atol` distance of the initial point `u0` is the minimal period.
At most `maxiter` crossings are checked.
"""
function minimal_period(ds::DynamicalSystem, po::PeriodicOrbit; kwargs...)
    type1 = isdiscretetime(ds)
    type2 = isdiscretetime(po)
    if type1 == type2
        newT = _minimal_period(ds, po.points[1], po.T; kwargs...)
        return _set_period(ds, po, newT)
    else
        throw(ArgumentError("Both the periodic orbit and the dynamical system have to be either discrete or continuous."))
    end
end

function _set_period(ds::DynamicalSystem, po, newT)
    if newT == po.T
        return po
    else
        # ensure that continuous po has the same amount of points
        isdiscretetime(ds) ? Δt = 1 : Δt = newT/(length(po.points)-1)
        return PeriodicOrbit(complete_orbit(ds, po.points[1], newT; Δt=Δt), newT, po.stable)
    end
end

function _minimal_period(ds::DiscreteTimeDynamicalSystem, u0, T; atol=1e-4)
    for n in 1:T-1
        T % n != 0 && continue
        reinit!(ds, u0)
        step!(ds, n)
        if norm(u0 - current_state(ds)) < atol
            return n
        end
    end
    return T
end

function _minimal_period(ds::ContinuousTimeDynamicalSystem, u0, T;atol=1e-4, maxiter=10, stepsize=1e-4)
    reinit!(ds, u0)
    step!(ds, stepsize)
    u1 = current_state(ds)
    a = u1 - u0
    b = dot(a, u0)
    pmap = PoincareMap(ds, [a..., b]; u0=u0)
    for _ in 1:maxiter
        step!(pmap)
        if norm(u0 - current_state(pmap)) <= atol
            return current_crossing_time(pmap)
        end
    end
    return T
end