# The API

## The PeriodicOrbit API

```@docs
PeriodicOrbit
minimal_period
isstable
```
For clarity, here is the docstring for a [`jacobian`](@ref) function from `DynamicalSystemsBase.jl` 
which is used for generating jacobians of dynamical rules via automatic differentiation.
```@docs
jacobian
```

```@docs
PeriodicOrbits.isdiscretetime
podistance
uniquepos
poequal
```

## The InitialGuess API

```@docs
InitialGuess
```

## The Detection of Periodic Orbits

```@docs
PeriodicOrbitFinder
periodic_orbit
periodic_orbits
```

### Algorithms for Discrete-Time Systems


### Algorithms for Continuous-Time Systems

- [`OptimizedShooting`](@ref)