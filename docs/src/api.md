# The Public API

## Main functions

```@docs
periodic_orbit
periodic_orbits
PeriodicOrbitFinder
InitialGuess
PeriodicOrbit
```

## Algorithms for Discrete-Time Systems

- [`SchmelcherDiakonos`](@ref)
- [`DavidchackLai`](@ref)

```@docs
SchmelcherDiakonos
lambdamatrix
lambdaperms
```

```@docs
DavidchackLai
```

## Algorithms for Continuous-Time Systems

- [`OptimizedShooting`](@ref)

```@docs
OptimizedShooting
```

## Utility functions

```@docs
minimal_period
postability
uniquepos
poequal
PeriodicOrbits.isdiscretetime
podistance
```


## Adding new algorithms

To implement a new periodic orbit finding algorithm, simply create a new type,
make it subtype `PeriodicOrbitFinder`, and then extend the function [`periodic_orbit`](@ref) for it.
