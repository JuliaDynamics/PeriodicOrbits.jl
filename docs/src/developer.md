# Developer's Docs

To implement a new periodic orbit finding algorithm, simply create a new type,
make it subtype `PeriodicOrbitFinder`, and then extend the function [`periodic_orbit`](@ref) for it.

The following functions may be useful for you when developing the code of the algorithm (you don't need to extend any of them):

```@docs
minimal_period
isstable
uniquepos
poequal
PeriodicOrbits.isdiscretetime
podistance
```