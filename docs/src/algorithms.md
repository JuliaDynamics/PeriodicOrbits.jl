# Algorithms

Here is the documentation of all available PO detection algorithms. To see examples of their usage, see [Examples](@ref).

## Optimized Shooting Method
```@docs
OptimizedShooting
```

## Schmelcher & Diakonos

```@docs
SchmelcherDiakonos
lambdamatrix
lambdaperms
```

## Davidchack & Lai

An extension of the [`SchmelcherDiakonos`](@ref) algorithm was proposed by Davidchack & Lai [Davidchack1999](@cite).
It works similarly, but it uses smarter seeding and an improved transformation rule.

```@docs
DavidchackLai
```
