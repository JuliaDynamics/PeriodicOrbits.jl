# Algorithms

## Optimized Shooting Method
```@docs
OptimizedShooting
```

## Example
```@example MAIN
using PeriodicOrbits
using CairoMakie
using OrdinaryDiffEq

@inbounds function roessler_rule(u, p, t)
    du1 = -u[2]-u[3]
    du2 = u[1] + p[1]*u[2]
    du3 = p[2] + u[3]*(u[1] - p[3])
    return SVector{3}(du1, du2, du3)
end

function plot_result(res, ds; azimuth = 1.3 * pi, elevation = 0.3 * pi)
    traj, t = trajectory(ds, res.T, res.points[1]; Dt = 0.01)
    fig = Figure()
    ax = Axis3(fig[1,1], azimuth = azimuth, elevation=elevation)
    lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)
    scatter!(ax, res.points[1])
    return fig
end

ig = InitialGuess(SVector(2.0, 5.0, 10.0), 10.2)
OSalg = OptimizedShooting(Δt=0.01, n=3)
ds = CoupledODEs(roessler_rule, [1.0, -2.0, 0.1], [0.15, 0.2, 3.5])
res = periodic_orbit(ds, OSalg, ig)
plot_result(res, ds; azimuth = 1.3pi, elevation=0.1pi)
```
## Schmelcher & Diakonos

```@docs
SchmelcherDiakonos
lambdamatrix
lambdaperms
```

### Standard Map example
For example, let's find the fixed points of the Standard map of period 2, 3, 4, 5, 6
and 8. We will use all permutations for the `signs` but only one for the `inds`.
We will also only use one `λ` value, and a 11×11 density of initial conditions.

First, initialize everything
```@example MAIN
using PeriodicOrbits

function standardmap_rule(x, k, n)
    theta = x[1]; p = x[2]
    p += k[1]*sin(theta)
    theta += p
    return SVector(mod2pi(theta), mod2pi(p))
end

standardmap = DeterministicIteratedMap(standardmap_rule, rand(2), [1.0])
xs = range(0, stop = 2π, length = 11); ys = copy(xs)
ics = InitialGuess[InitialGuess(SVector{2}(x,y), nothing) for x in xs for y in ys]

# All permutations of [±1, ±1]:
signss = lambdaperms(2)[2] # second entry are the signs

# I know from personal research I only need this `inds`:
indss = [[1,2]] # <- must be container of vectors!

λs = [0.005] # <- vector of numbers

periods = [2, 3, 4, 5, 6, 8]
ALLFP = Vector{PeriodicOrbit}[]

standardmap
```
Then, do the necessary computations for all periods

```@example MAIN
for o in periods
    SDalg = SchmelcherDiakonos(o=o, λs=λs, indss=indss, signss=signss, maxiters=30000)
    FP = periodic_orbits(standardmap, SDalg, ics)
    FP = uniquepos(FP; atol=1e-5)
    push!(ALLFP, FP)
end
```

Plot the phase space of the standard map
```@example MAIN
using CairoMakie
iters = 1000
dataset = trajectory(standardmap, iters)[1]
for x in xs
    for y in ys
        append!(dataset, trajectory(standardmap, iters, [x, y])[1])
    end
end

fig = Figure()
ax = Axis(fig[1,1]; xlabel = L"\theta", ylabel = L"p",
    limits = ((xs[1],xs[end]), (xs[1],xs[end]))
)
scatter!(ax, dataset[:, 1], dataset[:, 2]; markersize = 1, color = "black")
fig
```

and finally, plot the fixed points
```@example MAIN
markers = [:diamond, :utriangle, :rect, :pentagon, :hexagon, :circle]

for i in eachindex(ALLFP)
    FP = ALLFP[i]
    o = periods[i]
    points = Tuple{Float64, Float64}[]
    for po in FP
        append!(points, [Tuple(x) for x in po.points])
    end
    println(points)
    scatter!(ax, points; marker=markers[i], color = Cycled(i),
        markersize = 30 - 2i, strokecolor = "grey", strokewidth = 1, label = "period $o"
    )
end
axislegend(ax)
fig
```

Okay, this output is great, and we can tell that it is correct because:

1. Fixed points of period $n$ are also fixed points of period $2n, 3n, 4n, ...$
2. Besides fixed points of previous periods, *original* fixed points of
   period $n$ come in (possible multiples of) $2n$-sized pairs (see e.g. period 5).
   This is a direct consequence of the Poincaré–Birkhoff theorem.