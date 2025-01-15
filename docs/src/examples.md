# Examples

## Optimized Shooting Example
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
ds = CoupledODEs(roessler_rule, [1.0, -2.0, 0.1], [0.15, 0.2, 3.5]; diffeq=(abstol=1e-10, reltol=1e-10))
res = periodic_orbit(ds, OSalg, ig)
plot_result(res, ds; azimuth = 1.3pi, elevation=0.1pi)
```

## SchmelcherDiakonos example
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

## DavidchackLai example

### Logistic map example

The idea of periodic orbits can be illustrated easily on 1D maps. Finding all periodic orbits of period
$n$ is equivalent to finding all points $x$ such that $f^{n}(x)=x$, where $f^{n}$ is $n$-th composition of $f$. Hence, solving $f^{n}(x)-x=0$ yields such points. However, this is often impossible analytically. 
Let's see how [`DavidchackLai`](@ref) deals with it:

First let's start with finding periodic orbits with period $1$ to $9$ for the logistic map with parameter $3.72$.

```@example MAIN
using PeriodicOrbits
using CairoMakie

logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
ds = DeterministicIteratedMap(logistic_rule, SVector(0.4), [3.72])
seeds = InitialGuess[InitialGuess(SVector(i), nothing) for i in LinRange(0.0, 1.0, 10)]
alg = DavidchackLai(n=9, m=6, abstol=1e-6, disttol=1e-12)
output = periodic_orbits(ds, alg, seeds)
output = uniquepos(output);
```

Let's plot the periodic orbits of period $6$. 

```@example MAIN
function ydata(ds, period, xdata)
    ydata = typeof(current_state(ds)[1])[]
    for x in xdata
        reinit!(ds, x)
        step!(ds, period)
        push!(ydata, current_state(ds)[1])
    end
    return ydata
end

fig = Figure()
x = LinRange(0.0, 1.0, 1000)
period = 6
period6 = filter(po -> po.T == period, output)
fpsx = vcat([vec(po.points) for po in period6]...)
y = ydata(ds, period, [SVector(x0) for x0 in x])
fpsy = ydata(ds, period, fpsx)
axis = Axis(fig[1, 1])
axis.title = "Period $period"
lines!(axis, x, x, color=:black, linewidth=0.8)
lines!(axis, x, y, color = :blue, linewidth=1.7)
scatter!(axis, [i[1] for i in fpsx], fpsy, color = :red, markersize=15)
fig
```
Points $x$ which fulfill $f^{n}(x)=x$ can be interpreted as an intersection of the function 
$f^{n}(x)$ and the identity function $y=x$. Our result is correct because all the points of 
the intersection between the identity function and the sixth iterate of the logistic map 
were found.

### Henon Map example

Let's try to use [`DavidchackLai`](@ref) in higher dimension. We will try to detect 
all periodic points of Henon map of period `1` to `12`.

```@example MAIN
using PeriodicOrbits, CairoMakie
using LinearAlgebra: norm

function henon(u0=zeros(2); a = 1.4, b = 0.3)
    return DeterministicIteratedMap(henon_rule, u0, [a,b])
end
henon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])

ds = henon()
xs = LinRange(-3.0, 3.0, 10)
ys = LinRange(-10.0, 10.0, 10)
seeds = InitialGuess[InitialGuess(SVector{2}(x,y), nothing) for x in xs for y in ys]
n = 12
m = 6
alg = DavidchackLai(n=n, m=m, abstol=1e-7, disttol=1e-10)
output = periodic_orbits(ds, alg, seeds)
output = uniquepos(output)
output = [minimal_period(ds, po) for po in output]

markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :ltriangle, :rtriangle, :pentagon, :hexagon, :cross, :xcross, :star4]
fig = Figure(fontsize=18)
ax = Axis(fig[1,1])
for p = 12:-1:1
    pos = filter(x->x.T == p, output)
    for (index, po) in enumerate(pos)
        scatter!(ax, vec(po.points), markersize=10, color=Cycled(p), label = "T = $p", marker=markers[p])
    end
end
axislegend(ax, merge=true, unique=true, position=(0.0, 0.55))
fig

```
The theory of periodic orbits states that UPOs form sort of a skeleton of the chaotic attractor. Our results supports this claim since it closely resembles the Henon attractor.

Note that in this case parameter `m` has to be set to at least `6`. Otherwise, the algorithm 
fails to detect orbits of higher periods correctly.