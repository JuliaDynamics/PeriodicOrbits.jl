# Tutorial

Let's attempt to detect a periodic orbit of the Lorenz system. First we define the Lorenz 
system itself.

```@example MAIN
using PeriodicOrbits

function lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)
    return CoupledODEs(lorenz_rule, u0, [σ, ρ, β])
end
@inbounds function lorenz_rule(u, p, t)
    du1 = p[1]*(u[2]-u[1])
    du2 = u[1]*(p[2]-u[3]) - u[2]
    du3 = u[1]*u[2] - p[3]*u[3]
    return SVector{3}(du1, du2, du3)
end

ds = lorenz()
```

Next, we give initial guess of the location of the periodic orbit and its period.

```@example MAIN
u0_guess = SVector(3.5, 3.0, 0.0)
T_guess = 5.2
ig = InitialGuess(u0_guess, T_guess) 
```
Then we pick an appropriate algorithm that will detect the PO. In this case we can use 
any algorithm intended for continuous-time dynamical systems. We choose Optimized Shooting 
algorithm, for more information see [`OptimizedShooting`](@ref).

```@example MAIN
alg = OptimizedShooting(Δt=0.01, n=3)
```

Finally, we combine all the pieces to find the periodic orbit.

```@example MAIN
po = periodic_orbit(ds, alg, ig)
po
```

The closed curve of the periodic orbit can be visualized using plotting library such as 
[`Makie`](https://github.com/MakieOrg/Makie.jl).


```@example MAIN
using CairoMakie

u0 = po.points[1]
T = po.T
traj, t = trajectory(ds, T, u0; Dt = 0.01)
fig = Figure()
ax = Axis3(fig[1,1], azimuth = 0.6pi, elevation= 0.1pi)
lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)
scatter!(ax, u0)
fig
```

To ensure that the detected period is minimal, eg. it is not a multiple of the minimal 
period, we can use [`minimal_period`](@ref).

```@example MAIN
minT_po = minimal_period(ds, po)
minT_po
```

Whether two periodic orbits are equivalent up to some tolerance. Function [`poequal`](@ref) 
can be used.

```@example MAIN
equal = poequal(po, minT_po; dthres=1e-3, Tthres=1e-3)
"Detected periodic orbit had minimal period: $(equal)"
```

To determine whether found periodic orbit is stable or unstable, we can apply 
[`isstable`](@ref) function.

```@example MAIN
po = isstable(ds, po)
"Detected periodic orbit is $(po.stable ? "stable" : "unstable")."
```