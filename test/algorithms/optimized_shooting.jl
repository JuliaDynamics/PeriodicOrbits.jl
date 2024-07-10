using PeriodicOrbits
using CairoMakie

@inbounds function roessler_rule(u, p, t)
    du1 = -u[2]-u[3]
    du2 = u[1] + p[1]*u[2]
    du3 = p[2] + u[3]*(u[1] - p[3])
    return SVector{3}(du1, du2, du3)
end
function roessler_jacob(u, p, t)
    return SMatrix{3,3}(0.0,1.0,u[3],-1.0,p[1],0.0,-1.0,0.0,u[1]-p[3])          
end

#%%
a = 0.15; b=0.2; c=3.5
ds = CoupledODEs(roessler_rule, [1.0, -2.0, 0.1], [a, b, c]; diffeq = (abstol = 1e-16, reltol = 1e-16))
u0 = SVector(2.6286556703142154, 3.5094562051716300, 3.0000000000000000)
T = 5.9203402481939138
traj, t = trajectory(ds, T, u0; Dt = 0.01)


#%%
fig = Figure()
ax = Axis3(fig[1,1], azimuth = 1.3 * pi)
lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue)
display(fig)