using PeriodicOrbits

function logistic(x0=0.4; r = 4.0)
    return DeterministicIteratedMap(logistic_rule, SVector(x0), [r])
end
logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))
logistic_jacob(x, p, n) = @inbounds SMatrix{1,1}(p[1]*(1 - 2x[1]))

#%%
ds = logistic(r = 1+sqrt(8))
ig = InitialGuess(ds)
ig = InitialGuess(ds, 1.3)
ig = InitialGuess(ds, 1)
traj, t = trajectory(ds, 50, SVector{1}([0.5]))
u0 = traj[end]
jac = logistic_jacob
po1 = PeriodicOrbit(ds, u0, 3; jac=jac)
po2 = PeriodicOrbit(ds, SVector{1}([0.99]), 3; jac=jac)
po3 = po1
isdiscretetime(po1)
podistance(po1, po2)
poequal(po1, po2)
poequal(po1, po3)
po1 = PeriodicOrbit(ds, po1.points[1], po1.T*4; jac=jac)
po4 = minimal_period(ds, po1)
uniquepos([po1, po1, po2, po3, po4, po1], 1e-4)

r = 2.3
set_parameters!(ds, [r])
@assert isstable(ds, SVector{1}([(r-1)/r]), 1, jac) == true

r = 3.3
set_parameters!(ds, [r])
@assert isstable(ds, SVector{1}([(r-1)/r]), 1, jac) == false

function henon_rule(u, p, n) # here `n` is "time", but we don't use it.
    x, y = u # system state
    a, b = p # system parameters
    xn = 1.0 - a*x^2 + y
    yn = b*x
    return SVector(xn, yn)
end
henon_jacob(x, p, n) = SMatrix{2,2}(-2*p[1]*x[1], p[2], 1.0, 0.0)


u0 = [0.2, 0.3]
p0 = [1.4, 0.3]
henon = DeterministicIteratedMap(henon_rule, u0, p0)

isstable(henon, SVector{2}([0.2, 0.3]), 10, henon_jacob)


function standardmap(u0=[0.001245, 0.00875]; k = 0.971635)
    return DeterministicIteratedMap(standardmap_rule, u0, [k])
end
@inbounds function standardmap_rule(x, par, n)
    theta = x[1]; p = x[2]
    p += par[1]*sin(theta)
    theta += p
    while theta >= 2π; theta -= 2π; end
    while theta < 0; theta += 2π; end
    while p >= 2π; p -= 2π; end
    while p < 0; p += 2π; end
    return SVector(theta, p)
end
@inbounds standardmap_jacob(x, p, n) = SMatrix{2,2}(
    1 + p[1]*cos(x[1]), p[1]*cos(x[1]), 1, 1
)

smap = standardmap()
@assert isstable(smap, SVector(π, 0), 1, standardmap_jacob) == true
@assert isstable(smap, SVector(π, 0), 10, standardmap_jacob) == true
@assert isstable(smap, SVector(0, 0), 1, standardmap_jacob) == false
@assert isstable(smap, SVector(0, 0), 10, standardmap_jacob) == false