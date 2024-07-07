using PeriodicOrbits

function logistic(x0=0.4; r = 4.0)
    return DeterministicIteratedMap(logistic_rule, SVector(x0), [r])
end
logistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))

#%%
ds = logistic(r = 1+sqrt(8))
ig = InitialGuess(ds)
ig = InitialGuess(ds, 1.3)
ig = InitialGuess(ds, 1)
traj, t = trajectory(ds, 50, SVector{1}([0.5]))
u0 = traj[end]
po1 = PeriodicOrbit(ds, u0, 3)
po2 = PeriodicOrbit(ds, SVector{1}([0.99]), 3)
po3 = po1
isdiscretetime(po1)
podistance(po1, po2)
poequal(po1, po2)
poequal(po1, po3)
po1 = PeriodicOrbit(ds, po1.points[1], po1.T*4)
po4 = minimal_period(ds, po1)
uniquepos([po1, po1, po2, po3, po4, po1], 1e-4)