using PeriodicOrbits: lambdamatrix, lambdaperms
using PeriodicOrbits
using DynamicalSystems

#%%
ds = Systems.henon()
traj, t = trajectory(ds, 100)
igs = InitialGuess[InitialGuess(x, 0.0) for x in traj]


#%%
indss, signss = lambdaperms(dimension(ds))
alg2 = SchmelcherDiakonos(o=2, indss=indss, signss=signss, λs=[0.1])
alg3 = SchmelcherDiakonos(2, dimension(ds), 0.1; inftol=10.0, abstol=1e-8)
alg4 = SchmelcherDiakonos(2, [0.1], indss, signss)
@time orbits2 = periodic_orbits(ds, alg2, igs)
@time orbits3 = periodic_orbits(ds, alg3, igs)
@time orbits4 = periodic_orbits(ds, alg4, igs)

#%%
orbits = PeriodicOrbit[orbits2[1], orbits2[1], orbits2[1]]
unique(ds, orbits)

#%%
po = orbits2[1]
isdiscretetime(po)
complete_orbit!(ds, po)
true_period(ds, po)


#%%
po1 = orbits2[1]
po2 = orbits2[2]
distance(ds, po1, po2)