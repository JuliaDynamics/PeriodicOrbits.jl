using PeriodicOrbits: lambdamatrix, lambdaperms
using PeriodicOrbits
using DynamicalSystems

begin
    ds = Systems.henon()
    traj, t = trajectory(ds, 100)
    igs = InitialGuess[InitialGuess(x, 0.0) for x in traj]
end

begin
    alg1 = DavidchackLai(n=6, m=6)
    @time orbits1 = periodic_orbits(ds, alg1, igs)
    orbits1 = unique(orbits1)
end