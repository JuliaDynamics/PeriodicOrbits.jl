var documenterSearchIndex = {"docs":
[{"location":"references/#References","page":"References","title":"References","text":"","category":"section"},{"location":"references/","page":"References","title":"References","text":"Davidchack, R. L. and Lai, Y.-C. (1999). Efficient algorithm for detecting unstable periodic orbits in chaotic systems. Physical Review E 60, 6172–6175.\n\n\n\nDednam, W. and Botha, A. E. (2014). Optimized shooting method for finding periodic orbits of nonlinear dynamical systems.\n\n\n\nDiakonos, F. K.; Schmelcher, P. and Biham, O. (1998). Systematic Computation of the Least Unstable Periodic Orbits in Chaotic Attractors. Physical Review Letters 81, 4349–4352.\n\n\n\nPingel, D.; Schmelcher, P.; Diakonos, F. K. and Biham, O. (2000). Theory and applications of the systematic detection of unstable periodic orbits in dynamical systems. Physical Review E 62, 2119–2134.\n\n\n\nSchmelcher, P. and Diakonos, F. K. (1997). Detecting Unstable Periodic Orbits of Chaotic Dynamical Systems. Physical Review Letters 78, 4733–4736.\n\n\n\n","category":"page"},{"location":"api/#The-Public-API","page":"The Public API","title":"The Public API","text":"","category":"section"},{"location":"api/","page":"The Public API","title":"The Public API","text":"PeriodicOrbit\nInitialGuess\nperiodic_orbit\nperiodic_orbits","category":"page"},{"location":"api/#PeriodicOrbits.PeriodicOrbit","page":"The Public API","title":"PeriodicOrbits.PeriodicOrbit","text":"A structure that contains information about a periodic orbit.\n\npoints::StateSpaceSet - points of the periodic orbit. This container  always holds the complete orbit.\nT::Real - the period of the orbit\nstable::Union{Bool, Missing} - local stability of the periodic orbit. Unknown stability  is set to missing.\n\n\n\n\n\n","category":"type"},{"location":"api/#PeriodicOrbits.InitialGuess","page":"The Public API","title":"PeriodicOrbits.InitialGuess","text":"A structure that contains an initial guess for a periodic orbit detection algorithms.\n\nu0::AbstractArray{<:Real} - guess of a point in the periodic orbit\nT::Union{Real, Nothing} - guess of period of the orbit. Some algorithms do not require  the period guess to be given, in which case T is set to nothing.\n\n\n\n\n\n","category":"type"},{"location":"api/#PeriodicOrbits.periodic_orbit","page":"The Public API","title":"PeriodicOrbits.periodic_orbit","text":"periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, ig::InitialGuess = InitialGuess(ds)) → PeriodicOrbit\n\nTry to find single periodic orbit of the dynamical system ds using the algorithm alg given some initial guess ig. For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.\n\n\n\n\n\n","category":"function"},{"location":"api/#PeriodicOrbits.periodic_orbits","page":"The Public API","title":"PeriodicOrbits.periodic_orbits","text":"periodic_orbit(ds::DynamicalSystem, alg::PeriodicOrbitFinder, igs::Vector{InitialGuess} = InitialGuess(ds)) → Vector{PeriodicOrbit}\n\nTry to find multiple periodic orbits of the dynamical system ds using the algorithm alg given some initial guesses igs. For more details on the periodic orbit detection algorithms, see the documentation of the specific algorithm.\n\n\n\n\n\n","category":"function"},{"location":"api/#Algorithms-for-Discrete-Time-Systems","page":"The Public API","title":"Algorithms for Discrete-Time Systems","text":"","category":"section"},{"location":"api/","page":"The Public API","title":"The Public API","text":"SchmelcherDiakonos\nDavidchackLai","category":"page"},{"location":"api/#Algorithms-for-Continuous-Time-Systems","page":"The Public API","title":"Algorithms for Continuous-Time Systems","text":"","category":"section"},{"location":"api/","page":"The Public API","title":"The Public API","text":"OptimizedShooting","category":"page"},{"location":"developer/#Developer's-Docs","page":"Developer's Docs","title":"Developer's Docs","text":"","category":"section"},{"location":"developer/","page":"Developer's Docs","title":"Developer's Docs","text":"minimal_period\nisstable\nuniquepos\npoequal\nPeriodicOrbits.isdiscretetime\npodistance\nPeriodicOrbitFinder","category":"page"},{"location":"developer/#PeriodicOrbits.minimal_period","page":"Developer's Docs","title":"PeriodicOrbits.minimal_period","text":"minimal_period(ds::DynamicalSystem, po::PeriodicOrbit; kw...) → minT_po\n\nCompute the minimal period of the periodic orbit po of the dynamical system ds. Return the periodic orbit minT_po with the minimal period. In the literature, minimal  period is also called prime, principal or fundamental period.\n\nKeyword arguments\n\natol = 1e-4 : After stepping the point u0 for a time T, it must return to atol neighborhood of itself to be considered periodic.\nmaxiter = 40 : Maximum number of Poincare map iterations. Continuous-time systems only.  If the number of Poincare map iterations exceeds maxiter, but the point u0 has not  returned to atol neighborhood of itself, the original period po.T is returned.\nΔt = missing : The time step between points in the trajectory minT_po.points. If Δt  is missing, then Δt=minT_po.T/100 is used. Continuous-time  systems only. \n\nDescription\n\nFor discrete systems, a valid period would be any natural multiple of the minimal period.  Hence, all natural divisors of the period po.T are checked as a potential period.  A point u0 of the periodic orbit po is iterated n times and if the distance between  the initial point u0 and the final point is less than atol, the period of the orbit  is n.\n\nFor continuous systems, a point u0 of the periodic orbit is integrated for a very short  time. The resulting point u1 is used to create a normal vector a=(u1-u0) to a hyperplane  perpendicular to the trajectory at u0. A Poincare map is created using  this hyperplane. Using the Poincare map, the hyperplane crossings are checked. Time of the  first crossing that is within atol distance of the initial point u0 is the minimal  period. At most maxiter crossings are checked.\n\n\n\n\n\n","category":"function"},{"location":"developer/#PeriodicOrbits.isstable","page":"Developer's Docs","title":"PeriodicOrbits.isstable","text":"isstable(ds::CoreDynamicalSystem, po [, jac]) → new_po\n\nDetermine the local stability of the periodic orbit po using the jacobian rule jac.  Returns a new periodic orbit for which po.stable is set to true   if the periodic orbit  is stable or false if it is unstable.\n\nFor discrete-time systems, the stability is determined using eigenvalues of the jacobian  of po.T-th iterate of the dynamical system ds at the point po.points[1]. If the  maximum absolute value of the eigenvalues is less than 1, the periodic orbit is marked  as stable.\n\nFor continuous-time systems, the stability is determined by the Floquet multipliers of the  monodromy matrix. If the maximum absolute value of the Floquet multipliers is less than  1 (while neglecting the multiplier which is always 1), the periodic orbit is marked   as stable.\n\nThe default value of jacobian rule jac is obtained via automatic differentiation.\n\n\n\n\n\n","category":"function"},{"location":"developer/#PeriodicOrbits.uniquepos","page":"Developer's Docs","title":"PeriodicOrbits.uniquepos","text":"uniquepos(pos::Vector{<:PeriodicOrbit}; atol=1e-6) → Vector{PeriodicOrbit}\n\nReturn a vector of unique periodic orbits from the vector pos of periodic orbits. By unique we mean that the distance between any two periodic orbits in the vector is  greater than atol. To see details about the distance function, see podistance.\n\nKeyword arguments\n\natol : minimal distance between two periodic orbits for them to be considered unique.\n\n\n\n\n\n","category":"function"},{"location":"developer/#PeriodicOrbits.poequal","page":"Developer's Docs","title":"PeriodicOrbits.poequal","text":"poequal(po1::PeriodicOrbit, po2::PeriodicOrbit; kwargs...) → true/false\n\nReturn true if the periodic orbits po1 and po2 are equal within the given thresholds.\n\nKeyword arguments\n\nTthres=1e-3 : difference in periods of the periodic orbits must be less than this threshold\ndthres=1e-3 : distance between periodic orbits must be less than this threshold\ndistance : distance function used to compute the distance between the periodic orbits\n\nDistance between the orbits is computed using the given distance function distance. The default distance function is StrictlyMinimumDistance(true, Euclidean()) which finds  the minimal Euclidean distance between any pair of points where one point belongs to po1  and the other to po2. For other options of the distance function, see  StateSpaceSets.set_distance. Custom distance function can be provided as well.\n\n\n\n\n\n","category":"function"},{"location":"developer/#DynamicalSystemsBase.isdiscretetime","page":"Developer's Docs","title":"DynamicalSystemsBase.isdiscretetime","text":"isdiscretetime(po::PeriodicOrbit) → true/false\n\nReturn true if the periodic orbit belongs to a discrete-time dynamical system, false if  it belongs to a continuous-time dynamical system.\n\n\n\n\n\nisdiscretetime(ds::DynamicalSystem) → true/false\n\nReturn true if ds operates in discrete time, or false if it is in continuous time. This is information deduced from the type of ds.\n\n\n\n\n\n","category":"function"},{"location":"developer/#PeriodicOrbits.podistance","page":"Developer's Docs","title":"PeriodicOrbits.podistance","text":"podistance(po1::PeriodicOrbit, po2::PeriodicOrbit, [, distance]) → Real\n\nCompute the distance between two periodic orbits po1 and po2.  Periodic orbits po1 and po2 and the dynamical system ds all have to  be either discrete-time or continuous-time. Distance between the periodic orbits is computed using the given distance function distance. The default distance function is StrictlyMinimumDistance(true, Euclidean()) which finds the minimal  Euclidean distance between any pair of points where one point belongs to po1 and the other to po2.  For other options of the distance function, see StateSpaceSets.set_distance. Custom distance function can be provided as well.\n\n\n\n\n\n","category":"function"},{"location":"developer/#PeriodicOrbits.PeriodicOrbitFinder","page":"Developer's Docs","title":"PeriodicOrbits.PeriodicOrbitFinder","text":"PeriodicOrbitFinder\n\nSupertype for all the periodic orbit  detection algorithms. Each of the concrete subtypes of PeriodicOrbitFinder should  represent one given algorithm for detecting periodic orbits. This subtype will include  all the necessary parameters for the algorithm to work and optionally their default values. \n\n\n\n\n\n","category":"type"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#Optimized-Shooting-Example","page":"Examples","title":"Optimized Shooting Example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"using PeriodicOrbits\nusing CairoMakie\nusing OrdinaryDiffEq\n\n@inbounds function roessler_rule(u, p, t)\n    du1 = -u[2]-u[3]\n    du2 = u[1] + p[1]*u[2]\n    du3 = p[2] + u[3]*(u[1] - p[3])\n    return SVector{3}(du1, du2, du3)\nend\n\nfunction plot_result(res, ds; azimuth = 1.3 * pi, elevation = 0.3 * pi)\n    traj, t = trajectory(ds, res.T, res.points[1]; Dt = 0.01)\n    fig = Figure()\n    ax = Axis3(fig[1,1], azimuth = azimuth, elevation=elevation)\n    lines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)\n    scatter!(ax, res.points[1])\n    return fig\nend\n\nig = InitialGuess(SVector(2.0, 5.0, 10.0), 10.2)\nOSalg = OptimizedShooting(Δt=0.01, n=3)\nds = CoupledODEs(roessler_rule, [1.0, -2.0, 0.1], [0.15, 0.2, 3.5])\nres = periodic_orbit(ds, OSalg, ig)\nplot_result(res, ds; azimuth = 1.3pi, elevation=0.1pi)","category":"page"},{"location":"examples/#SchmelcherDiakonos-example","page":"Examples","title":"SchmelcherDiakonos example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"For example, let's find the fixed points of the Standard map of period 2, 3, 4, 5, 6 and 8. We will use all permutations for the signs but only one for the inds. We will also only use one λ value, and a 11×11 density of initial conditions.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"First, initialize everything","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using PeriodicOrbits\n\nfunction standardmap_rule(x, k, n)\n    theta = x[1]; p = x[2]\n    p += k[1]*sin(theta)\n    theta += p\n    return SVector(mod2pi(theta), mod2pi(p))\nend\n\nstandardmap = DeterministicIteratedMap(standardmap_rule, rand(2), [1.0])\nxs = range(0, stop = 2π, length = 11); ys = copy(xs)\nics = InitialGuess[InitialGuess(SVector{2}(x,y), nothing) for x in xs for y in ys]\n\n# All permutations of [±1, ±1]:\nsignss = lambdaperms(2)[2] # second entry are the signs\n\n# I know from personal research I only need this `inds`:\nindss = [[1,2]] # <- must be container of vectors!\n\nλs = [0.005] # <- vector of numbers\n\nperiods = [2, 3, 4, 5, 6, 8]\nALLFP = Vector{PeriodicOrbit}[]\n\nstandardmap","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Then, do the necessary computations for all periods","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"for o in periods\n    SDalg = SchmelcherDiakonos(o=o, λs=λs, indss=indss, signss=signss, maxiters=30000)\n    FP = periodic_orbits(standardmap, SDalg, ics)\n    FP = uniquepos(FP; atol=1e-5)\n    push!(ALLFP, FP)\nend","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Plot the phase space of the standard map","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using CairoMakie\niters = 1000\ndataset = trajectory(standardmap, iters)[1]\nfor x in xs\n    for y in ys\n        append!(dataset, trajectory(standardmap, iters, [x, y])[1])\n    end\nend\n\nfig = Figure()\nax = Axis(fig[1,1]; xlabel = L\"\\theta\", ylabel = L\"p\",\n    limits = ((xs[1],xs[end]), (xs[1],xs[end]))\n)\nscatter!(ax, dataset[:, 1], dataset[:, 2]; markersize = 1, color = \"black\")\nfig","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"and finally, plot the fixed points","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"markers = [:diamond, :utriangle, :rect, :pentagon, :hexagon, :circle]\n\nfor i in eachindex(ALLFP)\n    FP = ALLFP[i]\n    o = periods[i]\n    points = Tuple{Float64, Float64}[]\n    for po in FP\n        append!(points, [Tuple(x) for x in po.points])\n    end\n    println(points)\n    scatter!(ax, points; marker=markers[i], color = Cycled(i),\n        markersize = 30 - 2i, strokecolor = \"grey\", strokewidth = 1, label = \"period $o\"\n    )\nend\naxislegend(ax)\nfig","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Okay, this output is great, and we can tell that it is correct because:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Fixed points of period n are also fixed points of period 2n 3n 4n \nBesides fixed points of previous periods, original fixed points of period n come in (possible multiples of) 2n-sized pairs (see e.g. period 5). This is a direct consequence of the Poincaré–Birkhoff theorem.","category":"page"},{"location":"examples/#DavidchackLai-example","page":"Examples","title":"DavidchackLai example","text":"","category":"section"},{"location":"examples/#Logistic-map-example","page":"Examples","title":"Logistic map example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"The idea of periodic orbits can be illustrated easily on 1D maps. Finding all periodic orbits of period n is equivalent to finding all points x such that f^n(x)=x, where f^n is n-th composition of f. Hence, solving f^n(x)-x=0 yields such points. However, this is often impossible analytically.  Let's see how DavidchackLai deals with it:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"First let's start with finding periodic orbits with period 1 to 9 for the logistic map with parameter 372.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using PeriodicOrbits\nusing CairoMakie\n\nlogistic_rule(x, p, n) = @inbounds SVector(p[1]*x[1]*(1 - x[1]))\nds = DeterministicIteratedMap(logistic_rule, SVector(0.4), [3.72])\nseeds = InitialGuess[InitialGuess(SVector(i), nothing) for i in LinRange(0.0, 1.0, 10)]\nalg = DavidchackLai(n=9, m=6, abstol=1e-6, disttol=1e-12)\noutput = periodic_orbits(ds, alg, seeds)\noutput = uniquepos(output);","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Let's plot the periodic orbits of period 6. ","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"function ydata(ds, period, xdata)\n    ydata = typeof(current_state(ds)[1])[]\n    for x in xdata\n        reinit!(ds, x)\n        step!(ds, period)\n        push!(ydata, current_state(ds)[1])\n    end\n    return ydata\nend\n\nfig = Figure()\nx = LinRange(0.0, 1.0, 1000)\nperiod = 6\nperiod6 = filter(po -> po.T == period, output)\nfpsx = vcat([vec(po.points) for po in period6]...)\ny = ydata(ds, period, [SVector(x0) for x0 in x])\nfpsy = ydata(ds, period, fpsx)\naxis = Axis(fig[1, 1])\naxis.title = \"Period $period\"\nlines!(axis, x, x, color=:black, linewidth=0.8)\nlines!(axis, x, y, color = :blue, linewidth=1.7)\nscatter!(axis, [i[1] for i in fpsx], fpsy, color = :red, markersize=15)\nfig","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Points x which fulfill f^n(x)=x can be interpreted as an intersection of the function  f^n(x) and the identity function y=x. Our result is correct because all the points of  the intersection between the identity function and the sixth iterate of the logistic map  were found.","category":"page"},{"location":"examples/#Henon-Map-example","page":"Examples","title":"Henon Map example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Let's try to use DavidchackLai in higher dimension. We will try to detect  all periodic points of Henon map of period 1 to 12.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"using PeriodicOrbits, CairoMakie\nusing LinearAlgebra: norm\n\nfunction henon(u0=zeros(2); a = 1.4, b = 0.3)\n    return DeterministicIteratedMap(henon_rule, u0, [a,b])\nend\nhenon_rule(x, p, n) = SVector{2}(1.0 - p[1]*x[1]^2 + x[2], p[2]*x[1])\n\nds = henon()\nxs = LinRange(-3.0, 3.0, 10)\nys = LinRange(-10.0, 10.0, 10)\nseeds = InitialGuess[InitialGuess(SVector{2}(x,y), nothing) for x in xs for y in ys]\nn = 12\nm = 6\nalg = DavidchackLai(n=n, m=m, abstol=1e-7, disttol=1e-10)\noutput = periodic_orbits(ds, alg, seeds)\noutput = uniquepos(output)\noutput = [minimal_period(ds, po) for po in output]\n\nmarkers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :ltriangle, :rtriangle, :pentagon, :hexagon, :cross, :xcross, :star4]\nfig = Figure(fontsize=18)\nax = Axis(fig[1,1])\nfor p = 12:-1:1\n    pos = filter(x->x.T == p, output)\n    for (index, po) in enumerate(pos)\n        scatter!(ax, vec(po.points), markersize=10, color=Cycled(p), label = \"T = $p\", marker=markers[p])\n    end\nend\naxislegend(ax, merge=true, unique=true, position=(0.0, 0.55))\nfig\n","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"The theory of periodic orbits states that UPOs form sort of a skeleton of the chaotic attractor. Our results supports this claim since it closely resembles the Henon attractor.","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Note that in this case parameter m has to be set to at least 6. Otherwise, the algorithm  fails to detect orbits of higher periods correctly.","category":"page"},{"location":"#PeriodicOrbits.jl","page":"PeriodicOrbits.jl","title":"PeriodicOrbits.jl","text":"","category":"section"},{"location":"","page":"PeriodicOrbits.jl","title":"PeriodicOrbits.jl","text":"PeriodicOrbits","category":"page"},{"location":"#PeriodicOrbits","page":"PeriodicOrbits.jl","title":"PeriodicOrbits","text":"PeriodicOrbits.jl\n\n(Image: docsdev) (Image: docsstable) (Image: ) (Image: CI) (Image: codecov) (Image: Package Downloads)\n\nInterface and algorithms for finding both stable and unstable periodic orbits of dynamical systems based on the DynamicalSystems.jl ecosystem.\n\nTo install it, run import Pkg; Pkg.add(\"PeriodicOrbits\").\n\nAll further information is provided in the documentation, which you can either find online or build locally by running the docs/make.jl file.\n\nNote: Throughout the documentation, \"Periodic Orbit\" is often abbreviated as \"PO\" for brevity.\n\n\n\n\n\n","category":"module"},{"location":"tutorial/#Tutorial","page":"Tutorial","title":"Tutorial","text":"","category":"section"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Let's attempt to detect a periodic orbit of the Lorenz system. First we define the Lorenz  system itself.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using PeriodicOrbits\n\nfunction lorenz(u0=[0.0, 10.0, 0.0]; σ = 10.0, ρ = 28.0, β = 8/3)\n    return CoupledODEs(lorenz_rule, u0, [σ, ρ, β])\nend\n@inbounds function lorenz_rule(u, p, t)\n    du1 = p[1]*(u[2]-u[1])\n    du2 = u[1]*(p[2]-u[3]) - u[2]\n    du3 = u[1]*u[2] - p[3]*u[3]\n    return SVector{3}(du1, du2, du3)\nend\n\nds = lorenz()","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Next, we give initial guess of the location of the periodic orbit and its period.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"u0_guess = SVector(3.5, 3.0, 0.0)\nT_guess = 5.2\nig = InitialGuess(u0_guess, T_guess) ","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Then we pick an appropriate algorithm that will detect the PO. In this case we can use  any algorithm intended for continuous-time dynamical systems. We choose Optimized Shooting  algorithm, for more information see OptimizedShooting.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"alg = OptimizedShooting(Δt=0.01, n=3)","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Finally, we combine all the pieces to find the periodic orbit.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"po = periodic_orbit(ds, alg, ig)\npo","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"The closed curve of the periodic orbit can be visualized using plotting library such as  Makie.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"using CairoMakie\n\nu0 = po.points[1]\nT = po.T\ntraj, t = trajectory(ds, T, u0; Dt = 0.01)\nfig = Figure()\nax = Axis3(fig[1,1], azimuth = 0.6pi, elevation= 0.1pi)\nlines!(ax, traj[:, 1], traj[:, 2], traj[:, 3], color = :blue, linewidth=1.7)\nscatter!(ax, u0)\nfig","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To ensure that the detected period is minimal, eg. it is not a multiple of the minimal  period, we can use minimal_period.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"minT_po = minimal_period(ds, po)\nminT_po","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"Whether two periodic orbits are equivalent up to some tolerance. Function poequal  can be used.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"equal = poequal(po, minT_po; dthres=1e-3, Tthres=1e-3)\n\"Detected periodic orbit had minimal period: $(equal)\"","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"To determine whether found periodic orbit is stable or unstable, we can apply  isstable function.","category":"page"},{"location":"tutorial/","page":"Tutorial","title":"Tutorial","text":"po = isstable(ds, po)\n\"Detected periodic orbit is $(po.stable ? \"stable\" : \"unstable\").\"","category":"page"},{"location":"algorithms/#Algorithms","page":"Algorithms","title":"Algorithms","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Here is the documentation of all available PO detection algorithms. To see examples of their usage, see Examples.","category":"page"},{"location":"algorithms/#Optimized-Shooting-Method","page":"Algorithms","title":"Optimized Shooting Method","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"OptimizedShooting","category":"page"},{"location":"algorithms/#PeriodicOrbits.OptimizedShooting","page":"Algorithms","title":"PeriodicOrbits.OptimizedShooting","text":"OptimizedShooting(; kwargs...)\n\nA shooting method (Dednam and Botha, 2014) that uses Levenberg-Marquardt optimization  to find periodic orbits of continuous-time dynamical systems.\n\nKeyword arguments\n\nΔt::Float64 = 1e-6: step between the points in the residual R. See below for details.\nn::Int64 = 2: n*dimension(ds) is the number of points in the residual R. See below  for details.\nnonlinear_solve_kwargs = (reltol=1e-6, abstol=1e-6, maxiters=1000): keyword arguments  to pass to the solve function from  NonlinearSolve.jl. For details on the  keywords see the respective package documentation.\n\nDescription\n\nLet us consider the following continuous-time dynamical system\n\nfracdxdt = f(x p t)\n\nDednam and Botha (Dednam and Botha, 2014) suggest to minimize the residual R defined as\n\nR = (x(T)-x(0) x(T+Delta t)-x(Delta t) dots \nx(T+(n-1)Delta t)-x((n-1)Delta t))\n\nwhere T is unknown period of a periodic orbit and x(t) is a solution at time t  given some unknown initial point. Initial guess of the period T and the initial point  is optimized by the Levenberg-Marquardt algorithm.\n\nIn our implementation, the keyword argument n corresponds to n in the residual R.  The keyword argument Δt corresponds to Delta t in the residual R.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#Schmelcher-and-Diakonos","page":"Algorithms","title":"Schmelcher & Diakonos","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"SchmelcherDiakonos\nlambdamatrix\nlambdaperms","category":"page"},{"location":"algorithms/#PeriodicOrbits.SchmelcherDiakonos","page":"Algorithms","title":"PeriodicOrbits.SchmelcherDiakonos","text":"SchmelcherDiakonos(; kwargs...)\n\nDetect periodic orbits of ds <: DiscreteTimeDynamicalSystem using algorithm proposed by Schmelcher & Diakonos (Schmelcher and Diakonos, 1997).\n\nKeyword arguments\n\no : period of the periodic orbit\nλs : vector of λ parameters, see (Schmelcher and Diakonos, 1997) for details\nindss : vector of vectors of indices for the permutation matrix\nsignss : vector of vectors of signs for the permutation matrix\nmaxiters=10000 : maximum amount of iterations an initial guess will be iterated before  claiming it has not converged\ninftol=10.0 : if a state reaches norm(state) ≥ inftol it is assumed that it has  escaped to infinity (and is thus abandoned)\ndisttol=1e-10 : distance tolerance. If the 2-norm of a previous state with the next one  is ≤ disttol then it has converged to a fixed point\n\nDescription\n\nThe algorithm used can detect periodic orbits by turning fixed points of the original map ds to stable ones, through the transformation\n\nmathbfx_n+1 = mathbfx_n +\nmathbfLambda_kleft(f^(o)(mathbfx_n) - mathbfx_nright)\n\nThe index k counts the various possible mathbfLambda_k.\n\nPerformance notes\n\nAll initial guesses are evolved for all mathbfLambda_k which can very quickly lead to long computation times.\n\n\n\n\n\n","category":"type"},{"location":"algorithms/#PeriodicOrbits.lambdamatrix","page":"Algorithms","title":"PeriodicOrbits.lambdamatrix","text":"lambdamatrix(λ, inds::Vector{Int}, sings) -> Λk\n\nReturn the matrix mathbfLambda_k used to create a new dynamical system with some unstable fixed points turned to stable in the algorithm SchmelcherDiakonos.\n\nArguments\n\nλ<:Real : the multiplier of the C_k matrix, with 0 < λ < 1.\ninds::Vector{Int} : The i-th entry of this vector gives the row of the nonzero element of the ith column of C_k.\nsings::Vector{<:Real} : The element of the i-th column of C_k is +1 if signs[i] > 0 and -1 otherwise (sings can also be Bool vector).\n\nCalling lambdamatrix(λ, D::Int) creates a random mathbfLambda_k by randomly generating an inds and a signs from all possible combinations. The collections of all these combinations can be obtained from the function lambdaperms.\n\nDescription\n\nEach element of inds must be unique such that the resulting matrix is orthogonal and represents the group of special reflections and permutations.\n\nDeciding the appropriate values for λ, inds, sings is not trivial. However, in ref.(Pingel et al., 2000) there is a lot of information that can help with that decision. Also, by appropriately choosing various values for λ, one can sort periodic orbits from e.g. least unstable to most unstable, see (Diakonos et al., 1998) for details.\n\n\n\n\n\n","category":"function"},{"location":"algorithms/#PeriodicOrbits.lambdaperms","page":"Algorithms","title":"PeriodicOrbits.lambdaperms","text":"lambdaperms(D) -> indperms, singperms\n\nReturn two collections that each contain all possible combinations of indices (total of D) and signs (total of 2^D) for dimension D (see lambdamatrix).\n\n\n\n\n\n","category":"function"},{"location":"algorithms/#Davidchack-and-Lai","page":"Algorithms","title":"Davidchack & Lai","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"An extension of the SchmelcherDiakonos algorithm was proposed by Davidchack & Lai (Davidchack and Lai, 1999). It works similarly, but it uses smarter seeding and an improved transformation rule.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"DavidchackLai","category":"page"},{"location":"algorithms/#PeriodicOrbits.DavidchackLai","page":"Algorithms","title":"PeriodicOrbits.DavidchackLai","text":"DavidchackLai(; kwargs...)\n\nFind periodic orbits fps of periods 1 to n+1 for the dynamical system ds using the algorithm propesed by Davidchack & Lai (Davidchack and Lai, 1999).\n\nKeyword arguments\n\nn::Int64 : Periodic orbits of period up to n will be detected. Some (but not all) POs   of period n+1 will be detected. Keyword argument n must be a positive integer.\nm::Int64 : Initial guesses will be used to find POs of period 1 to m. These   periodic orbits will then be used to detect periodic orbits of periods from m+1 to   n+1. Keyword argument m must be a positive integer between 1 and n.\nβ = nothing: If it is nothing, then β(n) = 10*1.2^n. Otherwise can be a   function that takes period n and return a number. It is a parameter mentioned  in the paper(Davidchack and Lai, 1999).\nmaxiters = nothing: If it is nothing, then initial condition will be iterated max(100, 4*β(p)) times (where p is the period of the periodic orbit)  before claiming it has not converged. If it is an integer, then it is the maximum   amount of iterations an initial condition will be iterated before claiming   it has not converged.\ndisttol = 1e-10: Distance tolerance. If norm(f^{n}(x)-x) < disttol   where f^{n} is the n-th iterate of the dynamic rule f, then x   is an n-periodic point.\nabstol = 1e-8: A detected periodic point isn't stored if it is in abstol   neighborhood of some previously detected point. Distance is measured by   euclidian norm. If you are getting duplicate periodic points, increase this value.\n\nDescription\n\nThe algorithm is an extension of Schmelcher & Diakonos(Schmelcher and Diakonos, 1997) implemented as SchmelcherDiakonos.\n\nThe algorithm can detect periodic orbits by turning fixed points of the original dynamical system ds to stable ones, through the  transformation\n\nmathbfx_n+1 = mathbfx_n + \nbeta g(mathbfx_n) C^T - J(mathbfx_n)^-1 g(mathbfx_n)\n\nwhere\n\ng(mathbfx_n) = f^n(mathbfx_n) - mathbfx_n\n\nand\n\nJ(mathbfx_n) = fracpartial g(mathbfx_n)partial mathbfx_n\n\nThe main difference between SchmelcherDiakonos and  DavidchackLai is that the latter uses periodic points of previous period as seeds to detect periodic points of the next period. Additionally, SchmelcherDiakonos only detects periodic points of a given period,  while davidchacklai detects periodic points of all periods up to n.\n\nImportant note\n\nFor low periods n circa less than 6, you should select m = n otherwise the algorithm  won't detect periodic orbits correctly. For higher periods, you can select m as 6.  We recommend experimenting with m as it may depend on the specific problem.  Increase m in case the orbits are not being detected correctly.\n\nInitial guesses for this algorithm can be selected as a uniform grid of points in the state  space or subset of a chaotic trajectory.\n\n\n\n\n\n","category":"type"}]
}
