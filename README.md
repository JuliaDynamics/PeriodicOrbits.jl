# PeriodicOrbits.jl

[![docsdev](https://img.shields.io/badge/docs-dev-lightblue.svg)](https://juliadynamics.github.io/PeriodicOrbits.jl/dev/)
<!-- [![docsstable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/periodicorbits/stable/) -->
[![](https://img.shields.io/badge/DOI-10.1007%2F978--3--030--91032--7-purple)](https://link.springer.com/book/10.1007/978-3-030-91032-7)
[![CI](https://github.com/JuliaDynamics/PeriodicOrbits.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/PeriodicOrbits.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/JuliaDynamics/PeriodicOrbits.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDynamics/PeriodicOrbits.jl)
[![Package Downloads](https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/PeriodicOrbits)](https://pkgs.genieframework.com?packages=PeriodicOrbits)

**PeriodicOrbits.jl** provides both the interface and algorithm implementations for finding stable and unstable periodic orbits of discrete and continuous time dynamical systems based on the DynamicalSystems.jl ecosystem.

Currently this is work in progress and the interface is being finalized. The package is not registered yet. To install it, run
```
import Pkg; Pkg.add(url = "https://github.com/JuliaDynamics/PeriodicOrbits.j")
```

In the future this package will be a basis for local continuation integrated with DynamicalSystems.jl.

All further information is provided in the documentation, which you can either find online or build locally by running the `docs/make.jl` file.

**Note**: Throughout the documentation, "Periodic Orbit" is often abbreviated as "PO" for brevity.
