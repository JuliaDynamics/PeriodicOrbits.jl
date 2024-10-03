module PeriodicOrbits

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end PeriodicOrbits

using Reexport
@reexport using DynamicalSystemsBase

const default_Δt_partition = 100

# exports:
include("api.jl")
include("stability.jl")
include("minimal_period.jl")
include("pretty_printing.jl")
include("algorithms/discrete_time/lambdamatrix.jl")
include("algorithms/discrete_time/schmelcher_diakonos.jl")
include("algorithms/continuous_time/optimized_shooting.jl")
include("algorithms/discrete_time/davidchack_lai.jl")

end
