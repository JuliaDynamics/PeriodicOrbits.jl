module PeriodicOrbits

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end PeriodicOrbits

using Reexport
@reexport using DynamicalSystemsBase


# exports:
include("api.jl")
include("pretty_printing.jl")
include("algorithms/optimized_shooting.jl")

end
