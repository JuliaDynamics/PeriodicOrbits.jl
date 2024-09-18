using PeriodicOrbits
using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "PeriodicOrbits" begin
    testfile("api.jl")
    testfile("stability.jl")
    testfile("minimal_period.jl")
    testfile("algorithms/optimized_shooting.jl")
    testfile("algorithms/schmelcher_diakonos.jl")
    testfile("algorithms/davidchack_lai.jl")
end
