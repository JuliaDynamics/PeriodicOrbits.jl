using PeriodicOrbits
using Test

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname=defaultname(file)) = @testset "$testname" begin; include(file); end

@testset "PeriodicOrbits" begin
    testfile("api.jl")
    testfile("stability.jl")
end
