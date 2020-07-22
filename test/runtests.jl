using NoiseContrastiveEstimation
using Test
@testset "Basic Functional" begin
    lϕ(x,θ) = θ[1]*x + θ[2]*x*x/2
    data=randn(200).*0.3 .+ 1.5
    noised=foldl(hcat, x.+randn(20).*0.4  for x in data)
    J=CNCE(lϕ,data,noised)
    @test J((15,-10)) < J((5,-5)) < J((0,0))
end
