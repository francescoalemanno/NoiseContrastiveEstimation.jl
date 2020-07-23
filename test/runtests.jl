using NoiseContrastiveEstimation
using Test
@testset "Basic Functional" begin
    lϕ(x, θ) = θ[1] * x + θ[2] * x * x / 2
    gϕ(x, ϕ) = [x, x * x / 2]
    data = randn(200) .* 0.3 .+ 1.5
    noised = foldl(hcat, x .+ randn(20) .* 0.4 for x in data)
    J = CNCE(lϕ, gϕ, data, noised)
    @test J(Cost(), (15, -10)) < J(Cost(), (5, -5)) < J(Cost(), (0, 0))
    x = zeros(2)
    v = zeros(2)
    while true
        v = 0.9 .* v .- J(Grad(), x)
        x = x .+ v
        sum(abs2, v) < 1e-8 && break
    end
    @test J(Cost(), x) < J(Cost(), (15, -10))
    Jgless = CNCE(lϕ, data, noised)
    @test_throws ErrorException("Gradient is not available") Jgless.gϕ(1,2)
    @test J(Cost(), x) == Jgless(Cost(), x)
end
