using NoiseContrastiveEstimation
using Test, Random
@testset "Basic Functional" begin
    R = MersenneTwister(125)
    lϕ(x, θ) = θ[1] * x + θ[2] * x * x / 2
    gϕ(x, ϕ) = [x, x * x / 2]
    data = randn(R, 200) .* 0.3 .+ 1.5
    noised = foldl(hcat, x .+ randn(R, 20) .* 0.4 for x in data)
    J = CNCE(lϕ, gϕ, data, noised)
    @test J(Cost(), (15, -10)) < J(Cost(), (7.5, -5)) < J(Cost(), (0, 0))
    results = nesterov(J, zeros(2), 0.9, 7.0)
    @show results
    x = results.sol
    @test J(Cost(), x) < J(Cost(), (15, -10))
    Jgless = CNCE(lϕ, data, noised)
    @test_throws ErrorException("Gradient is not available") Jgless.gϕ(1, 2)
    @test J(Cost(), x) == Jgless(Cost(), x)
end

@testset "RBM training" begin
    R = MersenneTwister(125)
    function gϕ(σ, ξ)
        sc = ξ * σ
        g = similar(ξ)
        for i = 1:size(ξ, 1), j = 1:size(ξ, 2)
            g[i, j] = 2 * σ[j] * sc[i]
        end
        g
    end

    function lϕ(σ, ξ)
        sc = ξ * σ
        sc'sc
    end
    #=
    using ReverseDiff
    gϕ(σ,ξ)=ReverseDiff.gradient(x->lϕ(σ,x),ξ)
    =#
    data = [sign.(randn(R, 10)) for i = 1:2]
    noised = [sign.(rand(R, 10) .- 1 / 3) .* data[i] for j = 1:20, i in eachindex(data)]
    J = CNCE(lϕ, gϕ, data, noised)
    x = sign.(randn(R, 4, 10)) .* 0.01
    results = nesterov(J, x, 0.95, 0.05)
    @show results
    dx = sign.(results.sol)
    M = [maximum(abs.(dx * data[i] / length(data[i]))) for i = 1:length(data)]
    @test sum(M) == 2
end
