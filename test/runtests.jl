using NoiseContrastiveEstimation
using Test, Random

@testset "Basic Functional" begin
    R = MersenneTwister(125)
    lϕ(x, θ) = θ[1] * x + θ[2] * x * x / 2
    gϕ(x, ϕ) = [x, x * x / 2]
    data = randn(R, 200) .* 0.3 .+ 1.5
    noised = foldl(hcat, x .+ randn(R, 20) .* 0.4 for x in data)
    J = CNCE(f = lϕ, grad_f = gϕ, data = data, noised = noised)
    @test J((15, -10)).J < J((7.5, -5)).J < J((0, 0)).J
    results = nesterov(J, zeros(2), 0.9, 7.0)
    @show results
    x = results.sol
    @test J(x).J < J((15, -10)).J
    Jgless = CNCE(f = lϕ, data = data, noised = noised)
    @test Jgless.grad_f(1, 2) == 0
    @test J(x).J == Jgless(x).J
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

@testset "Estimate Covariate Model" begin
    R = MersenneTwister(125)
    gϕ(x, θ) = [x[1]*x[1], x[2]*x[2], x[1]*x[2], x[1], x[2]]
    lϕ(x, θ) = θ'gϕ(x, θ)

    data = [[x+1,y+1] for (x,y) in zip(randn(R,200),randn(R,200))]
    noised = [x+0.3*randn(2) for j=1:50, x in data]
    J = CNCE(f = lϕ, grad_f = gϕ, data = data, noised = noised)
    results = nesterov(J, zeros(5), 0.9, 2.5)
    @test results.sol[1]<0
    @test results.sol[2]<0
    @test results.sol[4]>0
    @test results.sol[5]>0
end