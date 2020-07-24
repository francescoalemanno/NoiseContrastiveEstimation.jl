module NoiseContrastiveEstimation

struct CNCE{T,F,dF}
    lϕ::F
    gϕ::dF
    data::Vector{T}
    noised::Matrix{T}
end

CNCE(lϕ, data, noised) =
    CNCE(lϕ, (x, y) -> error("Gradient is not available"), data, noised)

struct Grad end
struct Cost end

function (J::CNCE)(::Cost, θ)
    κ = size(J.noised, 1)
    N = length(J.data)
    foldl(+, begin
        bϕ = J.lϕ(J.data[i], θ)
        foldl(+, begin
            G = bϕ - J.lϕ(J.noised[j, i], θ)
            2 / (κ * N) * log(1 + exp(-G))
        end for j = 1:κ)
    end for i = 1:N)
end

function (J::CNCE)(::Grad, θ)
    κ = size(J.noised, 1)
    N = length(J.data)

    foldl(
        +,
        begin
            bϕ = J.lϕ(J.data[i], θ)
            dbϕ = J.gϕ(J.data[i], θ)
            foldl(+, begin
                G = bϕ - J.lϕ(J.noised[j, i], θ)
                dG = dbϕ - J.gϕ(J.noised[j, i], θ)
                2 / (κ * N) * (1 / (1 + exp(-G)) - 1) * dG
            end for j = 1:κ)
        end for i = 1:N
    )
end

function nesterov(J::CNCE, x0, μ, η; atol = 0, rtol = 1e-7)
    x = identity.(x0)
    i = 0
    v = -J(Grad(), x)
    C = J(Cost(), x)
    C0 = C
    while true
        v = μ * v - η * J(Grad(), x + μ * v)
        x = x + v
        i = i + 1
        n = J(Cost(), x) - C
        C += n
        (abs(n / C) <= rtol || abs(n) <= atol) && break
    end
    (sol = x, cost = C, cost_fractional_reduction = 1 - C / C0, fneval = i + 1, gneval = i)
end


export CNCE, Grad, Cost, nesterov
end
