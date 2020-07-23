module NoiseContrastiveEstimation

struct CNCE{T,F,dF}
    lϕ::F
    gϕ::dF
    data::Vector{T}
    noised::Matrix{T}
end

CNCE(lϕ, data, noised) = CNCE(lϕ, (x,y) -> error("Gradient is not available"), data, noised)

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

export CNCE, Grad, Cost
end
