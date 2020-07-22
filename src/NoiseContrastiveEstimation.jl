module NoiseContrastiveEstimation

struct CNCE{T,F}
    lϕ::F
    data::Vector{T}
    noised::Matrix{T}
end

function (J::CNCE)(θ)
    κ=size(J.noised,1)
    N=length(J.data)

    bϕ=J.lϕ(J.data[1],θ)
    G=bϕ-J.lϕ(J.noised[1,1],θ)
    jn=2/(κ*N)*log(1+exp(-G))
    for j=2:κ
        G=bϕ-J.lϕ(J.noised[j,1],θ)
        jn+=2/(κ*N)*log(1+exp(-G))
    end
    for i=2:N
        bϕ=J.lϕ(J.data[i],θ)
        for j=1:κ
            G=bϕ-J.lϕ(J.noised[j,i],θ)
            jn+=2/(κ*N)*log(1+exp(-G))
        end
    end
    jn
end

export CNCE
end
