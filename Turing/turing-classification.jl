"""
const path = "/Users/lunarcheung/Public/DataSets/clustering-datasets/"
"""

using Turing, GLMakie,Random,DataFrames,CSV,MLJ,JLSO
using Distributions,FillArrays,LinearAlgebra
Random.seed!(343434)


urls(str) = "/Users/lunarcheung/Public/DataSets/clustering-datasets/$str.csv"
get(str)=urls(str)|>CSV.File|>DataFrame
st="un"
df=get(st)

y, X= df[1:20:end,end],df[1:20:end,1:end-1]
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)
Xtrain=Matrix(Xtrain)
Xtest= Matrix(Xtest)


using Turing

@model function gaussian_mixture_model(x)
    # Draw the parameters for each of the K=2 clusters from a standard normal distribution.
    K = 2
    μ ~ MvNormal(Zeros(K), I)

    # Draw the weights for the K clusters from a Dirichlet distribution with parameters αₖ = 1.
    #w ~ Dirichlet(K, 1.0)
    # Alternatively, one could use a fixed set of weights.
    w = fill(1/K, K)

    # Construct categorical distribution of assignments.
    distribution_assignments = Categorical(w)

    # Construct multivariate normal distributions of each cluster.
    N,D= size(x)
    distribution_clusters = [MvNormal(Fill(μₖ, D), I) for μₖ in μ]

    # Draw assignments for each datum and generate it from the multivariate normal distribution.
    k = Vector{Int}(undef, N)
    for i in 1:N
        k[i] ~ distribution_assignments
        x[i, :] ~ distribution_clusters[k[i]]
    end

    return k
end


function eval_model()
    model = gaussian_mixture_model(Xtrain);
    sampler = Gibbs(PG(100, :k), HMC(0.05, 10, :μ))
    nsamples = 100
    nchains = 3
    chains = sample(model, sampler, MCMCThreads(), nsamples, nchains);
    return chains
end

#save_model=(chn)->JLSO.save("./Turing/classification.jlso", :chn => chn)
#chains=eval_model()|>save_model

#chn = JLSO.load("./Turing/classification.jlso")[:chn]

chn=eval_model()




