using MLJ, DataFrames,GLMakie,Random,CSV,JLSO,Turing,StatsPlots
using LinearAlgebra,FillArrays,Distributions
Random.seed!(1222)

fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing

df=fetch("sarcos_inv")

y, X= df[1:10:end,end],df[1:10:end,1:end-1]
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)
Xtrain=Matrix(Xtrain)
Xtest= Matrix(Xtest)
rows=length(ytest)  #for plot 

# Bayesian linear regression.

@model function linear_regression(x, y)
    # Set variance prior.
    σ² ~ truncated(Normal(0, 100); lower=0)

    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))

    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    #@info nfeatures
    coefficients ~ MvNormal(Zeros(nfeatures), 10.0 * I)

    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    return y ~ MvNormal(mu, σ² * I)
end

function eval_model(Xtrain, ytrain)
    model = linear_regression(Xtrain, ytrain)
    chain = sample(model, NUTS(), 3_000)
    return chain
end
#chain= eval_model(Xtrain, ytrain)
#JLSO.save("./Turing/sarco-inv.jlso", :chn => chain)
chn = JLSO.load("./Turing/sarco-inv.jlso")[:chn]

function prediction(chain, x)
    p = get_params(chain[200:end, :, :])
    targets = p.intercept' .+ x * reduce(hcat, p.coefficients)'
    return vec(mean(targets; dims=2))
end

yhat=prediction(chn,Xtest)

rms(yhat,ytest)   #5.528382637818355