"""
https://storopoli.io/Bayesian-Julia/pages/06_linear_reg/
"""

using DataFrames,CSV
using Turing
using LinearAlgebra: I
using Statistics: mean, std
using Random: seed!
seed!(123)

url="/Users/lunarcheung/Public/Julia-Code/🟢JuliaProject/1-JuliaMLProject/DataSource/kidiq.csv"


#===============data processing=======================-====#
    kidiq = CSV.read(url, DataFrame)

    #describe(kidiq)
    X = Matrix(select(kidiq, Not(:kid_score)))
    y = kidiq[:, :kid_score]
#===============data processing end=========================#




#===============bayes    workflow=========================#
    @model function linreg(X, y; predictors=size(X, 2))
        #priors
        α ~ Normal(mean(y), 2.5 * std(y))
        β ~ filldist(TDist(3), predictors)
        σ ~ Exponential(1)

        #likelihood
        return y ~ MvNormal(α .+ X * β, σ^2 * I)
    end;


    model = linreg(X, y);

    chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)

    summarystats(chain)
#===============bayes workflow end=========================#



