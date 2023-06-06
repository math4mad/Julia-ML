"""
BayesianLDA pkg=MultivariateStats
"""

import MLJ: fit!, predict,transform,predict_mode,fitted_params
using CSV
using DataFrames
using MLJ
using Plots
using Random


#=data processing=======#
    function data_prepare(str)
        urls(str) = "./DataSource/$str.csv"
        fetch(str) = urls(str) |> CSV.File |> DataFrame
        to_ScienceType(d)=coerce(d,:Outcome=> Multiclass)
        df = fetch(str)|>to_ScienceType
        return df
    end

    df=data_prepare("diabetes")
    y, X =  unpack(df, ==(:Outcome), rng=123);
    #schema(X)
    (Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)
#=data processing end=====#


BayesianLDA = @load BayesianLDA pkg=MultivariateStats

model = BayesianLDA()

mach = machine(model, Xtrain, ytrain)|>fit!

#Xproj = transform(mach, Xtrain)
#y_hat = predict(mach, Xtest)
y_hat = predict_mode(mach, Xtest)
accuracy(yhat,ytest)
