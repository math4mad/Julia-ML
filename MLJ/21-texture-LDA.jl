"""
https://online.stat.psu.edu/stat857/node/235/
"""


import MLJ:transform,predict,predict_mode,fitted_params
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:A41=> Multiclass)
    df = fetch(str)|>to_ScienceType
    return df
end

str="Texture"
df=data_prepare(str)
#first(df,10)
y, X =  unpack(df, ==(:A41), rng=123);

(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)


LDA = @load LDA pkg=MultivariateStats
model = LDA()

mach = machine(model, Xtrain, ytrain) |> fit!   

yhat= predict_mode(mach, Xtest)
accuracy(yhat,ytest)   #0.99

