"""
test metal.jl for gpu accelcerating
"""

import MLJ:transform,predict,predict_mode
import LIBSVM
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie
using Metal,LinearAlgebra



function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:label=>Multiclass)
    df = fetch(str)|>to_ScienceType
    ytrain, Xtrain=  unpack(df, ==(:label), rng=123);
    cat=ytrain|>levels|>unique
    return ytrain, Xtrain,cat
end

str="mnist_train"
y, X,cat=data_prepare(str)
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,shuffle=true)

arr=Xtrain|>Matrix.|>Float32|>MtlArray # gpu使用不了?

PCA = @load PCA pkg=MultivariateStats

maxdim=50
model=PCA(maxoutdim=maxdim)
@time mach = machine(model, Xtrain) |> fit!