"""
display mnist image
"""


import MLJ:transform,predict,predict_mode
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie



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

function plot_mnist(data;dim=28)
    imgs=data[1:25,:]|>Matrix
    fig=Figure()
    for i in 0:4
        for j in 1:5
            idx=i*5+j
            local img = imgs[idx, :] |> d ->reshape(d, dim, dim)
            local ax = Axis(fig[i, j],yreversed=true)
            image!(ax, img)
            hidespines!(ax)
            hidedecorations!(ax)
        end

    end
    fig
end

#plot_mnist(Xtrain)

PCA = @load PCA pkg=MultivariateStats

maxdim=324
model=PCA(maxoutdim=maxdim)
mach = machine(model, Xtrain) |> fit!
Ytr =transform(mach, Xtrain)

plot_mnist(Ytr;dim=18)



