"""
https://online.stat.psu.edu/stat857/node/11/
有完整数据
"""

import MLJ:transform,predict,predict_mode
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
PCA = @load PCA pkg=MultivariateStats


function different_pca_components(;n=3)
    acc_arr=[]
    for i in 1:n
        model1=PCA(maxoutdim=i)
        model2 = LDA()
        mach1 = machine(model1, Xtrain) |> fit!
        Ytr =transform(mach1, Xtrain)
        mach2 = machine(model2, Ytr, ytrain)|>fit!
        Yte=transform(mach1, Xtest)
        yhat = predict_mode(mach2, Yte)
        res=accuracy(yhat,ytest)
        push!(acc_arr,res)
    end
    return acc_arr
end



acc_arr=different_pca_components(;n=15)

function plot_accuracy(acc_arr)
    len=length(acc_arr)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel="pcs",ylabel="accuracy",title="LDA accruacy with n primary components")
    xs=1:len
    acc_arr=round.(acc_arr,digits=3)
    scatterlines!(ax,xs,acc_arr,markercolor = (:red,0.5))
    fig
    #save("16-texture-PCA-LDA.png",fig)
end

plot_accuracy(acc_arr)




