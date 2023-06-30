"""
reference :scikit-learn A demo of K-Means clustering on the handwritten digits data

就参考  26-NIR-spertra-milk-2的方法
原先的想法不可行, 在-2中实现.


"""


import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,Plots
gr()

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


PCA = @load PCA pkg=MultivariateStats


 model=PCA(maxoutdim=2)

 mach = machine(model, Xtrain) |> fit!

 Ytr =transform(mach, Xtrain)  #降维的训练数据

 
 
 positive=Ytr[ytrain.==1,:]
 negative=Ytr[ytrain.==0,:]

 scatter(positive[:,1],positive[:,2],marker=:circle,ms=2,label="positive",frame=:origin,xlims=(-500,100))
 scatter!(negative[:,1],negative[:,2],marker=:circle,ms=2,label="negative")

