
using Pkg
Pkg.activate("/Users/lunarcheung/Public/Julia-Code/🟢JuliaProject/1-JuliaMLProject")

import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,Plots
gr()

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:Column1=> Multiclass,Count=>Continuous)
    df = fetch(str)
    return df
end

str="diabetes"
df=data_prepare(str)


#data 
rows,cols=size(df)
ytrain, Xtrain =  unpack(df, ==(:Outcome), rng=123);

 PCA = @load PCA pkg=MultivariateStats


 model=PCA(maxoutdim=2)

 mach = machine(model, Xtrain) |> fit!

 Ytr =transform(mach, Xtrain)  #降维的训练数据

 
 
 positive=Ytr[ytrain.==1,:]
 negative=Ytr[ytrain.==0,:]

 scatter(positive[:,1],positive[:,2],marker=:circle,ms=2,label="positive",frame=:origin,xlims=(-500,100))
 scatter!(negative[:,1],negative[:,2],marker=:circle,ms=2,label="negative")

 Xtrain