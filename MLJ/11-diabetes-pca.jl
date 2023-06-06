import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,Plots
plotly()

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


 model=PCA(maxoutdim=3)

 mach = machine(model, Xtrain) |> fit!

 Ytr =transform(mach, Xtrain)

 
 
 positive=Ytr[ytrain.==1,:]
 negative=Ytr[ytrain.==0,:]

 scatter3d(positive[:,1],positive[:,2],positive[:,3],marker=:circle,ms=2,label="positive")
 scatter3d!(negative[:,1],negative[:,2],negative[:,3],marker=:circle,ms=2,label="negative")

