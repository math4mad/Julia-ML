"""
https://online.stat.psu.edu/stat857/node/11/
有完整数据
"""

import MLJ:transform,predict
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie



function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:Column1=> Multiclass,Count=>Continuous)
    df = fetch(str)
    return df
end

str="Texture"
df=data_prepare(str)
#first(df,10)
ytrain, Xtrain =  unpack(df, ==(:A41), rng=123);
rows,cols=size(Xtrain)

PCA = @load PCA pkg=MultivariateStats


res_arr=[]

# for i in 1:8
#     model=PCA(maxoutdim=i)
#     mach = machine(model, Xtrain) |> fit!
#     #tprincipalvar=report(mach)[:tprincipalvar]
#     #push!(res_arr,tprincipalvar)
#     @info report(mach)[:tprincipalvar]
# end


# plot(1:cols,res_arr,label="tprincipalvar")
# scatter!(res_arr,ms=3,color=:red, alpha=0.5,label=false,xlabel="Principal Components",ylabel="tprincipalvar")
#savefig("15-texture-pca-residualvar.png")



model=PCA(maxoutdim=2)
mach = machine(model, Xtrain) |> fit!

Ytr =transform(mach, Xtrain)

cat=levels(ytrain)|>unique



fig=Figure(resolution=(1400,1000))
ax=Axis(fig[1,1])

colors=[:red, :yellow,:purple,:lightblue,:black,:orange,:pink,:blue,:tomato,:lightgreen,:green]

for c in eachindex(cat)
    data=Ytr[ytrain.==cat[c],:]
    scatter!(ax,data[:,1], data[:,2],color=(colors[c],0.6),markersize=10)
        
end

fig
#save("15-texture-pca.png",fig)


