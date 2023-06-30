"""
在 执行 PCA 之前对属性做标准化  
使用 Z-score normalization  方法
dt=StatsBase.fit(ZScoreTransform, Xtrain,dims=1)
diabetes 的数据是 size(Xtrain) = (768, 8) 
有 8 个属性,以列的形式表示, 需要对属性进行缩放
dims=1 代表对 8 列数据进行标准化

"""

import MultivariateStats:fit,predict,reconstruct
using MultivariateStats, GLMakie,DataFrames,CSV,MLJ
using StatsBase

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:Column1=> Multiclass,Count=>Continuous)
    df = fetch(str)
    return df
end

str="diabetes"
df=data_prepare(str)

labels=[1,0]
colors=[:red,:green,]

#data 
rows,cols=size(df)
ytrain, Xtrain =  unpack(df, ==(:Outcome), rng=123);
Xtrain=Xtrain|>Matrix

dt=StatsBase.fit(ZScoreTransform, Xtrain,dims=1)
Xtrain=StatsBase.transform(dt, Xtrain)|>transpose


M3 = fit(PCA, Xtrain; maxoutdim=3)

Ytr3 = predict(M3, Xtrain)

M2 = fit(PCA, Xtrain; maxoutdim=2)

Ytr2 = predict(M2, Xtrain)

fig=Figure(resolution=(1200,600))
ax=Axis(fig[0,1:2],title="Diabetes PCA  Analysis",subtitle=" with data normalization",height=0,)
hidedecorations!(ax)
hidespines!(ax)
function plot_3PC(Ytr)
    ax=Axis3(fig[1,1],xlabel="PC1",ylabel="PC2",zlabel="PC3",title="3 components")
    for  (label, color) in zip(labels,colors)
        data=Ytr[:,ytrain.==label]
        si= label==1 ? "+" : "-"
        scatter!(ax, data[1,:],data[2,:],data[3,:],color=(color,0.6),markersize=16,label=si)
    end
    axislegend(ax)
end

function plot_2PC(Ytr)
    ax=Axis(fig[1,2],xlabel="PC1",ylabel="PC2",title="2 components")
    for  (label, color) in zip(labels,colors)
        data=Ytr[:,ytrain.==label]
        si= label==1 ? "+" : "-"
        scatter!(ax, data[1,:],data[2,:],color=(color,0.6),markersize=16,label=si)
    end
    axislegend(ax)
end

plot_3PC(Ytr3)
plot_2PC(Ytr2)
fig

#save("11-diabetes-pca-data-ZScoreTransform.png",fig)
 



