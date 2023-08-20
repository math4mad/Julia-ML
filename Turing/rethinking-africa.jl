

import MLJ:transform,fit!,machine,Continuous
using Turing, Distributions,CSV,DataFrames,MLJ,GLMakie
using ScientificTypes,StatsBase

str= "/Users/lunarcheung/Public/DataSets/rethinking-data/rugged.csv"
get(str)=(str)|>CSV.File|>DataFrame|>dropmissing
df=get(str)

data=select(df,[:cont_africa,:rugged,:rgdppc_2000])

#group_data=groupby(data,[:cont_africa])
zscore(data)=StatsBase.fit(ZScoreTransform, data, dims=1)|>dt->StatsBase.transform(dt, data)
col1=zscore(data[!,2])
col2=zscore(data[!,3])
scatter(col1,col2)