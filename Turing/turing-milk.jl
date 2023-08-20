import MLJ:transform,fit!,machine,Continuous
using Turing, Distributions,CSV,DataFrames,MLJ
using ScientificTypes

str= "/Users/lunarcheung/Public/DataSets/rethinking-data/milk.csv"
get(str)=(str)|>CSV.File|>DataFrame
df=get(str)
data=select(df,["kcal.per.g","mass","neocortex.perc"])
rename!(data, "kcal.per.g" => :kpg, "neocortex.perc" => :neopre)
#nameArr=names(data)
# Standardizer = @load Standardizer pkg=MLJModels
# stand1 =Standardizer()
# transform(fit!(machine(stand1, data)), data)


UnivariateFillImputer = @load UnivariateFillImputer pkg=MLJModels
imputer = UnivariateFillImputer()

#neocortex=select(data,[3])|>Matrix|>d->d[:,1]|>Array
coerce(data, :neopre=>Continuous)
#neo= machine(imputer, neocortex)|>fit!

data

