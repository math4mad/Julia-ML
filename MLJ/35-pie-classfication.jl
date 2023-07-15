"""
An Introduction to Machine Learning by Kubat, Miroslav.pdf  
pie example
"""


import MLJ:predict,predict_mode,measures,fit!,transform
using Plots, MLJ, CSV, DataFrames


urls(str) ="./DataSource/$str.csv"
get(str)=urls(str)|>CSV.File|>DataFrame|>dropmissing

df=get("pies")

df=coerce(df, autotype(df, :string_to_multiclass))
ytrain, Xtrain=  unpack(df, ==(:Class));
hot = OneHotEncoder(drop_last=true)
mach = fit!(machine(hot, Xtrain))
Wtrain = transform(mach, Xtrain)

(Xtrain, Xtest), (ytrain, ytest)  = partition((Wtrain, ytrain), 0.8, multi=true,  rng=123)



model = @load EvoTreeClassifier pkg=EvoTrees
mach2 = machine(model(), Xtrain, ytrain)|>fit!
preds = predict_mode(mach2, Xtrain)|>Array

accuracy(preds,ytrain)

