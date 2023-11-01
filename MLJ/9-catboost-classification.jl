"""
https://www.machinelearningplus.com/machine-learning
/an-introduction-to-gradient-boosting-decision-trees/

使用bsisc1.csv 数据, test 数据用于 决策边界生成

"""

import MLJ:predict,predict_mode,measures
using Plots, MLJ, CSV, DataFrames
using CatBoost.MLJCatBoostInterface

function data_prepare(str)
    urls(str) = "./DataSource/$str.csv"
    fetch(str) = urls(str) |> CSV.File |> DataFrame
    to_ScienceType(d)=coerce(d,:color=> Multiclass)
    df = fetch(str)|>to_ScienceType
    return df
end

df=data_prepare("basic1")
cat=df[:,:color]|>levels|>length # 类别
ytrain, Xtrain =  unpack(df, ==(:color), rng=123);

function boundary_data(df::AbstractDataFrame;n=200)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x])
    ylow,yhigh=extrema(df[:,:y])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    xtest=MLJ.table(x_test',names=[:x,:y])
    return tx,ty,xtest
end


tx,ty,xtest=boundary_data(df)

#catboost workflow

catboost = CatBoostClassifier(iterations=2,learning_rate=0.20)

mach = machine(catboost, Xtrain, ytrain)|>fit!

ytest = predict_mode(mach, xtest)[:,1]|>Array

contourf(tx,ty,ytest,levels=cat,color=cgrad(:redsblues),alpha=0.7)
p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)
#savefig("catboost-classify-basic1.png")



