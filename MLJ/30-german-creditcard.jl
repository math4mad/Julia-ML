"""
https://online.stat.psu.edu/stat857/node/215/
1. scitype 转换 参考:https://juliaai.github.io/DataScienceTutorials.jl/data/processing/ 直接使用 autotype(d, :few_to_finite)方法
2. 最后一个 feature 根据文章去掉, 但是!= 操作不成功
3. 优化方法到0.74
"""

import MLJ:predict,fit!,predict_mode,range
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,autotype(d, :few_to_finite))
    df = fetch(str)|>to_ScienceType
    ytrain, Xtrain=  unpack(df, ==(:Creditability));
    cat=levels(ytrain)
    return ytrain, Xtrain[:,1:end-1],cat
end


y, X,cat=data_prepare("german_creditcard")

(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.8, rng=123, multi=true)

LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
model=LogisticClassifier()
NuSVC = @load NuSVC pkg=LIBSVM
model2 = NuSVC()
KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model3 = KNNClassifier(weights = NearestNeighborModels.Inverse())

k1 =range(model, :gamma, lower=0.1, upper=1.2);
k2 =range(model, :lambda, lower=0.1, upper=1.2);
k3 =range(model, :penalty, values=([:l2, :l1,:en,:none]));
k4 =range(model, :fit_intercept, values=([true, false]));

tuning_logistic = TunedModel(model=model,
							 resampling = CV(nfolds=4, rng=1234),
							 tuning = Grid(resolution=8),
							 range = [k1,k2],
							 measure=accuracy)
mach = machine(tuning_logistic, Xtrain, ytrain)|>fit!
#entry = report(mach)
yhat=predict_mode(mach, Xtest)|>Array

accuracy(ytest,yhat)|>d->round(d,digits=3)
