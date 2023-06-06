"""
https://www.machinelearningplus.com/machine-learning
/an-introduction-to-gradient-boosting-decision-trees/

modeling tuning 还没有完成

"""

import MLJ:predict,predict_mode,measures,range
using GLMakie, MLJ, CSV, DataFrames
using CatBoost.MLJCatBoostInterface

function data_prepare(str)
    urls(str) = "./DataSource/$str.csv"
    fetch(str) = urls(str) |> CSV.File |> DataFrame
    to_ScienceType(d)=coerce(d,:Outcome=> Multiclass)
    df = fetch(str)|>to_ScienceType
    return df
end

df=data_prepare("diabetes")
y, X =  unpack(df, ==(:Outcome), rng=123);
#schema(X)
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,  rng=123)


#catboost workflow

catboost = CatBoostClassifier(iterations=5,learning_rate=0.20,depth=6,loss_function="Logloss",)
#loss_function 怎么定义?


# catboost_tuning = TunedModel(model=catboost,
# 							  resampling=CV(nfolds=3),
#                               range=r1,
# 							  tuning=Grid(resolution=10),measure=rms);

mach = machine(catboost, Xtrain, ytrain)|>fit!

# #probs = predict(mach, Xtrain)
yhat = predict_mode(mach, Xtest) 

# # misclassification_rate(yhat, ytest)
 accuracy(yhat,ytest)
