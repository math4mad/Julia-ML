"""
german credit 的 高斯混合模型
autotype 改为:
autotype(X, (:few_to_finite, :discrete_to_continuous))
"""

import MLJ:predict,predict_mode
import BetaML
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie

modelType= @load NeuralNetworkClassifier pkg = "BetaML"

layers= [BetaML.DenseLayer(19,8,f=BetaML.relu),BetaML.DenseLayer(8,8,f=BetaML.relu),BetaML.DenseLayer(8,2,f=BetaML.relu),BetaML.VectorFunctionLayer(2,f=BetaML.softmax)];
model= modelType(layers=layers,opt_alg=BetaML.ADAM())

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,autotype(d, (:few_to_finite, :discrete_to_continuous)))
    df = fetch(str)|>to_ScienceType
    ytrain, Xtrain=  unpack(df, ==(:Creditability));
    cat=levels(ytrain)
    return ytrain, Xtrain[:,1:end-1],cat
end


y, X,cat=data_prepare("german_creditcard")
Xtest,ytest=X[1:2:end,:],y[1:2:end]
#schema(X)

(fitResults, cache, report) = MLJ.fit(model, 1, X,y);

est_classes= predict_mode(model, fitResults, Xtest)

accuracy(ytest,est_classes)  #0,692
