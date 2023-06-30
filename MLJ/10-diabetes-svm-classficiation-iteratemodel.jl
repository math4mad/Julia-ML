"""
https://www.section.io/engineering-education/diagnose-diabetes-with-svm/

使用Iteratemodel 方法,目前有错误, 配置上有问题
"""


import MLJ: fit!, predict
using CSV
using DataFrames
using MLJ
using Plots
using Random
using KernelFunctions


# data processing
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


SVC = @load SVC pkg=LIBSVM
#define kernel function,调用 kernelfunctions 的方法
kernels=[PolynomialKernel(; degree=2, c=1), 
         SqExponentialKernel(),
         NeuralNetworkKernel(),
         LinearKernel(;c=1.0)
]

iterated_model = IteratedModel(model=SVC(),
                               resampling=Holdout(),
                               measures=log_loss,
                               controls=[Step(5),
                                         Patience(2),
                                         NumberLimit(100)],
                                iteration_parameter=nothing,
                               retrain=true)
#svc_mdl = SVC(kernel=kernels[4])
svc = machine(iterated_model, Xtrain, ytrain)
@time fit!(svc);

#生成决策边界预测值
yhat=predict(svc,Xtest)

accuracy(yhat,ytest)