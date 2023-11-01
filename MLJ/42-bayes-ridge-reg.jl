
#= 
http://127.0.0.1:8000/scikit-learn-docs/modules/linear_model.html#bayesian-regression
python 数据要做变换  数组转换为 mlj 的 table形式
=#

import MLJ:predict,table
using MLJ
BayesianRidgeRegressor = @load BayesianRidgeRegressor pkg=MLJScikitLearnInterface

X = [0. 0.; 1. 1.; 2. 2.; 3.  3.]|>table

y = [0., 1., 2., 3.]

x_test=table([1  0.;])
model=BayesianRidgeRegressor()

mach = machine(model, X, y) |> fit!
predict(mach, x_test)
