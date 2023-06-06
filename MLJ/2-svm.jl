"""
from   https://juliaai.github.io/DataScienceTutorials.jl/isl/lab-9/index.html
https://discourse.julialang.org/t/how-do-i-specify-a-
-kernel-in-svm-using-libsvm-or-mlj/61962

data from  kaggle 
"""

import MLJ: fit!, predict
using CSV
using DataFrames
using MLJ
using Plots
using Random
using KernelFunctions

"""
    data_prepare(str)
    生成 X,y数据,同时返回 df

`return df, X, y`
TBW
"""
function data_prepare(str)
    urls(str) = "./DataSource/$str.csv"
    f(str) = urls(str) |> CSV.File |> DataFrame
    df = f(str)
    rows, _ = size(df)
    x1 = df[1:2:rows, :x]
    x2 = df[1:2:rows, :y]
    X = hcat(x1, x2)
    y = df[1:2:rows, :color]
    X = MLJ.table(X)
    y = categorical(y)
    return df, X, y
end


"""
    boundary_data(df,;n=200)
    生成绘制决策边界的数据, 根据 df 的极值
    返回 grid 数据的tx,ty 范围和 x_test 数据
    x_test 用于生成预测结果
    示例:
```julia
    ypred=predict(svc, x_test)
    contourf(tx,ty,ypred)
```
    
TBW
"""
function boundary_data(df,;n=200)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x])
    ylow,yhigh=extrema(df[:,:y])
    tx = range(xlow,xhigh; length=n1)
    ty = range(ylow,yhigh; length=n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    x_test=MLJ.table(x_test')
    return tx,ty, x_test
end
 
str1="basic1"
str="boxes"
df, X, y=data_prepare(str)
tx,ty, x_test=boundary_data(df)
cat=df[:,:color]|>levels|>length # 类别


SVC = @load SVC pkg=LIBSVM


svc_mdl = SVC()
svc = machine(svc_mdl, X, y)
@time fit!(svc);

#生成决策边界预测值
ypred=predict(svc, x_test)

#plot

contourf(tx,ty,ypred,levels=cat,color=cgrad(:redsblues),alpha=0.7)
p1=scatter!(df[:,:x],df[:,:y],group=df[:,:color],label=false,ms=3,alpha=0.3)
#savefig(p1,"./$str-svm.png")








