"""
referece  scikit-learning  Classifier comparison
1. Generating Synthetic Data   MLJ doc
done!
"""

import MLJ:predict,predict_mode
using  MLJ,GLMakie,DataFrames,Random
Random.seed!(1222)
#colors=[:red,:blue]
colors=[:blue, :orange]
nums=100 #绘制决策边界的遍历值
names=["KNN","DecisionTree","RandomForest","CatBoost","BayesianLDA","SVC"]

"""
circle_data()
return df
"""
function circle_data()
    X, y = make_circles(400; noise=0.1, factor=0.3)
    df = DataFrame(X)
    df.y = y
    return df
end

"""
    moons_data()
    return df
TBW
"""
function moons_data()
    X, y = make_moons(400; noise=0.1)
    df = DataFrame(X)
    df.y = y
    return df
end

"""
    blob_data()
    return df
TBW
"""
function blob_data()
    X, y = make_blobs(400, 2; centers=2, cluster_std=[1.0, 2.0])
    df = DataFrame(X)
    df.y = y
    return df
end


function boundary_data(df,;n=nums)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x1])
    ylow,yhigh=extrema(df[:,:x2])
    tx = LinRange(xlow,xhigh,n1)
    ty = LinRange(ylow,yhigh,n2)
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    x_test=MLJ.table(x_test')
    return tx,ty,x_test
end

#  machine learning  workflow

function classfiy_model_collection()
    SVC = @load SVC pkg=LIBSVM   
    KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
    CatBoostClassifier = @load CatBoostClassifier pkg=CatBoost
    BayesianLDA = @load BayesianLDA pkg=MultivariateStats
    AdaBooster = @load AdaBoostStumpClassifier pkg=DecisionTree
     arr=[KNNClassifier,DecisionTreeClassifier,RandomForestClassifier,CatBoostClassifier,BayesianLDA,SVC]
    return arr
end


"""
    _fit(data::Tuple,m)
 data 的结构为 X,y,xtest=data,X,y 为训练数据和训练标签, xtest 为用于生成决策边界
 的测试数据

 根据训练数据和模型名称, 训练对应模型,
 并以矩阵形式返回 ytest
TBW
"""
function _fit(df::DataFrame,m)
    X,y=df[:,1:2],df[:,3]
    _,_,xtest=boundary_data(df;n=nums)
    local predict= m==MLJLIBSVMInterface.SVC  ? MLJ.predict : MLJ.predict_mode 
    model=m()
   mach = machine(model, X, y)|>fit!
   yhat=predict(mach, xtest)
   ytest=yhat|>Array|>d->reshape(d,nums,nums)
   return  ytest
end



function plot_desc_boudary(fig,ytest,i;df=df1,row=1)
    tx,ty,_=boundary_data(df)
    local y=df.y
    local ax=Axis(fig[row,i],title="$(names[i])")
    cat=y|>levels|>unique
    contourf!(ax, tx,ty,ytest,levels=length(cat),colormap=:redsblues)

    for (i,c) in enumerate(cat)
        d=df[y.==c,:]
        scatter!(ax, d[:,1],d[:,2],color=(colors[i],0.6))
    end
    hidedecorations!(ax)
    

end

function plot_comparsion(testdata,df;row=1)
    
    for (i,data) in enumerate(testdata)
        plot_desc_boudary(fig,data,i;df=df,row=row)
    end
    fig
end


models=classfiy_model_collection()

df1=circle_data()
ytest1=[_fit(df1,m) for (i,m) in enumerate(models)]

df2=moons_data()
ytest2=[_fit(df2,m) for (i,m) in enumerate(models)]

df3=blob_data()
ytest3=[_fit(df3,m) for (i,m) in enumerate(models)]

dfs=[df2,df1,df3]
ytests=[ytest2,ytest1,ytest3]

fig=Figure(resolution=(2100,1000))

for (df, data,i)  in zip(dfs,ytests,[1,2,3])
    plot_comparsion(data,df;row=i)
end

fig
#save("24-classifier-comparison.png",fig)
