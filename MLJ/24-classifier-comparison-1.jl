"""
referece  scikit-learning  Classifier comparison
1. Generating Synthetic Data   MLJ doc
"""

import MLJ:predict,predict_mode
using  MLJ,GLMakie,DataFrames,Random
Random.seed!(1222)

#X, y = make_circles(400; noise=0.05, factor=0.3)
X, y = make_moons(400; noise=0.1)
X2, y2 = make_blobs(400, 2; centers=2, cluster_std=[1.0, 2.0])

X3, y3 = make_circles(400; noise=0.05, factor=0.2)

df1 = DataFrame(X)
df1.y = y
X,y=df1[:,1:2],df1[:,3]

df2 = DataFrame(X2)
df2.y = y2
X2,y2=df2[:,1:2],df2[:,3]

df3 = DataFrame(X3)
df3.y = y3
X3,y3=df3[:,1:2],df3[:,3]

cat=df1.y|>levels|>unique
colors=[:green, :purple]


function plot_origin_data(df)
    fig=Figure()
    ax=Axis(fig[1,1])
    local cat=df.y|>levels|>unique
    @info cat
    local colors=[:green, :purple]
    for (i,c) in enumerate(cat)
        d=df[y.==c,:]
        scatter!(ax, d[:,1],d[:,2],color=(colors[i],0.6))
        #@show d
    end
    fig
end

#plot_origin_data()
nums=100
function boundary_data(df,;n=nums)
    n1=n2=n
    xlow,xhigh=extrema(df[:,:x1])
    ylow,yhigh=extrema(df[:,:x2])
    tx = LinRange(xlow,xhigh,n1)
    ty = LinRange(ylow,yhigh,n2)
    xs=[x for x in tx,y in ty]
    ys=[y for x in tx,y in ty]
    x_test = mapreduce(collect, hcat, Iterators.product(tx, ty));
    x_test=MLJ.table(x_test')
    return tx,ty,xs,ys, x_test
end





function classfiy_model_collection()
    SVC = @load SVC pkg=LIBSVM   
    KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
    DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
    RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
    CatBoostClassifier = @load CatBoostClassifier pkg=CatBoost
    BayesianLDA = @load BayesianLDA pkg=MultivariateStats
    Booster = @load AdaBoostStumpClassifier pkg=DecisionTree

    # models=Dict(:SVC=>SVC,
    #             :KNNClassifier=>KNNClassifier,
    #             :DecisionTreeClassifier=>DecisionTreeClassifier,
    #             :RandomForestClassifier=>RandomForestClassifier,
    #             :CatBoostRegressor=>CatBoostRegressor,
    #             :BayesianLDA=>BayesianLDA
    # )

    arr=[KNNClassifier,DecisionTreeClassifier,RandomForestClassifier,CatBoostClassifier,BayesianLDA,SVC]
    return arr
end

function _fit(data::Tuple,m)
    X,y,xtest=data
    local predict= m==MLJLIBSVMInterface.SVC  ? MLJ.predict : MLJ.predict_mode 
    
   model=m()
   mach = machine(model, X, y)|>fit!
   yhat=predict(mach, xtest)
   ytest=yhat|>Array|>d->reshape(d,nums,nums)
   return  ytest
end


names=["KNN","DecisionTree","RandomForest","CatBoost","BayesianLDA","SVC"]

function plot_desc_boudary(fig,ytest,i;df=df1,row=1)
    tx,ty,xs,ys, xtest=boundary_data(df)
    local ax=Axis(fig[row,i],title="$(names[i])")

    contourf!(ax, tx,ty,ytest,levels=length(cat),colormap=:phase)

    for (i,c) in enumerate(cat)
        d=df[y.==c,:]
        scatter!(ax, d[:,1],d[:,2],color=(colors[i],0.6))
    end
    hidedecorations!(ax)
    

end

#plot_desc_boudary()

models=classfiy_model_collection()

_,_,_,_ ,xtest=boundary_data(df1)
data=(X,y,xtest)
ytest_arr=[_fit(data,m) for (i,m) in enumerate(models)]

fig=Figure(resolution=(2100,1000))
function plot_comparsion(testdata,df,row=1)
    
    for i in eachindex(testdata)
        plot_desc_boudary(fig,testdata[i],i;df=df,row=row)
    end
    fig
end

plot_comparsion(ytest_arr,df1,1)

_,_,_,_ ,xtest3=boundary_data(df3)
data3=(X3,y3,xtest3)
ytest_arr3=[_fit(data3,m) for (i,m) in enumerate(models)]
plot_comparsion(ytest_arr3,df3,2)


_,_,_,_ ,xtest2=boundary_data(df2)
data2=(X2,y2,xtest2)
ytest_arr2=[_fit(data2,m) for (i,m) in enumerate(models)]
fig=plot_comparsion(ytest_arr2,df2,3)


#save("24-classifier-comparison.png",fig)

