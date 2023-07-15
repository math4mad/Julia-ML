
import MLJ:predict
using MLJ, DataFrames,GLMakie,Random,CSV,JLSO
Random.seed!(1222)


fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing

df=fetch("sarcos_inv")

y, X= df[!,end],df[!,1:end-1]
(Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.8, multi=true,  rng=123)
rows=length(ytest)  #for plot 

function define_modeltype()
    modelType1 = @load LinearRegressor pkg=MLJLinearModels
    modelType2 = @load EvoLinearRegressor pkg=EvoLinear
    modelType3 = @load ARDRegressor pkg=MLJScikitLearnInterface
    modelType4 =  @load BaggingRegressor pkg=MLJScikitLearnInterface
    modelType5 =  @load AdaBoostRegressor pkg=MLJScikitLearnInterface
    modelType6 =  @load CatBoostRegressor pkg=CatBoost
    modelType7 =  @load BayesianRidgeRegressor pkg=MLJScikitLearnInterface
    modelType8 =  @load ElasticNetRegressor pkg=MLJLinearModels
    modelType9 =  @load EpsilonSVR pkg=LIBSVM
    modelType10 = @load KNNRegressor pkg=NearestNeighborModels
    modelType11 = @load SVMLinearRegressor pkg=MLJScikitLearnInterface
    modelType12 = @load SVMRegressor pkg=MLJScikitLearnInterface
    modelType13 = @load RidgeRegressor pkg=MultivariateStats
    modelType14 = @load RandomForestRegressor pkg=DecisionTree
    modelType15 = @load NuSVR pkg=LIBSVM
    modelType16 = @load LassoRegressor pkg=MLJLinearModels
    modelType17 =  @load XGBoostRegressor pkg=XGBoost
    modelType18 = @load GradientBoostingRegressor pkg=MLJScikitLearnInterface
    modelType19 =  @load AdaBoostRegressor pkg=MLJScikitLearnInterface
    modelType20 = @load NeuralNetworkRegressor pkg = "BetaML"
    modelTypes=[modelType1,modelType2,modelType3,modelType4,modelType5,modelType6,modelType7,modelType8,modelType9,modelType10,modelType11,modelType12,modelType13,modelType14,modelType15,modelType16,modelType17,modelType18,modelType19,modelType20]

    
    return modelTypes
end
#models=define_modeltype()


#rms_arr=Vector{RMS}(undef, length(models));

#rms_arr=Vector{Tuple{AbstractString,Float64}}(undef, length(models))
#df=DataFrame(name =String[], rms = Float64[])


function plot_res()
    fig=Figure(resolution=(2400,300))
    ax=Axis(fig[1,1])
    scatter!(ax, 1:rows,ytest,label="obersvations",color=(:orange,0.8))
    scatter!(ax,1:rows,yhat,label="predictions",overdraw=true,color=(:blue,0.2))
    axislegend(ax)
    fig
end

function eval_models()
    for (idx, model) in  enumerate(models)
        local name=string(nameof(typeof(model())))
        if name=="NeuralNetworkRegressor"
            layers= [BetaML.DenseLayer(21,20,f=BetaML.relu),BetaML.DenseLayer(20,20,f=BetaML.relu),BetaML.DenseLayer(20,1,f=BetaML.relu)]
            model=model(layers=layers,opt_alg=BetaML.ADAM())
            (fitResults, _, _)=MLJ.fit(model, 0, Xtrain, ytrain)
            local yhat=predict(model, fitResults, Xtest)
        else
            local mach = machine(model(), Xtrain, ytrain)|>fit!
            local yhat=predict(mach, Xtest)
        
        end
        local r=rms(yhat,ytest)|>d->round(d,digits=3)
        #rms_arr[idx]=(name,r)
        insert!(df,idx, (name, r), promote=true)
        
    end
end

#eval_models()
#JLSO.save("./MLJ/mlj-linearreg-ression-comparison.jlso", :rms => df)
data= JLSO.load("./MLJ/mlj-linearreg-ression-comparison.jlso")[:rms]
data=filter(row -> row.rms<50, data)
(max,min)=extrema(data[:,2])
function plot_rms1()
    fig=Figure(resolution=(1800,600))
    ax=Axis(fig[1,1],xticks = (1:1:size(data,1), data[:,1]),xticklabelrotation=0.3,xticklabelcolor=:blue,xticklabelsize=20,title="sarcos-mlj-linreg-comparsion" )
    stem!(ax,1:size(data,1) ,data[:,2])
    hlines!(ax,[max,min],linestyle=:dot,color=(:red,0.5))
    save("./MLJ/imgs/mlj-linearreg-ression-comparison-rms.png",fig)
end

#plot_rms1()

function plot_rms2()
    fig=Figure(resolution=(2000,600))
    ax=Axis(fig[1,1],xticks = (1:1:size(data,1), data[:,1]),xticklabelrotation=0.3,xticklabelcolor=:blue,xticklabelsize=20,title="sarcos-mlj-linreg-comparsion")
    barplot!(ax,1:size(data,1) ,data[:,2],bar_labels = :y,flip_labels_at=10.00,width=0.8,color=data[:,2],strokecolor = :black, strokewidth = 1)
    save("./MLJ/imgs/mlj-linearreg-ression-comparison-barplot.png",fig)
end 

plot_rms2()

