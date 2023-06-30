"""
compare of BetalML method of classfiy german credit card 
"""

import MLJ:predict,predict_mode
import BetaML
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie
using CatBoost.MLJCatBoostInterface

#=================================data processing============================================#

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

#=================================data processing end=========================================#


#=================================ml model==================================================#

    function define_models()

        modelType1= @load NeuralNetworkClassifier pkg = "BetaML"

        layers= [BetaML.DenseLayer(19,8,f=BetaML.relu),BetaML.DenseLayer(8,8,f=BetaML.relu),BetaML.DenseLayer(8,2,f=BetaML.relu),BetaML.VectorFunctionLayer(2,f=BetaML.softmax)];
        nn_model= modelType1(layers=layers,opt_alg=BetaML.ADAM())

        modelType2= @load DecisionTreeClassifier pkg = "BetaML" verbosity=0
        dt_model= modelType2()

        modelType3= @load KernelPerceptron pkg = "BetaML"
        kp_model= modelType3()


        modelType4= @load LinearPerceptron pkg = "BetaML"
        lp_model= modelType4()

        modelType5= @load Pegasos pkg = "BetaML" verbosity=0
        peg_model=modelType5()


        modelType6= @load RandomForestClassifier pkg = "BetaML" verbosity=0
        rf_model=modelType6()

        
        cat_model=CatBoostClassifier(iterations=5)

        models=[nn_model,dt_model,kp_model,lp_model,peg_model,rf_model,cat_model]
        models_name=["nn","dt","kp","lp","peg","rf","cat"]
        return models,models_name
    end

    models,models_name=define_models()
#=================================ml model end==============================================#




function main()
    for (idx,model) in enumerate(models[1:6])
        local (fitResults, cache, report) = MLJ.fit(model, 0, X,y);
        local est_classes= predict_mode(model, fitResults, Xtest)
        local acc=accuracy(ytest,est_classes)|>d->round(d, digits=3)
        @info "$(models_name[idx])===>$(acc)"
    end
end

main()

# """
#     [ Info: nn===>0.692
#     [ Info: dt===>0.988
#     [ Info: kp===>1.0
#     [ Info: lp===>0.464
#     [ Info: peg===>0.692
#     [ Info: rf===>0.994
# """
# mach = machine(models[7], X, y)|>fit!

# preds = predict_mode(mach, Xtest)

# accuracy(ytest,preds)