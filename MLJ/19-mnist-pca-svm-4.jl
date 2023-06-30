"""
1,2,3的方法有点杂乱, 这里整理一下
svm 要用 yhat = predict(mach, Xnew) 注意
"""


import MLJ:transform,predict,predict_mode
import LIBSVM
using DataFrames,MLJ,CSV,MLJModelInterface,GLMakie


#==========================data processing=======================================#
    function data_prepare(str)
        fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
        to_ScienceType(d)=coerce(d,:label=>Multiclass)
        df = fetch(str)|>to_ScienceType
        ytrain, Xtrain=  unpack(df, ==(:label), rng=123);
        cat=ytrain|>levels|>unique
        return ytrain, Xtrain,cat
    end

    str="mnist_train"
    y, X,cat=data_prepare(str)
    (Xtrain, Xtest), (ytrain, ytest)  = partition((X, y), 0.7, multi=true,shuffle=true)
#==========================data processing end====================================#



#==================SVM workflow ================================================#
    PCA = @load PCA pkg=MultivariateStats
    SVC = @load SVC pkg=LIBSVM 
    
    """
        mnist_pca_svm(dim=50)
        定义 mnist 的 pca  svm 方法
    ## 参数:默认缩减维度至 50,30-80效果都比较好
    1.  模型:  model1=PCA(maxoutdim=dim) pca 模型
    2.  模型:  model2 = SVC()  svc模型

    !!! warning:MLJ 包装的多元统计的方法不能从缩减维度近似重建
        原始维度, 需要使用原始包, 在-3 文件中有重建方法

    流程是: 
    -  模型训练
        1. 构建模型 1,2
        2. 训练集拟合 pca模型
        3. 将训练集维度降低为 pca模型的维度
        4. 降低维度的训练集输入 svm 模型, 拟合降维的分类模型
        pca-svm 训练完成
    - 测试模型
        
        1. 测试集转换为降维的数据集
        2. 利用降维 svm模型预测结果 yhat

    - 输出accuracy
        accuracy(yhat,ytest)


    如果以DataFrame 数据框的形式来看, 数据框的行数在维度缩减中没有变化, 
    变化的只是列数. 
    同理 ytrain, ytest 的维度都没有改变


    TBW
    """
    function mnist_pca_svm(;dim=50)
        model1=PCA(maxoutdim=dim)
        model2 = SVC()
        mach1 = machine(model1, Xtrain) |> fit!
        Ytr =transform(mach1, Xtrain)
        mach2 = machine(model2, Ytr, ytrain)|>fit!
        Yte=transform(mach1, Xtest)
        yhat = predict(mach2, Yte)
        res=accuracy(yhat,ytest)
        return round(res,digits=3)
    end

    #acc=pca_svm(;dim=30)

#==================SVM workflow end ============================================#



#==================plot accuracy================================================#

    function plot_accuracy(acc_arr,dims2)
        len=length(acc_arr)
        fig=Figure()
        ax=Axis(fig[1,1],xlabel="pcs",ylabel="accuracy",title="SVM accruacy with n primary components",
        xticks=(1:len, dims2)
        )
        scatterlines!(ax,acc_arr,markercolor = (:red,0.5))
        fig
        
    end

    dims=[3,10,15,30,50,100,150]
    dims2=["3","10","15","30","50","100","150"] # 用于 xticks 的标识

    acc_arr=[mnist_pca_svm(;dim=i) for i in dims]
    plot_accuracy(acc_arr,dims2)

#==================plot accuracy end================================================#

