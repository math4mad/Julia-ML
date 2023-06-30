import MLJ:evaluate,MLJInterface
using MLJ,DataFrames
iris=load_iris()|>DataFrame

#first(iris,5)

y, X = unpack(iris, ==(:target); rng=123);

function  build_model(X,y)
    Tree = @load DecisionTreeClassifier pkg=DecisionTree
    tree = Tree()
    evaluate(tree, X, y, resampling=CV(shuffle=true),
        measures=[log_loss, accuracy],
        verbosity=0)
end

build_model(X,y)




