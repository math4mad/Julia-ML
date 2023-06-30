"""
MLJ  GaussianMixtureRegressor

botson 回归
"""

import MLJ:predict
using MLJ

modelType= @load GaussianMixtureRegressor pkg = "BetaML"

gmr= modelType()

X, y= @load_boston;
#schema(X)

(fitResults, cache, report) = MLJ.fit(gmr, 1, X, y);


y_res= predict(gmr, fitResults, X)

#rmse(y_res,y)

