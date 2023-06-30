"""
 参见 文档 learning curves
"""
using MLJ,Plots

X, y = @load_boston;

atom = (@load RidgeRegressor pkg=MLJLinearModels)()
ensemble = EnsembleModel(model=atom, n=1000)
mach = machine(ensemble, X, y)

r_lambda = range(ensemble, :(model.lambda), lower=1e-1, upper=100, scale=:log10)

curve = MLJ.learning_curve(mach;
                           range=r_lambda,
                           resampling=CV(nfolds=3),
                           measure=MeanAbsoluteError())

plot(curve.parameter_values,
     curve.measurements,
     xlab=curve.parameter_name,
     xscale=curve.parameter_scale,
     ylab = "CV estimate of RMS error")