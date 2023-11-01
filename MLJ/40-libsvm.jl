import MLJ:predict
using MLJ
import LIBSVM

NuSVR = @load NuSVR pkg=LIBSVM                 ## model type
model = NuSVR(kernel=LIBSVM.Kernel.Polynomial) ## instance

X, y = make_regression(rng=123) ## table, vector
mach = machine(model, X, y) |> fit!

Xnew, _ = make_regression(3, rng=123)
yhat = predict(mach, Xnew)