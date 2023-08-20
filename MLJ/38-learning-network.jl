"""
from Data Science tutorials learning network
"""

import MLJ:transform
using MLJ, StableRNGs
import DataFrames
Ridge = @load RidgeRegressor pkg=MultivariateStats

rng = StableRNG(551234) # for reproducibility

x1 = rand(rng, 300)
x2 = rand(rng, 300)
x3 = rand(rng, 300)
y = exp.(x1 - x2 -2x3 + 0.1*rand(rng, 300))

X = DataFrames.DataFrame(x1=x1, x2=x2, x3=x3)
#first(X, 3) |> pretty
test, train = partition(eachindex(y), 0.8);

Xs = source(X)
ys = source(y)

stand = machine(Standardizer(), Xs)
W = transform(stand, Xs)