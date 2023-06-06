
"""
Calculus_ Early Transcendentals. 9e-Cengage Learning (2020).pdf page 61
"""

using GaussianProcesses,CSV,GLMakie,Random,Distributions,LinearAlgebra,
      StatsBase

Random.seed!(1223)
ts=range(0.1,1,10)
height=Float64[450,445,431,408,375,332,279,216,143,61]

mZero = MeanZero()              
kern = PolynomialKernel()
logObsNoise = -1.0 

gp = GP(ts,height,mZero,kern)





