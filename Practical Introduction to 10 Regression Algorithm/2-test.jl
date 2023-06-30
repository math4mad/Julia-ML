using GLMakie, MLJ,CSV,DataFrames,StatsBase
include("data-processing.jl")

df=get_data()

first(df,10)