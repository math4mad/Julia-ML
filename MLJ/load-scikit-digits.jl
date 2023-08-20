"""
从 scikitlearning 导入 digits 数据然后转存为 csv 文件
"""


using ScikitLearn, Random, Printf,DataFrames
using PyCall,CSV
using ScikitLearn.Utils: meshgrid


@sk_import datasets: load_digits
@pyimport sklearn.metrics as metrics
@pyimport sklearn.preprocessing as preprocessing


Random.seed!(42)

digits = load_digits()

df=DataFrame(digits["data"],:auto)
df[:,:target]=digits["target"]
first(df,10)
isfile("fetch_digits.csv") ? (@info "already exists") : CSV.write("fetch_digits.csv", df)
#CSV.write("fetch_olivetti_faces.csv", df1)  写入 csv 文件