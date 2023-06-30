"""
从 scikitlearning 导入 digits 数据然后转存为 csv 文件
"""


using ScikitLearn, Random, Printf,DataFrames
using PyCall,CSV
using ScikitLearn.Utils: meshgrid


@sk_import datasets: fetch_olivetti_faces
@pyimport sklearn.metrics as metrics
@pyimport sklearn.preprocessing as preprocessing


Random.seed!(42)

face = fetch_olivetti_faces()

df1=DataFrame(face["data"],:auto)
df1.label=face["target"]

isfile("fetch_olivetti_faces.csv") ? (@info "already exists") : CSV.write("fetch_olivetti_faces.csv", df1)
#CSV.write("fetch_olivetti_faces.csv", df1)  写入 csv 文件