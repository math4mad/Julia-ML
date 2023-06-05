"""
data from :https://www.kaggle.com/datasets/abhishek14398
/salary-dataset-simple-linear-regression

注意数据的处理, MLJ 需要把 X 转为表格,并且是矩阵形式
X=MLJ.table(reshape(df[:,2],30,1))
y=Vector(df[:,3])
"""

using Plots, MLJ,CSV,DataFrames

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./DataSource/$str.csv") |> DataFrame |> dropmissing
    to_ScienceType(d)=coerce(d,:YearsExperience=>Continuous,:Salary=>Continuous)
    df = fetch(str)|>to_ScienceType
    return df
end

str="salary_dataset"
df=data_prepare(str)

 X=MLJ.table(reshape(df[:,2],30,1))
 y=Vector(df[:,3])


 # MLJ workflow 
 LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
 model=LinearRegressor()
 mach = MLJ.fit!(machine(model,X,y))

 fp=MLJ.fitted_params(mach)
 

 


#获取回归直线的参数
a=fp.coefs[1,1][2]
b=fp.intercept
f(t)=a*t+b

xspan=range(extrema(df[:,2])...,200)
plot(xspan,f.(xspan),label="fit-line")
scatter!(df[:,2],df[:,3],label="data",xlabel="YearsExperience",ylabel="Salary")

savefig("5-salary-dataset-linear-regression.png")
