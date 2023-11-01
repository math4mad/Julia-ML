"""
http://www.probability.ca/jeff/teaching/1617/sta130/lecturenotes/notesCposthand.pdf
https://rdrr.io/rforge/Lock5Data/man/CricketChirps.html#heading-3
https://www.britannica.com/animal/snowy-tree-cricket
雪树蟋蟀的鸣叫实际是大腿摩擦发出的声音, 经过数据收集,发现鸣叫的频率和环境温度正相关.

经过线性拟合得到的函数为`C(t)=4.25t-157.8`

Calculus Single Variable by Deborah Hughes-Hallett.pdf page 27
`𝐶 = 4𝑇 − 160`
"""

import FileIO:load
import MLJ:fit!,match,predict,table,fitted_params
using GLMakie, CSV,DataFrames,MLJ,FileIO
img=load("./test&exercise/imgs/snowy-cricket.jpg")

function data_prepare(str)
    urls(str) = "./DataSource/$str.csv"
    f(str) = urls(str) |> CSV.File |> DataFrame
    df = f(str)
    rows, _ = size(df)
    return df
end

df=data_prepare("CricketChirps")

X=MLJ.table(reshape(df[:,1],7,1))
y=Vector(df[:,2])

test_X=range(extrema(df[:,1])...,50)
test_X=MLJ.table(reshape(test_X,50,1))
cols=names(df)
fig=Figure()


function plot_origin_data(df)
    X=df[:,1]
    y=df[:,2]
    cols=names(df)
    ax=Axis(fig[1,1];xlabel="$(cols[1])",ylabel="$(cols[2])")
    scatter!(ax, X,y,markersize=16,color=(:red,0.8))
    fig
end
#plot_origin_data(df)

function LineReg(X,y)
    LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
    mach = fit!(machine(LinearRegressor(), X, y))
    return mach
end

mach=LineReg(X,y)
yhat=predict(mach,test_X).|>(d->round(d,digits=2))


function plot_fitting_curve(df,yhat)
    X=df[:,1]
    test_X=range(extrema(df[:,1])...,50)
    cols=names(df)
    fig=Figure()
    ax=Axis(fig[1:3,1:3];xlabel="$(cols[1])",ylabel="$(cols[2])",title="cricket-chirp")
    ax2 = Axis(fig[2,4],title="snowy-tree-cricket")
    scatter!(ax, X,y,markersize=16,color=(:red,0.8))
    lines!(ax, test_X,yhat,color=:blue)
    image!(ax2,img)
    hidespines!(ax2)
    hidedecorations!(ax2)
    fig
    
    #save("cricket-chirp-rate-linear-reg.png",fig)
end

plot_fitting_curve(df,yhat)

#params=fitted_params(mach)
#a,b=params
#params[:coefs][1][2]
#[:slope=>a[1][2],:intercep=>b]
#b=params[2]





