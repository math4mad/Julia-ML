

"""
James Stewart, Daniel K. Clegg, Saleem Watson, Lothar Redlin - Calculus_ Early Transcendentals. 9e-Cengage Learning (2020).pdf page4
"""
using Plots

ts=range(0,2pi,200)

# trigonometric_func=[sin,cos, tan,csc,sec,cot]
# plot_arr=[]

# for fun in trigonometric_func
#     p=plot(ts,fun.(ts), frame=:origin, lw=1, color=:blue,label=false)
#     push!(plot_arr,p)
# end
# plot(plot_arr...;layout=(3,3))

plot(ts,cot.(ts),frame=:origin)


