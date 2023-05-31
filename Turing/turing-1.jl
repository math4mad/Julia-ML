"""
参考:
https://storopoli.io/Bayesian-Julia/pages/04_Turing/

"""

using Turing, Distributions,Random
Random.seed!(13456)

#=
mean(Dirichlet(6, 1))
sum(mean(Dirichlet(6, 1)))
=#

@model function dice_throw(y)
    #Our prior belief about the probability of each result in a six-sided dice.
    #p is a vector of length 6 each with probability p that sums up to 1.
    p ~ Dirichlet(6, 1)

    #Each outcome of the six-sided dice has a probability p.
    for i in eachindex(y)
        y[i] ~ Categorical(p)
    end
end;

my_data = rand(DiscreteUniform(1, 6), 1_000);

#first(my_data,10)

model = dice_throw(my_data);

chain = sample(model, NUTS(), 1_000);

summaries,_ = describe(chain);

display(summaries)