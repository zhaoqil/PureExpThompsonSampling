using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions;
using DelimitedFiles

include("../thresholds.jl")
include("../peps.jl")
include("../elimination_rules.jl")
include("../stopping_rules.jl")
include("../sampling_rules.jl")
include("../runit.jl")
include("../experiment_helpers.jl")
include("../utils.jl")
include("../envelope.jl")

δ = 0.05
d = 12
K = 12
m = 3  # for topm

# for reproducibility
rng = MersenneTwister(123)

arms = Vector{Float64}[]
for k = 1:d
    v = zeros(d)
    v[k] = 1.0
    push!(arms, v)
end

alpha = 0.05
θ = [1 - i * alpha for i in 0:d-1]

println(arms)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/theta_topm.csv", θ)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/arms_topm.csv", arms)

μ = [arm'θ for arm in arms]
topm_arms = istar(Topm(arms, m), θ)
println("min abs value of μ: ", minimum(abs.(μ)))
println("min gap: ", minimum(maximum(μ) .- maximum(μ[setdiff(1:K, Set([argmax(μ)]))])))
println("min gap topm: ", minimum(minimum(μ[topm_arms]) .- maximum(μ[setdiff(1:K, topm_arms)])))

# β = LinearThreshold(d, 1, 1, 1)
# β = GK16()
β = HeuristicThreshold()

pep = Topm(arms, m);

w_star = optimal_allocation(pep, θ, false, 10000)
println("Optimal allocation: ", round.(w_star, digits=3))

max_samples = 1e6

repeats = 5;
seed = 123;

function run()

    # One fake run for each algorithm
    # This is to have fair comparison of compute times later since Julia compiles stuff at the first calls
    @time data = map(  # TODO: replace by pmap (it is easier to debug with map)
        ((sampling, stopping, elim),) -> runit(seed, sampling, stopping, elim, θ, pep, β, δ), 
        zip(sampling_rules, stopping_rules, elim_rules)
    );

    @time data = map(  # TODO: replace by pmap (it is easier to debug with map)
        (((sampling, stopping, elim), i),) -> runit(seed + i, sampling, stopping, elim, θ, pep, β, δ), 
        Iterators.product(zip(sampling_rules, stopping_rules, elim_rules), 1:repeats),
    );
    
    data = collect(data)
    writedlm("/home/jupyter-zli9/PureExpThompsonSampling/results_topm_500_oracle.txt", data, '\t')

#     dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);

#     # save
#     isdir("experiments/results") || mkdir("experiments/results")
#     @save isempty(ARGS) ? "experiments/results/lin_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

#################################################
# LinGapE
#################################################
# elim_rules = [NoElim()]
# stopping_rules = [Force_Stopping(30000, NoStopping())]
# sampling_rules = [LinGapE(NoElimSR)]

# run()

#################################################
# LinGame
#################################################
# elim_rules = [NoElim()]
# stopping_rules = [Force_Stopping(30000, NoStopping())]
# sampling_rules = [LinGame(CTracking, NoElimSR, false)]

# run()

# #################################################
# # LinGIFA
# #################################################
# elim_rules = [NoElim(), CompElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [LinGIFA(), LinGIFA()]

# run()

#################################################
# Oracle
#################################################
elim_rules = [NoElim()]
stopping_rules = [Force_Stopping(30000, NoStopping())]
sampling_rules = [FixedWeights(w_star)]

run()

# #################################################
# # LazyTaS
# #################################################
# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [LazyTaS(NoElimSR), LazyTaS(NoElimSR), LazyTaS(ElimSR)]

# run()

# #################################################
# # FWS
# #################################################
# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [FWSampling(NoElimSR), FWSampling(NoElimSR), FWSampling(ElimSR)]

# run()