using JLD2;
using Distributed;
using Printf;
using IterTools;
using Distributions
using Random;
using JLD;
using JSON;
using HDF5;
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

δ = 0.01
d = 6
K = 20
m = 5  # for topm

# for reproducibility
rng = MersenneTwister(123)


arms = Vector{Float64}[]
for i in 1:20
    v = randn(rng, d)
    v /= norm(v)
    push!(arms, v)
end

min_norm = 10
for i in 1:length(arms)
    for j in (i+1):length(arms)
        norm_ij = norm(arms[i] - arms[j])
        if norm_ij < min_norm
            global min_pair = [i, j]
            global min_norm = norm_ij
        end
    end
end


θ = arms[min_pair[1]] + 0.01*(arms[min_pair[2]] - arms[min_pair[1]])
println(arms)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/theta.csv", θ)
writedlm("/home/jupyter-zli9/PureExpThompsonSampling/arms.csv", arms)

# # β = LinearThreshold(d, 1, 1, 1)
# # β = GK16()
# β = HeuristicThreshold()
β = 5

pep = BAI(arms);

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

    # dump_stats(pep, θ, δ, β, stopping_rules, sampling_rules, elim_rules, data, repeats);
    
    data = collect(data)
    writedlm("/home/jupyter-zli9/PureExpThompsonSampling/results_sphere_500_LinGame.txt", data, '\t')

#     # save
#     isdir("experiments/results") || mkdir("experiments/results")
#     @save isempty(ARGS) ? "experiments/results/lin_$(typeof(pep))_$(typeof(sampling_rules[1]))_K$(K)_d$(d).dat" : ARGS[1] θ pep stopping_rules sampling_rules elim_rules data δ β repeats seed

end

# #################################################
# # LinGapE
# #################################################
# # elim_rules = [NoElim(), CompElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [LinGapE(NoElimSR), LinGapE(NoElimSR), LinGapE(ElimSR)]

# # run()

#################################################
# LinGame
#################################################

elim_rules = [NoElim()]
stopping_rules = [Force_Stopping(3000, NoStopping())]
sampling_rules = [LinGame(CTracking, NoElimSR, false)]

# elim_rules = [NoElim(), CompElim(), CompElim()]
# stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping()),  Force_Stopping(max_samples, Elim_Stopping())]
# sampling_rules = [LinGame(CTracking, NoElimSR, false), LinGame(CTracking, NoElimSR, false), LinGame(CTracking, ElimSR, false)]

run()


# #################################################
# # Oracle
# #################################################
# # elim_rules = [NoElim(), CompElim()]
# # stopping_rules = [Force_Stopping(max_samples, LLR_Stopping()), Force_Stopping(max_samples, Elim_Stopping())]
# # sampling_rules = [FixedWeights(w_star), FixedWeights(w_star)]

# # run()