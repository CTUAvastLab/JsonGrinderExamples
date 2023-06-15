####
# Allow to use more threads using -t switch, as for example below
# julia --project=. -t 28
# 
####
using Folds
using Flux
using StatsBase
using Random
using JSON
using JsonGrinder
using Mill
using IterTools
using Serialization
using FileIO
using PrayTools
using Setfield
using Flux.Losses: logitcrossentropy
using GradientBoost
using GradientBoost.LossFunctions: LogisticLoss
import GradientBoost.GBBaseLearner: learner_fit, learner_predict

include("prepare_data.jl")

struct MillLearner end 

function learner_fit(loss, learner::MillLearner, x, wy)
	y = Int.(wy .> 0)
	w = StatsBase.Weights(abs.(wy))
	function mb() 
		ii = sample(1:length(y), w, 128, replace = false)
		(reduce(catobs, x[ii]), Flux.onehotbatch(y[ii], [0,1]))
	end
	model = reflectinmodel(x[1], 
		d -> Chain(Dense(d, 32, relu)),
		SegmentedMeanMax, 
		fsm = Dict("" => d -> Chain(Dense(d, 32, relu), Dense(32, 2))),
		single_scalar_identity = false,
		all_imputing = true,
	)
	function l(x::AbstractMillNode, y)
		logitcrossentropy(model(x), y)
	end
	opt = AdaBelief()
	cby, history = PrayTools.initevalcby()
	ps = Flux.params(model);
	Flux.testmode!(false)
	PrayTools.train!(l, ps, mb, opt, 20000; cby)
	Flux.testmode!(true)
	@set model.m = Chain(model.m.layers..., x -> (2softmax(x)[2,:] .- 1))
end

function learner_predict(loss, learner::MillLearner,  model, x)
	Flux.testmode!(true)
	o = Folds.map(Iterators.PartitionIterator(x, 100)) do xs 
		model(reduce(catobs,xs))
	end
	reduce(vcat, o)
end

function accuracy(gb_model, x, y)
	Flux.testmode!(true)
	ŷ = Folds.map(Iterators.PartitionIterator(x, 100)) do xs 
		xx = reduce(catobs,xs)
		ŷ = zeros(Float32, length(xs))
		for f in gb_model.base_funcs[2:end]
			ŷ .+= f.model_const .* f.model(xx)
		end
		ŷ
	end
	ŷ = reduce(vcat, ŷ)
	mean((ŷ .> 0) .== y)
end


((trn_x, trn_y), (tst_x, tst_y)) = load_cached_data();
gbbl = GradientBoost.ML.GBBL(MillLearner();loss_function = LogisticLoss(), num_iterations = 5, learning_rate=1)
gbl = GradientBoost.ML.GBLearner(gbbl, :class)
gbl.model = GradientBoost.ML.fit(gbl.algorithm, trn_x, 2trn_y .- 1)

tst_accuracy = accuracy(gbl.model, tst_x, tst_y)
trn_accuracy = accuracy(gbl.model, trn_x, trn_y)

@info "Results" trn_accuracy tst_accuracy
