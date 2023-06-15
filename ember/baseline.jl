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
using Flux.Losses: logitcrossentropy

include("prepare_data.jl")

function StatsBase.predict(model::Mill.AbstractMillModel, x::Vector; batchsize = 100)
	Flux.testmode!(true)
	o = Folds.map(Iterators.PartitionIterator(x, length(x) ÷ batchsize)) do xs 
		Flux.onecold(model(reduce(catobs,xs)), 0:1)
	end
	reduce(vcat, o)
end

function prepare_minibatch(x, y, n) 
	ii = sample(1:length(y), n, replace = false)
	(reduce(catobs, x[ii]), Flux.onehotbatch(y[ii], [0,1]))
end

((trn_x, trn_y), (tst_x, tst_y)) = load_cached_data();

model = reflectinmodel(trn_x[1],
   d -> Dense(d, 32, relu),
   SegmentedMeanMax,
   fsm = Dict("" => d -> Chain(Dense(d, 32, relu), Dense(32, 2))),
   single_scalar_identity = false,
   all_imputing = true,
)


opt = ADAM()
cby, history = PrayTools.initevalcby()
ps = Flux.params(model);
Flux.testmode!(false)
mb = () -> prepare_minibatch(trn_x, trn_y, 128)
loss(x,y) =  logitcrossentropy(model(x), y)
PrayTools.train!(loss, ps, mb, opt, 20000; cby)

tst_accuracy = mean(predict(model, tst_x) .== tst_y)
trn_accuracy = mean(predict(model, trn_x) .== trn_y)

@info "Results" trn_accuracy tst_accuracy
