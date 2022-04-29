####
# Allow to use more threads using -t switch, as for example below
# julia --project=. -t 28
# 
####
using Folds, Flux, StatsBase, Random, JSON, JsonGrinder, Mill, IterTools, Serialization,  FileIO

cachedir(s...) = joinpath("/home/tomas.pevny/data/cache/ember_2018/jls_1.7.2/", s...)
jsondir(s...) = joinpath("/home/tomas.pevny/data/cache/ember_2018/jsonl/", s...)

include("prepare_data.jl")

function predict(model, x; batchsize = 100)
	Flux.testmode!(true)
	base_funcs = base_funcs[2:end] 
	o = Folds.map(Iterators.PartitionIterator(x, length(x) ÷ batchsize)) do xs 
		Flux.onecold(model(reduce(catobs,xs)), 0:1)
	end
	reduce(vcat, o)
end

function prepare_minibatch(x, y, n) 
	ii = sample(1:length(y), n, replace = false)
	(reduce(catobs, x[ii]), Flux.onehotbatch(y[ii], [0,1]))
end


((trn_x, trn_y), (tst_x, tst_y)) = prepare_data();

model = reflectinmodel(x[1],
   d -> Chain(Dense(d, 32, relu), BatchNorm(32)),
   SegmentedMeanMax,
   fsm = Dict("" => d -> Chain(Dense(d, 32, relu), BatchNorm(32), Dense(32, 2))),
   single_scalar_identity = false,
   all_imputing = true,
)

opt = AdaBelief()
cby, history = PrayTools.initevalcby()
ps = Flux.params(model);
Flux.testmode!(false)
PrayTools.train!((x, y) -> logitcrossentropy(model(x), y), ps, mb, opt, 20000; cby)

mean(predict(model, trn_x) .= trn_y)
mean(predict(model, tst_x) .= tst_y)