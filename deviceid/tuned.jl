using JsonGrinder, Mill, Flux, MLDatasets, JSON, CSV, DataFrames, Statistics, Random, Setfield

Random.seed!(42)

include("prepare_data.jl")

BATCH_SIZE = 100

x_train, y_train = load_train_data()
y_train_smoothed = Flux.Losses.label_smoothing(Flux.onehotbatch(y_train, 1:length(LABELS)), 0.1)
x_train = split_ip(map_mac_vendor(x_train))
schema = JsonGrinder.schema(x_train)
extractor = suggestextractor(schema)
Setfield.@set! extractor.dict[:services].item.dict[:port] =
    ExtractCategorical(schema[:services].items[:port])
ds_train = map(extractor, x_train)

encoder = reflectinmodel(schema, extractor, d -> Chain(Dense(d, 50, gelu), Dropout(0.5)))
model = Dense(50, length(LABELS)) ∘ encoder
loss = (ds, y) -> Flux.Losses.logitcrossentropy(model(ds), y)
accuracy = (ds, y) -> mean(Flux.onecold(model(ds), eachindex(LABELS)) .== y)

opt = AdaBelief()
ps = Flux.params(model)
data_loader = Flux.Data.DataLoader((ds_train, y_train_smoothed), batchsize=BATCH_SIZE, shuffle=true)
Flux.@epochs 20 begin
    Flux.Optimise.train!(loss, ps, data_loader, opt)
    @show accuracy(ds_train, y_train)
end

x_test, y_test = load_test_data()
x_test = split_ip(map_mac_vendor(x_test))
ds_test = map(extractor, x_test)
@show accuracy(ds_test, y_test)
