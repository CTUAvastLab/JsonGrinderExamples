using JsonGrinder, Mill, Flux, MLDatasets, JSON, CSV, DataFrames, Statistics, Random

Random.seed!(42)

include("prepare_data.jl")

BATCH_SIZE = 50

x_train, y_train = load_train_data()
schema = JsonGrinder.schema(x_train)
extractor = suggestextractor(schema)
ds_train = map(extractor, x_train)

encoder = reflectinmodel(schema, extractor, d -> Chain(Dense(d, 50, relu), Dropout(0.25)))
model = Dense(50, length(LABELS)) ∘ encoder
loss = (ds, y) -> Flux.Losses.logitcrossentropy(model(ds), Flux.onehotbatch(y, eachindex(LABELS)))
accuracy = (ds, y) -> mean(Flux.onecold(model(ds), eachindex(LABELS)) .== y)

opt = AdaBelief()
ps = Flux.params(model)
data_loader = Flux.Data.DataLoader((ds_train, y_train), batchsize=BATCH_SIZE, shuffle=true)
Flux.@epochs 3 begin
    Flux.Optimise.train!(loss, ps, data_loader, opt)
    @show accuracy(ds_train, y_train)
end

x_test, y_test = load_test_data()
ds_test = map(extractor, x_test)
@show accuracy(ds_test, y_test)
