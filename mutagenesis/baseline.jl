using JsonGrinder, Mill, Flux, MLDatasets, Statistics, Random

Random.seed!(42)

BATCH_SIZE = 10

x_train, y_train = MLDatasets.Mutagenesis.traindata();
schema = JsonGrinder.schema(x_train)
extractor = suggestextractor(schema)
ds_train = map(extractor, x_train)

encoder = reflectinmodel(schema, extractor)
model = Dense(10, 2) ∘ encoder
loss = (ds, y) -> Flux.Losses.logitbinarycrossentropy(model(ds), Flux.onehotbatch(y .+ 1, 1:2))
accuracy = (ds, y) -> mean(Flux.onecold(model(ds)) .== y .+ 1)

opt = AdaBelief()
ps = Flux.params(model)
data_loader = Flux.Data.DataLoader((ds_train, y_train), batchsize=BATCH_SIZE, shuffle=true)
Flux.@epochs 3 begin
    Flux.Optimise.train!(loss, ps, data_loader, opt)
    @show accuracy(ds_train, y_train)
end

x_test, y_test = MLDatasets.Mutagenesis.testdata();
ds_test = map(extractor, x_test)
@show accuracy(ds_test, y_test)
