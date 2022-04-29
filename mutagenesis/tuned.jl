using JsonGrinder, Mill, Flux, MLDatasets, Statistics, Random

Random.seed!(42)

BATCH_SIZE = 50
P_DROPOUT=0.05

x_train, y_train = MLDatasets.Mutagenesis.traindata();
data_dropout(d::Dict) = Dict(k => data_dropout(v) for (k, v) in randsubseq(collect(d), 1-P_DROPOUT))
data_dropout(v::Vector) = data_dropout.(randsubseq(v, 1-P_DROPOUT))
data_dropout(x) = x
x_train = [x_train; data_dropout.(x_train); data_dropout.(x_train)]
y_train = Flux.onehotbatch(repeat(y_train, 3) .+ 1, 1:2)
schema = JsonGrinder.schema(x_train)
extractor = suggestextractor(schema)
ds_train = map(extractor, x_train)

encoder = reflectinmodel(schema, extractor,
    d -> Chain(Dense(d, 50, relu)),
)
model = Dense(50, 2) ∘ encoder
loss = (ds, y) -> Flux.Losses.logitbinarycrossentropy(model(ds), y)
accuracy = (ds, y) -> mean(Flux.onecold(model(ds)) .== y .+ 1)
opt = AdaBelief()
ps = Flux.params(model)
data_loader = Flux.Data.DataLoader((ds_train, y_train), batchsize=BATCH_SIZE, shuffle=true)
Flux.Optimise.train!(loss, ps, data_loader, opt)

x_test, y_test = MLDatasets.Mutagenesis.testdata();
ds_test = map(extractor, x_test)
@show accuracy(ds_test, y_test)
