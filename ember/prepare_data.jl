using Downloads
using TranscodingStreams
using CodecBzip2
using Downloads
using Tar
const EMBER_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

cachedir(s...) = joinpath("/home/tomas.pevny/data/cache/ember_2018/jls_1.7.2/",s...)
jsondir(s...) = joinpath("/home/tomas.pevny/data/cache/ember_2018/jsonl/",s...)

"""
	function prepare_raw_data()

	download ember data from the central repository, 
	unpack it to jsons. Everything is performed only 
	if needed
"""
function prepare_raw_data()
	if !isfile(cacheddatadir("ember_dataset_2018_2.tar.bz2"))
		!isdir(cacheddatadir()) && mkpath(cacheddatadir())
		@info "downloading ember data from $(EMBER_URL)"
		Downloads.download(EMBER_URL, cacheddatadir("ember_dataset_2018_2.tar.bz2"))
	end
	json_files = [
	 "test_features.jsonl",
	 "train_features_0.jsonl",
	 "train_features_1.jsonl",
	 "train_features_2.jsonl",
	 "train_features_3.jsonl",
	 "train_features_4.jsonl",
	 "train_features_5.jsonl",
	 ]
	if !all((isfile ∘ jsondir)(s) for s in json_files)
		@info "extracting downloaded data (we use Julia's internal tooling for platform independece"
		raw_data = read(cacheddatadir("ember_dataset_2018_2.tar.bz2"))
		decoded_data = transcode(Bzip2Decompressor, raw_data)
		write(cacheddatadir("ember_dataset_2018_2.tar"), decoded_data)
		Tar.extract(cacheddatadir("ember_dataset_2018_2.tar"), jsondir())
		for l in filter(s -> endswith(s, ".jsonl"), readdir(jsondir("ember2018")))
			mv(jsondir("ember2018", l), jsondir(l))
		end
		rm(jsondir("ember2018"), recursive=true)
	end
end

function makeschema(jsons::Vector{T}) where {T<:Dict}
	JsonGrinder.updatemaxkeys!(typemax(Int))
	schemas = Folds.map(schema, Iterators.partition(jsons, div(length(jsons), Threads.nthreads())));
	merge(schemas...)
end

function fixsample(ds)
	ds[:histogram].data ./= max.(sum(ds[:histogram].data, dims = 1), 1)
	ds[:byteentropy].data ./= max.(sum(ds[:byteentropy].data, dims = 1), 1)
	ds[:strings][:printabledist].data ./= max.(sum(ds[:strings][:printabledist].data, dims = 1), 1)
	ds
end

function prepare_data()
	trn_files = filter(s -> startswith(s, "train_"), readdir(jsondir()))
	trn_jsons = map(s -> Folds.map(l -> JSON.parse(l),readlines(jsondir(s))), trn_files);
	trn_jsons = reduce(vcat, trn_jsons);

	trn_targets = Folds.map(d -> d["label"], trn_jsons);
	trn_jsons = trn_jsons[trn_targets .!= -1]

	!isdir(cachedir()) && mkpath(cachedir())
	if !isfile(cachedir("extractor.jls"))
		sch = makeschema(trn_jsons);
		ks = [:exports,:section,:general,:header,:histogram,:datadirectories,:byteentropy,:strings]
		# exs = map(k -> k => suggestextractor(sch[k], (scalar_extractors = extractor_rules(), key_as_field = 500)), ks)
		exs = map(k -> k => suggestextractor(sch[k], (;key_as_field = 500)), ks)
		push!(exs, :imports => ExtractKeyAsField(ExtractString(), ExtractArray(ExtractString())))
		extractor = ExtractDict(Dict(exs))
		serialize(cachedir("extractor.jls"), extractor)
		serialize(cachedir("schema.jls"), sch)
	end

	extractor = deserialize(cachedir("extractor.jls"))

	####
	# export the training data
	####
	trn_targets = Folds.map(d -> d["label"], trn_jsons);
	class_indexes = classindexes(trn_targets);
	mbindices = map(x -> vcat(x...), zip(map(v -> Iterators.partition(shuffle(v), 100), values(class_indexes))...))
	trn_dsy = Folds.map(mbindices) do ii
		(ds = map(fixsample ∘ extractor, trn_jsons[ii]),
		y = trn_targets[ii],
		)
	end
	trn_x = reduce(vcat, [x[1] for x in trn_dsy])
	trn_y = reduce(vcat, [x[2] for x in trn_dsy])

	####
	# export the testing data
	####
	tst_jsons = Folds.map(l -> JSON.parse(l), readlines(jsondir("test_features.jsonl")))
	tst_targets = map(d -> d["label"], tst_jsons);
	mbindices = Iterators.partition(1:length(tst_jsons), 100)
	tst_dsy = Folds.map(mbindices) do ii
		(ds = map(fixsample ∘ extractor, tst_jsons[ii]),
		y = tst_targets[ii],
		)
	end
	tst_x = reduce(vcat, [x[1] for x in tst_dsy])
	tst_y = reduce(vcat, [x[2] for x in tst_dsy])

	((trn_x, trn_y), (tst_x, tst_y))
end
