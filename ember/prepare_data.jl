using JSON
using JsonGrinder
using Mill
using IterTools
using Serialization
using FileIO
using PrayTools
using Downloads
using TranscodingStreams
using CodecBzip2
using Downloads
using Tar
using Folds
using Random
const EMBER_URL = "https://ember.elastic.co/ember_dataset_2018_2.tar.bz2"

cachedir(s...) = joinpath("/tmp/ember/",s...)
jlsdir(s...) = cachedir("jls", s...)

"""
	function prepare_raw_data()

	download ember data from the central repository, 
	unpack it to is. Everything is performed only 
	if needed
"""
function prepare_raw_data()
	if !isfile(cachedir("ember_dataset_2018_2.tar.bz2"))
		!isdir(cachedir()) && mkpath(cachedir())
		@info "downloading ember data from $(EMBER_URL)"
		Downloads.download(EMBER_URL, cachedir("ember_dataset_2018_2.tar.bz2"))
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
	if !all((isfile ∘ cachedir)(s) for s in json_files)
		@info "extracting downloaded data (we use Julia's internal tooling for platform independece"
		raw_data = read(cachedir("ember_dataset_2018_2.tar.bz2"))
		decoded_data = transcode(Bzip2Decompressor, raw_data)
		write(cachedir("ember_dataset_2018_2.tar"), decoded_data)
		Tar.extract(cachedir("ember_dataset_2018_2.tar"), cachedir("raw"))
		for l in filter(s -> endswith(s, ".jsonl"), readdir(cachedir("raw","ember2018")))
			mv(cachedir("raw","ember2018", l), cachedir(l))
		end
		rm(cachedir("raw"), recursive=true)
	end
end

function makeschema(jsons::Vector{T}) where {T<:Dict}
	JsonGrinder.updatemaxkeys!(typemax(Int))
	schemas = Folds.map(schema, Iterators.partition(jsons, div(length(jsons), Threads.nthreads())));
	merge(schemas...)
end

function export_data(jsons, extractor, prefix)
	Threads.@threads for (i, jss) in collect(enumerate(Iterators.partition(jsons, 100)))
		ds = map(fixsample ∘ extractor, jss)
		y = map(d -> d["label"], jss);
		serialize(cachedir("jls", "$(prefix)_$(i).jls"), (reduce(catobs, ds), y))		
	end
end

function extractor_rules()
	[(e -> (keys_len = length(keys(e)); keys_len / e.updated < 0.1 && keys_len <= 1000),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	 (e -> JsonGrinder.is_intable(e),
		(e, uniontypes) -> extractscalar(Int32, e, uniontypes)),
	 (e -> JsonGrinder.is_floatable(e),
	 	(e, uniontypes) -> JsonGrinder.extractscalar(FloatType, e, uniontypes)),
	(e -> true,
		(e, uniontypes) -> JsonGrinder.extractscalar(JsonGrinder.unify_types(e), e, uniontypes)),]
end

function prepare_data()
	if !isdir(cachedir()) || (length(filter(l -> endswith(l,".jsonl"), readdir(cachedir()))) ≠ 7)
		prepare_raw_data()
	end
	trn_files = filter(s -> startswith(s, "train_"), readdir(cachedir()))
	@assert length(trn_files) == 6
	jsons = map(s -> Folds.map(l -> JSON.parse(l),readlines(cachedir(s))), trn_files);
	jsons = reduce(vcat, jsons);

	targets = Folds.map(d -> d["label"], jsons);
	jsons = jsons[targets .!= -1]
	!isdir(jlsdir()) && mkpath(jlsdir())
	if !isfile(jlsdir("extractor.jls"))
		JsonGrinder.updatemaxkeys!(typemax(Int))
		sch = makeschema(jsons);
		ks = [:label,:exports,:section,:general,:header,:avclass,:histogram,:datadirectories,:byteentropy,:appeared,:sha256,:strings,:md5]
		exs = map(k -> k => suggestextractor(sch[k], (scalar_extractors = extractor_rules(), key_as_field = 500)), ks)
		push!(exs, :imports => ExtractKeyAsField(ExtractString(), ExtractArray(ExtractString())))
		extractor = ExtractDict(Dict(exs))
		serialize(jlsdir("extractor.jls"), extractor)
		serialize(jlsdir("schema.jls"), sch)
	end

	extractor = deserialize(jlsdir("extractor.jls"))
	delete!(extractor.dict, :sha256);
	delete!(extractor.dict, :md5);
	delete!(extractor.dict, :label);
	delete!(extractor.dict, :appeared);

	####
	# export the trainig data
	####
	trn_targets = Folds.map(d -> d["label"], jsons);
	class_indexes = classindexes(trn_targets);
	mbindices = map(x -> vcat(x...), zip(map(v -> Iterators.partition(shuffle(v), 100), values(class_indexes))...))
	Folds.map(enumerate(mbindices)) do iii
		i, ii = iii
		data = map(extractor, jsons[ii]);
		ds = reduce(catobs, data)
		y = trn_targets[ii]
		serialize(jlsdir("train_$(i).jls"), (ds, y))
		nothing
	end

	####
	# export the testing data
	####
	tst_jsons = Folds.map(l -> JSON.parse(l), readlines(cachedir("test_features.jsonl")))
	tst_targets = map(d -> d["label"], tst_jsons);
	mbindices = Iterators.partition(1:length(tst_jsons), 100)
	Folds.map(collect(enumerate(mbindices))) do iii
		i, ii = iii
		data = map(extractor, tst_jsons[ii])
		ds = reduce(catobs, data)
		y = tst_targets[ii]
		serialize(jlsdir("test_$(i).jls"), (ds, y))
		nothing
	end

end

function fixsample(ds)
	p = collect(pairs(ds.data))
	p = filter(i -> i.first ∉ [:avclass, :md5, :sha256, :label, :appeared], p)
	ds = ProductNode((;p...), nothing)
	ds[:histogram].data ./= max.(sum(ds[:histogram].data, dims = 1), 1)
	ds[:byteentropy].data ./= max.(sum(ds[:byteentropy].data, dims = 1), 1)
	ds[:strings][:printabledist].data ./= max.(sum(ds[:strings][:printabledist].data, dims = 1), 1)
	ds
end

function loadsamples(files)
	dsy = Folds.map(files) do f
		ds, y = deserialize(jlsdir(f))
		dss = [fixsample(ds[i]) for i in 1:nobs(ds)]
		dss, y
	end
	dss = reduce(vcat, [i[1] for i in dsy])
	y = reduce(vcat, [i[2] for i in dsy])
	dss, y 
end

function load_cached_data()
	trnfiles = filter(s -> startswith(s, "train_") && endswith(s, ".jls"), readdir(jlsdir()))
	trn_x, trn_y = loadsamples(trnfiles);
	tstfiles = filter(s -> startswith(s, "test_") && endswith(s, ".jls"), readdir(jlsdir()))
	tst_x, tst_y = loadsamples(tstfiles);
	((trn_x, trn_y), (tst_x, tst_y))
end

