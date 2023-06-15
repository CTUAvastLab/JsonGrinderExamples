# JsonGrinderExamples

This repo contains examples of use of the [JsonGrinder.jl](https://github.com/CTUAvastLab/JsonGrinder.jl) library from *JsonGrinder.jl: automated differentiable neural architecture for embedding arbitrary JSON data, Mandlík, Račinský, Lisý, and Tomáš Pevný, 2022*.

Julia v1.7.2 was used in all experiments. Each directory contains the `Project.toml` and `Manifest.toml` for reproducibility. To make sure that all dependencies have same versions, run
```
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

in the directory.

To reproduce the results, it is sufficient to run in appropriate directory. For baseline results, run
```
julia --project=. baseline.jl
```
for tuned results, run
```
julia --project=. tuned.jl
```

## Mutagenesis
Mutagenesis is a small dataset from biology, which describes molecules trialed for mutagenicity on
*Salmonella typhimurium*. It is very small and contains only 100 training samples.

## Ember
Ember is a problem from a computer security, where the goal is to classify samples to malware and clean. Data are available at [https://github.com/endgameinc/ember](https://github.com/endgameinc/ember). The problem is large, therefore the scripts are adapted to take advantage of multi-threadding  in the data preparation phase (hence run julia with `-t` option with correct number of threads). Because of this, scripts depart a bit from the usual examples. The tuned versions uses a local copy of no longer developed package [GradientBoost.jl](https://github.com/svs14/GradientBoost.jl)

**Warning: The ember dataset is large. By default, it will be downloaded to a temporary directory in `/tmp`. Make sure you have at least 10Gb of free space.**

Execute `julia -t auto --project=. -e 'include("prepare_data.jl"); prepare_data()'` to download the data and cache the result of extraction.
