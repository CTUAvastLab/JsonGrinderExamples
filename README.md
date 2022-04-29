# JsonGrinderExamples.jl

This repo contains examples of use of JsonGrinder.jl library use in the *JsonGrinder.jl: automated differentiable neural architecture for embedding arbitrary JSON data, Mandlík, Račinský, Lisý, and Tomáš Pevný, 2022*.

Each directory contains the `Project.toml` and `Manifest.toml` for improved reproducibility. To reproduce the results, it is sufficient to run in appropriate directory. For baseline results, run
```
julia --project=. baseline.jl
```
for tuned results, run
```
julia --project=. tuned.jl
```

## Ember
Ember is a problem from a computer security, where the goal is to classify samples to malware and clean. Data are available at (https://github.com/endgameinc/ember)[https://github.com/endgameinc/ember]. The problem is large, therefore the scripts are adapted to take advantage of multi-threadding  in the data preparation phase (hence run julia with `-t` option with correct number of threads). Because of this, scripts depart a bit from the usual examples. 

**Warning: The ember dataset is large. By default, it will be downloaded to a temporary directory in `/tmp`. Make sure you have at least 10Gb of free space.**
