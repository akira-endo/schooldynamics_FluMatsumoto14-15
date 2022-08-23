# Within and between classroom transmission patterns of seasonal influenza and pandemic management strategies at schools
Source code accompanying Endo et al. "Within and between classroom transmission patterns of seasonal influenza and pandemic management strategies at schools".

## Main files
* src/inference.jl:
Script for inference of transmission patterns of seasonal influenza in Matsumoto city
* src/simulation.jl:
Script for school outbreak simulation
* src/outputs_inference.jl
Script for plotting inference results
* src/clusteringanalysis.jl
Analysis of temporal clustering of students' onset dates
* src/householdsimulation.jl
Model validation by simulated of household-school outbreaks
* src/hetLK.jl
Function definitions for household likelihood
* src/SchoolOutbreak.jl
Function and structure definitions for inference and simulation
* src/schooloutbreaks_Texas.jl
Comparison between empirical (Texas, US) and simulated school outbreak sizes
* data/anonymizedstudents.jld2
JLD2 data file containing anonymized data
* output/Matsumoto_MCMCsamples.csv
Posterior MCMC samples generated by inference.jl

## Licence

[MIT](https://github.com/akira-endo/Intro-PMCMC/blob/master/LICENSE)

## Dependencies
* Julia ver. 1.5.2
* CSV.jl ver. 0.8.3
* DataFrames.jl ver. 0.22.5
* JLD2.jl ver. 0.2.4
* Distributions.jl ver. 0.23.2
* Mamba.jl ver. 0.12.2
* Optim.jl ver. 1.2.3
* StatsPlots.jl ver. 0.14.19
* InvertedIndices.jl ver. 1.0.0
* DataStructures.jl ver.0.18.9
* StatsBase.jl ver. 0.33.2
* StatsFuns.jl ver. 0.9.6
* Plots.jl ver. 1.10.3
* IJulia.jl ver. 1.23.1
* RCall.jl ver. 0.13.10
* R ver. 4.1.0
* {IRkernel} ver. 1.2
* {distr} ver. 2.8.0
* {royston} ver. 1.2
* Python ver. 3.7.4
* Jupyter Notebook ver. 4.5.0

## Authors

[Akira Endo](https://github.com/akira-endo), 
Mitsuo Uchida,
Naoki Hayashi,
[Yang Liu](https://github.com/yangclaraliu),
[Katie Atkins](https://github.com/katiito),
[Adam Kucharski](https://github.com/adamkucharski),
[Sebastian Funk](https://github.com/sbfnk)
