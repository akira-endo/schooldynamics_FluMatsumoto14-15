# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Julia 1.2.0
#     language: julia
#     name: julia-1.2
# ---

# ## Load data

# +
using Plots, InvertedIndices, StatsPlots, CSV, LinearAlgebra, StatsBase, Distributions, DataFrames
using Mamba:Mamba, quantile, mean, mm

chain1 = CSV.read("output/Matsumoto_MCMCsamples.csv",DataFrame)
nclassmat=CSV.read("../data/NumberOfClasses.csv",DataFrame)[:,2:7]|>Matrix
nstudentsmat=CSV.read("../data/NumberOfClasses.csv",DataFrame)[:,[8:13...]]|>Matrix;

# -

gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,grid=false, tick_direction=:out,)

function ribbons(x,mode=[:midpoints,:ribbons])
    hdiffs=diff([x zeros(size(x,1))],dims=2)./2
    getindex((midpoints=x.+hdiffs,ribbons=hdiffs),mode)
end
###### infection hazard
using RCall
@rimport distr

## COVID-19
incperiod=distr.Lnorm(1.434065,0.6612) # He et al.
infprof=distr.Norm(0.53,sqrt(7)) # Aschroft et al.
gtime=incperiod+infprof
ylim=(0,0.151);xlim=(1,20)
## COVID-19 (updated, based on Ferretti et al.)
#=incperiod=distr.UnivarMixingDistribution(
    distr.Lnorm(meanlog = 1.621, sdlog = 0.418),
    distr.Lnorm(meanlog = 1.425, sdlog = 0.669),
    distr.Lnorm(meanlog = 1.57, sdlog = 0.65),
    distr.Lnorm(meanlog = 1.53, sdlog = 0.464),
    distr.Lnorm(meanlog = 1.611, sdlog = 0.472),
    distr.Lnorm(meanlog = 1.54, sdlog = 0.47),
    distr.Lnorm(meanlog = 1.857, sdlog = 0.547)) # Ferretti et al.
gtime=distr.Weibull(3.2862,6.1244) # Ferretti et al.
=#
ylim=(0,0.151);xlim=(1,20)

## Pandemic influenza
#incperiod=distr.Lnorm(0.34,0.42) # Lessler et al. 2009 (https://dx.doi.org/10.1016%2FS1473-3099(09)70069-6
#gtime=((mean,sd)->distr.Gammad(mean^2/sd^2, sd^2/mean))(1.7,1)
#ylim=(0,0.5);xlim=(1,10)
#

# plot gtime
gtimevec_plot=distr.p(gtime)((0:20))|>RCall.unsafe_vec|>diff
gtimevec_plot=gtimevec_plot./sum(gtimevec_plot)
symratio=[0 0.25 0.5 0.75 1]
hazards_symsc=gtimevec_plot.*(1 .- symratio.*(distr.p(incperiod)((1:20).-1)|>RCall.unsafe_vec))
surv=[0 0.1 0.2 0.3 0.4 0.5]
hazards_randsc=[hazard.*((1 .-surv).^(0:19)) for hazard in eachcol(hazards_symsc)]

include("infprofile_Ferretti.jl")

@show sum(hazards_symsc,dims=1)
plot_sym=plot(ribbons(hazards_symsc,:midpoints),ribbon=ribbons(hazards_symsc,:ribbons),fillalpha=0.8,linealpha=0,palette=cgrad(:blues, 5, categorical = true),color=[1,4,3,5,2]',ylim=ylim,
     guidefontsize=9,xlabel="days after infection",ylabel="infection profile",xlim=xlim,
label="transmission attributed to symptomatic: ".*string.([0,25,50,75,100]').*"%",legend=(0.4,0.95),size=(450,250))
display(plot_sym)
plot_rand=[plot(ribbons(hazards,:midpoints),ribbon=ribbons(hazards,:ribbons),fillalpha=1,linealpha=0,palette=cgrad(:reds, 16, categorical = true),color=[1,14,8,11,5,2]|>permutedims,ylim=ylim,
     guidefontsize=8,xlabel="days after infection",ylabel="infection profile",xlim=xlim,
label="test rate: ".*string.((0:5:25)').*"%",size=(450,300)) for hazards in hazards_randsc[1:end-1]]
plot(plot_rand...,legend=[:none (0.65,1) fill(:none,1,2)],legendfontsize=7,title="transm. attr. symptomatic: ".*string.([0 25 50 75]).*"%")|>display
plot(reduce(vcat,sum.(hazards_randsc,dims=1))',ylim=(0,1),color=palette(:viridis,6)[5:-1:1]|>permutedims,size=(450,130),ylabel="relative Rs",xlabel="effective daily testing rate",guidefontsize=8,bottom_margin=2mm,legend=:none)
xticks!(1:6,string.((0:5:25)).*"%")|>display
reduce(vcat,sum.(hazards_randsc,dims=1))

## Simulation results
using JLD2
@load "simulation_weekend_covid.jld2"
#@load "covidsimulation_overdispersion02.jld2" scenarios scenarios_dayoff scenarios_classisolation
#@load "covidsimulation_flu.jld2" scenarios scenarios_dayoff scenarios_classisolation
scenarios=first.(scenarios_strategies_notest);
scenarios_dayoff1=first.(scenarios_strategies_dayoff1);
scenarios_dayoff2=first.(scenarios_strategies_dayoff2);

# Class interventions
Base.:+(x::AbstractArray,y::Number)=x.+y
Base.:-(x::AbstractArray,y::Number)=x.-y
outbreakmaps=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario))./[40,20,20,40] for scenario in scenarios]
heatmaps=heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
vline!.(heatmaps,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
plot(heatmaps[1:12]...,layout=(3,4),size=(800,450),xlim=(0,200),
left_margin=2mm,xlabel=[fill("",1,8) fill("day",1,4)],ylabel=repeat(["grade/class (sorted)" "" "" ""],1,3),guidefontsize=7,tickfontsize=7, title=["baseline" "split class" "staggered (within class)"  "staggered (between class)" fill("",1,8)], titlefontsize=8)#,colorbar=[fill(:none,1,3) (0.5,1)])

# Screening
function transposelayout(panels,layout) reshape(panels,layout...)|>permutedims|>vec end
plot(transposelayout(heatmaps[1:4:33],(3,3))...,layout=(3,3),size=(600,450),left_margin=2mm, xlim=(0,200),xlabel=[fill("",1,6) fill("day",1,6)],ylabel=repeat(["grade/class (sorted)" "" ""],1,3),guidefontsize=7,tickfontsize=7, title=["baseline (no screening)" "symptom screening (50%)" "symptom + 10% regular test" fill("",1,6)], titlefontsize=8)#,colorbar=[fill(:none,1,2) (0.5,1)])

# Setting regular day off
outbreakmaps_dayoff=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario))./[40,20,20,40] for scenario in [scenarios_dayoff1[:,1:1] scenarios_dayoff2[:,2:-1:1]]]
heatmaps_dayoff=heatmap.(reduce(vcat,vec(outbreakmaps_dayoff)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
vline!.(heatmaps_dayoff,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps_dayoff,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_dayoff,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
plot(transposelayout(heatmaps_dayoff[1:4:33],(3,3))...,xlim=(0,200),layout=(3,3),size=(600,450),left_margin=2mm,  xlabel=[fill("",1,6) fill("day",1,3)],ylabel=repeat(["grade/class (sorted)" "" ""],1,3),guidefontsize=7,tickfontsize=7, title=["1 day on : 1 day off" "symptom + 2 day on : 1 day off" "2 day on : 1 day off" fill("",1,6)], titlefontsize=8)#,colorbar=[fill(:none,1,2) (0.5,1)])

@load "simulation_weekend_covid_classisolation.jld2"
scenarios_CI=first.(scenarios_classisolation);
# Class isolation
outbreakmaps_classisolation=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario))./[40,20,20,40] for scenario in scenarios_CI]
heatmaps_classisolation=heatmap.(reduce(vcat,vec(outbreakmaps_classisolation)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
vline!.(heatmaps_classisolation,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps_classisolation,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,12),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_classisolation,repeat(Ref(1:5).*[2,4,2,1].+0.5,12),linetype=:hline,color=:black,linealpha=0.2)
plot(transposelayout(heatmaps_classisolation[1:4:45],(3,4))...,xlim=(0,100),layout=(3,4),size=(800,450),left_margin=2mm,  xlabel=[fill("",1,8) fill("day",1,4)],ylabel=repeat(["grade/class (sorted)" "" "" ""],1,4),guidefontsize=7,tickfontsize=7, title=["50% outside-class interaction" "50% outside/120% inside-class" "10% outside-class interaction" "10% outside/140% inside-class" fill("",1,8)], titlefontsize=8)#,colorbar=[fill(:none,1,2) (0.5,1)])

# cohorting + class structure
heatmaps=heatmap.(reduce(vcat,vec(outbreakmaps_classisolation)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)[1:36]
vline!.(heatmaps,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
plot(heatmaps[1:12]...,layout=(3,4),size=(800,450),xlim=(0,100),
left_margin=2mm,xlabel=[fill("",1,8) fill("day",1,4)],ylabel=repeat(["grade/class (sorted)" "" "" ""],1,3),guidefontsize=7,tickfontsize=7, title=["baseline" "split class" "staggered (within class)"  "staggered (between class)" fill("",1,8)], titlefontsize=8)#,colorbar=[fill(:none,1,3) (0.5,1)])

# get final size distribution
function outbreakf(outbreaks::AbstractVector{<:AbstractMatrix}, f) f(outbreaks) end
function outbreakf(outbreaks,f) outbreakf.(outbreaks,f) end
function getquantile(x,q) 0==length(x) ? NaN .+ zero(q) : quantile(x,q) end
# Prob and size of major outbreaks
function finalsizes(outbreaks)
    obsum=outbreakf(outbreaks,sum)
    obfirst=outbreakf(obsum,x->first.(x))
    withinclass=reduce.(hcat,obfirst).|>eachrow.|>collect
    oball=outbreakf(obsum,x->sum.(x))
    allschool=reduce.(hcat,oball).|>eachrow.|>collect
    (withinclass=withinclass,allschool=allschool)
end
riskoverthreshold(outbreaks,threshold)=((x,threshold)->mean(x.≥threshold)).(outbreaks,threshold)
quantileoverthreshold(outbreaks,threshold)=((outbreak,threshold)->getquantile(filter(x->x≥threshold,outbreak),[0.025,0.25,0.5,0.75,0.975])).(outbreaks,threshold)
finsizes=(baseline=finalsizes(scenarios),dayoff=finalsizes(scenarios_dayoff),classisolation=finalsizes(scenarios_CI));

@rimport base as r
function vconvolve(v,w)
    [dot(v[1:x],reverse(w[1:x])) for x in 1:length(v)]
end
multiconvolve(v,times)=reduce(vconvolve,(v for t in 1:times))
hcatv(x::AbstractArray{<:AbstractArray},y)=hcatv.(x,y)
hcatv(x::AbstractArray{<:Number},y)=hcat(x,y)
function multiintroductions(dist,times,threshold)
    (1 .-(multiconvolve(dist,times)|>cumsum))[floor(Int,threshold)]
end
multiintroductions(v::AbstractArray{<:AbstractArray},times,threshold)=multiintroductions.(v,Ref(times),Ref(threshold))
 
function outbreakrisk(finsizes=(fs,fs_d,fs_c),threshold=20,inits=1:10)
    (begin
    findist=(x->(rcopy.(r.table.(vcat.(x,Ref(1:480)))).-1)./length.(x)).(f.allschool)
    multipleintros=(x->first.(x)).(multiintroductions.(Ref(findist),inits,threshold))
    multiv=reduce((x,y)->vcat.(x,y),multipleintros)
    (x->hcat(x...)).(eachcol(multiv))
            end for f in finsizes)|>Tuple
end
outbreakrisks=outbreakrisk(finsizes,10)
outbreakrisks_30=outbreakrisk(finsizes,30);

sumgtimes=sum.([hazards_symsc[:,1],hazards_symsc[:,3], hazards_randsc[3][:,3]]|>permutedims.|>(x->x.*repeat([4,3,3,3,3,4,5],3)[1:20]./5))
sumgtimes_dayoff=sum.([hazards_symsc[:,1].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],#=hazards_symsc[:,3].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],=#hazards_symsc[:,3].*repeat([1,1,0],size(hazards_symsc,1))[1:size(hazards_symsc,1)]]|>permutedims)
Rvalues=(sumgtimes).*[2,1.5,0.8]#.*5/7
Rvalues_dayoff=[2,1.5,0.8] .*[sum(repeat([0,2,1,1,2,0,3],3)[1:20]./3 .*hazards_symsc[:,1]) sum(repeat([2,1,3,3,1,2,4],3)[1:20]./4 .*hazards_symsc[:,3])]#sumgtimes_dayoff#.*[2,1.5,0.8];

plot(plot.([[vcat(outbreakrisks[1:2]...)] [vcat(outbreakrisks_30[1:2]...)]],line_z=vec([Rvalues Rvalues_dayoff])',color=cgrad([:lightblue,:royalblue,:purple,:firebrick,:tomato]),linestyle=[:solid #=:dash :dashdot=#],linewidth=2)...,ylim=(0,1.01),clim=(0.5,1.5),
    xlabel="initial cases",ylabel="risk of outbreak (> ".*string.([10 30]).*" secondary cases)",legend=false, size=(400,520),layout=(2,1),colorbar_title="effective reproduction number",guidefontsize=8,bottom_margin=2mm)
    
 # Undetected cases when the first case is found
function atdetection(incidence,testrate,symptom;ipdist=LogNormal(1.434065,0.6612),nsamples=10,positivitylength=20)
    allschool=(sum.(incidence)|>cumsum).-1
    #outsideclass=allschool.-(first.(incidence)|>cumsum).+1 # outside the seeded class: not ideal for class-closure analysis
    outsideclass=allschool.-sample.(cumsum(incidence),Weights.(vec.(incidence))).+1 # outside the detected class
    detectrate=.-log.(1 .-symptom.*(cdf.(ipdist,0:length(incidence)-1).-cdf.(ipdist,-1:length(incidence)-2))) # symptom screening
    detectrate[1:positivitylength].+=testrate
    cumdetectrate=[0;vconvolve(sum.(incidence),detectrate) |>cumsum]
    probs=[0;(diff(cumdetectrate).*(exp.(.-cumdetectrate[1:end-1])))[1:end-1]]
    probs[end]+=1-sum(probs)
    obsize=sample(vcat.(allschool,outsideclass),Weights(probs),nsamples)
end
function samplesatdetection(incidences,testrate,symptom;ipdist,nsamples=10,positivitylength=20)
    resultvec=atdetection.(incidences,testrate,symptom,ipdist=ipdist,nsamples=nsamples,positivitylength=positivitylength)
    results=vcat.(reduce(vcat,resultvec))
    (all=first.(results),out=last.(results))
end
ipdist=LogNormal(1.434065,0.6612) #COVID-19
#ipdist=LogNormal(0.34,0.42) #Flu
rawincs=(x->first.(x)).([scenarios[:,1] scenarios_CI[:,[1,3]]])
@time casesatdetection=[samplesatdetection.(rawincs,testrate,psymptom;ipdist=ipdist) for testrate in [0.0,0.1,0.2,0.3], psymptom in [0.5,0.25,0.1]];

tribar(x)=plot(bar.(Ref(0:100),shiftcol.(eachcol(x),1:3,3),color=[2,1,3]')...,layout=(3,1),xlim=(-1,20),ylim=(-0.05,1),size=(800,800),linealpha=0#=,title="Rs = ".*string.([1.8 1.2 0.8])=#,legend=false)
shiftcol(col,nth,outof)=begin ret=zeros(length(col),outof);ret[:,nth].=col;ret end
function plotcasesatdetection(cases,bins)
    den_all=(x->(([first(x);0:100]|>r.table.|>rcopy).-1)./length(first(x))).(values(cases))
    den_out=(x->(([last(x);0:100]|>r.table.|>rcopy).-1)./length(last(x))).(values(cases))
    
    ranges=range.(bins.+1,[bins[2:end];100],step=1)
    @views cumbin_all=getindex.(den_all,permutedims(ranges[:,:,:],(2,3,1))).|>sum#|>(x->reverse(x,dims=2))
    band_all=groupedbar.(eachslice(cumbin_all,dims=2).|>(x->reverse(x,dims=2)),bar_position=:stack,color=[:white;1:42:42*4+1]|>reverse|>permutedims,palette=:dense)
    @views cumbin_out=getindex.(den_out,permutedims(ranges[:,:,:],(2,3,1))).|>sum
    band_out=groupedbar.(eachslice(cumbin_out,dims=2).|>(x->reverse(x,dims=2)),bar_position=:stack,color=[:white;1:42:42*4+1]|>reverse|>permutedims,palette=:amp)
    
    (all=band_all,out=band_out)
end
bands_screening=getindex.(casesatdetection,:,1)
bandsplot=reduce(vcat,first.(plotcasesatdetection.(bands_screening,Ref([0,1,6,11,16,21]))));
bandsplot_out=reduce(vcat,last.(plotcasesatdetection.(bands_screening,Ref([0,1,6,11,16,21]))));

bands_classisolation=reduce(vcat,eachcol.(casesatdetection[[1,5,9]]).|>collect)
bandsplot_classisolation=reduce(vcat,(x->[first.(x)[reshape(1:6,3,2)'|>vec] last.(x)[reshape(1:6,3,2)'|>vec]]|>permutedims|>vec)(plotcasesatdetection.(bands_classisolation,Ref([0,1,6,11,16,21]))));
bandsplot_classisolation_full=reduce(vcat,(x->[first.(x)[reshape(1:9,3,3)'|>vec] last.(x)[reshape(1:9,3,3)'|>vec]]|>permutedims|>vec)(plotcasesatdetection.(bands_classisolation,Ref([0,1,6,11,16,21]))));

ylab=hcat(vcat.(#=string.([50 25 10]).*"% symptomatic \n=#"probability","","","", "","")...)|>vec|>permutedims
xlab=vcat(hcat.("","",repeat(["baseline Rs"],1,6))...)|>vec|>permutedims
title=vcat(hcat.(repeat(["overall\n " "spillover"],1,3),"","","")...)|>vec|>permutedims

pl=(([[reshape(bandsplot[[1:2;5:6;9:10]],2,3) ;bandsplot_classisolation_full[[7,9,11]]|>permutedims]|>vec [reshape(bandsplot_out[[1:2;5:6;9:10]],2,3) ;bandsplot_classisolation_full[[8,10,12]]|>permutedims]|>vec]|>permutedims)) 
xticks!.(vec(pl),Ref(1:3),[fill(fill("",3),12);fill(string.([2.0,1.5,0.8]),6)],xflip=true,xlabel=xlab, guidefontsize=8,titlefontsize=8,ylabel=ylab,title=title,size=(650,350))
plot(pl...,layout=(3,6),legend=false,bottom_margin=1.0mm,top_margin=-2.5mm,left_margin=0.5mm,right_margin=-2mm,ytick=[true fill(false,1,5)],title=title,ylab=ylab,xlab=xlab)|>display

ylab=hcat(vcat.(#=string.([50 25 10]).*"% symptomatic \n=#"probability","","","")...)|>vec|>permutedims
xlab=vcat(hcat.("","",repeat(["baseline Rs"],1,4))...)|>vec|>permutedims
title=vcat(hcat.(repeat(["overall\n " "spillover"],1,2),"","","")...)|>vec|>permutedims

pl=(([[bandsplot[[3 7 11]] ;bandsplot_classisolation_full[[13 15 17]]]|>vec [bandsplot_out[[3 7 11]] ;bandsplot_classisolation_full[[14 16 18]]]|>vec]|>permutedims)) 
xticks!.(vec(pl),Ref(1:3),[fill(fill("",3),8);fill(string.([2.0,1.5,0.8]),4)],xflip=true,xlabel=xlab, guidefontsize=8,titlefontsize=8,ylabel=ylab,title=title,size=(450,350))

plot(pl...,layout=(3,4),legend=false,bottom_margin=1.0mm,top_margin=-2.5mm,left_margin=0.5mm,right_margin=-2mm,ytick=[true fill(false,1,3)],title=title,ylab=ylab,xlab=xlab)|>display

## Class closure strategy
@load "closurestrategy_covid.jld2" scenarios_strategies_notest scenarios_strategies_10test
#@load "closurestrategy_flu.jld2" scenarios_strategies_notest scenarios_strategies_10test
#@load "closurestrategy_covid_verdispersion02.jld2" scenarios_strategies_notest scenarios_strategies_10test

outbreakmaps_notest=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_notest)]
#outbreakmaps_notest=outbreakmaps_notest[:,[1,2,4,3,5]]
heatmaps_notest=heatmap.(reduce(vcat,vec(outbreakmaps_notest)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
vline!.(heatmaps_notest,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps_notest,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_notest,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

outbreakmaps_10test=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_10test)]
heatmaps_10test=heatmap.(reduce(vcat,vec(outbreakmaps_10test)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
vline!.(heatmaps_10test,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(heatmaps_10test,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_10test,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

closuremaps_notest=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario)) for scenario in last.(scenarios_strategies_notest)]
coldmaps_notest=heatmap.(reduce(vcat,vec(closuremaps_notest)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
vline!.(coldmaps_notest,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(coldmaps_notest,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(coldmaps_notest,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

closuremaps_10test=[(x->reduce(hcat,vec.(x'))[:,filter(!x->x%7 in (6,0),1:360)]).(mean(scenario)) for scenario in last.(scenarios_strategies_10test)]
coldmaps_10test=heatmap.(reduce(vcat,vec(closuremaps_10test)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
vline!.(coldmaps_10test,Ref([5:5:360]),color=:gray,alpha=0.3)
yticks!.(coldmaps_10test,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(coldmaps_10test,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)


plot(transposelayout([heatmaps_notest[1:4:9];coldmaps_notest[1:4:9];heatmaps_10test[1:4:9];coldmaps_10test[1:4:9];heatmaps_notest[13:4:21];coldmaps_notest[13:4:21]],(3,6))...,
    layout=(3,6),size=(1200,450),xlim=(0,100),left_margin=4mm,right_margin=0mm,bottom_margin=4mm,  
xlabel=[fill("",1,12) fill("day",1,6)],ylabel=repeat(["grade/class (sorted)" fill("",1,5)],1,5),guidefontsize=8,tickfontsize=7,title=[vec(hcat(vcat.(fill("incidence",1,3),"prob of class closure")...));fill("",12)]|>permutedims, titlefontsize=8)
plot(transposelayout([heatmaps_notest[25:4:33];coldmaps_notest[25:4:33];heatmaps_notest[37:4:45];coldmaps_notest[37:4:45];heatmaps_notest[49:4:57];coldmaps_notest[49:4:57]],(3,6))...,
    layout=(3,6),size=(1200,450),left_margin=4mm,right_margin=0mm,bottom_margin=4mm, xlim=(0,100),
xlabel=[fill("",1,12) fill("day",1,6)],ylabel=repeat(["grade/class (sorted)" fill("",1,5)],1,5),guidefontsize=8,tickfontsize=7,title=[vec(hcat(vcat.(fill("incidence",1,3),"closure")...));fill("",12)]|>permutedims, titlefontsize=8)

obsizes_notest=[first(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_notest]
closuresizes_notest=[last(scenario).|>first.|>(x->[max.(y,0) for y in x]).|>sum.|>sum for scenario in scenarios_strategies_notest]
obsizes_10test=[first(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_10test]
closuresizes_10test=[last(scenario).|>first.|>(x->[max.(y,0) for y in x]).|>sum.|>sum for scenario in scenarios_strategies_10test]
obsizes_baseline=[first.(scenario).|>sum.|>sum for scenario in scenarios]
obsizes_classisolation=[first.(scenario).|>sum.|>sum for scenario in scenarios_CI]
ste(v)=std(v)/sqrt(length(v))
multiplot(x,yerror;kw...)=bar(x,yerror=yerror,kw...)
obsizes_closure=[obsizes_notest[:,1] obsizes_10test[:,1] obsizes_notest[:,2]]
closuresizes=[closuresizes_notest[:,1] closuresizes_10test[:,1] closuresizes_notest[:,2]]
obsizes=[obsizes_baseline[:,[1,3]] obsizes_classisolation[:,1]]
#((b,c)->bar.(vcat.(hcat.(eachrow(mean.(b)),eachrow(mean.(c))),hcat.(eachrow(quantile.(b,0.95)),eachrow(quantile.(c,0.95))))))(obsizes,obsizes_closure).|>display
function closureeffects(size_baseline, size_closure, closuredays)
    means=[mean.(x) for x in (size_baseline, size_closure, closuredays)]./[480,480,1]
    upp95s=[quantile.(x,0.975) for x in (size_baseline, size_closure, closuredays)]./[480,480,1]
    maxs=[quantile.(x,0.995) for x in (size_baseline, size_closure, closuredays)]./[480,480,1]
    obbars=reduce.(hcat,vcat.(upp95s[1],means[1],upp95s[2],means[2])|>eachrow)
    @show clbars=reduce.(hcat,vcat.(upp95s[3],means[3])|>eachrow)
    obmax=reduce.(hcat,vcat.(maxs[1],NaN,maxs[2],NaN)|>eachrow)
    @show clmax=reduce.(hcat,vcat.(maxs[3],NaN)|>eachrow)

    [((obvalue,maxvalue)->bar(obvalue|>permutedims,color=repeat([600 8],inner=(1,2)),palette=:matter,linealpha=[1 0.5],label=["isolation only" :none "class closure" :none],
    xticks=(1:3,["naive","regular\ntests","class\ndistancing"]),xrotation=0, yerror=(Tuple.(vcat.(Ref([0.0]),maxvalue.-obvalue)))|>permutedims,ms=0)).(obbars,obmax);
    ((clvalue,maxvalue)->bar(clvalue|>permutedims,color=:lightblue,linealpha=[1 0.5],label=:none#=["days closed" :none]=#,
    xticks=(1:3,["naive","regular\ntests","class\ndistancing"]),xrotation=0,yerror=(Tuple.(vcat.(Ref([0.0]),maxvalue.-clvalue)))|>permutedims,ms=0)).(clbars,clmax)]
end
plot(closureeffects(obsizes,obsizes_closure,closuresizes)...,layout=(2,3), legend=[:none :none (0.5,1)],ylim=[(0,0.65) (0,0.25) (0,0.05) (0,50) (0,50) (0,50)],size=(800,600),title=["Rs = ".*string.([2.0 1.5 0.8]) fill("",1,3)],titlefont=10,
ylabel=repeat(["proportion infected" "total days of class closure"],inner=(1,3)),left_margin=2mm,tickfontsize=10)

struct Nth
    num::Int
end
(nth::Nth)(x)=getindex(x,nth.num)
obsizes_notest=[[first(scenario).|>Ref(Nth(n)).|>sum.|>sum for scenario in scenarios_strategies_notest] for n in 1:4]
closuresizes_notest=[[last(scenario).|>Ref(Nth(n)).|>(x->[max.(y,0) for y in x]).|>sum.|>sum for scenario in scenarios_strategies_notest] for n in 1:4]
obsizes_10test=[[first(scenario).|>Ref(Nth(n)).|>sum.|>sum for scenario in scenarios_strategies_10test] for n in 1:4]
closuresizes_10test=[[last(scenario).|>Ref(Nth(n)).|>(x->[max.(y,0) for y in x]).|>sum.|>sum for scenario in scenarios_strategies_10test] for n in 1:4]
obsizes_baseline=[[Nth(n).(scenario).|>sum.|>sum for scenario in scenarios] for n in 1:4]
obsizes_classisolation=[[Nth(n).(scenario).|>sum.|>sum for scenario in scenarios_CI] for n in 1:4]

obsizes_closure=[[obsizes_notest[n][:,1] obsizes_10test[n][:,1] obsizes_notest[n][:,2]] for n in 1:4]
closuresizes=[[closuresizes_notest[n][:,1] closuresizes_10test[n][:,1] closuresizes_notest[n][:,2]] for n in 1:4]
obsizes=[[obsizes_baseline[n][:,[1,3]] obsizes_classisolation[n][:,1]] for n in 1:4]
#((b,c)->bar.(vcat.(hcat.(eachrow(mean.(b)),eachrow(mean.(c))),hcat.(eachrow(quantile.(b,0.95)),eachrow(quantile.(c,0.95))))))(obsizes,obsizes_closure).|>display
for n in 1:4
plot(closureeffects(obsizes[n],obsizes_closure[n],closuresizes[n])[1:3]...,layout=(3,1), legend=[:none :none (0.5,1)],ylim=[(0,1) (0,1) (0,0.1) (0,50) (0,50) (0,50)],size=(300,900),#title=["Rs = ".*string.([1.8 1.2 0.8]) fill("",1,3)],titlefont=10,
ylabel=repeat(["proportion infected" "total days of class closure"],inner=(1,3)),left_margin=10mm,tickfontsize=10) |>display
end
for n in 1:4
plot(closureeffects(obsizes[n],obsizes_closure[n],closuresizes[n])[4:6]...,layout=(3,1), legend=[:none :none (0.5,1)],ylim=[(0,100) (0,100) (0,100)],size=(300,900),#title=["Rs = ".*string.([1.8 1.2 0.8]) fill("",1,3)],titlefont=10,
ylabel=repeat(["total days of class closure"],inner=(1,3)),left_margin=10mm,tickfontsize=10) |>display
end
                                                        
# final sizes
obsizes_full=[[obsizes_baseline[n][:,1:3] obsizes_classisolation[n][:,1:4]] for n in 1:4]
obsizebars=closureeffects([reduce(hcat,getindex.(obsizes_full,:,1)) obsizes_full[1][:,2:end]],fill([0],3,10),fill([0],3,10),schoolsize=[[480,480,240,240];fill(480,6)]',
barcolors=[fill(2,4);fill(13,2);fill(16,4)],barpalette=[:auto],xlabels=["baseline","split\nclasses", "staggered\n(within)","staggered\n(between)","50%\nsymptom\nscreening","10%\nregular\ntests","class\ncohorting\n50%/100%", "class\ncohorting\n50%/120%","class\ncohorting\n10%/100%", "class\ncohorting\n10%/140%"])
plot(obsizebars[1:3]...,layout=(3,1), legend=[:none :none :none],ylim=[(0,1) (0,1) (0,0.1) (0,50) (0,50) (0,50)],size=(1000,700),bottom_margin=3mm,
ylabel=repeat(["proportion infected" "total days of class closure"],inner=(1,3)),left_margin=10mm,tickfontsize=10) |>display

# peak week sizes
sumbyweek(vecofmats)=[0;cumsum(sum.(vecofmats))[7:7:end]]|>diff|>maximum # takes the maximum week sum
peaksizes_baseline=[[Nth(n).(scenario).|>sumbyweek for scenario in scenarios] for n in 1:4]
peaksizes_classisolation=[[Nth(n).(scenario).|>sumbyweek for scenario in scenarios_CI] for n in 1:4]
peaksizes_full=[[peaksizes_baseline[n][:,1:3] peaksizes_classisolation[n][:,1:4]] for n in 1:4]
peaksizebars=closureeffects([reduce(hcat,getindex.(peaksizes_full,:,1)) peaksizes_full[1][:,2:end]],fill([0],3,10),fill([0],3,10),schoolsize=[[480,480,240,240];fill(480,6)]',
barcolors=[fill(2,4);fill(13,2);fill(16,4)],barpalette=[:auto],xlabels=["baseline","split\nclasses", "staggered\n(within)","staggered\n(between)","50%\nsymptom\nscreening","10%\nregular\ntests","class\ncohorting\n50%/100%", "class\ncohorting\n50%/120%","class\ncohorting\n10%/100%", "class\ncohorting\n10%/140%"])
plot(peaksizebars[1:3]...,layout=(3,1), legend=[:none :none :none],ylim=[(0,0.35) (0,0.25) (0,0.05) (0,50) (0,50) (0,50)],size=(1000,700),bottom_margin=3mm,
ylabel=repeat(["peak weekly attack rate" "total days of class closure"],inner=(1,3)),left_margin=10mm,tickfontsize=10) |>display
