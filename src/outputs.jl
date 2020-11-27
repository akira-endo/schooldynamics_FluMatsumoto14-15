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

chain1 = read("Matsumoto_chain.jls", Mamba.ModelChains)
nclassmat=CSV.read("../data/NumberOfClasses.csv",DataFrame)[:,2:7]|>Matrix
nstudentsmat=CSV.read("../data/NumberOfClasses.csv",DataFrame)[:,[8:13...]]|>Matrix;

meansuscovar=[-0.010262748329213619, 0.4767005401446489, 0.5135036162226495, 0.8005126796667582]
meaninfcovar=[-0.010262748329213619, 0.4767005401446489, 0.5135036162226495, 0.8005126796667582, 0.013732491073880802]
# -

# ## Class/grade distributuion

gr(fontfamily="Helvetica")
using Random;Random.seed!(2020)
pl=scatter(ceil.(nclassmat).+(rand(29,6).-0.5)./4,nstudentsmat./ceil.(nclassmat), grid=false,
    color=randperm(29),markersize=4,markerstrokewidth=0,legend=false,tick_direction=:out,
    size=(400,300),xguidefontsize=8,yguidefontsize=8)
xlabel!("number of classes per grade")
ylabel!("class size")
plot!(pl,[10.0],linetype=:hline,color=:gray,linestyle=:dash)

function posterior(chain, parameter::Symbol)
    if parameter==:β0 parameter=:βs;fn=β0
    elseif parameter==:rcom0 parameter=:rcom;fn=rcom0
    elseif parameter in (:suscoef,:infcoef) fn=(slice->slice[:,parameter,:].value.|>exp|>vec)
        else fn = (slice->slice[:,parameter,:].value|>vec) end
    v=[fn(chain[x,:,:]) for x in chain.range]
    names=[chain[:,parameter,:].names[x] for x in 1:size(first(v),1)]
    (names=names ,samples=[getindex.(v,x) for x in 1:size(first(v),1)])
end
function β0(slice)
    vec(slice[:,:βs,:].value) end
function rcom0(slice)
    vec(slice[:,:rcom,:].value) end
bclog(x)=log.(x)
function R_nm(chain,nclasses,sizerange, meansuscovar=0,meaninfcovar=0)
    βs=reduce(hcat,posterior(chain,:β0).samples)
    βsm=βs[:,1:3];βsm[:,3].=vec(mean(βs[:,3:4],dims=2))
    βsm.*=exp.(sum(bclog.(posterior(chain,:suscoef).samples).*meansuscovar)+sum(bclog.(posterior(chain,:infcoef).samples).*meaninfcovar))
    if nclasses==1 βsm[:,2].=0 end
    γs=reduce(hcat,posterior(chain,:γs).samples)
    δs=reduce(hcat,posterior(chain,:δs).samples)
    [quantile.((βsm./(classsize/30).^γs./(nclasses./[3,3,3]').^δs).*[5*nclasses*classsize (nclasses-1)*classsize classsize] |>eachcol,Ref([0.5,0.025,0.975])) for classsize in sizerange]
end
function R_nm(parm::Vector{<:AbstractFloat},nclasses,sizerange,meansuscovar=0,meaninfcovar=0)
    βs=parm[1:4]
    βsm=βs[1:3];βsm[3]=mean(βs[3:4])
    suscoef=parm[12:15];infcoef=parm[16:20]
    βsm.*=exp.(sum(log.(suscoef).*meansuscovar)+sum(log.(infcoef).*meaninfcovar))
    if nclasses==1 βsm[2]=0 end
    γs=parm[5:7]
    δs=parm[8:10]
    [(βsm./(classsize/30).^γs./(nclasses./[3,3,3]).^δs).*[5*nclasses*classsize,(nclasses-1)*classsize,classsize] for classsize in sizerange]
end
function R_reduction(chain,nmbefore,nmafter,coefafter=ones(3),coefbefore=ones(3))
    βs=reduce(hcat,posterior(chain,:β0).samples)
    βsm=βs[:,1:3];βsm[:,3].=vec(mean(βs[:,3:4],dims=2))
    if nmbefore[2]==1 coefbefore[2]=0 end
    if nmafter[2]==1 coefafter[2]=0 end
    γs=reduce(hcat,posterior(chain,:γs).samples)
    δs=reduce(hcat,posterior(chain,:δs).samples)
    popbefore=[5*prod(nmbefore) prod(nmbefore)-nmbefore[1] nmbefore[1]]
    popafter=[5*prod(nmafter) prod(nmafter)-nmafter[1] nmafter[1]]
    quantile(sum(popafter.*coefafter'.*βsm./(nmafter[1]/30).^γs./(nmafter[2]./[3,3,3]').^δs,dims=2)./sum((popbefore.*coefbefore'.*βsm./(nmbefore[1]/30).^γs./(nmbefore[2]./[3,3,3]').^δs),dims=2) |>vec, [0.5,0.025,0.975])
end

# ## Posteriors

@time posteriors=posterior.(Ref(chain1),[:β0,:rcom0,:γs,:δs,:suscoef,:infcoef])
posteriors=reduce(vcat,posteriors)
reduce(vcat,(x->[x.names quantile.(x.samples,Ref([0.5,0.025,0.975]))]).(posteriors))

# # School reproduction number

@time Rnms=(R_nm(chain1,nclasses,1:40,meansuscovar,meaninfcovar) for nclasses in 1:6)
medRnms=[reduce(hcat,(x->first.(x)).(Rnm))' for Rnm in Rnms];
function split5(medRnm)
    [fill(medRnm[1],5)./5;(medRnm)[2:3]]
end
function medR(medRnms,n,m)
    (getindex.(medRnms,n,:)[m])|>transpose
end

gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,grid=false, tick_direction=:out,)
bars=groupedbar.(Ref(20:40),[reduce(vcat,medR.(Ref(medRnms),20:40,m)) for m in 1:6], bar_position = :stack, bar_width=1, 
    ylim=(0,1),color=reverse([1 :skyblue 9],dims=2),linealpha=0.5)
xlabel!.(bars,"class size")
ylabel!.(bars,"school reproduction number")
plot(bars...,layout=(2,3),size=(800,400),xguidefontsize=8,yguidefontsize=8,
    label=["classmate" "grademate" "schoolmate" fill("",1,4)],legend=[:none#=(0.75,1.01)=# fill(:none,1,5)],legendfontsize=6,
    title= "number of classes per grade: " .* string.((1:6)') ,format=:svg)

barsbyclass=groupedbar.(Ref(1:5),[reduce(vcat,medR.(Ref(medRnms),n,1:5)) for n in [20,30,40]], bar_position = :stack, bar_width=0.9, 
    ylim=(0,1),color=reverse([1 :skyblue 9],dims=2),linealpha=0.8)
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"school reproduction number")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false, 
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.09)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)

Rreductions=reduce(hcat,R_reduction.(Ref(chain1),Ref((40,2)),repeat([(40,2),(20,4),(20,2),(40,1)],3),repeat([ones(3),[0.5,0.5,1],[0.1,0.1,1]],inner=4)))
colors=[:black,:royalblue,3]

plot([Rreductions[1,:] fill(NaN,12,2)],
line=0,marker=([:circle#=,:pent,:hex,:oct=#],6),markerstrokewidth=0,ylim=(0,2),color=[repeat(colors,inner=4) repeat(colors[2:3]|>permutedims,12)], bottom_margin=10mm,
xticks=(1:12,repeat(["no change","split class","staggered attendance\n(within class)", "staggered attendance\n(between class)"],3)),xrotation=60,
label=string.([100 50 10]).*"% outside-class interaction", legend=(0.67,1))
ylabel!("relative change in Rs",yguidefontsize=8)
plot!([1.0],linetype=:hline,color=:gray,linestyle=:dash,label="")
ohlc!(OHLC[Tuple([NaN;Rreductions[[3,2],c];NaN]) for c in 1:size(Rreductions,2)],color=repeat(colors,inner=9*4),width=2,label="")

# ## Sensitivity analysis

include("SchoolOutbreak.jl")

# +
# Sensitivity analysis
si35=[0.0004182791889357821, 0.0013997166772905499, 0.012764913562876365, 0.014341352452209204, 1.7603861643667735, 0.22772679501257262, 1.1009663035454162, 1.087297661793457, 0.5236847953683836, 0.16153548562287995, 0.02502274220887011, 1.011524807365527, 0.8878481727992324, 0.7669077050250066, 1.566782110990794, 0.8911392108241846, 1.1082451847239987, 0.537808383165279, 1.2921332907020806, 0.19157989847800885]
hh0=[0.0004424034113794797, 0.0012950950961191253, 0.012821486344136287, 0.014398924023715658, 1.5987836717952695, 0.7237036279875796, 1.0059064870791439, 0.9083047813405848, 0.09760648305915211, 0.1895139293536357, 0.03855273309171171, 1.1680064422642666, 0.8148397957824194, 0.7189616866946185, 1.9218172138911396, 0.9045824995519222, 0.9471839859475095, 0.641211364708997, 1.2443357779554214, 0.37156672370843424]
hhwindow7=[0.00027220195001409177, 0.001174211469233752, 0.014394350613973568, 0.01707277689303326, 1.3525151933446318, 1.0173208669060467, 1.1062492245896793, 1.2837865108249, -0.12873981647695484, 0.17090440257359354, 0.016630011639652525, 1.2937438240585482, 0.8895805676385737, 0.7536937247600156, 1.548041477840481, 0.6753539911840748, 0.9977350770406842, 0.6646662098050453, 1.2639029376500923, 0.19280811713618695]
nocov=[0.0003523310381098889, 0.0014266145618400457, 0.016631050116624295, 0.019778772046246423, 0.7757036894078635, 1.331436865728501, 1.1366610122995209, 1.2662053209646682, -0.07627558642533991, 0.21787785967271264,0.01731264010447371,1,1,1,1,1,1,1,1,1]

medRnms=(reduce(hcat,R_nm(si35,nclasses,1:40,meansuscovar,meaninfcovar))' for nclasses in 1:6)

barsbyclass=groupedbar.(Ref(1:5),[reduce(vcat,medR.(Ref(medRnms),n,1:5)) for n in [20,30,40]], bar_position = :stack, bar_width=0.9, 
    ylim=(0,1),color=reverse([1 :skyblue 9],dims=2))
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"school reproduction number")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false, 
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.07)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)
# -

# # COVID-19 simulation
# ## infection profile

# +
###### infection hazard
using RCall
@rimport distr

## COVID-19
incperiod=distr.Lnorm(1.434065,0.6612) # He et al.
infprof=distr.Norm(0.53,sqrt(7)) # Aschroft et al.
gtime=incperiod+infprof
ylim=(0,0.151);xlim=(1,20)
## Pandemic influenza
#incperiod=distr.Lnorm(0.34,0.42) # Lessler et al. 2009 (https://dx.doi.org/10.1016%2FS1473-3099(09)70069-6
#gtime=((mean,sd)->distr.Gammad(mean^2/sd^2, sd^2/mean))(1.7,1)
#ylim=(0,0.5);xlim=(1,10)
##

# plot gtime
gtimevec_plot=distr.p(gtime)((0:20))|>RCall.unsafe_vec|>diff
gtimevec_plot=gtimevec_plot./sum(gtimevec_plot)
symratio=[0 0.25 0.5 0.75 1]
hazards_symsc=gtimevec_plot.*(1 .- symratio.*(distr.p(incperiod)((1:20).-1)|>RCall.unsafe_vec))
@show sum(hazards_symsc,dims=1)
plot_sym=plot(ribbons(hazards_symsc,:midpoints),ribbon=ribbons(hazards_symsc,:ribbons),fillalpha=0.8,linealpha=0,palette=:blues,color=[1,4,3,5,2]',ylim=ylim,
     guidefontsize=9,xlabel="days after infection",ylabel="infection profile",xlim=xlim,
label="proportion symptomatic: ".*string.([0,25,50,75,100]').*"%",legend=(0.55,0.9),size=(450,250))
display(plot_sym)
sens=[0.4,0.6,0.8,0.9] # Czumbei et al 10.3389/fmed.2020.00465
surv=[0 0.1 0.2 0.3 0.4 0.5]
hazards_randsc=[hazard.*((1 .-surv).^(0:19)) for hazard in eachcol(hazards_symsc)][1:end-1]
plot_rand=[plot(ribbons(hazards,:midpoints),ribbon=ribbons(hazards,:ribbons),fillalpha=1,linealpha=0,palette=:reds,color=[1,14,8,11,5,2]|>permutedims,ylim=ylim,
     guidefontsize=8,xlabel="days after infection",ylabel="infection profile",xlim=xlim,
label="test rate: ".*string.([0,10,20,30,40,50]').*"%",size=(450,300)) for hazards in hazards_randsc]
plot(plot_rand...,legend=[:none (0.65,1) fill(:none,1,2)],legendfontsize=7,title="proportion symptomatic: ".*string.([0 25 50 75]).*"%")|>display
reduce(vcat,sum.(hazards_randsc,dims=1))
# -

# ## Simulation results

using JLD2
@load "covidsimulation.jld2" scenarios scenarios_dayoff scenarios_classisolation
#@load "covidsimulation_overdispersion02.jld2" scenarios scenarios_dayoff scenarios_classisolation
#@load "covidsimulation_flu.jld2" scenarios scenarios_dayoff scenarios_classisolation


# Class interventions
Base.:+(x::AbstractArray,y::Number)=x.+y
Base.:-(x::AbstractArray,y::Number)=x.-y
outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios]
heatmaps=heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
plot(heatmaps[1:12]...,layout=(3,4),size=(800,450),left_margin=2mm, xlabel="time",ylabel="grade/class (sorted)",guidefontsize=6,tickfontsize=6, title=" \n".*["baseline" "split class" "staggered attendance (within class)"  "staggered attendance (between class)"], titlefontsize=6)#,colorbar=[fill(:none,1,3) (0.5,1)])

# Screening
function transposelayout(panels,layout) reshape(panels,layout...)|>permutedims|>vec end
plot(transposelayout(heatmaps[1:4:33],(3,3))...,layout=(3,3),size=(600,450),left_margin=2mm, xlabel="time",ylabel="grade/class (sorted)",guidefontsize=7,tickfontsize=6, title=["baseline (no screening)" "symptom screening (50%)" "symptom + 10% regular test"], titlefontsize=7)#,colorbar=[fill(:none,1,2) (0.5,1)])

# Setting regular day off
outbreakmaps_dayoff=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios_dayoff]
heatmaps_dayoff=heatmap.(reduce(vcat,vec(outbreakmaps_dayoff)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps_dayoff,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_dayoff,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
plot(transposelayout(heatmaps_dayoff[1:4:33],(3,3))...,layout=(3,3),size=(600,450),left_margin=2mm,  xlabel="time",ylabel="grade/class (sorted)",guidefontsize=7,tickfontsize=6, title=["1 day on : 1 day off" "symptom + 1 day on : 1 day off" "symptom + 2 day on : 1 day off"], titlefontsize=7)#,colorbar=[fill(:none,1,2) (0.5,1)])

# Class isolation
outbreakmaps_classisolation=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios_classisolation]
heatmaps_classisolation=heatmap.(reduce(vcat,vec(outbreakmaps_classisolation)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps_classisolation,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,12),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_classisolation,repeat(Ref(1:5).*[2,4,2,1].+0.5,12),linetype=:hline,color=:black,linealpha=0.2)
plot(transposelayout(heatmaps_classisolation[1:4:45],(3,4))...,layout=(3,4),size=(800,450),left_margin=2mm,  xlabel="time",ylabel="grade/class (sorted)",guidefontsize=7,tickfontsize=6, title=["50% outside-class interaction" "50% outside-class/120% inside-class" "10% outside-class interaction" "10% outside-class/140% inside-class"], titlefontsize=7)#,colorbar=[fill(:none,1,2) (0.5,1)])

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
finsizes=(baseline=finalsizes(scenarios),dayoff=finalsizes(scenarios_dayoff),classisolation=finalsizes(scenarios_classisolation));

# +
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
# -

sumgtimes=sum.([hazards_symsc[:,1],hazards_symsc[:,3], hazards_randsc[3][:,2]]|>permutedims)
sumgtimes_dayoff=sum.([hazards_symsc[:,1].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],hazards_symsc[:,3].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],hazards_symsc[:,3].*repeat([1,1,0],size(hazards_symsc,1))[1:size(hazards_symsc,1)]]|>permutedims)
Rvalues=sumgtimes.*[1.8,1.2,0.8]
Rvalues_dayoff=sumgtimes_dayoff.*[1.8,1.2,0.8];

plot(plot.([[vcat(outbreakrisks[1:2]...)] [vcat(outbreakrisks_30[1:2]...)]],line_z=vec([Rvalues Rvalues_dayoff])',color=cgrad([:lightblue,:royalblue,:purple,:firebrick,:tomato]),linestyle=[:solid #=:dash :dashdot=#],linewidth=2)...,ylim=(0,1.01),clim=(0.5,1.5),
    xlabel="initial cases",ylabel="risk of outbreak (> ".*string.([10 30]).*" secondary cases)",legend=false, size=(800,260),colorbar_title="effective reproduction number",guidefontsize=8,bottom_margin=2mm)

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
rawincs=(x->first.(x)).([scenarios[:,1] scenarios_classisolation[:,[1,3]]])
@time casesatdetection=[samplesatdetection.(rawincs,testrate,psymptom;ipdist=ipdist) for testrate in [0.0,0.1,0.2,0.3], psymptom in [0.5,0.25,0.1]];

# +
tribar(x)=plot(bar.(Ref(0:100),shiftcol.(eachcol(x),1:3,3),color=[2,1,3]')...,layout=(3,1),xlim=(-1,20),ylim=(-0.05,1),size=(800,800),linealpha=0#=,title="Rs = ".*string.([1.8 1.2 0.8])=#,legend=false)
shiftcol(col,nth,outof)=begin ret=zeros(length(col),outof);ret[:,nth].=col;ret end
function plotcasesatdetection(cases,bins)
    den_all=(x->(([first(x);0:100]|>r.table.|>rcopy).-1)./length(first(x))).(values(cases))
    den_out=(x->(([last(x);0:100]|>r.table.|>rcopy).-1)./length(last(x))).(values(cases))
    
    ranges=range.(bins.+1,[bins[2:end];100],step=1)
    @views cumbin_all=getindex.(den_all,permutedims(ranges[:,:,:],(2,3,1))).|>sum
    band_all=groupedbar.(eachslice(cumbin_all,dims=2),bar_position=:stack,color=[:white,1,6,8,3,5]|>permutedims,palette=:dense)
    @views cumbin_out=getindex.(den_out,permutedims(ranges[:,:,:],(2,3,1))).|>sum
    band_out=groupedbar.(eachslice(cumbin_out,dims=2),bar_position=:stack,color=[:white,1,6,8,3,5]|>permutedims,palette=:amp)
    
    (all=band_all,out=band_out)
end
bands_screening=getindex.(casesatdetection,:,1)
bandsplot=plot(reduce(vcat,first.(plotcasesatdetection.(bands_screening,Ref([0,1,6,11,16,21]))))...,legend=false);
bandsplot_out=plot(reduce(vcat,last.(plotcasesatdetection.(bands_screening,Ref([0,1,6,11,16,21]))))...,legend=false);

bands_classisolation=reduce(vcat,eachcol.(casesatdetection[[1,5]]).|>collect)
bandsplot_classisolation=plot(reduce(vcat,(x->[first.(x)[reshape(1:6,3,2)'|>vec] last.(x)[reshape(1:6,3,2)'|>vec]]|>permutedims|>vec)(plotcasesatdetection.(bands_classisolation,Ref([0,1,6,11,16,21]))))...,legend=false);

# -

plot(bandsplot)
ylab=hcat(vcat.(string.([50 25 10]).*"% symptomatic\n \nprobability","","","")...)|>vec|>permutedims
xlab=vcat(hcat.("","",repeat(["baseline Rs"],1,4))...)|>vec|>permutedims
title=vcat(hcat.(["no tests" string.([10 20 30]).*"% regular tests"],"","","")...)|>vec|>permutedims
xticks!(1:3,string.([1.8,1.2,0.8]),xflip=true,xlabel=xlab, guidefontsize=8,titlefontsize=8,ylabel=ylab,title=title,bottom_margin=0mm,right_margin=0mm)

plot(bandsplot_out)
ylab=hcat(vcat.(string.([50 25 10]).*"% symptomatic\n \nprobability","","","")...)|>vec|>permutedims
xlab=vcat(hcat.("","",repeat(["baseline Rs"],1,4))...)|>vec|>permutedims
title=vcat(hcat.(["no tests" string.([10 20 30]).*"% regular tests"],"","","")...)|>vec|>permutedims
xticks!(1:3,string.([1.8,1.2,0.8]),xflip=true,xlabel=xlab, guidefontsize=8,titlefontsize=8,ylabel=ylab,title=title,bottom_margin=0mm,right_margin=0mm)

plot(bandsplot_classisolation)
ylab=hcat(vcat.("outside-class ".*string.([100 50 10]).*"%\n \nprobability","","","")...)|>vec|>permutedims
xlab=vcat(hcat.("","",repeat(["baseline Rs"],1,4))...)|>vec|>permutedims
title=vcat(hcat.(["50% symptomatic" "spillover" "25% symptomatic" "spillover"],"","","")...)|>vec|>permutedims
xticks!(1:3,string.([1.8,1.2,0.8]),xflip=true,xlabel=xlab, guidefontsize=8,titlefontsize=8,ylabel=ylab,title=title,bottom_margin=0mm,right_margin=0mm)

@load "closurestrategy_covid.jld2" scenarios_strategies_notest scenarios_strategies_10test
#@load "closurestrategy_flu.jld2" scenarios_strategies_notest scenarios_strategies_10test
#@load "closurestrategy_covid_verdispersion02.jld2" scenarios_strategies_notest scenarios_strategies_10test

# +
outbreakmaps_notest=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_notest)]
heatmaps_notest=heatmap.(reduce(vcat,vec(outbreakmaps_notest)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps_notest,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_notest,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

outbreakmaps_10test=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_10test)]
heatmaps_10test=heatmap.(reduce(vcat,vec(outbreakmaps_10test)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps_10test,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(heatmaps_10test,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

closuremaps_notest=[(x->reduce(hcat,vec.(x'))).(mean(scenario)) for scenario in last.(scenarios_strategies_notest)]
coldmaps_notest=heatmap.(reduce(vcat,vec(closuremaps_notest)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
yticks!.(coldmaps_notest,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(coldmaps_notest,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)

closuremaps_10test=[(x->reduce(hcat,vec.(x'))).(mean(scenario)) for scenario in last.(scenarios_strategies_10test)]
coldmaps_10test=heatmap.(reduce(vcat,vec(closuremaps_10test)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
yticks!.(coldmaps_10test,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
plot!.(coldmaps_10test,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)


plot(transposelayout([heatmaps_notest[1:4:9];coldmaps_notest[1:4:9];heatmaps_10test[1:4:9];coldmaps_10test[1:4:9];heatmaps_notest[13:4:21];coldmaps_notest[13:4:21]],(3,6))...,
    layout=(3,6),size=(1200,450),left_margin=2mm,bottom_margin=4mm,  xlabel="time",ylabel="grade/class (sorted)",guidefontsize=8,tickfontsize=7,title=vec(hcat(vcat.(["naive strategy" "10% regular test" "50% outside-class interaction"],"closure")...))|>permutedims, titlefontsize=8)#,colorbar=[fill(:none,1,2) (0.5,1)])
# -

plot(transposelayout([heatmaps_notest[25:4:33];coldmaps_notest[25:4:33];heatmaps_notest[37:4:45];coldmaps_notest[37:4:45];heatmaps_notest[49:4:57];coldmaps_notest[49:4:57]],(3,6))...,
    layout=(3,6),size=(1200,450),left_margin=2mm,bottom_margin=4mm,  xlabel="time",ylabel="grade/class (sorted)",guidefontsize=8,tickfontsize=7,title=vec(hcat(vcat.(["50% outside/120% inside" "10% outside-class interactions" "10% outside/140% inside"],"closure")...))|>permutedims, titlefontsize=8)#,colorbar=[fill(:none,1,2) (0.5,1)])

obsizes_notest=[first(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_notest]
closuresizes_notest=[last(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_notest]
obsizes_10test=[first(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_10test]
closuresizes_10test=[last(scenario).|>first.|>sum.|>sum for scenario in scenarios_strategies_10test]

obsizes_baseline=[scenario.|>first.|>sum.|>sum for scenario in scenarios]
obsizes_classisolation=[scenario.|>first.|>sum.|>sum for scenario in scenarios_classisolation]

@load "strategies_outbreaksize.jld2" obsizes_notest closuresizes_notest obsizes_10test closuresizes_10test obsizes_baseline obsizes_classisolation

ste(v)=std(v)/sqrt(length(v))
multiplot(x,yerror;kw...)=bar(x,yerror=yerror,kw...)
obsizes_closure=[obsizes_notest[:,1] obsizes_10test[:,1] obsizes_notest[:,2]]
closuresizes=[closuresizes_notest[:,1] closuresizes_10test[:,1] closuresizes_notest[:,2]]
obsizes=[obsizes_baseline[:,[1,3]] obsizes_classisolation[:,1]]
#((b,c)->bar.(vcat.(hcat.(eachrow(mean.(b)),eachrow(mean.(c))),hcat.(eachrow(quantile.(b,0.95)),eachrow(quantile.(c,0.95))))))(obsizes,obsizes_closure).|>display
function closureeffects(size_baseline, size_closure, closuredays)
    means=[mean.(x) for x in (size_baseline, size_closure, closuredays)]./[480,480,1]
    upp95s=[quantile.(x,0.95) for x in (size_baseline, size_closure, closuredays)]./[480,480,1]
    #barbase=reduce.(hcat,vcat.(hcat(upp95s[1:2]...)[:,[1,4,2,5,3,6]],hcat(means[1:2]...)[:,[1,4,2,5,3,6]])|>eachrow)
    
    obbars=reduce.(hcat,vcat.(upp95s[1],means[1],upp95s[2],means[2])|>eachrow)
    clbars=reduce.(hcat,vcat.(upp95s[3],means[3])|>eachrow)
    
    #barclosure=reduce.(hcat,vcat.(upp95s[2],means[2])|>eachrow)
    [bar.(obbars.|>permutedims,color=repeat([8 6],inner=(1,2)),palette=:matter,linealpha=[1 0.5],label=["isolation only" :none "class closure" :none],
    xticks=(1:3,["naive","regular\ntests","class\ndistancing"]),xrotation=0);
    bar.(clbars.|>permutedims,color=:lightblue,linealpha=[1 0.5],label=:none#=["days closed" :none]=#,
    xticks=(1:3,["naive","regular\ntests","class\ndistancing"]),xrotation=0)]
end
plot(closureeffects(obsizes,obsizes_closure,closuresizes)...,layout=(2,3), legend=[:none :none (0.5,1)],ylim=[(0,1) (0,0.5) (0,0.1) (0,500) (0,100) (0,50)],size=(800,600),title="Rs = ".*string.([1.8 1.2 0.8]),titlefont=10,
ylabel=repeat(["proportion infected" "total days of class closure"],inner=(1,3)),left_margin=2mm,tickfontsize=10)
