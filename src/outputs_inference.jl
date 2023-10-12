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

# Q-Q plot
using RCall
RCall.rcall_p(:options, rcalljl_options=Dict(:width => 1000, :height => 1000))
chainmat=Matrix(chain)[:,1:end-1]
@rput chainmat
R"library(royston);par(cex=2);royston.test(chainmat)"
# -


## Class/grade distributuion

gr(fontfamily="Helvetica")
using Random;Random.seed!(2020)
pl=scatter(ceil.(nclassmat).+(rand(29,6).-0.5)./4,nstudentsmat./ceil.(nclassmat), grid=false,
    color=randperm(29),markersize=4,markerstrokewidth=0,legend=false,tick_direction=:out,
    size=(400,300),xguidefontsize=8,yguidefontsize=8)
xlabel!("number of classes per grade")
ylabel!("class size")
plot!(pl,[10.0],linetype=:hline,color=:gray,linestyle=:dash)

## Class/grade against attack ratio
using InvertedIndices
attackrate=[7.9, 11.5, 16.2, 33.5, 50.0, 21.0, 14.8, 23.6, 19.4, 15.7, 10.1, 37.0, 13.5, 16.0, 16.0, 23.1, 18.5, 21.2, 12.5, 40.2, 0.0, 8.2, 29.1, 27.2, 12.6, 30.6, 19.7, 29.9, 21.6]
meanclasssize=[31.833333333333332, 32.19166666666666, 6.166666666666667, 27.833333333333332, 4.666666666666667, 30.0, 28.875, 31.708333333333332, 30.758333333333336, 25.416666666666668, 33.625, 25.666666666666668, 29.0, 29.041666666666668, 28.166666666666668, 33.44444444444445, 31.26388888888889, 21.583333333333332, 30.805555555555554, 21.166666666666668, 5.833333333333333, 29.333333333333332, 32.23055555555556, 25.916666666666668, 30.333333333333332, 29.972222222222225, 30.222222222222218, 30.708333333333332, 36.333333333333336]
meannclasses=[2.0, 4.833333333333333, 1.0, 1.0, 0.5, 2.3333333333333335, 3.8333333333333335, 4.166666666666667, 4.833333333333333, 2.0, 4.5, 1.0, 2.6666666666666665, 3.8333333333333335, 2.0, 3.0, 3.5, 2.0, 2.3333333333333335, 1.0, 0.8333333333333334, 2.0, 4.666666666666667, 2.1666666666666665, 2.0, 2.5, 2.8333333333333335, 3.8333333333333335, 2.0]
attackrate, meanclasssize, meannclasses = getindex.((attackrate, meanclasssize, meannclasses),Ref(Not([3,5,21])))
@rimport stats as st
@show st.cor_test(meanclasssize,attackrate)
@show st.cor_test(meannclasses,attackrate)
plot(scatter.((meanclasssize,meannclasses),Ref(attackrate))...,ylim=(0,50),xlim=[(15,40) (0.5,5)],labels=["r = -0.39 (-0.68,-0.01)" "r = -0.30 (-0.62, 0.10)"],legend=(0.4,0.95),
    xlabel=["mean class size" "mean # classes per grade"], ylabel=["attack ratio %" "attack ratio %"], markercolor=:black,size=(250,500),layout=(2,1))

# Plotting Rs
## Functions
function posterior(chain, parameter::Symbol)
    if parameter==:β0 parameter=:βs;fn=β0
    elseif parameter==:rcom0 parameter=:rcom;fn=rcom0
    elseif parameter in (:suscoef,:infcoef) fn=(slice->slice[parameter].|>exp)
        else fn = (slice->slice[parameter]) end
    v=fn(chain)
    (names=parameter ,samples=v)#[getindex.(v,x) for x in 1:size(first(v),1)])
end

function β0(slice)
    (slice[:logβs]).|>exp end
function rcom0(slice)
    (slice[:logrcom]).|>exp end
bclog(x)=log.(x)
function R_nm(chain,nclasses,sizerange, meansuscovar=0,meaninfcovar=0)
    βs=posterior(chain,:β0).samples
    βsm=βs[:,1:3];βsm[:,3].=vec(mean(βs[:,3:4],dims=2))
    βsm.*=exp.(sum((posterior(chain,:suscoef).samples).*meansuscovar',dims=2)+sum((posterior(chain,:infcoef).samples).*meaninfcovar',dims=2))
    if nclasses==1 βsm[:,2].=0 end
    γs=posterior(chain,:γs).samples
    δs=posterior(chain,:δs).samples
    [quantile.((βsm./(classsize/30).^γs./(nclasses./[3,3,3]').^δs).*[5*nclasses*classsize (nclasses-1)*classsize classsize] |>eachcol,Ref([0.5,0.025,0.975])) for classsize in sizerange]
end
function R_nm(parm::Vector{<:AbstractFloat},nclasses,sizerange,meansuscovar=0,meaninfcovar=0)
    βs=parm[1:4]
    βsm=βs[1:3];βsm[3]=mean(βs[3:4])
    suscoef=parm[12:15];infcoef=parm[16:20]
    if nclasses==1 βsm[2]=0 end
    γs=parm[5:7]
    δs=parm[8:10]
    [(βsm./(classsize/30).^γs./(nclasses./[3,3,3]).^δs).*[5*nclasses*classsize,(nclasses-1)*classsize,classsize] for classsize in sizerange]
end
function R_reduction(chain,nmbefore,nmafter,coefafter=ones(3),coefbefore=ones(3))
    βs=posterior(chain,:β0).samples
    βsm=βs[:,1:3];βsm[:,3].=vec(mean(βs[:,3:4],dims=2))
    if nmbefore[2]==1 coefbefore[2]=0 end
    if nmafter[2]==1 coefafter[2]=0 end
    γs=posterior(chain,:γs).samples
    δs=posterior(chain,:δs).samples
    popbefore=[5*prod(nmbefore) prod(nmbefore)-nmbefore[1] nmbefore[1]]
    popafter=[5*prod(nmafter) prod(nmafter)-nmafter[1] nmafter[1]]
    quantile(sum(popafter.*coefafter'.*βsm./(nmafter[1]/30).^γs./(nmafter[2]./[3,3,3]').^δs,dims=2)./sum((popbefore.*coefbefore'.*βsm./(nmbefore[1]/30).^γs./(nmbefore[2]./[3,3,3]').^δs),dims=2) |>vec, [0.5,0.025,0.975])
end
function Rtotal_nm(chain,nclasses,sizerange, meansuscovar=0,meaninfcovar=0)
    βs=posterior(chain,:β0).samples
    βsm=βs[:,1:3];βsm[:,3].=vec(mean(βs[:,3:4],dims=2))
    βsm.*=exp.(sum((posterior(chain,:suscoef).samples).*meansuscovar',dims=2)+sum((posterior(chain,:infcoef).samples).*meaninfcovar',dims=2))
    if nclasses==1 βsm[:,2].=0 end
    γs=posterior(chain,:γs).samples
    δs=posterior(chain,:δs).samples
    [quantile.(sum((βsm./(classsize/30).^γs./(nclasses./[3,3,3]').^δs).*[5*nclasses*classsize (nclasses-1)*classsize classsize],dims=2) |>eachcol,Ref([0.5,0.025,0.975])) for classsize in sizerange]
end

# ## Posteriors

parnames=["βs[".*string.(1:4).*"]";"rcom";"γs[".*string.(1:3).*"]";"δs[".*string.(1:3).*"]";"suscoef[".*string.(1:4).*"]";"infcoef[".*string.(1:5).*"]";"loglikelihood"]
@time posteriors=posterior.(Ref(chain1),[:β0,:rcom0,:γs,:δs,:suscoef,:infcoef,:loglikelihood])
posteriors=reduce(vcat,posteriors)
hcat(parnames,reduce(vcat,(x->quantile.(x.samples|>eachcol,Ref([0.5,0.025,0.975]))).(posteriors)))

# Plotting school reproduction number

@time Rnms=(R_nm(chain1,nclasses,1:40) for nclasses in 1:6)
medRnms=[reduce(hcat,(x->first.(x)).(Rnm))' for Rnm in Rnms];
uppRnms=[reduce(hcat,(x->last.(x)).(Rnm))' for Rnm in Rnms];
lowRnms=[reduce(hcat,(x->getindex.(x,2)).(Rnm))' for Rnm in Rnms];
function split5(medRnm)
    [fill(medRnm[1],5)./5;(medRnm)[2:3]]
end
function medR(medRnms,n,m)
    (getindex.(medRnms,n,:)[m])|>transpose
end

gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,grid=false, tick_direction=:out,)
# median
barsbyclass=groupedbar.(Ref(1:5),[reduce(vcat,medR.(Ref(medRnms),n,1:5)) for n in [20,30,40]], bar_position = :stack, bar_width=0.9, 
    ylim=(0,1),color=reverse([1 :skyblue 9],dims=2),linealpha=0.8)
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"median (stacked)")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false, 
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.09)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,left_margin=2mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)

# by proximity
medv=[reduce(vcat,reverse.(medR.(Ref(medRnms),n,1:5),dims=2)) for n in [20,30,40]]
uerr=[reduce(vcat,reverse.(medR.(Ref(uppRnms),n,1:5),dims=2)) for n in [20,30,40]].-medv
lerr=medv.-[reduce(vcat,reverse.(medR.(Ref(lowRnms),n,1:5),dims=2)) for n in [20,30,40]]
yerr=((x,y)->Tuple.(vcat.(x,y))).(lerr,uerr)
barsbyclass=[groupedbar(1:5,medv[n], bar_width=0.9, 
    ylim=(0,1),color=[1 :skyblue 9],linealpha=0.8,yerrors=yerr[n]) for n in 1:3]
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"contribution by proximity")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false, 
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.09)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,left_margin=2mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)

# overall Rs
@time Rtot=(Rtotal_nm(chain1,nclasses,1:40) for nclasses in 1:6)
medRtot=[reduce(hcat,(x->first.(x)).(Rnm))' for Rnm in Rtot];
uppRtot=[reduce(hcat,(x->last.(x)).(Rnm))' for Rnm in Rtot];
lowRtot=[reduce(hcat,(x->getindex.(x,2)).(Rnm))' for Rnm in Rtot];
medvtot=[reduce(vcat,reverse.(medR.(Ref(medRtot),n,1:5),dims=2)) for n in [20,30,40]]
uerrtot=[reduce(vcat,reverse.(medR.(Ref(uppRtot),n,1:5),dims=2)) for n in [20,30,40]].-medvtot
lerrtot=medvtot.-[reduce(vcat,reverse.(medR.(Ref(lowRtot),n,1:5),dims=2)) for n in [20,30,40]]
yerrtot=((x,y)->Tuple.(vcat.(x,y))).(lerr,uerrtot)
barsbyclass=[groupedbar(1:5,medvtot[n], bar_width=0.9, 
    ylim=(0,1.5),color=[6],linealpha=0.8,yerrors=yerrtot[n]) for n in 1:3]
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"school reproduction number")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false,
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.09)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,left_margin=2mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)  
Rreductions=reduce(hcat,R_reduction.(Ref(chain1),Ref((40,2)),repeat([(40,2),(20,4),(20,2),(40,1)],3),repeat([ones(3),[0.5,0.5,1],[0.1,0.1,1]],inner=4)))
colors=[:black,:royalblue,3]

plot([Rreductions[1,:] fill(-1.,12,2)],
line=0,marker=([:circle#=,:pent,:hex,:oct=#],6),markerstrokewidth=0,ylim=(0,2),color=[repeat(colors,inner=4) repeat(colors[2:3]|>permutedims,12)], bottom_margin=10mm,
xticks=(1:12,repeat(["no change","split class","staggered attendance\n(within class)", "staggered attendance\n(between class)"],3)),xrotation=60,
label=string.([100 50 10]).*"% outside-class interaction", legend=(0.67,1))
ylabel!("relative change in Rs",yguidefontsize=8)
plot!([1.0],linetype=:hline,color=:gray,linestyle=:dash,label="")
ohlc!(OHLC[Tuple([NaN;Rreductions[[3,2],c];NaN]) for c in 1:size(Rreductions,2)],color=repeat(colors,inner=4),width=2,label="")

# ## Sensitivity analysis

include("SchoolOutbreak.jl")

# +
# Sensitivity analysis

# estim (input): vector of parameter estimates for each sensitivity analysis

medRnms=(reduce(hcat,R_nm(estim,nclasses,1:40))' for nclasses in 1:6)
barsbyclass=groupedbar.(Ref(1:5),[reduce(vcat,medR.(Ref(medRnms),n,1:5)) for n in [20,30,40]], bar_position = :stack, bar_width=0.9, 
    ylim=(0,1),color=reverse([1 :skyblue 9],dims=2))
xlabel!.(barsbyclass,"number of classes per grade")
ylabel!.(barsbyclass,"school reproduction number")
plot(barsbyclass...,layout=(1,3),size=(800,200),xguidefontsize=8,yguidefontsize=8,grid=false, 
    label=["classmate" "grademate" "schoolmate" fill("",1,4)], legend=[:none#=(0.75,1.07)=# fill(:none,1,3)],legendfontsize=6,
    top_margin=4mm,bottom_margin=5mm,title="class size: ".* string.([20 30 40]),tick_direction=:out,format=:svg)
# -

