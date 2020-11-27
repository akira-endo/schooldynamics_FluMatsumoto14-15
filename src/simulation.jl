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

# # Simulation of COVID-19/pandemic influenza outbreaks in school
#
# ## Preparation

include("SchoolOutbreak.jl")

# +
using Distributions, Plots, InvertedIndices, LinearAlgebra, StatsBase
gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,grid=false, tick_direction=:out)

classes=[2,4,2,1]
sizes=[40,20,20,40]

nclassmat=repeat(classes,1,6)
nstudentsmat=repeat(sizes,1,6).*nclassmat
struct ImproperUniform<: Distribution{Univariate,Discrete} end
Distributions.cdf(d::ImproperUniform, x::Integer)=1.0
Distributions.pdf(d::ImproperUniform, x::Integer)=1.0

rcomdist=Uniform(0,365)
hhfoi=-Inf
Gammameansd(mean,sd)=Gamma(mean^2/sd^2,sd^2/mean)
#gtimedist=Gammameansd(3,1)
gtimedist=Gammameansd(1.7,1) # H3N2 serial interval from Vink et al 2014 (https://doi.org/10.1093/aje/kwu209)
N0=[30,30,30,30,30] # normalizing constant to avoid correlation between β and γ
C0=[0,6*3,3,3,3]

sus_covlabels=[:Grade, :Vaccine_curr, :NPI_mask,:NPI_wash]
inf_covlabels=[:Grade, :Vaccine_curr, :NPI_mask,:NPI_wash,:duringbreak]

strata=vcat.(1:4,sortslices.(repeat.((x->reduce(hcat,vec(collect.(x)))).(Iterators.product.(Ref(1:6),range.(1,classes,step=1)).|>collect),1,sizes),dims=2))
strata=vcat.(strata,repeat.(Ref((1:2)'),1,size.(strata,2).÷2))
strata=reduce(hcat,strata)
parameters=SchoolOutbreak.makeparameters(ones(5),ones(5)/2,ones(5)/2,0.01,
    ones(size(sus_covlabels)),ones(size(inf_covlabels)),gtimedist,20,rcomdist,strata,nclassmat,nstudentsmat,N0,C0,Inf)
@time students=SchoolOutbreak.studentsfromdata(parameters,false,fill(typemax(Int),size(strata,2)),eachcol(strata),
    Ref(zeros(length(sus_covlabels))),Ref(zeros(length(inf_covlabels))),nclassmat,nstudentsmat,-Inf,1);
@time simstudents=SchoolOutbreak.Students.([filter(x->x.stratum[1]==schoolid,students) for schoolid in 1:4],Ref(parameters));
# -

using Mamba
chain1 = read("Mastumoto_chain1.jls", ModelChains)

# preparation for simulation
function parametersfromchain(chain, iter)
    (β=SchoolOutbreak.posterior(chain,:βs,iter).samples[:,[1;1:4]]|>vec,
    γ=SchoolOutbreak.posterior(chain,:γs,iter).samples[:,[1;1:3;3]]|>vec,
    δ=SchoolOutbreak.posterior(chain,:δs,iter).samples[:,[1;1:3;3]]|>vec,
    rcom=SchoolOutbreak.posterior(chain,:rcom,iter).samples|>first|>fill,
    suscoef=SchoolOutbreak.posterior(chain,:suscoef,iter).samples|>vec,
    infcoef=SchoolOutbreak.posterior(chain,:infcoef,iter).samples|>vec)
end
nsample=500
@time updateparameters=parametersfromchain.(Ref(chain1),(1:nsample).*(length(chain1.range)÷(2*nsample)));
(x->x.rcom.=0).(updateparameters)
(x->x.members[1].isinfected=true).(simstudents)
(x->x.members[1].onset=1).(simstudents)

# ## Parameter settings

# +
# Adjust R0
function R0set!(parameters,R0,nclasses=2,classsize=30)
    parameters.β.*=R0/sum((parameters.β./(classsize/30).^parameters.γ./(nclasses./[3,3,3,3,3]).^parameters.δ).*[0,5*nclasses*classsize,(nclasses-1)*classsize,classsize/2,classsize/2])
    
end
R0set!.(updateparameters,2);

function R0(students)
    sum((x->x.pairwiseβ[]/exp(x.pairwiselogN[]*x.γ[]+x.pairwiselogM[]*x.δ[])).(students.pairs),dims=2)
end

# +
###### infection hazard
using RCall
@rimport distr

## COVID-19
incperiod=distr.Lnorm(1.434065,0.6612) # He et al.
infprof=distr.Norm(0.53,sqrt(7)) # Aschroft et al.
gtime=incperiod+infprof

## Pandemic influenza
#incperiod=distr.Lnorm(0.34,0.42) # Lessler et al. 2009 (https://dx.doi.org/10.1016%2FS1473-3099(09)70069-6)
#gtime=((mean,sd)->distr.Gammad(mean^2/sd^2, sd^2/mean))(1.7,1)
#

gtimevec=distr.p(gtime)((0:20))|>RCall.unsafe_vec|>diff
gtimevec=gtimevec./sum(gtimevec)

# set hazard function
symratio=(0:0.25:1)'
hazards_symsc=gtimevec.*(1 .- symratio.*(distr.p(incperiod)((1:20).-1)|>RCall.unsafe_vec))
surv=(0:0.1:0.5)'
hazards_randsc=[hazard.*((1 .-surv).^(0:19)) for hazard in eachcol(hazards_symsc)]

# enbed generation time distribution in parameters
function setgtime!(parameters,newpdf)
    SchoolOutbreak.updateArray!(parameters.pdfgtime,[0;newpdf;0])
    SchoolOutbreak.updateArray!(parameters.cdfgtime,[0;0;cumsum(newpdf)])
end
setgtime!(parameters,hazards_randsc[3][:,2]); # 50% symptomatic + 10% testing rate
# -

# ## Outbreak simulation

incidence(outbreak)=length(outbreak)
incidence(outbreak::AbstractArray{<:AbstractVector})=incidence.(outbreak)
incidence(outbreak::AbstractArray{<:AbstractMatrix})=sum.(outbreak)
function getgradeclass(student) return((student.stratum[2],student.stratum[3])) end
function reorderedclass(class,gradeorder,classorder) # returns reordered class id
    grnum=findfirst(x->x==class[1],gradeorder)
    clnum=findfirst(x->x==class[2],classorder[class[1]])
    (grnum,clnum)
end
function updateorders!(gradeorder,classorder,newstudents)
    gcs=getgradeclass.(newstudents)
    push!(gradeorder,filter(x->!(x in gradeorder),unique(getindex.(gcs,1)))...) # push nonregistered grades
    for grade in 1:6
        push!(classorder[grade],unique(getindex.(filter(x->(x[1]==grade)&&!(x[2] in classorder[grade]),gcs),2))...)
    end
end
function incidencebyclass_reorder!(members,idsvector,ngrades,nclasses, closuretime=nothing)
    gradeorder=Int[]
    classorder=[Int[] for x in 1:6]
    reorderedincidence=[((members, ids)->
        begin
            casesbyclass=zeros(Int,ngrades,nclasses)
            updateorders!(gradeorder,classorder, members[ids])
            reorderedgcs=reorderedclass.(getgradeclass.(members[ids]),Ref(gradeorder),Ref(classorder))
            for gc in reorderedgcs
                casesbyclass[CartesianIndex(gc)].+=1
            end
            casesbyclass
        end)(members,ids)
    for ids in idsvector]
    if isnothing(closuretime) return(reorderedincidence) end # return reordered incidence only if closuretime is not passed
    reorderedclosuretime=[zeros(Int,ngrades,nclasses) for t in 1:length(idsvector)]
    for (g,closure,corder) in zip(1:6,collect(eachrow(closuretime)),classorder)
        if g in gradeorder
            setindex!.(reorderedclosuretime, (getindex.(closure[corder],t) for t in 1:length(idsvector)), findfirst(x->x==g,gradeorder), Ref(corder))
        end
    end
    (incidence=reorderedincidence, closuretime=reorderedclosuretime)
end
function readoutbreakbyclass_reorder(outbreak, students::Vector{<:SchoolOutbreak.Students}, ngrades,nclasses, closuretime=nothing)
    reordered=incidencebyclass_reorder!.(getfield.(students,:members),outbreak,ngrades,nclasses, closuretime)
    if isnothing(closuretime) return(reordered) end
    (incidence=getindex.(reordered,:incidence), closuretime=getindex.(reordered,:closuretime))
end
function outbreakscenarios!(simstudents,updateparameters,R0,gtime)
    R0set!.(updateparameters,Ref(R0))
    setgtime!(simstudents[1].parameters,gtime)
    outbreaks=SchoolOutbreak.simulateoutbreaks!.(Ref(simstudents), updateparameters, Ref(1:360))
    R0set!.(updateparameters,Ref(1 ./R0)) # to fix the breaking change when R0 is a vector
    readoutbreakbyclass_reorder.(outbreaks,Ref(simstudents),6,Ref([2,4,2,1]))
end

gtimes=[hazards_symsc[:,1],hazards_symsc[:,3], hazards_randsc[3][:,2]]|>permutedims
@time scenarios=outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),[1.8,1.2,0.8],gtimes);
gtimes_dayoff=[hazards_symsc[:,1].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],hazards_symsc[:,3].*repeat(0:1,size(hazards_symsc,1))[1:size(hazards_symsc,1)],hazards_symsc[:,3].*repeat([1,1,0],size(hazards_symsc,1))[1:size(hazards_symsc,1)]]|>permutedims
@time scenarios_dayoff=outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),[1.8,1.2,0.8],gtimes_dayoff);
classisolation=[[fill(0.5,3);1;1], [fill(0.5,3);1.2;1.2], [fill(0.1,3);1;1], [fill(0.1,3);1.4;1.4]]|>permutedims
@time scenarios_classisolation=outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),[1.8,1.2,0.8].*classisolation,Ref(gtimes[1]));

# ## Simulation results

Base.:+(x::AbstractRange,y::Number)=x.+y
Base.:-(x::AbstractRange,y::Number)=x.-y
outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios]
heatmaps=Plots.heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(9,4),size=(800,1200),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios_dayoff]
heatmaps=Plots.heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,9),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,9),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(9,4),size=(800,1200),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in scenarios_classisolation]
heatmaps=Plots.heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,12),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,12),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(12,4),size=(800,1600),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

# ## Responsive management strategy simulation

Base.:^(x,y::AbstractDict)=Dict(keys(y).=>x.^values(y))
detectbinom(d::AbstractDict,prob)=Dict(keys(d).=>rand.(Binomial.(values(d),prob)))
struct Strategy{V,F,D}
    detectdist::V
    detectprob::F
    closurelength::Integer
    closuretime::D
end
Broadcast.broadcastable(m::Strategy) = Ref(m)
function getclosuretime(strategy::Strategy,student::SchoolOutbreak.Student)
    strategy.closuretime[student.stratum[2:3]...]
end
function canceltransmission!(student,frompairs,strategy,currenttime)
    infectedpairs=filter(x->x.isinfected,frompairs)
    if length(infectedpairs)==0 return(nothing) end
    infectiondates=getfield.(getfield.(infectedpairs,:from),:onset).+getfield.(infectedpairs,:serialinterval)
    
    fromid=(infectiondates.==currenttime)
    if getclosuretime(strategy,student)[currenttime]!=0 || all(getindex.(getclosuretime.(strategy,getfield.(infectedpairs[fromid],:from)),currenttime).!=0)
        student.onset=minimum([filter(x->x>currenttime,infectiondates);typemax(Int)])
        student.isinfected=length(filter(x->x>currenttime,infectiondates))!=0 #sum(fromid)!=length(infectiondates)
    end
end
function (strategy::Strategy)(students,currenttime)
    infectorsid=findall(x->x.onset==currenttime,students.members)
    if length(infectorsid)==0 return(nothing) end # abort if no infected person  
    infectors=students.members[infectorsid]
    frompairs=eachcol(@view students.pairs[:,infectorsid])
    
    canceltransmission!.(infectors, frompairs, strategy, currenttime) # cut transmissions during closure
    
    detectedsdict=detectbinom(countmap(getfield.(infectors,:stratum)),strategy.detectprob)
    for classdetected in detectedsdict
        if last(classdetected)!=0 # if at least one case was detected
            closurefrom=currenttime+minimum(rand(strategy.detectdist,last(classdetected)))
            closuretimes=strategy.closuretime[first(classdetected)[2:3]...]
            if closurefrom>length(closuretimes)||closuretimes[closurefrom]!=0 continue end # if already closed, no change
            closuretimes[closurefrom+1:min(closurefrom+strategy.closurelength,length(closuretimes))].=reverse(1:min(strategy.closurelength,length(closuretimes)-closurefrom))
            if closurefrom+strategy.closurelength<length(closuretimes) && closuretimes[closurefrom+strategy.closurelength+1]!=0
                @views closuretimes[closurefrom+1:min(closurefrom+closuretimes[closurefrom+1],length(closuretimes))].=0
            end
        end
    end
    (detectedsdict, countmap(getfield.(infectors,:stratum)),strategy.detectprob)
end
function outbreakscenarios!(simstudents,updateparameters,R0,gtime,strategy)
    R0set!.(updateparameters,Ref(R0))
    setgtime!(simstudents[1].parameters,gtime)
    outbreaks=SchoolOutbreak.simulateoutbreaks!.(Ref(simstudents), updateparameters, Ref(1:360), statuscheck! = strategy)
    R0set!.(updateparameters,Ref(1 ./R0)) # to fix the breaking change when R0 is a vector
    incidences=(x->getindex.(x,:incidence)).(outbreaks)
    closuretimes=(x->getindex.(x,:closuretime)).(outbreaks)
    reordered=readoutbreakbyclass_reorder.(incidences,Ref(simstudents),6,Ref([2,4,2,1]),closuretimes)
    (incidence=getindex.(reordered,:incidence), closuretime=getindex.(reordered,:closuretime))
end

# +
gtimes=[hazards_symsc[:,1]] 
ipdist=LogNormal(1.434065,0.6612) #COVID-19
#ipdist=LogNormal(0.34,0.42) #Flu

function detectdistribution(ipdist, testingrate, symptomaticratio, trange=0:19)
    probmass=1 .-(1 .-symptomaticratio*cdf.(ipdist,trange)).*(1-testingrate).^(trange)
    probmass=[0;probmass]|>diff
    (dist=Categorical(normalize(probmass,1)), prob=sum(probmass))
end
detect=(x->[first.(x),last.(x)])(detectdistribution.(ipdist, [0,0.1], 0.5)) # replace 0.5 with 0.25 for lower symptomatic ratio
strategies=[Strategy.(detect[1],detect[2], 14, Ref([zeros(Int,360) for i in 1:6, j in 1:classsize])) for classsize in (2,4,2,1)] # replace 14 with 7 for flu
@time scenarios_strategies_notest=outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),[1.8,1.2,0.8].*[1 classisolation],gtimes,Ref(first.(strategies)));
@time scenarios_strategies_10test=outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),[1.8,1.2,0.8].*[1 classisolation],gtimes,Ref(last.(strategies)));
# -

using JLD2
#@save "closurestrategy_covid.jld2" scenarios_strategies_notest scenarios_strategies_10test

# ## Simulation results
# ### No regular testing

outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_notest)]
heatmaps=Plots.heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(15,4),size=(800,2000),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

closuremaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario)) for scenario in last.(scenarios_strategies_notest)]
heatmaps=Plots.heatmap.(reduce(vcat,vec(closuremaps)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(15,4),size=(800,2000),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

# ### With 10% regular testing

outbreakmaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario))./[40,20,20,40] for scenario in first.(scenarios_strategies_10test)]
heatmaps=Plots.heatmap.(reduce(vcat,vec(outbreakmaps)),clim=((0,0.2/40)),color=cgrad([:white,:red,:purple,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(15,4),size=(800,2000),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)

closuremaps=[(x->reduce(hcat,vec.(x'))).(mean(scenario)) for scenario in last.(scenarios_strategies_10test)]
heatmaps=Plots.heatmap.(reduce(vcat,vec(closuremaps)),clim=((0,0.2)),color=cgrad([:white,:skyblue,:blue,:indigo,:black]),legend=false)
yticks!.(heatmaps,repeat(Ref((1:6) .-0.5).*[2,4,2,1].+0.5,15),Ref(string.(1:6).*""),tickfontsize=6)
Plots.plot!.(heatmaps,repeat(Ref(1:5).*[2,4,2,1].+0.5,15),linetype=:hline,color=:black,linealpha=0.2)
Plots.plot(heatmaps...,layout=(15,4),size=(800,2000),left_margin=2mm,bottom_margin=-2mm, xlabel="time",ylabel="classes/grades",guidefontsize=6,tickfontsize=6)
