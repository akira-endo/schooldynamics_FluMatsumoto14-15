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

# # School network model
# This model uses matrix expressions of school network to provide a samplewise likelihood function for primary school influenza data in Matsumoto city, Japan.

# ## Load and prepare data

# +
# School incidence data
using Dates, Distributions
const diZero=Dates.value(Date("2014-09-30")) # day integer for t=0
const diEnd=Dates.value(Date("2015-05-01")) # day integer for t=end
const diLen=diEnd-diZero
function d2di(d::Date) Dates.value(d)-diZero end
function di2d(di::Int) Date(Dates.UTInstant(Day(di+diZero))) end

winterbreak=d2di(Date("2014-12-27")):d2di(Date("2015-1-7")) # 27 Dec - 7 Jan was winter break in school in Matsumoto
duringbreak=(x->x.+1).(winterbreak);
logispar_est=[64.21977, 113.60699, 10.15022]
# -

# ## Construct school data

# +
include("src/SchoolOutbreak.jl")

# override function for recall bias adjustment
function SchoolOutbreak.updatesusinf!(student::Student,parameters::NT where NT<:NamedTuple)
    if student.isinfected
        student.susceptibility.=linregval(student.suscovar,parameters.suscoef)
        student.infectiousness.=linregval(student.infcovar,parameters.infcoef)
    else
        @views student.susceptibility.= exp(student.suscovar[1]*parameters.suscoef[1][]+
            sum(log.( (1 .-student.suscovar[2:end]) .+ (student.suscovar[2:end].*exp.(first.(parameters.suscoef[2:end]))) )))
        @views student.infectiousness.= exp(student.infcovar[1]*parameters.infcoef[1][]+
            sum(log.( (1 .-student.infcovar[2:end]) .+ (student.infcovar[2:end].*exp.(first.(parameters.infcoef[2:end]))) )))

    end
end
# -

using JLD2
@load "Matsumoto_studentsdata.jld2" students
@time schooldata=SchoolOutbreak.StudentsData.([filter(x->x.stratum[1]==schoolid,students) for schoolid in 1:29],Ref(parameters),Ref((schoolclosure=duringbreak,)));

# ## MLE (sensitivity analysis only)

# +
function ll(lparms,schooldata=schooldata)
    parms=copy(lparms)
    parms[1:8].=exp.(lparms[1:8])
    @views SchoolOutbreak.updateArray.(schooldata[1].parameters.β,[0;parms[1:4]])
    @views SchoolOutbreak.updateArray.(schooldata[1].parameters.γ,[0;parms[[5:7;7]]])
    @views SchoolOutbreak.updateArray.(schooldata[1].parameters.γ,[0;parms[[8:10;10]]])
    @views SchoolOutbreak.updateArray.(schooldata[1].parameters.rcom,parms[11])
    @views SchoolOutbreak.updateArray(schooldata[1].parameters.suscoef,length(parms)>12&&parms[12:15])
    @views SchoolOutbreak.updateArray(schooldata[1].parameters.infcoef,parms[end-(length(parms)-12)÷2:end])
    -SchoolOutbreak.llfunc!(schooldata)
end
@time @show ll(fill(-1.,20),schooldata)

# Baseline
parmlen=11+length(sus_covlabels)+length(inf_covlabels)

# No covariate effects
#parmlen=12

# MLE
#@time opt=optimize(ll,fill(-20.0,parmlen),fill(0.0,parmlen),fill(-4.0,parmlen))
#exp.(opt.minimizer)
# -

# ## MCMC

using Mamba
function schoolmodel(schooldata,
        usecovars=(sus=zeros(Bool,length(sus_covlabels)),inf=[zeros(Bool,length(inf_covlabels)-1);true]),
        ;likelihood! = SchoolOutbreak.llfunc!,
        regularize=(x)->0.0,
        invtemp=1.0,
        dimγ=3,dimδ=3,
        γfix=1.0,δfix=1.0)
    
    dotter=[100]
    inputs=Dict{Symbol,Any}(
        :schooldata=>schooldata,
        #:parameters=>parameters,
        :zerotrick=>0.0,
        :invtemp=>invtemp,
        :counter=>([0],dotter),
        :usecovars=>usecovars
    )
    parms=Dict{Symbol,Any}(
        :logβs=>.-ones(4).*5,
        :γs=>ones(dimγ)./2,
        :δs=>ones(dimδ)./2,
        :logrcom=>fill(-4.0),
        :suscoef=>zeros(sum(usecovars.sus)),
        :infcoef=>zeros(sum(usecovars.inf))
    )
    if(length(parms[:suscoef])==0) push!(inputs,:suscoef=>pop!(parms,:suscoef)) end
    if(length(parms[:infcoef])==0) push!(inputs,:infcoef=>pop!(parms,:infcoef)) end
    if(length(parms[:γs])==0) push!(inputs,:γs=>push!(pop!(parms,:γs),γfix)) end
    if(length(parms[:δs])==0) push!(inputs,:δs=>push!(pop!(parms,:δs),δfix)) end
    priors=Dict{Symbol,Any}()
    for parname in keys(parms)
        priors[parname]=Stochastic(length(size(parms[parname])),()->Uniform(-20,20))
    end

    inits=merge(parms,inputs)
    inits=[inits]

    model=Model(
        βs=Logical(1,(logβs)->exp.(logβs)),
        rcom=Logical((logrcom)->exp.(logrcom)),
        llfunc=Logical((βs,γs,δs,rcom,suscoef,infcoef,usecovars,schooldata)->begin
            parameters=schooldata[1].parameters
            SchoolOutbreak.updateArray.(parameters.γ,[γs[1];γs.+zeros(3);γs[end]]) # to enable a single common γs
            SchoolOutbreak.updateArray.(parameters.δ,[γs[1];δs.+zeros(3);δs[end]]) # to enable a single common γs
            SchoolOutbreak.updateArray.(parameters.β,[0;βs])
            SchoolOutbreak.updateArray.(parameters.rcom,rcom)
            SchoolOutbreak.updateArray(parameters.suscoef[usecovars.sus],suscoef)
            SchoolOutbreak.updateArray(parameters.suscoef[.!usecovars.sus],0.0)
            SchoolOutbreak.updateArray(parameters.infcoef[usecovars.inf],infcoef)
            SchoolOutbreak.updateArray(parameters.infcoef[.!usecovars.inf],0.0)
            ll=likelihood!(schooldata)
            ll-regularize(sum((suscoef).^2)+sum((infcoef).^2))
        end, false),
        llvalue=Logical((llfunc,invtemp)->llfunc*invtemp),
        zerotrick=Stochastic((llvalue)->Poisson(-llvalue),false),
        count=Logical((counter,rcom)->0);
        priors...
    )
    countup=Sampler([:count],
        (count,counter)->begin
            counter[1].+=1
            if counter[1][1]==counter[2][1]
                counter[1].=0
                print(".")
            end
            count+=1
        end
    )
    setsamplers!(model,[AMM(collect(keys(parms)),Matrix{Float64}(I,fill(length(reduce(vcat,values(parms))),2)...).*0.0005),countup])
    return(model=model,inputs=inputs,inits=inits)
end

# +
usecovars=(sus=ones(Bool,length(sus_covlabels)),inf=ones(Bool,length(inf_covlabels))) # include all covars
schoolmodel1 = schoolmodel(schooldata[Not([3,5,21])],usecovars) # exclude school 3, 5, 21
mcmclen=200000
burn=div(mcmclen,2)

chain1 = mcmc(schoolmodel1..., mcmclen, burnin=burn, thin=max(1,div(mcmclen,2000)), chains=1, verbose=true);
# -

Base.exp(chain::ModelChains)=begin newchain=deepcopy(chain);newchain.value.=exp.(chain.value);newchain end
showparam=[:logβs,:γs,:logrcom,:infcoef]
@show Mamba.draw(Mamba.plot(chain1[:,showparam,:]))
describe(exp(chain0[:,[:logβs,:logrcom,:infcoef],:]))
describe(chain1[:,[:γs,:llvalue],:])

write("Matsumoto_chain1.jls", chain1)
