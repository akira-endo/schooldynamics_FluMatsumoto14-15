include("src/SchoolOutbreak.jl")

using Distributions, Plots, InvertedIndices, LinearAlgebra, StatsBase, Random
gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,grid=false, tick_direction=:out)

# setup
struct ImproperUniform<: Distribution{Univariate,Discrete} end
Distributions.cdf(d::ImproperUniform, x::Integer)=1.0
Distributions.pdf(d::ImproperUniform, x::Integer)=1.0

rcomdist=Logistic(113.63,10.15)#Uniform(0,365)
hhfoi=-Inf
Gammameansd(mean,sd)=Gamma(mean^2/sd^2,sd^2/mean)
#gtimedist=Gammameansd(3,1)
gtimedist=Gammameansd(1.7,1) # H3N2 serial interval from Vink et al 2014 (https://doi.org/10.1093/aje/kwu209)
N0=[1,1,1,1,1].*5 # normalizing constant to avoid correlation between β and γ
C0=[0,30,30,30,30]

sus_covlabels=[:Grade, :Vaccine_curr, :NPI_mask,:NPI_wash]
inf_covlabels=[:Grade, :Vaccine_curr, :NPI_mask,:NPI_wash,:duringbreak]

classes=repeat([20,30,40],inner=100)
sizes=fill(5,300)
nclassmat=repeat(classes,1,1)
nstudentsmat=repeat(sizes,1,1).*nclassmat
strata=vcat.(1:300,sortslices.(repeat.((x->reduce(hcat,vec(collect.(x)))).(Iterators.product.(Ref(1:1),range.(1,classes,step=1)).|>collect),1,sizes),dims=2))
strata=vcat.(strata,repeat.(Ref([1]),1,size.(strata,2)))
strata=reduce(hcat,strata)

parameters=SchoolOutbreak.makeparameters(ones(5),ones(5)/2,ones(5)/2,0.02,
    ones(size(sus_covlabels)),ones(size(inf_covlabels)),gtimedist,20,rcomdist,strata,nclassmat,nstudentsmat,N0,C0,Inf)
@time students=SchoolOutbreak.studentsfromdata(parameters,false,fill(typemax(Int16)|>Int,size(strata,2)),eachcol(strata),
    Ref(zeros(length(sus_covlabels))),Ref(zeros(length(inf_covlabels))),nclassmat,nstudentsmat,-Inf,1);
@time simstudents=SchoolOutbreak.Students.([filter(x->x.stratum[1]==schoolid,students) for schoolid in 1:300],Ref(parameters));
# Ovetwrite beta
for schoolid in 1:length(simstudents)
    setfield!.(simstudents[schoolid].pairs[1:5:end,1:5:end],:pairwiseβ,Ref(simstudents[1].parameters.β[4]))
    setfield!.(simstudents[schoolid].pairs[1:5:end,1:5:end],:δ,Ref(simstudents[1].parameters.δ[4]))
end

#parameter settings
## infection hazard
###### infection hazard
using RCall
@rimport distr

## COVID-19
#incperiod=distr.Lnorm(1.434065,0.6612) # He et al.
#infprof=distr.Norm(0.53,sqrt(7)) # Aschroft et al.
#gtime=incperiod+infprof

## Pandemic influenza
incperiod=distr.Lnorm(0.34,0.42) # Lessler et al. 2009 (https://dx.doi.org/10.1016%2FS1473-3099(09)70069-6)
gtime=((mean,sd)->distr.Gammad(mean^2/sd^2, sd^2/mean))(1.7,1)
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
gtimes=[hazards_symsc[:,1]]#,hazards_symsc[:,3], hazards_randsc[3][:,2]]|>permutedims # no intervention, symptom screening, symptom+test


# Outbreak simulation
## Functions
incidence(outbreak)=length(outbreak)
incidence(outbreak::AbstractArray{<:AbstractVector})=incidence.(outbreak)
incidence(outbreak::AbstractArray{<:AbstractMatrix})=sum.(outbreak)

function outbreakscenarios!(simstudents,updateparameters,gtime)
    setgtime!(simstudents[1].parameters,gtime)
    outbreaks=SchoolOutbreak.simulateoutbreaks!.(Ref(simstudents), updateparameters, Ref(1:360))
    outbreaks#readoutbreakbyclass_reorder.(outbreaks,Ref(simstudents),20,Ref([1]))
end

include("src/hetLK.jl")
function hh_lkhratio(i,n,diOnset,rcomdist,rcomdays,rcom=0.02)
    sumi=sum(i)
    if sumi==1||i[1]==0 return(-Inf) end
    εh=-log.(1 .- rcom)
    εh*=rcomdays*pdf(rcomdist,diOnset)
    R = ones(2,2).*0.1
    ll_notfromschool = hetLK.ll(i,n,εh.*[0.0;ones(1)],R,0.)
    ll_fromschool = hetLK.ll(i,n,[1e20;εh.*ones(1)],R,0.) #+ log(εh[1])
    return(ll_notfromschool-ll_fromschool)
end

function ll(lparms,schooldata=householddata)
    parms=exp.(copy(lparms))
    @views SchoolOutbreak.updateArray!.(schooldata[1].parameters.β,[0;0;parms[1];0;0])
    @views SchoolOutbreak.updateArray!.(schooldata[1].parameters.γ,zeros(5))
    @views SchoolOutbreak.updateArray!.(schooldata[1].parameters.δ,zeros(5))
    @views SchoolOutbreak.updateArray!(schooldata[1].parameters.rcom,parms[2])
    #@views SchoolOutbreak.updateArray!(schooldata[1].parameters.suscoef,length(parms)>12&&parms[12:15])
    #@views SchoolOutbreak.updateArray!(schooldata[1].parameters.infcoef,parms[end-(length(parms)-12)÷2:end])
    -SchoolOutbreak.llfunc!(schooldata)
end

# repeated simulation
using Optim, LinearAlgebra, Random, Calculus
function repeatestimate(students,parameters, updateparameters,gtimes,assumedrcom=0.02,rcomdays=3,len=50)
    repeatedests=Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}[]
    for x in 1:len
        Random.seed!(x)
        #SchoolOutbreak.updateArray!.(getfield.(students[1:5:end],:HHllkh),-Inf)
        outbreakscenarios!.(Ref(simstudents),Ref(updateparameters),gtimes);
        householddat=SchoolOutbreak.StudentsData.([filter(x->x.stratum[1]==schoolid,students[1:5:end]) for schoolid in 1:300],Ref(parameters),Ref(Dict()));
        for data in householddat data.βmat.=Ref(data.parameters.β[4]) end
        for data in householddat data.δmat.=Ref(data.parameters.δ[4]) end

        hhfoi=[hh_lkhratio([Int(students[n].isinfected) sum(getfield.((@view students[n.+(1:4)]),:isinfected))]|>permutedims,[1 4]|>permutedims,students[n].onset,rcomdist,rcomdays,assumedrcom) for n in 1:5:length(students)]; # 3 days window period
        SchoolOutbreak.updateArray!.(getfield.(students[1:5:end],:HHllkh),hhfoi)  
        for n in 1:length(householddat), member in householddat[n].members
            member.cdfcommunity.=cdf(rcomdist,member.onset)
            member.pdfcommunity.=pdf(rcomdist,member.onset)
        end
print("$x,")
        @time opt=optimize(p->ll(p,householddat),fill(-5.,3),fill(1.,3),fill(-1.,3))
        hess=Calculus.hessian(p->ll(p,householddat),opt.minimizer)
print(eigen(hess))
        sd=1.96.*sqrt.(diag(inv(hess)))
        #=βnorm=Normal(0.04,0.005)
        rcomnorm=Normal(0.02,0.001)
        ns=10000
        candidatepars=zip(log.(abs.(rand(βnorm,ns))),log.(abs.(rand(rcomnorm,ns))))
        lls=[ll(pars|>collect,householddata)-logpdf(βnorm,pars[1])-logpdf(βnorm,pars[1]) for pars in candidatepars]
        pos=collect(candidatepars)[sample(1:ns,weights(exp.(lls.-logsumexp(lls))),ns)]
        qs=quantile.([first.(pos).|>exp,last.(pos).|>exp],Ref([0.5,0.025,0.975]))=#
        push!(repeatedests,[vec((opt.minimizer.|>exp).*hcat(ones(3),exp.(sd),exp.(.-sd)));minimum(eigen(hess).values)]|>Tuple)
    end
    repeatedests
end
function estplot(repeatedests,colors)
    plots=[plot((1:50).+[0 0.33 0.66][:,1:size(repeatedests,2)],getindex.(repeatedests,1),ylim=(0.0,0.1),yerror=zip(getindex.(repeatedests,4).-getindex.(repeatedests,1),.-getindex.(repeatedests,7).+getindex.(repeatedests,1))|>collect,linewidth=0,markerstrokecolor=colors,color=colors,marker=:circle,markersize=1.5),
    plot((1:50).+[0 0.33 0.66][:,1:size(repeatedests,2)],getindex.(repeatedests,2),ylim=(0,1),yerror=zip(getindex.(repeatedests,5).-getindex.(repeatedests,2),.-getindex.(repeatedests,8).+getindex.(repeatedests,2))|>collect,linewidth=0,marker=:circle,markerstrokecolor=colors,color=colors,markersize=1.5),
    plot((1:50).+[0 0.33 0.66][:,1:size(repeatedests,2)],getindex.(repeatedests,3),ylim=(0,0.05),yerror=zip(getindex.(repeatedests,6).-getindex.(repeatedests,3),.-getindex.(repeatedests,9).+getindex.(repeatedests,3))|>collect,linewidth=0,marker=:circle,markerstrokecolor=colors,color=colors,markersize=1.5)]
    hline!.(plots,[[0.05],[0.5],[0.02]],color=:black,linestyle=:dash,label="")
end


## Simulation set 1
updateparameters=fill((β=[0,0,0,0.05,0.1], γ=zeros(5), δ=[0,0,0,0.5,0],
    rcom=0.02, suscoef=ones(4), infcoef=ones(5)),1)
repeatests1=repeatestimate.(Ref(students),Ref(parameters),Ref(updateparameters),Ref(gtimes),[0.02,0.02,0.,0.02],[1,3,3,7],50);
plot([plot(estplot(repeatests1[1],:gray40)...,title=["within-school transmissibility" "exponent parameter" "risk of infection from community"],layout=(1,3));
plot(estplot(reduce(hcat,repeatests1[[3,4]]),[1 2 3])...,xlabel="simulation runs",guidefontsize=9,layout=(1,3))]...,layout=(2,1),size=(650,400),legend=:none)


## Simulation set 2
updateparameters=fill((β=[0,0,0,0.08,0.1], γ=zeros(5), δ=[0,0,0,0.8,0],
    rcom=0.05, suscoef=ones(4), infcoef=ones(5)),1)
repeatests2=repeatestimate.(Ref(students),Ref(parameters),Ref(updateparameters),Ref(gtimes),[0.05,0.05,0.,0.05],[1,3,3,7],50);

plot([plot(estplot(repeatests2[1],:gray40)...,title=["within-school transmissibility" "exponent parameter" "risk of infection from community"],layout=(1,3));
plot(estplot(reduce(hcat,repeatests2[3:4]),[1 2 3])...,xlabel="simulation runs",guidefontsize=9,layout=(1,3))]...,layout=(2,1),legend=:none,size=(650,400))
    
## Save results
using JLD2
@save "repeatests.jld2" repeatests1 repeatests2