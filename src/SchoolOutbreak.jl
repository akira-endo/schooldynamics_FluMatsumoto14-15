module SchoolOutbreak
using Distributions, LinearAlgebra, StatsFuns
Base.isless(n::Nothing,f)=false;Base.isless(f, n::Nothing)=true;Base.isless(n1::Nothing, n2::Nothing)=false
# utils
updateArray!(x::A,y) where A<:AbstractArray =begin x.=y end
updateArray!(x::A,y) where A<:AbstractArray{<:AbstractArray} =begin updateArray!.(x,y) end
interval(x,lo=-Inf,up=Inf)=min(max(x,lo),up)

function qfindall(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        @inbounds if f(a[i])
            b[j] = i
            j += 1
        end
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

function getstdistance(from,to)
    Int.(to.stratum.==from.stratum) |> cumprod |> sum |> x->x+one(x) # look for the match level betweeen stX and stY
end
function getstratum(student)
    student.stratum
end
function getlogNbylevels(stratum, nclassmat, nstudentsmat, denom=ones(5)) 
    begin
        N=Vector{Float64}(undef,5)
        N[1]=sum(nstudentsmat)/denom[1] |> log # outside school
        N[2]=(sum(@view nstudentsmat[stratum[1],:])/sum(nclassmat[stratum[1],:]))/denom[2] |> log # different grade
        N[3]=nstudentsmat[stratum[1:2]...]/nclassmat[stratum[1:2]...]/denom[3] |> log # different class same grade
        N[4] = N[5] = nstudentsmat[stratum[1:2]...]/nclassmat[stratum[1:2]...]/denom[4] |> log # same class
        N .|>fill
    end
end
function getlogMbylevels(stratum, nclassmat, nstudentsmat, denom=ones(5)) 
    begin
        N=Vector{Float64}(undef,5)
        N[1]=sum(nstudentsmat)/denom[1] |> log # outside school
        N[2]=(sum(@view nclassmat[stratum[1],:]))/denom[2] |> log # different grade
        N[3]=nclassmat[stratum[1:2]...]/denom[3] |> log # different class same grade
        N[4] = N[5] = nclassmat[stratum[1:2]...]/denom[4] |> log # same class
        N .|>fill
    end
end
function linregval(covar,coef)
    exp(dot(first.(covar),first.(coef)))
end

## Estimation
# Types and related functions
Scalar{T}=Array{T,0}
AbstractScalar{T}=AbstractArray{T,0}
mutable struct
    Student{I<:Integer, R<:Real, VI<:AbstractVector,VR<:AbstractVector,SR<:AbstractScalar}
    isinfected::Bool
    onset::I
    stratum::VI
    suscovar::VR
    infcovar::VR
    logN::Vector{SR}
    logM::Vector{SR}
    HHllkh::SR
    susceptibility::SR
    infectiousness::SR
    pdfcommunity::SR
    cdfcommunity::SR
    βs::Vector{SR}
    γs::Vector{SR}
    δs::Vector{SR}
    pdfgtime::Vector{SR}
    cdfgtime::Vector{SR}
    sampleweight::R
end
function studentsfromdata(parameters,isinfecteds,onsets,strata,
            suscovars,infcovars,nclasses,nstudents,HHlkhs,samplingrate)
    logNs=getindex.(Ref(parameters.logN),strata)
    logMs=getindex.(Ref(parameters.logM),strata)
    students=Student.(isinfecteds,onsets,strata,suscovars,infcovars,logNs,logMs, fill.(HHlkhs),
        fill.(linregval.(suscovars,Ref(parameters.suscoef))),fill.(linregval.(infcovars,Ref(parameters.infcoef))),
        fill.(cdf.(parameters.rcomdist,onsets).-cdf.(parameters.rcomdist,onsets.-1)), fill.(cdf.(parameters.rcomdist,onsets.-1)),
        Ref(parameters.β), Ref(parameters.γ),Ref(parameters.δ), Ref(parameters.pdfgtime), Ref(parameters.cdfgtime), 1 ./samplingrate)
end
struct StudentsData{VStudent,NT1,NT2,MR<:Matrix,VR<:Vector,V<:Vector,M<:Matrix}
    members::VStudent
    parameters::NT1
    data::NT2
    isinfected::Vector{Bool}
    βmat::MR
    γmat::MR
    δmat::MR
    logNmat::MR
    logMmat::MR
    pdfgtimemat::MR
    cdfgtimemat::MR
    pdfcommunity::VR
    cdfcommunity::VR
    infectiousness::VR
    susceptibility::VR
    HHllkh::VR
    sampleweight::V
    temps::NamedTuple{(:cdfmat,:pdfmat,:pdfvec),Tuple{M,M,V}}
    StudentsData(members, parameters = NamedTuple(), data = NamedTuple())=new{typeof(members),typeof(parameters),typeof(data),Matrix{Scalar{Float64}},Vector{Scalar{Float64}},Vector{Float64},Matrix{Float64}}(members, parameters, data, makemats(members)...)
end
function makemats(members::T where T<:Vector{<:Student})
    (isinfected=[from.isinfected for from in members],
    βmat=[to.βs[getstdistance(from,to)] for from in members, to in members],
    γmat=[to.γs[getstdistance(from,to)] for from in members, to in members],
    δmat=[to.δs[getstdistance(from,to)] for from in members, to in members],
    logNmat=[to.logN[getstdistance(from,to)] for from in members, to in members],
    logMmat=[to.logM[getstdistance(from,to)] for from in members, to in members],
    pdfgtimemat=[from.pdfgtime[interval(to.onset-from.onset+1,1,length(from.pdfgtime))] for from in members, to in members],
    cdfgtimemat=[from.cdfgtime[interval(to.onset-from.onset+1,1,length(from.cdfgtime))] for from in members, to in members],
    pdfcommunity=[to.pdfcommunity for to in members],
    cdfcommunity=[to.cdfcommunity for to in members],
    infectiousness=[from.infectiousness for from in members],
    susceptibility=[to.susceptibility for to in members],
    HHllkh=[to.HHllkh for to in members],
    sampleweight=[to.sampleweight for to in members],
    temps=(cdfmat=[0.0 for from in members[getfield.(members,:isinfected)], to in members],
            pdfmat=[0.0 for from in members[getfield.(members,:isinfected)], to in members[getfield.(members,:isinfected)]],
            pdfvec=[0.0 for from in members[getfield.(members,:isinfected)]]))
end

# main functions
function makeparameters(βs,γs,δs,rcom,suscoef,infcoef,gtimedist,gtimemax,rcomdist,strata,nclasses,nstudents,Ndenom,Mdenom,overdispersion)
    uniquestrata=unique(eachcol(strata))
    logN=getlogNbylevels.(uniquestrata,Ref(nclasses),Ref(nstudents),Ref(Ndenom))
    dlogN=Dict{eltype(uniquestrata),eltype(logN)}(uniquestrata.=>logN)
    logM=getlogMbylevels.(uniquestrata,Ref(nclasses),Ref(nstudents),Ref(Mdenom))
    dlogM=Dict{eltype(uniquestrata),eltype(logM)}(uniquestrata.=>logM)
    β=fill.(βs)
    γ=fill.(γs)
    δ=fill.(δs)
    pdfgtime=[0;cdf.(gtimedist,1:gtimemax).-cdf.(gtimedist,0:gtimemax-1)]
    cdfgtime=[0;cdf.(gtimedist,0:gtimemax-1)]
    push!(pdfgtime,0.0);push!(cdfgtime,1.0) # add 0 to the end
    rcom=fill.(rcom)
    (β=β, γ=γ,δ=δ, rcom=rcom,suscoef=suscoef,infcoef=infcoef, pdfgtime=fill.(pdfgtime),cdfgtime=fill.(cdfgtime), rcomdist=fill(rcomdist),logN=dlogN,logM=dlogM,Ndenom=Ndenom,Mdenom=Mdenom,overdispersion=fill.(overdispersion))
end
function ll!(studentsdata::StudentsData)
    updatesusinf!.(studentsdata.members, Ref(studentsdata.parameters)) # reflect covariates
    
    @views studentsdata.temps.cdfmat.=first.(studentsdata.βmat[studentsdata.isinfected,:]).*exp.(.-first.(studentsdata.γmat[studentsdata.isinfected,:]).*first.(studentsdata.logNmat[studentsdata.isinfected,:]).-first.(studentsdata.δmat[studentsdata.isinfected,:]).*first.(studentsdata.logMmat[studentsdata.isinfected,:])) # βN⁻ᵞ
    @views studentsdata.temps.pdfmat.=studentsdata.temps.cdfmat[:,studentsdata.isinfected] # copy for pdf computation
    
    @views studentsdata.temps.cdfmat.*=first.(studentsdata.cdfgtimemat[studentsdata.isinfected,:])
    
    @views ll= - first.(studentsdata.infectiousness[studentsdata.isinfected])'*(studentsdata.temps.cdfmat)*(first.(studentsdata.susceptibility).*first.(studentsdata.sampleweight)) # siβN⁻ᵞcdf
    ll-= (studentsdata.parameters.rcom)sum(first.(studentsdata.cdfcommunity).*first.(studentsdata.susceptibility).*first.(studentsdata.sampleweight))
    
    studentsdata.temps.pdfmat.*=first.(studentsdata.pdfgtimemat[studentsdata.isinfected,studentsdata.isinfected])
    studentsdata.temps.pdfvec.= (studentsdata.temps.pdfmat)'*first.(studentsdata.infectiousness[studentsdata.isinfected])
    
    @views studentsdata.temps.pdfvec.+=(studentsdata.parameters.rcom).*first.(studentsdata.pdfcommunity[studentsdata.isinfected])
    @views studentsdata.temps.pdfvec.*=first.(studentsdata.susceptibility[studentsdata.isinfected])
    @views studentsdata.temps.pdfvec.=log.((1 .-exp.(.-studentsdata.temps.pdfvec)).*(1 .-exp.(first.(studentsdata.HHllkh[studentsdata.isinfected]))).+exp.(first.(studentsdata.HHllkh[studentsdata.isinfected])))
    
    ll+=dot(studentsdata.temps.pdfvec, first.(@view studentsdata.sampleweight[studentsdata.isinfected]))
    ll
end
function llfunc!(studentsdatasets::AS) where AS<:Array{<:StudentsData}
    sum((ll!(dataset) for dataset in studentsdatasets))
end
function updatesusinf!(student::Student,parameters::NT where NT<:NamedTuple)
    student.susceptibility.=linregval(student.suscovar,parameters.suscoef)
    student.infectiousness.=linregval(student.infcovar,parameters.infcoef)
end


## Simulation

mutable struct StudentPair{I<:Integer,SR<:AbstractScalar,VR<:AbstractVector,ST<:Student}
    isinfected::Bool
    serialinterval::I
    stratadistance::I
    γ::SR
    δ::SR
    pairwiseβ::SR
    pairwiselogN::SR
    pairwiselogM::SR
    infectiousness::SR
    susceptibility::SR
    pdfgtime::VR
    cdfgtime::VR
    from::ST
    to::ST
    StudentPair(from,to)=begin
        stdistance=getstdistance(from,to)
        new{typeof(stdistance),typeof(to.susceptibility),typeof(to.pdfgtime),typeof(from)}(
            from.isinfected,
            to.onset.-from.onset,
            stdistance,
            to.γs[stdistance],
            to.δs[stdistance],
            to.βs[stdistance],
            to.logN[stdistance],
            to.logM[stdistance],
            from.infectiousness,
            to.susceptibility,
            from.pdfgtime,
            from.cdfgtime,
            from, to
        )
    end
end

mutable struct Students{NT1<:NamedTuple,D<:Dict,ST<:Student,SP<:StudentPair}
    members::Vector{ST}
    pairs::Matrix{SP}
    parameters::NT1
    data::D
    Students(members, parameters = NamedTuple(), data = Dict())=new{typeof(parameters),typeof(data),eltype(members),typeof(StudentPair(members[1],members[1]))}(members, StudentPair.(members,permutedims(members[:,:],(2,1))), parameters, data)
end
function Rmean(students)
    size(students.pairs,1)mean((pair->pair.pairwiseβ[]*exp(-pair.pairwiselogN[]*pair.γ[]-pair.pairwiselogM[]*pair.δ[])),students.pairs)
end
function llpair(pair::StudentPair)
    first(.-pair.pairwiseβ.*exp.(.-pair.γ.*pair.pairwiselogN.-pair.δ.*pair.pairwiselogM)
        .*(@view pair.cdfgtime[interval(pair.serialinterval,1,length(pair.cdfgtime))]).*pair.infectiousness.*pair.susceptibility
  .* ifelse(pair.isinfected,(@view pair.pdfgtime[interval(pair.serialinterval,1,length(pair.cdfgtime))]),1) )
    ## sampling, hhfoi
end
function initialize!(students::Students, updateparameters=NamedTuple())
    for par in keys(updateparameters)
        updateArray!(students.parameters[par],updateparameters[par])
    end
    onsets=Vector{Union{Nothing,typeof(students.members[1].onset)}}(undef,length(students.members))
    onsets.=getfield.(students.members,:onset)
    onsets[.!getfield.(students.members,:isinfected)].=nothing
    setfield!.(students.members,:isinfected,false)
    setfield!.(students.members,:onset,typemax(Int))
    (x->x.infcovar[end]=0.0).(students.members) # remove winterbreak flag
    setfield!.(students.pairs,:isinfected,false)
    setfield!.(students.pairs,:serialinterval,typemax(Int))
    updatesusinf!.(students.members, Ref(students.parameters))
    onsets # returns the currently recorded onset dates
end
function initialcases!(students::Students, id, onset)
    setfield!.(students.members[id],:onset,onset)
    setfield!.(students.members[id],:isinfected,true)
end
function communitytransmission!(students::Students)
    infected=rand.(fill(Bernoulli(1-exp(-first(students.parameters.rcom))),length(students.members)))
    setfield!.(students.members[infected], :isinfected, true)
    setfield!.(students.members[infected],:onset,floor.(typeof(students.members[1].onset),rand(students.parameters.rcomdist[],sum(infected))))
end
function pairtransmit!(pair, parameters, data)
    # update winterbreak
    if haskey(data,:duringbreak) && (pair.from.infcovar[end]>0)!=(pair.from.onset in data.duringbreak)
        pair.from.infcovar[end]=(pair.from.onset in data.duringbreak)
        pair.from.infectiousness.*=exp(parameters.infcoef[end]*(2pair.from.infcovar[end]-1)) #updatesusinf!(pair.from, parameters)
    end
    
    if rand(Bernoulli(exp(llpair(pair)))) || pair.to.onset≤pair.from.onset 
        return end
    pair.isinfected=true
    pair.to.isinfected=true
    if(haskey(parameters,:overdispersion)&&0<parameters.overdispersion[]<Inf)
        pair.to.infectiousness.*=rand(Gamma(parameters.overdispersion[],1/parameters.overdispersion[]))
    end
    SI=1:length(pair.cdfgtime)-1   
    lls=[begin pair.serialinterval=si
        llpair(pair) end 
            for si in SI]
    lls.=exp.(lls.-logsumexp(lls))
    pair.serialinterval=rand(Categorical(lls))
    pair.to.onset=min(pair.to.onset,pair.from.onset+pair.serialinterval) # update onset
end
function transmission!(students::Students, onset::Integer, statuscheck!)
    statuscheck!(students,onset)
    infectorids=qfindall(x->getfield(x,:onset) == onset ,students.members)
    @views pairtransmit!.(students.pairs[infectorids,:], Ref(students.parameters), Ref(students.data))
    return(infectorids)
end
function transmission!(students::Students, onsets, warning, statuscheck!)
    if hasfield(typeof(statuscheck!),:closuretime) updateArray!(statuscheck!.closuretime,0) end
    incidence=transmission!.(Ref(students), onsets, statuscheck!)
    if warning && ((x->ifelse(x.isinfected, x.onset, -Inf)).(students.members) |>maximum > last(onsets)) @warn "Simulation ended halfway (time step "*string(last(onsets))*", unprocessed onset at "* ((x->ifelse(x.isinfected, x.onset, -Inf)).(students.members) |>maximum |> string) *"). Restart simulation from the following step if necessary." end
    if hasfield(typeof(statuscheck!),:closuretime)
            return((incidence=incidence, closuretime=(x->min.(1,x)).(statuscheck!.closuretime))) end
    return(incidence)
end
function posterior(chain, parameter::Symbol)
    mat=chain[:,parameter,1].value[:,:,1]
    names=chain[:,parameter,1].names
    (names=names ,samples=mat)
end
function posterior(chain, parameter::Symbol,iter)
    mat=chain[(x->first(x):last(x))(chain.range[iter]),parameter,1].value[:,:,1]
    names=chain[:,parameter,1].names
    (names=names ,samples=mat)
end
function simulateoutbreaks!(students, parameters, times; initialize=true, warning=false, statuscheck! = (x...)->nothing)
    if initialize
        onsets=SchoolOutbreak.initialize!.(students,Ref(parameters))
        seedonsets=minimum.(onsets)
        initialcases=findall.((x->!isnothing(x) && x==seedonset for seedonset in seedonsets),onsets)
        SchoolOutbreak.initialcases!.(students,initialcases,seedonsets)
        SchoolOutbreak.communitytransmission!.(students)
    end
    result=SchoolOutbreak.transmission!.(students,Ref(times), warning, statuscheck!)
    #incidences=(x->length.(x)).(result)
    #totalincidence=sum(incidences)
    result
end
function incidence(outbreak)
    length(outbreak)
end
function incidence(outbreak::AbstractArray{<:AbstractArray})
    incidence.(outbreak)
end
end
