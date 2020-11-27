module hetLK
struct VIterable
    upp::Array{Int}
    low::Array{Int}
    function VIterable(upp::Array{Int},low=fill(0,size(upp)))
        if length(upp)!=length(low) error("upp and low has to have the same length") end
        new(upp,low)
    end
end

function viterate(upp::Vector{Int},low=fill(0,size(upp)))
    Iterators.product(range.(low,upp,step=1)...)
end

# Vector Iterator
function Base.iterate(iter::VIterable,state=(iter.low,1))
    el, count = state
    if count>prod(iter.upp.-iter.low.+1) return nothing end
    nextel=copy(el)
    nextel[1]+=1
    j=1
    while j<length(nextel)
        if nextel[j]>iter.upp[j]
            nextel[j+1]+=1
            nextel[j]=iter.low[j] 
        else
            break
        end
        j+=1
    end   
    return (el, (nextel,count+1))
end
Base.length(iter::VIterable)=prod(iter.upp.-iter.low.+1)
Base.eltype(iter::VIterable)=Array{Int}

using Distributions
using LinearAlgebra
using StatsFuns
function LKlatent(i::Array{Int,1},n::Array{Int,1},lambda::Array{Float64,1},Rmat::Array{Float64,2},gamma::Float64,p::Array{Float64,1},memo::Dict{Pair{Vector{Int},Vector{Float64}},Float64},linememo::Dict{Pair{Vector{Int},Vector{Int}},Float64})
    if all(p.==1)
        return LK1(i,n,lambda,Rmat,gamma,memo,linememo)
    else
        ll=0.0
        p=fill(0.0,size(i)).+p
        for lat_i in VIterable(n,i)
            ll+=LK1(lat_i,n,lambda,Rmat,gamma,memo,linememo)+sum(logpdf.(Binomial.(lat_i,p),i))
            return ll
        end
    end
end

function LKlatent(i::SubArray{Int,1},n::SubArray{Int,1},lambda::Array{Float64,1},Rmat::Array{Float64,2},gamma::Float64,p::Array{Float64,1},memo::Dict{Pair{Vector{Int},Vector{Float64}},Float64},linememo::Dict{Pair{Vector{Int},Vector{Int}},Float64})
    if all(p.==1)
        return LK1(i,n,lambda,Rmat,gamma,memo,linememo)
    else
        ll=0.0
        p=fill(0.0,size(i)).+p
        for lat_i in VIterable(n,i)
            ll+=LK1(lat_i,n,lambda,Rmat,gamma,memo,linememo)+sum(logpdf.(Binomial.(lat_i,p),i))
            return ll
        end
    end
end

function LKnn(n::Array{Int,1},lambda::Array{Float64,1},Rmat::Matrix{Float64},nfactor::Array{Float64,1},gamma::Float64,memo::Dict{Pair{Vector{Int},Vector{Float64}},Float64})
	if haskey(memo,n=>nfactor) return memo[n=>nfactor] end
    if all(n.==0) return 0.0 end
	
	#foi=Rmat*k.*nfactor.+lambda
	
    deduct = [sum(log.(binomial.(n,k)))-sum((n-k).* (Rmat*k.*nfactor.+lambda) )+LKnn(k,lambda,Rmat,nfactor,gamma,memo) for k in VIterable(n) if k!=n]
    lpr=log1mexp(logsumexp(deduct))
	memo[n=>nfactor]=lpr
    return(lpr)
end

function LK1(i::Array{Int,1},n::Array{Int,1},lambda::Array{Float64,1},Rmat::Array{Float64,2},gamma::Float64,memo::Dict{Pair{Vector{Int},Vector{Float64}},Float64},linememo::Dict{Pair{Vector{Int},Vector{Int}},Float64})
    if haskey(linememo,n=>i) return linememo[n=>i] end
    Neff=(Rmat*n-diag(Rmat).*min.(n,1))/(sum(Rmat)/size(Rmat,1))
    nfactor=1.0./Neff.^gamma
	nfactor[isnan.(nfactor)].=0.0
    foi=Rmat*i.*nfactor.+lambda
    rtn=sum(log.(binomial.(n,i)))-sum((n-i).*foi)+LKnn(i,lambda,Rmat,nfactor,gamma,memo)
    linememo[n=>i]=rtn
    return rtn
end

function ll(i::Array{Int,2},n::Array{Int,2},lambda::Array{Float64,1},Rmat::Array{Float64,2},gamma:: Float64=0.0,p::Array{Float64,1}=[1.0])
	memo=Dict{Pair{Vector{Int},Vector{Float64}},Float64}() #memoize LKnn by n => nfactor
    linememo=Dict{Pair{Vector{Int},Vector{Int}},Float64}()
    if size(i)!=size(n) error("sizes of n and i don't match") end
    if !(size(i,1)==size(lambda,1)==size(Rmat,1)==size(Rmat,2)) error("the number of classes don't match") end
    llik=0.0
    for c in 1:size(n,2)
        llik+=LKlatent(i[:,c],n[:,c],lambda,Rmat,gamma,p,memo,linememo)
    end
    return llik
end
end
