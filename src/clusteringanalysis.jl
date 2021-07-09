using CSV, JLD2, DataStructures, Statistics, Plots
include("src/SchoolOutbreak.jl")
@load "data/anonymizedstudents.jld2"

function temporalpatterns(students)
    strata=(student->getfield.(Ref(student),(:schoolID,:gradeID,:classID)).|>first).(filter(x->x.isinfected,students))
    onsets=getfield.(students,:onset)
    stindex=unique(strata)|>sort
    OrderedDict(stindex.=>[onsets[findall(x->st==x,strata)]|>sort for st in stindex])
end
@time onsetbystratum=temporalpatterns(anonymizedstudents)

gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,legendfontsize=9,guidefontsize=10,grid=false, tick_direction=:out)
function onsetbylevel(onsetbystratum, level,origin=mean)
    ukeys=unique(getindex.(keys(onsetbystratum),Ref(1:level)))
    vectors=(vcat((filter(x->x[1][1:level]==ukey,onsetbystratum)|>values)...) for ukey in ukeys)
    filter(x->x!=0,vcat(((v->v.-origin(v)).(vectors))...))
end
devonsets=onsetbylevel.(Ref(onsetbystratum),0:3)
plot(((v,c)->histogram(v,xlim=(-100,100),color=c,normalize=true)).(devonsets,[6,9,:skyblue,1])...,size=(550,550),layout=(4,1),titles=["all" "within-school" "within-grade" "within-class"],xlabel=[fill("",1,3) "deviation of onset dates from mean"],labels="SD = ".*string.(round.(std.(devonsets)',digits=1)))

        
## Clustering simulation
using JLD2
@load "Matsumoto_studentsdata.jld2" students
duringbreak=(d2di(Date("2014-12-27")):d2di(Date("2015-1-7"))).+1 # 27 Dec - 7 Jan was the final day of school in Matsumoto, shifted one day to account for presymptomatic infection
@time simstudents=SchoolOutbreak.Students.([filter(x->x.stratum[1]==schoolid,students) for schoolid in (1:29)[Not([3,5,21])]],Ref(parameters),[Dict(duringbreak=>duringbreak#=,duringclosure=duringclosure[x]=#) for x in (1:29)[Not([3,5,21])]]);
chain1=CSV.read("output/Matsumoto_MCMCsamples.csv",DataFrame)

function parametersfromchain(chain, iter)
    (β=SchoolOutbreak.posterior(chain,:βs,iter).samples[:,[1;1:4]]|>vec,
    γ=SchoolOutbreak.posterior(chain,:γs,iter).samples[:,[1;1:3;3]]|>vec,
    δ=SchoolOutbreak.posterior(chain,:δs,iter).samples[:,[1;1:3;3]]|>vec,
    rcom=SchoolOutbreak.posterior(chain,:rcom,iter).samples|>first|>fill,
    suscoef=SchoolOutbreak.posterior(chain,:suscoef,iter).samples|>vec,
    infcoef=SchoolOutbreak.posterior(chain,:infcoef,iter).samples|>vec)
end
function parametersfromchain(chain::DataFrame, iter)
    chainmat=Matrix(chain)
    (β=(chainmat[iter,[2;2:5]]|>vec) .|>exp,
    γ=chainmat[iter,[6;6:8;8]]|>vec,
    δ=chainmat[iter,[9;9:11;11]]|>vec,
    rcom=((chainmat[iter,1])|>exp).*1,
    suscoef=chainmat[iter,12:15]|>vec,
    infcoef=chainmat[iter,16:20]|>vec)
end
@time updateparameters=parametersfromchain.(Ref(chain1),(1:50).*20);
Random.seed!(2021)
@time outbreaks=SchoolOutbreak.simulateoutbreaks!.(Ref(simstudents), updateparameters, Ref(1:250),initcases=0);
incidences=SchoolOutbreak.incidence(outbreaks);

# plot group-level deviation of onset dates
function onset(outbreak,students)
    onsetpairs=((ob,st)->reduce(vcat,(Tuple.(getindex.(getfield.(st.members[ids],:stratum),Ref(1:3))).=>date) for (date, ids) in enumerate(ob))).(outbreak,students) # Pairs: stratum => onsetdate
    onsetpairs=vcat(onsetpairs...)
    ukeys=unique(first.(onsetpairs))|>sort
    Dict(ukeys.=>[filter(x->first(x)==ukey,onsetpairs).|>last for ukey in ukeys])
end
onsets=[onset(outbreaks[n],simstudents) for n in 1:50]
gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,legendfontsize=9,guidefontsize=10,grid=false, tick_direction=:out)
#=joinbylevel(v::Array{<:Array},level)=level==0 ? [joinbylevel(vcat(v...),0)] : joinbylevel.(v,level-1)
joinbylevel(v::Array{<:Number},level)=v
function onsetbylevel(simonsets, level,origin=mean)
    vectors=joinbylevel(simonsets,level)
    vcat(((v->v.-origin(v)).(vectors))...)
    end=#
function onsetbylevel(onsetbystratum, level,origin=mean)
    ukeys=unique(getindex.(keys(onsetbystratum),Ref(1:level)))
    vectors=[vcat((filter(x->x[1][1:level]==ukey,onsetbystratum)|>values)...) for ukey in ukeys]
    vcat(((v->v.-origin(v)).(filter(x->length(x)>1,vectors)))...)
end
devonsets=vcat.((onsetbylevel.(Ref(onsets[n]),0:3) for n in (1:50).*1)...)
Plots.plot(((v,c)->histogram(v,xlim=(-100,100),color=c,normalize=true)).(devonsets,[6,9,:skyblue,1])...,size=(550,550),layout=(4,1),titles=["all" "within-school" "within-grade" "within-class"],xlabel=[fill("",1,3) "deviation of onset dates from mean"],labels="SD = ".*string.(round.(std.(devonsets)',digits=1)))
