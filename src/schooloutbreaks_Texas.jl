using CSV
using DataFrames
using Dates
using JLD2
using Plots
using Plots.PlotMeasures
using StatsBase
@load "../output/outbreaksizes_delta.jld2" # load obsizes_baseline
ts=CSV.read("../data/texas_studentcases.csv",DataFrame)
df=CSV.read("../data/texas_schooldata_AugtoNov2021.csv",DataFrame)
df_el=filter(row->!ismissing(row.CampusName) && !ismissing(row.TotalSchoolEnrollment) &&
    split(row.CampusName," ")[end] in ("EL", "ELEMENTARY") &&
    !(tryparse(Int,row.TotalSchoolEnrollment)|>isnothing) && tryparse(Int,row.TotalSchoolEnrollment)>0, df)
first(df_el,5)

gr(fontfamily="Helvetica",foreground_color_legend = nothing,background_color_legend = nothing, titlefontsize=9,guidefontsize=9,grid=false, tick_direction=:out,)
incidence=bar(ts[:,1],ts[:,2],ylim=(0,1e5),label=:none,linealpha=0,color=ifelse.(in.(ts[:,1],Ref(Date("2021-8-2"):Day(1):Date("2021-11-21"))),2,1),ylabel="weekly incidence")

function parseIntStar(x)
    if(x==" ") return(0) end
    if(x=="*") return(rand(1:4)) end
    Base.Fix1(parse,Int)(x)
end
@show mean(df_el.TotalSchoolEnrollment.|>parseIntStar)
@show quantile(df_el.TotalSchoolEnrollment.|>parseIntStar,(0.25,0.75))
histogram(df_el.TotalSchoolEnrollment.|>parseIntStar)

cases=df_el.TotalStudentCases.|>parseIntStar
denoms=df_el.TotalSchoolEnrollment.|>parseIntStar
atkrates=cases[cases.>0]./denoms[cases.>0];

obmeans=mean.(obsizes_baseline[1,:])./480
obups=quantile.(obsizes_baseline[1,:]./480,Ref([0.975,0.995]))
texmeans=mean(atkrates)
texups=quantile(atkrates,[0.975,0.995])
obmetrics=[vcat.(obmeans,obups);[[texmeans;texups]]].|>reverse
obmetrics=reduce(hcat,obmetrics)|>permutedims
bars=bar(obmetrics[:,2:3],color=[600,600,600, 8],palette=:matter,xticks=(1:4,["baseline" "symptom\nscreening" "regular tests" "Texas schools"]),
label=:none,yerror=hcat((Tuple.(vcat.(Ref([0.0]),obmetrics[:,1].-obmetrics[:,2]))),fill((0.,0.),4)),ms=0,ylim=(0,0.7),
ylabel="proportion infected")
