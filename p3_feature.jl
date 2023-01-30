### A Pluto.jl notebook ###
# v0.19.18

using Markdown
using InteractiveUtils

# ╔═╡ ac97e0d6-2cfa-11ed-05b5-13b524a094e3
begin
	import Pkg
	Pkg.activate(Base.current_project())

	using Cairo
	using Fontconfig

	using CSV
	using DataFrames
	using DataStructures
	using Dates
	using Gadfly
	using GeoDataFrames
	using Statistics
	using UnicodePlots
	using YAML
end;

# ╔═╡ d75e2748-fa12-45f2-a380-e53ba1826893
using ColorSchemes

# ╔═╡ 3bf1db0d-fb5e-4212-8dac-5e9c009ad77f
using Plots

# ╔═╡ aeb3597f-ef6a-4d21-bff5-d2ad359bc1a2
using Chain

# ╔═╡ 9d5f897f-7f25-43ed-a30c-064f21e50174
using Pipe: @pipe

# ╔═╡ 786c2441-7abb-4caa-9f50-c6078fff0f56
using ArchGDAL

# ╔═╡ 1c7bfba6-5e1d-457d-bd92-8ba445353e0b
using MLJ

# ╔═╡ 5bfda8cf-487b-4930-b212-9cd1622367de
using Latexify

# ╔═╡ a95245b0-afa8-443a-9a16-b01de7294f13
using LinearAlgebra

# ╔═╡ 0486a516-1210-4f42-9eac-b78433065365
using Random

# ╔═╡ ba8c8058-7efc-48f2-b3c5-38500524856d
using EvoTrees

# ╔═╡ 01312a8c-8205-4c63-b617-84067b5c0bc2
using MLJModelInterface

# ╔═╡ 97657a35-0407-4acc-b761-e6ebc06a3764
using StatsBase

# ╔═╡ f2c3d4ac-5843-47ad-b78b-ca08c9c4df2a
using ComputationalResources

# ╔═╡ 7d312518-a642-4e28-942d-e1eb4de69121
using ShapML

# ╔═╡ 9b3790d3-8d5d-403c-8495-45def2c6f8ba
md"""
### Section 1: Data Cleansing and Loading
---
This preliminary section will first load each of the data elements used in the analysis and prepare them for merging prior to a learning pipeline
"""

# ╔═╡ bf772ea4-c9ad-4fe7-9436-9799dcb0ad04
date_f = "yyyy-mm-dd HH:MM:SS"

# ╔═╡ 020b96e3-d218-470d-b4b0-fc9b708ffdf3
begin
	sources_file = joinpath(pwd(), "sources.yml")
	sources = YAML.load_file(sources_file)
	data_destination = sources["output-destination"]
	data_path = joinpath(data_destination, "data", "nyc")

	input_dir = joinpath(data_path, "p1_o")
	input_dir_environmental = joinpath(data_path, "p2_o")
	output_dir = joinpath(data_path, "p3_o")
end;

# ╔═╡ 52ee36d9-eb72-4f2a-a3ed-a48e8e0d7931
buildings = GeoDataFrames.read(joinpath(input_dir, "buildings.geojson"))

# ╔═╡ 6daef876-af37-4284-9489-181d8320b653
begin
energydata = CSV.read(joinpath(input_dir, "alldata.csv"), DataFrame)
energydata.date = Date.(energydata.date)

select!(energydata, Not([
	:council_region,
	:electricity_kbtu,
	:naturalgas_kbtu
]))
end;

# ╔═╡ ec1c88ff-bb90-4de9-bff2-6c7f531b725e
function splitdf(df, pct)
   @assert 0 <= pct <= 1
   ids = collect(axes(df, 1))
   shuffle!(ids)
   sel = ids .<= nrow(df) .* pct
   return view(df, sel, :), view(df, .!sel, :)
end

# ╔═╡ 2a94df22-2a14-4ba6-821f-a0c125c93e07
unique_buildings = unique(energydata[:,"Property Id"])

# ╔═╡ b8a9e591-1318-4ec4-b608-164c6425187c
test_buildings = sample(
	unique_buildings, 
	trunc(Int, length(unique_buildings) * 0.20)
);

# ╔═╡ a5ad08d5-08ee-46d9-b2fd-af8ee904e766
@info "Number of Test Buildings:" length(test_buildings)

# ╔═╡ 0b9c6880-9912-4268-9245-4cc8c7cad66f
train_buildings = setdiff(unique_buildings, test_buildings);

# ╔═╡ af422ea0-0f62-4233-a28d-d96a46009a14
@info "Number of Train Buildings:" length(train_buildings)

# ╔═╡ 60943880-2aa3-4d39-adc9-9d10338427e7
begin
train = filter( x -> x["Property Id"] ∈ train_buildings, energydata)
test = filter( x -> x["Property Id"] ∈ test_buildings, energydata)

coerce!(train, 
	:zone => Multiclass, 
	:council_region => Multiclass,
	:bbl => Multiclass,
	# Symbol("Property Id") => Multiclass,
	:month => OrderedFactor
)

coerce!(test,
	:zone => Multiclass, 
	:council_region => Multiclass,
	:bbl => Multiclass,
	# Symbol("Property Id") => Multiclass,
	:month => OrderedFactor
)
end;

# ╔═╡ 9aa06073-d43e-4658-adb9-bbc11425978d
# begin
# 	t = CSV.read(joinpath(input_dir, "train.csv"), DataFrame)
# 	v = CSV.read(joinpath(input_dir, "validate.csv"), DataFrame)

# 	train = vcat(t, v)
# 	validate = v
# 	test = CSV.read(joinpath(input_dir, "test.csv"), DataFrame)
# 	@info unique(sort(filter(x -> x.date >= DateTime(2020), test).date))

# 	coerce!(train, 
# 		:zone => Multiclass, 
# 		:council_region => Multiclass,
# 		:bbl => Multiclass,
# 		# Symbol("Property Id") => Multiclass,
# 		:month => OrderedFactor
# 	)
	
# 	coerce!(test, 
# 		:zone => Multiclass, 
# 		:council_region => Multiclass,
# 		:bbl => Multiclass,
# 		# Symbol("Property Id") => Multiclass,
# 		:month => OrderedFactor
# 	)
# end;

# ╔═╡ 9deff2bf-61a3-48c8-be41-5c3b501d604f


# ╔═╡ c8a8830a-a861-4b35-ae13-2a5cccecbe50
building_zones = DataFrames.select(unique(vcat(train,test), Symbol("Property Id")), [Symbol("Property Id"),:zone]);

# ╔═╡ ced7e799-c5e9-490a-943e-533d2d1b4f2a
testcouncils = DataFrames.select(test, ["Property Id","zipcode"]);

# ╔═╡ 8c97ee63-98a3-4640-8e1c-69fa0cf3810b
1e4

# ╔═╡ 7b53849c-d51e-48a3-a43c-a20392356400
Gadfly.plot(
	train,
	x=:weather_station_distance,
	Geom.histogram(bincount=30)
)

# ╔═╡ b1526350-b68a-4899-adb6-bf981adf26e0
minimum(train.date)

# ╔═╡ 4f1c0eae-e637-40f8-95a9-61088e423725
# want this brief section to extract a system for collecting building metadata used in analysis

# ╔═╡ a8a03990-fd30-48b9-9b76-ce33dd90ceb3
# validation_metadata = DataFrames.select(unique(validate, "Property Id"), [
# 	"Property Id",
# 	"heightroof",
# 	"cnstrct_yr",
# 	"groundelev",
# 	"area",
# 	"council_region",
# 	"weather_station_distance"
# ]);

# ╔═╡ 44f02f33-9044-4355-ba82-b35595a82bdd
test_metadata = DataFrames.select(unique(test, "Property Id"), [
	"Property Id",
	"heightroof",
	"cnstrct_yr",
	"groundelev",
	"area",
	"zipcode",
	"weather_station_distance"
]);

# ╔═╡ c2560a7b-ec09-4cdb-bdec-68a65301249f
# cmip_p = joinpath(input_dir_environmental, "cmip.csv");

# ╔═╡ 56f43ba6-568b-436d-85a5-a8da5a0a3956
begin
	# path for the weather station data
	epw_p = joinpath(input_dir, "epw.csv")
	# building the paths for all of the environmental features
	era5_p = joinpath(input_dir_environmental, "era5.csv")
	landsat8_p = joinpath(input_dir_environmental, "landsat8.csv")
	lst_aqua_p = joinpath(input_dir_environmental, "lst_aqua.csv")
	lst_terra_p = joinpath(input_dir_environmental, "lst_terra.csv")
	noaa_p = joinpath(input_dir_environmental, "noaa.csv")
	sentinel_1C_p = joinpath(input_dir_environmental, "sentinel_1C.csv")
	sentinel_2A_p = joinpath(input_dir_environmental, "sentinel_2A.csv")
	viirs_p = joinpath(input_dir_environmental, "viirs.csv")
end;

# ╔═╡ 8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
epw_r = CSV.read(epw_p, DataFrame);

# ╔═╡ 36699ba6-6472-4cf3-bef6-7c4757fb69b7
epw_sample = CSV.read(joinpath(input_dir, "epw_sample.csv"), DataFrame);

# ╔═╡ 5c8708ff-bc87-44ed-b39d-8f6a87365a3a
property_map = CSV.read(joinpath(input_dir, "property_mapping.csv"), DataFrame);

# ╔═╡ 79fed5b6-3842-47b7-8918-63f918e070bb
begin
	dynam_p = joinpath(input_dir_environmental, "dynamicworld.csv")
	dynam_r = CSV.read(dynam_p, DataFrame; dateformat=date_f)
end;

# ╔═╡ cfc6d000-3338-468a-a1f8-e3d0b3c9881d
describe(dynam_r, :nmissing)

# ╔═╡ dc5167ab-6c65-4916-be18-c635b04f0c0d
sar_p = joinpath(input_dir_environmental, "sar.csv")

# ╔═╡ e33201c0-678e-4ffc-9310-2420ea65aced
sar_r = CSV.read(sar_p, DataFrame; dateformat=date_f);

# ╔═╡ 4b2ce2f1-fdcd-4d30-80b2-38d909270975
sar_ids = unique(sar_r[:,"Property Id"])[400:800]

# ╔═╡ 5b7ae19f-d341-466a-996d-e709e9006a42
sar_sample = dropmissing(filter( x -> x["Property Id"] ∈ sar_ids, sar_r ), :VV);

# ╔═╡ f204fb81-916c-4c85-b13d-9ccf4a70fa43
sar_sample.month = Dates.month.(sar_sample.date);

# ╔═╡ 179e28b1-4102-4e8e-bd54-12baea4044f7
group_sar = combine(groupby(sar_sample, "Property Id"), [:VV,:VH] .=> mean, renamecols=false)

# ╔═╡ 412dd805-7fc4-4c63-9854-0ab98fcb2c8a
U, s, V = svd(Matrix(group_sar[:,[:VV,:VH]]));

# ╔═╡ a0180e4b-4f32-42a7-bd93-9a8aaa0d4979
begin
sᵢ = 1
group_sar[:,:colorgroups] = U[:,sᵢ].*s[sᵢ]
sar_colorsample = leftjoin(
	sar_sample,
	group_sar[:,["Property Id","colorgroups"]],
	on="Property Id"
);
end;

# ╔═╡ 38c7cb84-cb66-4108-8ef3-06767ee15110
# Gadfly.plot(
# 	sar_colorsample,
# 	x=:date,
# 	y=:VV,
# 	color=:colorgroups,
# 	# Geom.beeswarm(padding=1pt),
# 	Geom.smooth(smoothing=0.2),
# 	Guide.title("VV Polarization Over Time"),
# 	Guide.Theme(default_color="black", point_size=0.5pt, line_width=0.5pt),
# 	# Coord.cartesian(ymin=-10, ymax=10),
# )

# ╔═╡ 1a37f7f0-b112-4b0b-b869-713b84f0a1a8
# begin
# monthly_sar_sample = combine(groupby(sar_sample, :month), :VV => mean, renamecols=false)
# Gadfly.plot(
# 	monthly_sar_sample,
# 	x=:month,
# 	y=:VV,
# 	color="Property Id",
# 	Geom.point,
# 	Geom.smooth(smoothing=0.9),
# 	Guide.Theme(default_color="black"),
# 	Guide.xticks(ticks=1:12),
# 	Guide.Title("Monthly Averages")
# )
# end

# ╔═╡ a2e624f7-5626-431a-9680-f62ed86b61aa
# sar_i = Impute.interp(sar_r) |> Impute.locf() |> Impute.nocb();

# ╔═╡ db7c092f-5fa8-4038-b9cf-d40d822a4b9a
# Set(values(countmap(dropmissing(sar_r, [:HH] )[:,"Property Id"])))

# ╔═╡ d3d814ee-ad4f-47bf-966a-08cabc79bf90
# Gadfly.plot(
# 	filter( x -> x.date == DateTime("2019-04-20T22:51:04"), sar_r ),
# 	x=:VV,
# 	Geom.histogram,
# 	Theme(default_color="black")
# )

# ╔═╡ 00acb065-2378-4181-b76a-488071f43a7e


# ╔═╡ fe529a32-ab71-4e5d-a593-45085d69f580
# Gadfly.plot(
# 	dropmissing(sar_r, :VV),
# 	x=:date,
# 	y=:VV,
# 	color="Property Id",
# 	Geom.point,
# 	Geom.line,
# 	# Coord.cartesian(ymin=37.57, ymax=37.6),
# 	Theme(default_color="black", point_size=2pt)
# )

# ╔═╡ a536094d-3894-4ce7-95cd-f38a3666e07e
# VV_councilmedian = combine(groupby(innerjoin(dropmissing(sar_r, :VH), train[:,["Property Id","council_region"]], on="Property Id"), ["date","council_region"]), :VH => mean, renamecols=false)

# ╔═╡ 175ec879-2c34-477a-9359-38f8f9992b72
# Gadfly.plot(
# 	filter(x -> 
# 		x.date > DateTime("2020-09-15T22:58:27") &&
# 		x.council_region > 45, VV_councilmedian),
# 	x=:date,
# 	y=:council_region,
# 	color=:VH,
# 	# Geom.point,
# 	Geom.rectbin
# )

# ╔═╡ 217c69fd-380b-4240-8078-68a54e8eafde
describe(sar_r, :nmissing)

# ╔═╡ 067fc936-5eac-4082-80f8-c50f194f1721
era5_r = CSV.read(era5_p, DataFrame; dateformat=date_f);

# ╔═╡ f589009c-fd53-4f5a-a5e3-844442665e8b
begin
landsat8_r = CSV.read(landsat8_p, DataFrame; dateformat=date_f);
landsat8_r.ST_B10 = landsat8_r.ST_B10 .* 0.00341802 .+ 149.0 .- 273.15
end

# ╔═╡ 06672789-8a1f-44f4-b901-903c3abec016
begin
landsat8_ro = CSV.read(landsat8_p, DataFrame; dateformat=date_f);
dropmissing!(landsat8_ro, :ST_B10)
end

# ╔═╡ 5e053e99-2501-4167-8368-da1720c6d0dd
landsat_capturetimes = Dates.DateTime.(Dates.Hour.(landsat8_ro.date), Dates.Minute.(landsat8_ro.date), Dates.Second.(landsat8_ro.date))

# ╔═╡ b0547ee9-51cc-4427-8269-d69c4b263bbb
begin
landsat_capturetimesₛ = sample(landsat_capturetimes, 200)
end

# ╔═╡ 2c29917b-d1ae-40b5-9eff-14c672bf6a1e
timeticks = collect(Dates.DateTime(0,1,1,15,30, 30):Dates.Minute(1):Dates.DateTime(0,1,1,15,40,30))

# ╔═╡ 7faa0480-bf01-4786-ab6f-4bb772775750


# ╔═╡ 4df4b299-b9ee-46a2-9622-37c430e867a1
lst_aqua_r = CSV.read(lst_aqua_p, DataFrame; dateformat=date_f);

# ╔═╡ fdd41dc5-1439-458c-ad41-3d10f3a8478f
lst_terra_r = CSV.read(lst_terra_p, DataFrame; dateformat=date_f);

# ╔═╡ 5e26802a-0a84-4dc1-926d-d51ac589dc5e
begin
lst_r = vcat(lst_aqua_r, lst_terra_r);
lst_r[!,"LST_Day_1km"] = lst_r[:,"LST_Day_1km"] .* 0.02 .-273.15
lst_r[!,"LST_Night_1km"] = lst_r[:,"LST_Night_1km"] .* 0.02 .-273.15
end;

# ╔═╡ 65c72331-c36e-4e15-b530-13069b8cc070
lst_r

# ╔═╡ cc20d207-016e-408c-baf1-83d68c4c0fde
noaa_r = DataFrames.select(CSV.read(noaa_p, DataFrame; dateformat=date_f), Not(:ACPC01));

# ╔═╡ 0a22f19d-c662-4071-b4e2-6e8103a0f359
sentinel_1C_r = CSV.read(sentinel_1C_p, DataFrame; dateformat=date_f);

# ╔═╡ a9f2d94d-cbf8-4d47-a4b2-438f451882e5
sentinel_2A_r = CSV.read(sentinel_2A_p, DataFrame; dateformat=date_f);

# ╔═╡ ca12dd08-29af-4ce3-a2cc-d3bf1fa9e3c7
viirs_r = CSV.read(viirs_p, DataFrame; dateformat=date_f)

# ╔═╡ 348c4307-94dc-4d5f-82b0-77dc535c1650
function strip_month!(data::DataFrame)
	data[!,"date"] = Date.(Dates.Year.(data.date), Dates.Month.(data.date))
end

# ╔═╡ 09a4789c-cbe7-496e-98b5-a2c2db3102b6
begin
	# also want to get the building data in a uniform format for matching
	strip_month!(train)
	# strip_month!(validate)
	strip_month!(test)

	strip_month!(epw_r)

	strip_month!(era5_r)
	strip_month!(landsat8_r)
	strip_month!(lst_r)
	strip_month!(noaa_r)
	strip_month!(sentinel_1C_r)
	strip_month!(sentinel_2A_r)
	strip_month!(viirs_r)
	strip_month!(sar_r)
end;

# ╔═╡ 22494217-9254-4374-8a7d-02528bdd0df3
strip_month!(dynam_r)

# ╔═╡ d23500b2-53ba-436c-b26d-187f60821a43


# ╔═╡ dcde8c56-7294-47fd-aad1-2204de6c904b
function skip_function(values)
	valuearr::Vector{Any} = (collect∘skipmissing)(values)
	return valuearr
end

# ╔═╡ e87d641a-9555-4a93-9fe8-f39f8964ce84
begin
	function f₁(x)
		y = (collect ∘ skipmissing)(x)
		if length(y) > 0
			return percentile(y, 5)
		else
			return missing
		end
	end
	
	function f₂(x)
		y = (collect ∘ skipmissing)(x)
		if length(y) > 0
			return percentile(y, 25)
		else
			return missing
		end

	end
	
	function f₃(x)
		y = (collect ∘ skipmissing)(x)
		if length(y) > 0
			return percentile(y, 50)
		else
			return missing
		end
	end
	function f₄(x)
		y = (collect ∘ skipmissing)(x)
		if length(y) > 0
			return percentile(y, 75)
		else
			return missing
		end
	end
	function f₅(x)
		y = (collect ∘ skipmissing)(x)
		if length(y) > 0
			return percentile(y, 95)
		else
			return missing
		end
	end
end;

# ╔═╡ a6e81108-c335-4e14-8c23-dd33867e45a9
function monthly_aggregation(
	data::DataFrame, 
	agg_terms::Vector{String},
	functional_terms::Matrix{Function}
)
	agg_terms = ["Property Id", "date"]
	numericterms = filter(
		x -> x ∉ agg_terms, 
		names(data, Union{Real, Missing})
	)
	g = combine(
		groupby(
			data, 
			agg_terms
			), 
		numericterms .=> functional_terms,
		renamecols=true
	)
	g.date = Date.(g.date)
	g
end

# ╔═╡ e03565fe-83eb-4076-ac21-f425ded9f39b
function monthly_aggregation(
	data::DataFrame, 
	agg_terms::Vector{String},
	functional_terms::Function
)
	agg_terms = ["Property Id", "date"]
	numericterms = filter(
		x -> x ∉ agg_terms, 
		names(data, Union{Real, Missing})
	)
	g = combine(
		groupby(
			data, 
			agg_terms
			), 
		numericterms .=> functional_terms,
		renamecols=false
	)
	g.date = Date.(g.date)
	g
end

# ╔═╡ 2f1fec21-76a2-4365-b305-0f24505b1ccc
describe(sar_r, :nmissing)

# ╔═╡ 86d465e3-7916-479d-a29c-2b93ae54ab6d
agg_terms = 	["Property Id","date"]

# ╔═╡ 637220ba-c76a-4210-8c08-fde56b86366a
functional_terms = [f₁ f₃ f₅]

# ╔═╡ cbd44ab9-6346-46f0-bb22-1c19b471ccf7
# cmip = 			monthly_aggregation(cmip_r, agg_terms, functional_terms);

# ╔═╡ 8191b635-e556-47fe-bb39-93ce96ee82bd
length(unique(epw_r[:,"Property Id"]))

# ╔═╡ 3da877c2-159b-4d0d-8b97-34da4dbf2ac3
epw = 			monthly_aggregation(epw_r, agg_terms, functional_terms);

# ╔═╡ b5b2e5b1-af5e-46fd-9c63-2251b9539f62
countmap(epw[:,"Property Id"])

# ╔═╡ 6f73795c-cb72-4a75-adde-17f7361cc452
era5 = 			monthly_aggregation(era5_r, agg_terms, functional_terms);	

# ╔═╡ 766cadfe-acac-402d-8c80-d186b6fef2e6
landsat8 = 		monthly_aggregation(landsat8_r, agg_terms, functional_terms);

# ╔═╡ e9c65a69-4bf1-48df-8190-f988c7442e38
lst = 			monthly_aggregation(lst_r, agg_terms, functional_terms);

# ╔═╡ c614fd3b-363f-436e-b6e9-2366d1cf87b9
noaa = 			monthly_aggregation(noaa_r, agg_terms, functional_terms);

# ╔═╡ f40de5ec-5260-498d-a102-461e4ba24178
sentinel_1C = 	monthly_aggregation(sentinel_1C_r, agg_terms, functional_terms);

# ╔═╡ 0c7528dd-f857-47b8-a1c9-1a7fd5a316a9
sentinel_2A = 	monthly_aggregation(sentinel_2A_r, agg_terms, functional_terms);

# ╔═╡ 073ca242-0c60-4f54-a0f2-f5d2bed88421
viirs = 		monthly_aggregation(viirs_r, agg_terms, functional_terms);

# ╔═╡ 8d0a2791-0f06-48d7-8be9-d7663226f9c2
sar = 			monthly_aggregation(
					DataFrames.select(sar_r, Not([:HH,:HV])), 
					agg_terms, 
					functional_terms
);

# ╔═╡ a7d8d342-0829-45fe-ae5e-f4b74cfc29b4
describe(sar, :nmissing)

# ╔═╡ 0cea9f55-9486-4166-a987-67db0c09da2a
dynam = 		monthly_aggregation(dynam_r, agg_terms, functional_terms);

# ╔═╡ 3e648045-0915-4eb6-b037-4c09f1f0036e
sample_sar = innerjoin(train, sar, on=["Property Id", "date"])

# ╔═╡ 5f739960-1b18-4804-ba3e-986207146849
# Gadfly.plot(
# 	sample_sar,
# 	x=:date,
# 	y=:angle,
# 	color=:council_region,
# 	Geom.point,
# 	Geom.line
# )

# ╔═╡ e4cda53e-7fe2-4cdb-8a9c-c9bdf67dd66f
md""" 
Getting a flavor of what kind of data we have now
"""

# ╔═╡ 3b980465-9b75-404d-a41f-06ad351d12ae
@info "Training data points prior to merge" nrow(train)

# ╔═╡ a693145c-8552-44ff-8b46-485c8c8fb738
# begin
# 	environmental_terms = [era5, noaa, lst, landsat8, viirs]
# 	@info "# Environmental Datasets" length(environmental_terms)
# end

# ╔═╡ b6da63b0-417c-4a84-83f2-7357bf81fd4d
md"""
This section is composed of auxillary functions which will help with the structure of the ML system
"""

# ╔═╡ 17218b4f-64d2-4de3-a5a9-1bbe9194e9d6
names(train)

# ╔═╡ 3d66f852-2a68-4804-bc73-0747b349cf22
base_terms = [
	agg_terms...,
	"heightroof",
	"cnstrct_yr",
	"groundelev",
	"area",
	"month",
	"weather_station_distance",
	"zone",
	"zipcode"
];

# ╔═╡ dcfc75b1-1951-4db1-9d4c-18b46fb21ffd
electricity_terms = [base_terms...,"electricity_mwh"];

# ╔═╡ 5eddb220-0264-4ce0-856f-ef040c9af06b
naturalgas_terms = [base_terms...,"naturalgas_mwh"];

# ╔═╡ 86bd203f-83ca-42d9-9150-a141ec163e3f
function clean(data)
	filter(row -> all(x -> !(x isa Number && isnan(x)), row), data)
end

# ╔═╡ 5bd94ed4-a413-418b-a67c-21e3c73ea36b
function prepare_terms(
	data::DataFrame, 
	augmentation::DataFrame,
	join_on::Union{Vector{String}, Vector{Symbol}, Symbol, String}
)

data_augmented = innerjoin(data, augmentation, on=join_on)
println(nrow(data_augmented))
	
data_clean = clean(dropmissing(data_augmented))

X = DataFrames.select(data_clean, Not(join_on))
aux = DataFrames.select(data_clean, join_on)

return X, aux
end

# ╔═╡ 31705a91-8f1b-4e1c-a912-d79e5ce349b4
function prepare_terms(
	data::DataFrame, 
	augmentation::Vector{DataFrame},
	join_on::Union{Vector{String}, Vector{Symbol}, Symbol, String}
)

data_augmented = innerjoin(data, augmentation..., on=join_on)
data_clean = clean(dropmissing!(data_augmented))

X = DataFrames.select(data_augmented, Not(join_on))
aux = DataFrames.select(data_augmented, join_on)

return X, aux
end

# ╔═╡ a7b58da6-5e51-4c31-8dbc-82b9fe54e609
nrow(era5)

# ╔═╡ 32e235c8-212e-40bb-9428-74ea2a78b900
function training_pipeline(
	training_data::DataFrame, 
	# augmentation::Union{DataFrame, Vector{}},
	# join_on::Union{String, Symbol, Vector{String}, Vector{Symbol}},
	prediction_term::Union{String, Symbol},
	model
)
# initial_size = nrow(training_data)

# println(initial_size)
# if size(augmentation)[1] > 0
# 	training_prepared, auxiliary = prepare_terms(
# 		training_data,
# 		augmentation,
# 		join_on
# 	)
# else
# 	training_prepared = DataFrames.select(training_data, Not(join_on))
# end

# println(nrow(training_prepared))
	
X = DataFrames.select(training_data, Not(prediction_term))
y = training_data[:,prediction_term]

# final_size = nrow(X)

custom_machine = machine(model, X, y; cache=false)
MLJ.fit_only!(custom_machine)

# println(100 * final_size / initial_size)

return custom_machine
end

# ╔═╡ 9e3961f4-fa02-4ce6-9655-484fb7f7e6dc
function validation_pipeline(
	validation_data::DataFrame, 
	prediction_term::Union{String, Symbol},
	# distinct_terms::Union{String, Symbol},
	custom_machine
)

# initial_size = nrow(validation_data)
	
# if size(augmentation)[1] > 0
# 	validating_prepared, auxiliary = prepare_terms(
# 		validation_data,
# 		augmentation,
# 		join_on
# 	)
# else
# 	validating_prepared = DataFrames.select(validation_data, Not(join_on))
# 	auxiliary = DataFrames.select(validation_data, join_on)
# end

X = DataFrames.select(validation_data, Not(prediction_term))
ŷ = MLJ.predict(custom_machine, X)
# X[!,"recorded"] = y

# # Ẋ = hcat(X, auxiliary)	
# vᵧ = groupby(Ẋ, "Property Id")
# results = combine(vᵧ) do vᵢ
# (
# cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
# nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
# cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
# )
# end

return ŷ
end

# ╔═╡ 055aaebf-84ea-49dc-8f15-85a0e2e48b49
a = false

# ╔═╡ 36cc54b3-1e48-4f9a-9b34-b3bac0394eb8
!a

# ╔═╡ b5f8ce76-008a-491d-9537-f1b8f4943880
begin
tₐ = clean(dropmissing(DataFrames.select(train, electricity_terms)))
tₐ.month = coerce(tₐ.month, OrderedFactor)
tₐ.electricity_mwh = tₐ.electricity_mwh ./ Dates.daysinmonth.(tₐ.date)

tᵧ = clean(dropmissing(DataFrames.select(train, naturalgas_terms)))
tᵧ.month = coerce(tᵧ.month, OrderedFactor)
tᵧ.naturalgas_mwh = tᵧ.naturalgas_mwh ./ Dates.daysinmonth.(tᵧ.date)
end;

# ╔═╡ 77a3bb94-19f5-417a-ba34-9270a9054b5c
minimum(tₐ.date)

# ╔═╡ a62b17d9-3d2b-4985-b982-78b557a5889c
maximum(tₐ.date)

# ╔═╡ 5ddf5811-9d4c-4375-8ae9-268bcb9e7ed5
naturalgas_terms

# ╔═╡ 36bf3b43-4cf6-4a48-8233-1f49b9f7fac3
# begin
# vₐ = clean(dropmissing(DataFrames.select(validate, electricity_terms)))
# vₐ.month = coerce(vₐ.month, OrderedFactor)

# vᵧ = clean(dropmissing(DataFrames.select(validate, naturalgas_terms)))
# vᵧ.month = coerce(vᵧ.month, OrderedFactor)
# end;

# ╔═╡ 94d6e626-965d-4101-b31c-08c983678f92
begin
teₐ = clean(dropmissing(DataFrames.select(test, electricity_terms)))
teₐ.month = coerce(teₐ.month, OrderedFactor)
teₐ.electricity_mwh = teₐ.electricity_mwh ./ Dates.daysinmonth.(teₐ.date)

teᵧ = clean(dropmissing(DataFrames.select(test, naturalgas_terms)))
teᵧ.month = coerce(teᵧ.month, OrderedFactor)
teᵧ.naturalgas_mwh = teᵧ.naturalgas_mwh ./ Dates.daysinmonth.(teᵧ.date)
end;

# ╔═╡ 10620dd0-2ef4-40d0-97db-3b6dc1411036
teᵧ.zone

# ╔═╡ 6a02a599-d4f7-4505-b4a8-be8253b47374
dataₑ = vcat(tₐ, teₐ);

# ╔═╡ 74046c2f-7ad2-41b3-bb9f-2c6e30dc6a5a
dataᵧ = vcat(tᵧ, teᵧ);

# ╔═╡ f73ae203-056b-400b-8457-6245e9283ead
schema(tₐ)

# ╔═╡ f3da6fb8-a85b-4dfc-a307-bc987239abc6
# schema(vₐ)

# ╔═╡ 54f438b0-893e-42d3-a0f5-2364723be84e
md"""
##### honestly just playing with the data a bit here
want to explore generalized trends
"""

# ╔═╡ c6f4308d-6dce-4a57-b411-6327f4aa87a7
# p₂ = Gadfly.plot(
# 	train,
# 	x=:council_region,
# 	y=:weather_station_id,
# 	Guide.xlabel("Council Region"),
# 	Guide.ylabel("Weather Station ID"),
# 	Guide.title("Data Points Histogram"),
# 	Guide.yticks(ticks=725020:4:725060),
# 	Geom.histogram2d
# )

# ╔═╡ c6c038f0-cf8b-4234-a64c-75616fdc07a5
# draw(
# 	PNG(
# 		joinpath(output_dir, "datapoint_histogram.png"), 
# 		20cm, 
# 		10cm,
# 		dpi=500
# 	), p₂
# )

# ╔═╡ 1a5ab262-0493-470f-ab58-baa5fa1a69af
begin
ᵞ₁ = combine(groupby(tᵧ, :month), :naturalgas_mwh => mean => "Natural Gas");
ᵞ₁.month = convert.(Int, ᵞ₁.month)

p₁ = combine(groupby(tₐ, :month), :electricity_mwh => mean => "Electricity");
p₁.month = convert.(Int, p₁.month)
end;

# ╔═╡ b5dff4f2-3088-4301-8dfe-96fb8c6999c7
begin
Η₀ = leftjoin(
	ᵞ₁,
	p₁,
	on="month"
)
Η₀.month = convert.(Int, Η₀.month)
end;

# ╔═╡ df3549d2-4183-4690-9e04-b665a9286792
Η₁ = stack(Η₀, 2:3);

# ╔═╡ 992c77d0-1b6d-4e07-bd3a-f648ef023870
begin
ᵞ₂ = combine(groupby(tᵧ, [:zone,:month]), :naturalgas_mwh => mean => "Natural Gas");
p₂ = combine(groupby(tₐ, [:zone,:month]), :electricity_mwh => mean => "Electricity");

Η₀₂ = leftjoin(
	ᵞ₂,
	p₂,
	on=[:zone,:month]
)
Η₀₂.month = convert.(Int, Η₀₂.month)

Η₂ = stack(Η₀₂, 3:4);
end;

# ╔═╡ 70576040-eec4-4b80-8d23-c2da2e65b2d2
Η₂

# ╔═╡ e6ea0a65-b79f-400f-be61-951b08b5ce88
p₃ = Gadfly.plot(
	Η₂,
	x=:month,
	y=:value,
	color=:variable,
	xgroup=:zone,
	Scale.color_discrete(),
	Guide.xlabel("Month"),
	Guide.ylabel("Avg. Energy - MWh"),
	# Guide.xticks(ticks=1:12),
	Guide.colorkey(title="Term"),
	Guide.title("New York Energy Consumption Habits by Zone"),
	Theme(point_size=2.5pt, key_position=:bottom),
	Geom.subplot_grid(
		Geom.point, 
		Geom.line,
		# Guide.yticks(ticks=0.0:0.025:0.15),
		Guide.xticks(ticks=1:12)
	),
	Scale.color_discrete_manual("indianred","lightblue")
)

# ╔═╡ 77aad1ea-16cf-4c5f-9a6b-345f7168afb3
draw(
	PNG(
		joinpath(output_dir, "energy_trends.png"), 
		20cm, 
		10cm,
		dpi=500
	), p₃
)

# ╔═╡ fa79e50c-5d5e-47a1-b2e7-fe3341ecbf4f
## still playing around a little bit with the visualizations

# ╔═╡ 8ef3db5b-f7cd-4d09-8817-35dbf96629d8
sar

# ╔═╡ 055a08c0-abc9-465b-bb64-24744e87da5d
sar′ = dropmissing(clean(innerjoin(
	tₐ,
	sar, 
	lst,
	on=["Property Id", "date"]
)));

# ╔═╡ 4947232e-568d-4724-8ebd-5f000bdc9483
names(sar′, Union{Real, Missing})

# ╔═╡ c9e49960-f659-432b-9338-8390e41b447a
selected_weather_station = 725033

# ╔═╡ 0429f5c6-d5c7-461b-b42c-04bfd0f205ab
weatherstation_buildings = filter( x -> x.weather_station_id == selected_weather_station, property_map)[:,"Property Id"]

# ╔═╡ 07a187c7-bc2e-47ce-ba06-55620c9f516b
sample_sars = sample(filter( x -> x ∈ weatherstation_buildings, unique(sar′[:,"Property Id"])), 100)

# ╔═╡ 46df5f17-15f0-4646-b3f7-653122f0d3b7
begin
sample_filtered = filter( x-> x["Property Id"] in sample_sars, landsat8_r );
dropmissing!(sample_filtered, :ST_B10)
end

# ╔═╡ 1256b894-473c-42a8-8451-2b9bb0142d23
begin
single_building = filter( x-> x["Property Id"] == 6405415, landsat8_r );
dropmissing!(single_building, :ST_B10)
end;

# ╔═╡ 4a5e2b32-fa51-4146-b0bd-793078b30b95
begin
sample_epw_region = filter( x -> x.weather_station_id == selected_weather_station, epw_sample )
sample_epw_region.year = Dates.Year.(sample_epw_region.date)
sample_epw_region.month = Dates.Month.(sample_epw_region.date)

sample_epw_aggregation = combine(groupby(sample_epw_region, [:month, :year]),
	# :median_temp => median,
	# :min_temp => minimum,
	# :max_temp => maximum,
	names(sample_epw_region, Real) .=> mean,
	renamecols=false
)
sample_epw_aggregation.date = Dates.Date.(sample_epw_aggregation.year, sample_epw_aggregation.month)
sample_epw_aggregation
end;

# ╔═╡ 413cbb00-bc81-4d17-948c-76707bfb589a
t₁ = Gadfly.plot(
	# Gadfly.layer(
	# 	x=:date,
	# 	y=:ST_B10,
	# 	# intercept=[mean(sample_filtered.LST_Day_1km_f₃ .- sample_filtered.LST_Night_1km_f₃) - 5], 
	# 	# slope=[1],
	# 	# Geom.abline(color="pink", style=:solid),
	# ),
	Gadfly.layer(
		single_building,
		x=:date,
		y=:ST_B10,
		Geom.point,
		Geom.smooth(smoothing=0.15),
		# Geom.line,
		Theme(default_color="lightcoral", line_width=1.2pt, point_size=2pt)
	),
	Gadfly.layer(
		sample_epw_aggregation,
		x=:date,
		y=:median_temp,
		ymin=:min_temp,
		ymax=:max_temp,
		# Geom.point,
		Geom.line,
		Theme(
			default_color="gray", 
			line_width=1.1pt, 
			point_size=2pt, 
			alphas=[0.8],
			# Theme(lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.4))
		),
	),
	Gadfly.layer(
		sample_filtered,
		x=:date,
		y=:ST_B10,
		color="Property Id",
		# Geom.line,
		# Geom.point,
		# Geom.line,
		Geom.smooth(smoothing=0.15),
		Theme(line_width=0.08pt, alphas=[0.2])
	),
	Scale.discrete_color_hue,
	Guide.ylabel("Temperature °C"),
	Guide.xlabel("Date"),
	Guide.yticks(ticks=-10:5:50),
	# Coord.cartesian(xmin=Date("2018-01-01"), xmax=Date("2019-12-01")),
	Guide.xticks(ticks=DateTime("2018-01-1"):Month(1):DateTime("2020-12-31"), orientation=:vertical),
	Guide.title("Temperature Discrepancies near Central Park - Landsat8 vs. EPW"),
	Theme(default_color="black", key_position = :none),
	Guide.manual_color_key("Legend", ["Landsat8", "EPW"], ["lightcoral", "gray"])
)

# ╔═╡ 4b37d389-bc6d-4d02-83f3-4d5879646c16
draw(
	PNG(
		joinpath(output_dir, "temperature_nonlinearity.png"), 
		20cm, 
		12cm,
		dpi=600
	), t₁
)

# ╔═╡ 6c088aab-1d2b-435b-8a13-a82d5879200e
# begin
# bincount = 100
# Gadfly.plot(
# 	x=sample_filtered.VH_f₃,
# 	y=sample_filtered.LST_Day_1km_f₃,
# 	Geom.histogram2d(xbincount=bincount, ybincount=bincount),
# 	Guide.xlabel("Polarization"),
# 	Guide.ylabel("Daytime Temperature - °C"),
# 	Guide.title("Polarization vs. Daytime Temps"),
# 	Guide.Theme(panel_line_width=0pt),
# 	# Coord.cartesian(xmin=-18, xmax=-13),
# )
# end

# ╔═╡ 40af6472-77ee-419e-b1b1-d77c29f31c46
lst′ = dropmissing(clean(innerjoin(
	tₐ,
	lst,
	dynam,
	on=["Property Id", "date"]
)))

# ╔═╡ d34c91f0-75e5-4174-96c0-1ecb3e26d335
lst_names = unique(lst′[:,"Property Id"])[1:100]

# ╔═╡ bdcb3e10-f7b5-49a6-b424-0185f6908301
sort(countmap(lst′[:,"Property Id"]); byvalue=true, rev=true)

# ╔═╡ 3c945664-6536-428b-a7eb-5f85f189c12f
# Gadfly.plot(
# 	filter( x -> x["Property Id"] in lst_names, lst′ ),
# 	x=:date,
# 	y=:trees_f₃,
# 	color="Property Id",
# 	# Geom.point,
# 	Geom.smooth(smoothing=0.2),
# 	Scale.discrete_color,
# 	Theme(key_position = :none)
# )

# ╔═╡ 3892a1f6-e205-43ef-957e-3f7ba4fa1bd1


# ╔═╡ de82b60b-9335-4acb-bd1f-84c5539ef04c


# ╔═╡ 217bfbed-1e13-4147-b4e4-c58eaed29382
md"""
# Training Pipeline
"""

# ╔═╡ 61238caa-0247-4cf6-8fb7-3b9db84afcee
md"""
Base model - this m₁ term will be used for almost all of the anlaysis
"""

# ╔═╡ 0bc7b14b-ccaf-48f6-90fa-3006737727ed
rng = MersenneTwister(500);

# ╔═╡ 58b4d31f-5340-4cd9-8c9e-6c504479897f
EvoTree = @load EvoTreeRegressor pkg=EvoTrees verbosity=0

# ╔═╡ 850ddd0b-f6ea-4743-99ca-720d9ac538a0
Tree = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# ╔═╡ 90227375-6661-4696-8abd-9562e08040bf
"Invalid loss: EvoTrees.L1(). Only [`:linear`, `:logistic`, `:L1`, `:quantile`] are supported at the moment by EvoTreeRegressor.";

# ╔═╡ aeeab107-8e68-4d0f-a3e3-8b1e0c12e8e6


# ╔═╡ c5e5a5e3-e324-4c8b-8449-caccc87cf8b6
CUDALibs()

# ╔═╡ ea1a797c-da0a-4133-bde5-366607964754
begin
m_tree = EvoTree( 
 	max_depth=2,
 	lambda=0.2,
 	gamma=0.05,
	eta=0.175,
 	nrounds=2,
 	rowsample=0.95, 
 	colsample=0.95,
 	device="gpu"
 )
# m = EnsembleModel(model=m_tree, n=100, rng=rng)
end

# ╔═╡ e5f7e860-4802-42a1-822d-51fcc983dfae
#m_tree = Tree(
#	rng=rng,
#	max_depth=3,
#	n_trees=20,
#	sampling_fraction=0.85
#)

# ╔═╡ 8edfe547-f860-418f-8c68-e8fe0c16162f
md"""
###### Electric Results
"""

# ╔═╡ 6d5c77c8-2596-45eb-8163-c8e14249948d
# baselineₑ = filter(x -> x.model == "Null", modelresultsₑ).rmse[1]

# ╔═╡ 49f04d2e-f8fc-4d4e-b9a9-bb97bdf559a6


# ╔═╡ 1b63c7e3-3021-469e-b3d8-d898fa3a755b
importance

# ╔═╡ 8a462521-c2bc-4b8a-b58f-4fc5743fe09c
EvoTrees.Linear

# ╔═╡ ef2f7716-f622-4bf2-ada2-14e76900619b
# feature_importances(mₑ)[1:10]

# ╔═╡ 6c387352-ae9c-405d-8f16-562b562c4a4b
md"""
###### Gas Results
"""

# ╔═╡ a89ec65a-951c-445c-a0de-8e465dd5c6b7


# ╔═╡ eea75399-d25c-46b6-a471-2cb8676e4db7
md"""
##### 1. Initial Results - Comparison to Baseline
"""

# ╔═╡ 40907e02-e67d-453d-ae28-6db4543db9d8
# begin
# tᵥ₅ = DataFrames.select(
# 	DataFrames.select(tₐ′, term₅), 
# 	exclusion_terms
# )

# X₅ = DataFrames.select(tᵥ₅, Not("electricity_mwh"))
# # y = tᵥ[:,"electricity_mwh"]
	
# explain₅ = copy(X₅)[sample(1:nrow(X₅), 50),:]
# # reference₅ = copy(X₅)[1:5000,:]

# data_shap₅ = ShapML.shap(
# 	explain = explain₅,
# 	# reference = reference₅,
# 	model = m₅,
# 	predict_function = predict_function,
# 	sample_size = sample_size,
# 	seed = 1
# )
# end;

# ╔═╡ 9b0f5fdb-450e-4a51-a1e9-c8c4a884a4e7
# unique(data_shap₅.feature_name)

# ╔═╡ 3d130a66-4d0a-4d02-bec5-ad19a1ebabff
# Gadfly.plot(
# 	filter(x -> x.feature_name ∈ ["month"], data_shap₅),
# 	x=:feature_value,
# 	y=:shap_effect,
# 	color=:feature_name,
# 	Geom.point,
# 	Geom.smooth(smoothing=0.08),
# 	# Theme(point_size=1pt)
# )

# ╔═╡ 0546ace9-f503-4340-9c1d-2ae58460f32f
# begin
# tᵥₒ = DataFrames.select(
# 	DataFrames.select(tₐ′, term₇), 
# 	exclusion_terms
# )

# Xtoeₑ = DataFrames.select(tᵥₒ, Not("electricity_mwh"))
# # y = tᵥ[:,"electricity_mwh"]
	
# explain = copy(Xtoeₑ)[sample(1:nrow(Xtoeₑ), 200),:]
# reference = copy(Xtoeₑ)[1:5000,:]
# end;

# ╔═╡ 026dd361-efd3-42dc-a6c9-b27efe6dfd47
# begin
# data_shap = ShapML.shap(explain = explain,
#                         model = m₇,
#                         predict_function = predict_function,
#                         sample_size = sample_size,
#                         seed = 1
#                         )
# end

# ╔═╡ 657008c0-997a-4e01-b9a2-f514dc3550b6
# unique(data_shap.feature_name)

# ╔═╡ d1a85a16-dd24-4ce8-9d8f-df575a585dfa
# Gadfly.plot(
# 	filter(x -> x.feature_name ∈ ["VV_f₁","VV_f₃","VV_f₅"], data_shap),
# 	x=:feature_value,
# 	y=:shap_effect,
# 	color=:feature_name,
# 	# Geom.point,
# 	Geom.smooth(smoothing=0.08),
# 	Theme(point_size=1pt)
# )

# ╔═╡ 1ddb5277-57cb-4fd6-8fe7-632e23d702f1
# Gadfly.plot(
# 	filter(x -> x.feature_name ∈ ["VH_f₁","VH_f₃","VH_f₅"], data_shap),
# 	x=:feature_value,
# 	y=:shap_effect,
# 	color=:feature_name,
# 	# Geom.point,
# 	Geom.smooth(smoothing=0.08),
# 	Theme(point_size=1pt)
# )

# ╔═╡ c27b1508-f7f6-47d0-a04e-f816b4b619de
# Gadfly.plot(
# 	filter(x -> x.feature_name ∈ ["month"], data_shap),
# 	x=:feature_value,
# 	y=:shap_effect,
# 	# Geom.point,
# 	Geom.point,
# 	Theme(point_size=3pt)
# )

# ╔═╡ 9fe1f154-8174-4ae7-82ca-542b123a5201
# begin
# # Compute stochastic Shapley values.
# data_shap = ShapML.shap(
# 	explain = explain,
#     reference = reference,
# 	model = m₇,
#     predict_function = predict_function,
# 	sample_size = sample_size,
# 	parallel = :features,
#     seed = 1
# )

# data_plot = sort(
# 	combine(
# 		groupby(dropmissing(data_shap), :feature_name), :shap_effect .=> x -> mean(abs.(x)), renamecols=false), :shap_effect, rev=true
# )
# end

# ╔═╡ 521a5979-e206-44a1-9b4e-e5348e070125


# ╔═╡ a8d738bb-0424-4df0-aa22-2a299fd994b1
# begin
# imp = feature_importances(mₑ);
# imp_vars = map(x -> x[1], imp);
# imp_values = map(x ->x[2], imp);

# n_imp = 8
# end;

# ╔═╡ dc8fb980-6500-4849-b498-b39454dd3ffa
# Gadfly.plot(
# 	x=imp_vars[1:n_imp],
# 	y=imp_values[1:n_imp],
# 	Geom.bar
# )

# ╔═╡ bb078875-611c-46f6-8631-1befde358054
md"""
before jumping into the training process, we want to first try and standardize the data which is used throughout the learning process by keeping the buildings fairly uniform
"""

# ╔═╡ df753575-b121-4c3b-a456-cfe4e535c2aa
md"""
###### Electricity Data Cleaning and Prep
"""

# ╔═╡ a157969b-100a-4794-a78e-2f40439e28d9
comprehensive_datalist = [
	# cmip,
	dynam,
	noaa,
	# era5,
	epw, 
	# lst,
	landsat8,
	# sentinel_1C,
	sar,
	viirs
];

# ╔═╡ 09b8527a-afb9-4453-80ff-1026fe2f1bfa
length(unique(tₐ[:,"Property Id"])) * 12 * 3

# ╔═╡ 9239f8dc-fd07-435f-9d88-e24bb1f6faa2
for term in comprehensive_datalist
	@info nrow(term)
end

# ╔═╡ d8395037-8305-4852-8d13-d0fff7005a5a
innerjoin(
	tₐ,
	sar,
	on=["Property Id","date"]
)

# ╔═╡ f8478f59-7393-472b-b7d0-5cf18a3d4b29
sar

# ╔═╡ ea14aaee-f5ae-4449-b319-7e20416d2b72
innerjoin(
	tₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)

# ╔═╡ 0c89dbf1-c714-43da-be89-3f82ccc2373a
# here is where we want to inject all of the data and drop missing terms
tₐ′ = clean(dropmissing(innerjoin(
	tₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

# ╔═╡ 8cbf28f7-a3a9-4fde-9156-935c25b001e7
nrow(tₐ′)

# ╔═╡ 960d4a47-c701-42c5-934e-a80d74b7ddbd
# property_counts = countmap(complete_training_data[:,"Property Id"])

# ╔═╡ 6d873e8f-d3ee-4423-86e7-e8d75abaad38
# months_represented = convert(Int, percentile(values(property_counts), 90))

# ╔═╡ cb23bb0f-87ad-4a1b-8491-c6db384824da
# months_represented = maximum(values(property_counts))

# ╔═╡ 28ac0a1f-e8cd-4e27-9852-38e712488b81
# bₜₑ = collect(keys(filter( x -> x[2] == months_represented, property_counts )))

# ╔═╡ 0c9c56b4-4394-4457-b184-23ac8c35d7e6
# dₜₑ = sort(collect(Set(complete_training_data[:,"date"])))

# ╔═╡ b81a31cc-5024-418c-8b69-61a6011385ff
# tₐ̇ = filter(
# 	x -> x["Property Id"] ∈ bₜₑ && x["date"] ∈ dₜₑ,
# 	tₐ
# );

# ╔═╡ 3d513e7f-a1f1-4668-9a0f-87cf7e7a68c6
# # here is where we want to inject all of the data and drop missing terms
# vₐ′ = clean(dropmissing(innerjoin(
# 	vₐ,
# 	comprehensive_datalist...,
# 	on=["Property Id", "date"]
# )));

# ╔═╡ e81ce097-79e0-4e5e-b7f9-3956b14d5db3
# describe(filter(
# 	x -> x.date > Date(2020,01,01),
# 	innerjoin(
# 		teₐ,
# 		comprehensive_datalist...,
# 		on=["Property Id", "date"]
# 	)
# ), :nmissing)

# ╔═╡ bfab2fc0-aee6-4e8d-a7e6-dcbac516dedd
# countmap(dropmissing(innerjoin(
# 	teₐ,
# 	comprehensive_datalist...,
# 	on=["Property Id", "date"]
# )).date)

# ╔═╡ 98da20c8-c530-4ed6-acb2-3c98019a36a9
teₐ′ = clean(dropmissing(innerjoin(
	teₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

# ╔═╡ 7ff9cc78-44ae-4baa-b3af-d10487c11920
# begin
# 	property_counts_validation = countmap(complete_validation_data[:,"Property Id"])
# 	months_represented_validation = maximum(values(property_counts_validation))
# end

# ╔═╡ 5f260a93-7a7e-4c62-8a52-55371a2093c9
# bᵥₑ = collect(keys(filter( x -> x[2] == months_represented_validation, property_counts_validation )))

# ╔═╡ 294cd133-af00-4025-a182-6e2f057107a6
# dᵥₑ = sort(collect(Set(complete_validation_data[:,"date"])))

# ╔═╡ 7073082d-a2fa-4419-996c-a38cf3ee044c
# vₐ̇ = filter(
# 	x -> x["Property Id"] ∈ bᵥₑ && x["date"] ∈ dᵥₑ,
# 	vₐ
# );

# ╔═╡ ec365574-6e66-4284-9723-5634ba73d52a
md"""
###### Natural Gas Data Cleaning and Prep
"""

# ╔═╡ b7983d6f-2dab-4279-b7a5-222e40ce968b
joining_terms = ["Property Id", "date"]

# ╔═╡ ce3e2b3b-a6d6-4494-8168-665e878edae3
extra_omission_features = [
	"weather_station_distance",
	"zone",
	"zipcode",
	"month"
]

# ╔═╡ 617aa3dd-045f-4e78-bfcf-2d6c45dfe138
tᵧ′ = dropmissing(clean(innerjoin(
	tᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

# ╔═╡ f2bbbe98-9b3f-46c2-9630-2b3997549742
# vᵧ′ = clean(dropmissing(innerjoin(
# 	vᵧ,
# 	comprehensive_datalist...,
# 	on=joining_terms
# )));

# ╔═╡ 4e0f81be-141f-4882-83f6-a443febc0592
maximum(teᵧ.date)

# ╔═╡ 4a0f370f-4331-46de-8636-f93dc6172251
teᵧ.zone

# ╔═╡ d37ff7b0-fb28-4f55-b5c1-f4040f1387f6
teᵧ′ = clean(dropmissing(innerjoin(
	teᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

# ╔═╡ 5cbbd305-513c-42e3-b6b4-5d969d62a3f3
exclusion_terms = Not([joining_terms..., extra_omission_features...])

# ╔═╡ ea359342-7685-441d-876e-6562f1770504


# ╔═╡ 0cc43266-fd12-447c-ab55-4e8c3d359062


# ╔═╡ fd61e191-3b66-4ebc-b5ec-02a128548ffb
md"""
#### Electricity
"""

# ╔═╡ 7ea0c4fd-a2b2-4a4b-975e-8fd010007788
md"""
Addionally will be exploring aggregation effects, and the quality of predictions using an aggregated predictor. Here are the counts of each building in the aggregation, prior to identifying the quality of their prediction:
"""

# ╔═╡ 82121c42-b4eb-4495-90f9-3eb34f986d15
electricityₘ = 	combine(groupby(teₐ′, :date), 
		:electricity_mwh => mean,
		:electricity_mwh => std,
		renamecols=true
);

# ╔═╡ a084f5b5-05f4-421c-aa22-dc61511c8002


# ╔═╡ 764c858a-8810-4498-b0fe-300a3ecf8488
test

# ╔═╡ 38c79d97-e5da-415b-a1f1-45a093aeeb2f
sort(unique(filter( x -> x.date >= DateTime(2020), test).date))

# ╔═╡ 47827a31-25cd-4cd3-845d-62c7bf4491ba
DataFrames.select(tₐ′, exclusion_terms)

# ╔═╡ 3a2cf01a-ce41-42c6-9cc2-8a6ff0da2016
exclusion_terms

# ╔═╡ 61705c8f-a3e2-4d4b-bd52-a334a0a9f5bd
# function electrictrain(modelname::String, terms::Vector{String})
# 	tᵥ = DataFrames.select(tₐ′, terms)
# 	mᵥ = training_pipeline(
# 		DataFrames.select(tᵥ, exclusion_terms),
# 		"electricity_mwh",
# 		m
# 	)

# 	# validation
# 	vᵥ = DataFrames.select(vₐ′, terms)
# 	vᵥ.prediction = validation_pipeline(
# 		DataFrames.select(vᵥ, exclusion_terms),
# 		"electricity_mwh",
# 		mᵥ
# 	);
	
# 	vᵥ.recorded = vᵥ.electricity_mwh
# 	vᵥ.model = repeat([modelname], nrow(vᵥ))
	
# 	# test
# 	teᵥ = DataFrames.select(teₐ′, terms)
# 	teᵥ.prediction = validation_pipeline(
# 		DataFrames.select(teᵥ, exclusion_terms),
# 		"electricity_mwh",
# 		mᵥ
# 	);
	
# 	teᵥ.recorded = teᵥ.electricity_mwh
# 	teᵥ.model = repeat([modelname], nrow(teᵥ))

# 	leftjoin!(
# 		teᵥ,
# 		building_zones,
# 		on="Property Id"
# 	)

# 	return mᵥ, vᵥ, teᵥ
# end

# ╔═╡ 3c41548c-da78-49c5-89a3-455df77bf4fa
md"""
Null model
"""

# ╔═╡ a149faa9-bee4-42a0-93fe-5adff459e0e9
## as a preliminary introduction - thinking about the overall dataset
@info "Number of data points" nrow(tₐ′)

# ╔═╡ 748321c3-0956-4439-90d4-e74598d83f20
term₀ = unique([electricity_terms...])

# ╔═╡ 10abc1fa-c350-4473-9d72-8e9333a0bee4
names(tₐ′, Real)

# ╔═╡ e10d17be-95f8-4ef9-9387-23a46702c27f
# cor(Matrix(DataFrames.select(tₐ′, names(tₐ′, Real))))

# ╔═╡ 19c8d652-f42c-4952-ac16-d4752dafdfea
# Gadfly.plot(
# 	tₐ′[sample(1:nrow(tₐ′), 1000),:],
# 	x=:heightroof,
# 	y=:VH,
# 	Theme(alphas=[0.5], default_color="black")
# )

# ╔═╡ 7f4f0941-495d-4309-bccf-da82aab641da
n_buildings = trunc(Int, length(train_buildings) * 0.15)

# ╔═╡ f99924cb-8619-4cf5-b026-499f3fd7289e


# ╔═╡ 6dd0acce-a14c-41d5-b85d-c2945751983f
begin
sampling_strategy::Vector{Tuple{Vector{Int64}, Vector{Int64}}} = []
sampling_strategyᵧ::Vector{Tuple{Vector{Int64}, Vector{Int64}}} = []

training_indexlist = collect(1:nrow(tₐ′))
training_indexlistᵧ = collect(1:nrow(tᵧ′))

for i in 1:10
sample_buildings = sample(
	MersenneTwister(i),
	train_buildings, 
	n_buildings
)
@info sample_buildings[1:5]
	
building_boolmap = map( x -> x ∈ sample_buildings, tₐ′[:,"Property Id"] );
building_boolmapᵧ = map( x -> x ∈ sample_buildings, tᵧ′[:,"Property Id"] );

building_cv = training_indexlist[building_boolmap]
building_cvᵧ = training_indexlistᵧ[building_boolmapᵧ]

building_train = training_indexlist[.!building_boolmap]
building_trainᵧ = training_indexlistᵧ[.!building_boolmapᵧ]

push!(sampling_strategy, (building_train, building_cv))
push!(sampling_strategyᵧ, (building_trainᵧ, building_cvᵧ))
end
end

# ╔═╡ a346a657-5fe7-4f63-a331-c469e51e34c1
begin
# Xtoe = DataFrames.select(DataFrames.select(toe₁, exclusion_terms), Not("electricity_mwh"))
# ytoe = DataFrames.select(toe₁, exclusion_terms)[:,"electricity_mwh"]

# mtoe = machine(m_tree, Xtoe, ytoe; cache=false)
# # MLJ.fit_only!(mtoe)
# evaluate(m_tree, Xtoe, ytoe, resampling=sampling_strategy, measure=rmse, verbosity=0)
# println(100 * final_size / initial_size)
end

# ╔═╡ 9ec28f12-d242-48fa-be75-c96af3c9a324
m_tree

# ╔═╡ 88e0db0e-28bb-457a-9a12-3f020074f4a4
begin
n = 120
roundsᵣ = range(m_tree, :nrounds, lower=20, upper=90, scale=:linear);
max_depthᵣ = range(m_tree, :max_depth, lower=3, upper=7, scale=:linear);
ηᵣ = range(m_tree, :eta, values=[0.15, 0.175]);
γᵣ = range(m_tree, :gamma, lower=0, upper=1.0, scale=:linear);
αᵣ = range(m_tree, :alpha, lower=0, upper=15, scale=:linear);
λᵣ = range(m_tree, :lambda, lower=0.0, upper=1.0, scale=:linear);
rowsampleᵣ = range(m_tree, :rowsample, lower=0.35, upper=1.0, scale=:linear);
colsampleᵣ = range(m_tree, :colsample, lower=0.35, upper=1.0, scale=:linear);
nbinsᵣ = range(m_tree, :nbins, lower=16, upper=64, scale=:linear);

fitting_terms = [
	roundsᵣ,
	max_depthᵣ,
	nbinsᵣ,
	# ηᵣ,
	γᵣ,
	αᵣ,
	λᵣ,
	rowsampleᵣ,
	colsampleᵣ
]
end;

# ╔═╡ fc367d8c-0f9b-4d01-a130-be869801412f
ComputationalResources.CPUThreads()

# ╔═╡ 16792372-04b2-4a63-b869-c6e37521a324
m_tree_tuning = TunedModel(
	model=m_tree,
	tuning=LatinHypercube(),
	n=n,
	resampling=sampling_strategy,
	range=fitting_terms,
	measure=rms,
    # acceleration=CPUThreads(),
    # acceleration_resampling=CPUThreads(),
	train_best=true
);

# ╔═╡ 1a524b30-c267-4551-813a-4fa5eda2bee1
function electrictrain(modelname::String, terms::Vector{String})
tᵥ = DataFrames.select(
	DataFrames.select(tₐ′, terms), 
	exclusion_terms
)

X = DataFrames.select(tᵥ, Not("electricity_mwh"))
y = tᵥ[:,"electricity_mwh"]

mach = machine(m_tree_tuning, X, y);
fit!(mach, verbosity=0)

# test
teᵣ = DataFrames.select(teₐ′, terms)
teᵥ = DataFrames.select(
	teᵣ, 
	exclusion_terms
)
Xₑ = DataFrames.select(teᵥ, Not("electricity_mwh"))
yₑ = teᵥ[:,"electricity_mwh"]

teᵣ.prediction = MLJ.predict(mach, teᵥ)
teᵣ.recorded = yₑ
teᵣ.model = repeat([modelname], nrow(teᵥ))

return (mach, 0, teᵣ)
end

# ╔═╡ f301972b-feff-485a-b385-943fb434f0fd
m_tree_tuningᵧ = TunedModel(
	model=m_tree,
	tuning=LatinHypercube(),
	n=n,
	resampling=sampling_strategyᵧ,
	range=fitting_terms,
	measure=rms,
    # acceleration=CPUThreads(),
    # acceleration_resampling=CPUThreads(),
	train_best=true
);

# ╔═╡ 425e4473-5e8b-4264-8c24-6e06ec2a053b
# fitted_params(mach).best_model

# ╔═╡ a138dfc8-5e9b-4b64-864a-5c292b75186d
# report(mach).best_history_entry

# ╔═╡ 8653930c-43d6-4062-8be0-764ea5303abe
# # begin
# toeₑ = DataFrames.select(teₐ′, term₇)
# Xtoeₑ = DataFrames.select(DataFrames.select(toeₑ, exclusion_terms), Not("electricity_mwh"))

# explain = copy(Xtoeₑ)
# reference = copy(Xtoeₑ)
# sample_size = 100
# end

# ╔═╡ 22660260-d2d5-4f3c-bb8f-173f06a4689e
function predict_function(model, data)
  data_pred = DataFrame(y_pred = MLJ.predict(model, data))
  return data_pred
end

# ╔═╡ a43eb841-aeb8-4ee3-b8b2-72a83741b04f
# begin
# # Compute stochastic Shapley values.
# data_shap = ShapML.shap(
# 	explain = explain[2000:4000,:],
#     reference = reference,
# 	model = mach,
#     predict_function = predict_function,
# 	sample_size = sample_size,
# 	parallel = :features,
#     seed = 1
# )

# data_plot = sort(
# 	combine(
# 		groupby(dropmissing(data_shap), :feature_name), :shap_effect .=> x -> mean(abs.(x)), renamecols=false), :shap_effect, rev=true
# )
# end

# ╔═╡ 0d3141a3-6fd9-4a8a-9531-56d15fae5d37
# Gadfly.plot(
# 	data_plot, 
# 	y = :feature_name, 
# 	x = :shap_effect,
# 	Coord.cartesian(yflip = true),
# 	Scale.y_discrete, 
# 	Geom.bar(
# 		position = :dodge, 
# 		orientation = :horizontal
# 	),
# 	Theme(bar_spacing = 1mm),
# 	Guide.ylabel(nothing),
# 	Guide.title("Feature Importance - Mean Absolute Shapley Value"),
# 	Theme(
# 		default_color="black"
# 	)
# )

# ╔═╡ 801fdb3b-698e-4555-8686-98822ef0028b
# begin
# data_plot₂ = data_shap[data_shap.feature_name .== "VH", :]  # Selecting 1 feature for ease of plotting.

# baseline = round(data_shap.intercept[1], digits = 1)
# p_points = Gadfly.layer(data_plot₂, x = :feature_value, y = :shap_effect, Geom.point(), Theme(alphas=[0.5]))
# p_line = Gadfly.layer(data_plot₂, x = :feature_value, y = :shap_effect, Geom.smooth(method = :loess, smoothing = 0.5),
#                style(line_width = 0.75mm,), Theme(default_color = "black"))
# p = Gadfly.plot(p_line, p_points, Guide.xlabel("Feature value"),
#          Guide.title("Feature Effect - $(data_plot₂.feature_name[1])"))
# end

# ╔═╡ 0c3fbc2d-f6cf-4bd8-be25-492ae1bffc07
# Tree = @load DecisionTreeRegressor pkg=DecisionTree verbosity=0;

# ╔═╡ 0767fb27-a9d2-4382-9c87-cfe230351a90
m_tree

# ╔═╡ e7e41d5f-8319-433e-8cfa-be502f84bfd5
Standardizer = @load Standardizer pkg=MLJModels

# ╔═╡ 901c6640-feba-4242-aaeb-687fd2a9f371
#

# ╔═╡ dbae822c-24fa-4767-aaaa-d6bd8ad700ac
begin
m₀, vₐ′₀, teₐ′₀ = electrictrain("Null", term₀);
@info "NUll Model:" fitted_params(m₀).best_model
end

# ╔═╡ b22da8d5-9899-4569-bb82-482ee8f72253
@info "Full Data Model:" fitted_params(m₀).best_model

# ╔═╡ 808adca1-648b-4831-968f-5951eb024477
t₁₁ = combine(groupby(teₐ′₀, :date), [:prediction, :recorded] .=> sum, renamecols=false);

# ╔═╡ 325d32d2-646f-494b-b747-8e0cc703cb3d
t₁₁

# ╔═╡ d566fcaf-53ed-4a4d-bf71-891a6fdfe311
md"""
##### Individual Building vs Aggregation
"""

# ╔═╡ dee3a9ec-79f5-4917-ad40-2e4ddcdd423d
md"""
###### ERA5
"""

# ╔═╡ ad174123-117e-4365-9ede-50456d445fce
term₁ = unique([names(era5)..., electricity_terms...])

# ╔═╡ ee57d5d5-545e-4b70-91d9-b82a108f854b
md"""
###### NOAA
"""

# ╔═╡ ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
term₂ = unique([names(noaa)..., electricity_terms...])

# ╔═╡ 30e3c82a-fc70-4922-a7f7-cc1bec0e7d1c
begin
m₂, vₐ′₂, teₐ′₂ = electrictrain("NOAA", term₂);
@info "NOAA Model:" fitted_params(m₂).best_model
end

# ╔═╡ 448c4152-ca34-458e-a35d-3a3a569d96ec
t₁₂ = combine(groupby(teₐ′₂, :date), [:prediction, :recorded] .=> sum, renamecols=false);

# ╔═╡ af170bc6-cbd9-4f45-aca8-0a1900dd4ccf
# p₁₂ = Gadfly.plot(
# 	t₁₂,
# 	Gadfly.layer(
# 		x=:date,
# 		y=:prediction,
# 		Theme(default_color="indianred"),
# 		Geom.point,
# 		Geom.line
# 	),
# 	Gadfly.layer(
# 		x=:date,
# 		y=:recorded,
# 		Geom.point,
# 		Geom.line
# 	),
# 	Guide.title("Prediction vs Recorded"),
# 	Guide.ylabel("Electricity MWh")
# )

# ╔═╡ 51ecb564-06d5-4767-aa41-3030ca08a6c7
md"""
###### MODIS
"""

# ╔═╡ 2119a637-98d1-4e1f-b25a-27d3dc42e636
nrow(tₐ′)

# ╔═╡ 763edcce-0696-4670-a8cf-4963cfe70975
term₃ = unique([names(lst)..., electricity_terms...])

# ╔═╡ 4c5c4ad3-2bbc-4e26-bea9-ff254b737ca8
# m₃, vₐ′₃, teₐ′₃ = electrictrain("Landsat8", term₃);

# ╔═╡ cde0836c-3dbd-43a9-90fd-c30e5985acf7
md"""
###### EPW
"""

# ╔═╡ 5caecfab-3874-4989-8b3d-c65b53361c62
term₄ = unique([names(epw)..., electricity_terms...])

# ╔═╡ 7fdd86d3-1520-4ff4-8d84-87b4585bce65
m₄, vₐ′₄, teₐ′₄ = electrictrain("EPW", term₄);

# ╔═╡ b9b8a050-1824-414e-928d-b7797760f176
md"""
###### Landsat8
"""

# ╔═╡ e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
term₅ = unique([names(landsat8)..., electricity_terms...])

# ╔═╡ e783c591-ea50-4508-8d28-524287466621
begin
m₅, vₐ′₅, teₐ′₅ = electrictrain("Landsat8", term₅);
@info "Landsat Model:" fitted_params(m₅).best_model
end

# ╔═╡ a6cada88-c7c9-495d-8806-2503e674ec39
md"""
###### VIIRS
"""



# ╔═╡ 7c84422b-d522-4f11-9465-058f41a4266f
term₆ = unique([names(viirs)..., electricity_terms...])

# ╔═╡ e44de134-6bb6-4d26-9657-963cd587e40a
m₆, vₐ′₆, teₐ′₆ = electrictrain("VIIRS", term₆);

# ╔═╡ 03d2381d-e844-4809-b5a9-048c7612b7e2
md"""
###### SAR
"""

# ╔═╡ 98d04357-be23-4882-b5b5-8a6d924b7876
term₇ = unique([names(sar)..., electricity_terms...])

# ╔═╡ f1306758-b100-4fca-a030-a0eb2efc34ac
toe₁ = DataFrames.select(tₐ′, term₇)

# ╔═╡ 8f3bcc9f-80e4-455a-bea1-c52feea40191
begin
m₇, vₐ′₇, teₐ′₇ = electrictrain("SAR", term₇);
end

# ╔═╡ 5440fd5f-dfc4-40ed-a614-076904f10220
@info "SAR Model:" fitted_params(m₇).best_model

# ╔═╡ 447e0a54-a6b4-492b-8db5-aea294c6d45d


# ╔═╡ e84033ac-3b34-4e1f-a72b-9dfd937382c1
md"""
###### Dynamic World
"""

# ╔═╡ 13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
term₈ = unique([names(dynam)..., electricity_terms...])

# ╔═╡ 183175cd-b909-4e03-84d0-e7169a822f89
m₈, vₐ′₈, teₐ′₈ = electrictrain("Dynamic World", term₈);

# ╔═╡ 0c1da8de-971b-44dd-84b6-0a236fafe027
md"""
###### CMIP
"""

# ╔═╡ 7b95effc-8729-425e-8443-c9ebf7b02b97
# term₉ = unique([names(cmip)..., electricity_terms...])

# ╔═╡ 3962b939-ff46-4362-a30a-dda8cf84133d
# m₉, vₐ′₉, teₐ′₉ = electrictrain("CMIP", term₉);

# ╔═╡ 6c09c553-0c77-410a-aae9-2004c9768d8b
md"""
###### CMIP + SAR
"""

# ╔═╡ 4c6dbaa4-0549-4775-b2d8-7c58cdafc24d
# cmip_α = rename(
# 	DataFrames.select(cmip, ["Property Id","date","tasmin_median_f₃","tasmax_median_f₃"]),
# 	:tasmin_median_f₃ => :tmin,
# 	:tasmax_median_f₃ => :tmax
# )

# ╔═╡ 291d3eb9-df30-4df4-bd9a-dce4603f6fd2
# termᵪₛ = unique([names(cmip_α)...,names(sar)...,electricity_terms...])

# ╔═╡ 7fbfb83f-e9e1-4701-a7ec-e60cb79b806c
# tₔ = clean(dropmissing(innerjoin(
# 	tₐ,
# 	[cmip_α, sar]...,
# 	on=["Property Id", "date"]
# )));

# ╔═╡ ff177d30-ecd8-4935-ba10-dc7ea0b79a06
# teₔ = clean(dropmissing(innerjoin(
# 	teₐ,
# 	[cmip_α, sar]...,
# 	on=["Property Id", "date"]
# )));

# ╔═╡ 82a444a7-6bdb-4c1a-b9dd-aa85e5056faa
exclusion_terms

# ╔═╡ bb329d66-c51e-423d-b28b-a15cb38d7ed7
md"""
###### Full Dataset
"""

# ╔═╡ 590138f1-7d6f-4e78-836d-258f7b4f617e
termₑ = names(tₐ′)

# ╔═╡ e548a3d5-578d-477f-a192-14d8e4c83853
termₑ

# ╔═╡ 1e09be56-ef2a-4ef1-a181-d1f07dcc4ced
begin
mₑ, vₐ′ₑ, teₐ′ₑ = electrictrain("Full Data", termₑ);
@info "Full Data Model:" fitted_params(mₑ).best_model
end

# ╔═╡ 685169a0-e65c-439c-a5ba-068c83258200
test_terms = [teₐ′₀,teₐ′₂,teₐ′₄,teₐ′₅,teₐ′₆,teₐ′₇,teₐ′₈];

# ╔═╡ d0c234cc-9620-4c9c-bfae-f32b68b9d31f
common_terms = [
	"Property Id",
	"date",
	"month",
	"prediction",
	"recorded",
	"model",
	"zone"
]

# ╔═╡ ae507fc7-5a13-42ed-a477-79d0b11c2efb
begin
resultsₑ = vcat([ DataFrames.select(x, common_terms) for x in test_terms ]...);
resultsₑ.diff = abs.(resultsₑ.prediction .- resultsₑ.recorded);
end;

# ╔═╡ 982b25d3-fba6-4dac-9340-d5e87075f95a
begin
benchmark_improvements = leftjoin(
	filter( x -> x.model == "EPW", resultsₑ),
	DataFrames.select(filter( x -> x.model == "SAR", resultsₑ), ["Property Id", "date","diff"]),
	on=["Property Id","date"],
	makeunique=true
)
benchmark_improvements.improvement = -100 .* (benchmark_improvements.diff_1 .- benchmark_improvements.diff) ./ benchmark_improvements.diff
benchmark_propertydelta = combine(groupby(benchmark_improvements, "Property Id"), :improvement => mean, renamecols=false)
end

# ╔═╡ 19ffec1f-43f7-49af-91be-2553d8998951
begin
resultsₑ_combined = combine(groupby(resultsₑ, ["month","model","zone"]), 
	:diff => mean,
	nrow,
	renamecols=false
)
resultsₑ_combined.fuel = repeat(["Electricity"], nrow(resultsₑ_combined))
end;

# ╔═╡ 0773b0fe-5a8a-4079-9e14-1ea5520a4bdb
r₁ = Gadfly.plot(
	resultsₑ_combined,
	x=:month,
	y=:diff,
	color=:model,
	ygroup=:zone,
	Geom.subplot_grid(Geom.smooth, free_y_axis=true),
	Guide.title("Seasonality Benefits by Zone - Electricity"),
	Guide.xlabel("Month"),
	Guide.ylabel("Prediction Error - ΔMWh")
)

# ╔═╡ fe9f7509-2ba7-45f2-bcf5-84312660d754
draw(
	PNG(
		joinpath(output_dir, "seasonality_benefits.png"), 
		15cm, 
		15cm,
		dpi=500
	), r₁
)

# ╔═╡ 2be996a7-d960-4939-91c7-9f1a42d35b49
names(resultsₑ)

# ╔═╡ 9b3f7260-58f5-45d5-8d53-cb43d739acd9
building_metrics = filter(x -> x.improvement > 0, dropmissing(leftjoin(
	buildings,
	benchmark_propertydelta,
	on="Property Id"
), :improvement))

# ╔═╡ 5eebab7e-7cc4-410e-a9ae-a628644edbeb
maximum(building_metrics.improvement)

# ╔═╡ 55915fe9-b1f4-4b45-b418-65c7af973f4f
standardize(
	ZScoreTransform, 
	building_metrics.improvement
)

# ╔═╡ a5a9d0e7-2a1f-4b83-a790-6148a2029490
begin
distance_colors = cgrad(:Greens)
distance_colormap = distance_colors[standardize(
	UnitRangeTransform, 
	building_metrics.improvement
)]
end

# ╔═╡ d395b477-b536-4603-bc45-b105b3463042
begin
	source_num = 4326
	target_num = 32618
	pm_num = 3857
	source = ArchGDAL.importPROJ4("+proj=longlat +datum=WGS84 +no_defs +type=crs")
	target = ArchGDAL.importPROJ4("+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs +type=crs")
	pm = ArchGDAL.importEPSG(pm_num)

	@info "Source EPSG: " source
	@info "Target EPSG: " target
end

# ╔═╡ ec420749-2c5e-4b70-a476-ff8be50f6fca
 # combine(groupby(filter( x -> x.cvrmse < Inf, test_suiteₜₑ ), :model), [:cvrmse, :nmbe, :cvstd, :rmse] .=> mean, renamecols=false)

# ╔═╡ ed696113-737b-46f5-bfb1-c06d430a83ac
seasons = Dict(
	1 => "Winter",
	2 => "Winter",
	3 => "Spring",
	4 => "Spring",
	5 => "Spring",
	6 => "Summer",
	7 => "Summer",
	8 => "Summer",
	9 => "Autumn",
	10 => "Autumn",
	11 => "Autumn",
	12 => "Autumn"
)

# ╔═╡ 8b3175d4-6938-4823-8a91-77cde7c31c2a
prediction_terms = [:prediction, :recorded, :month, :model, :zone]

# ╔═╡ 9d1ea09f-3c42-437f-afbc-21a70296a3ba
interest_terms = [:weather_station_distance]

# ╔═╡ 05a040cf-f6ac-42b2-bcb1-599af5a26038
test_termsₑ = vcat([ DataFrames.select(x, prediction_terms, interest_terms) for x in test_terms ]...);

# ╔═╡ 41751922-2646-45fd-b6e0-7ca2109cc642
function mse(x)
	(sum(x.^2) / length(x))^0.5
end

# ╔═╡ c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
begin
# Null
Ξ₀ = hcat(DataFrames.select(teₐ′₀, prediction_terms), DataFrames.select(teₐ′, interest_terms))

# Landsat8
Ξ₅ = hcat(DataFrames.select(teₐ′₅, prediction_terms), DataFrames.select(teₐ′, interest_terms))

# NOAA
Ξ₂ = hcat(DataFrames.select(teₐ′₂, prediction_terms), DataFrames.select(teₐ′, interest_terms))
	
#EPW
Ξ₄ = hcat(DataFrames.select(teₐ′₄, prediction_terms), DataFrames.select(teₐ′, interest_terms))
	
Ξ = vcat(Ξ₀, Ξ₅, Ξ₂, Ξ₄)

Ξ.error = Ξ.prediction .- Ξ.recorded
Ξ′ = combine(groupby(Ξ, [:month, :model]), :error => mean, renamecols=false)
Ξ′.color = map(x-> seasons[x], Ξ′.month)

p₄ = Gadfly.plot(
	sort(Ξ′, [:model,:month]),
	x=:month,
	y=:error,
	color=:color,
	ygroup=:model,
	yintercept=[0],
	Guide.ylabel("Δ Prediction - Recorded"),
	Guide.xlabel("Month"),
	Guide.title("Mean Electricity Error by Season"),
	Guide.colorkey(title="Season"),
	Geom.subplot_grid(
		# Guide.yticks(ticks=0.0:0.001:0.01),
		Guide.xticks(ticks=1:1:12),
		Geom.point,
		Geom.line,
		Geom.hline(color=["pink"], style=:dash),
		# Geom.point,
		# free_y_axis=true,
	),	# Coord.cartesian(ymin=-0.1, ymax=0.1, aspect_ratio=1.5),
	# Theme(default_color="black")
)
end

# ╔═╡ 9156e2e0-1a15-4176-a664-5a03b725c24e
sort(Ξ′, [:model,:month])

# ╔═╡ e98994c4-78ab-4d08-a1a0-e8fb3f3e79a7
md"""##### This gives a snapshot of the relative performance of the model with increasing distance from the weather station"""

# ╔═╡ 11277ca6-3bbf-403f-8cfc-2aabf94069f7
p₇ = Gadfly.plot(
	test_termsₑ,
	color=:model,
	x=:weather_station_distance,
	y=abs.(test_termsₑ.prediction .- test_termsₑ.recorded),
	xgroup=:zone,
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Electricity - MAE vs Weather Station Distance"),
	Geom.smooth(smoothing=0.9),
	Scale.y_log,
	# Coord.cartesian(ymin=-3.3, ymax=-2.95),
	# Guide.yticks(ticks=-3.3:0.05:-2.95),
	# Geom.subplot_grid(
	# 	Geom.smooth(smoothing=0.7),
	# 	# Guide.yticks(ticks=-0.15:0.1:0.15),
	# 	# Coord.cartesian(ymin=-0.2, ymax=0.2),
	# ),
)

# ╔═╡ b0f7e467-684b-41dd-b046-ed378dded683
resₑ = filter( x -> x.zone == "Residential", test_termsₑ );

# ╔═╡ 38b3911a-8db5-47f4-bd1b-0d77a207188f
Gadfly.plot(
	resₑ,
	x=:weather_station_distance,
	y=resₑ.prediction .- resₑ.recorded,
	color=:model,
	Geom.smooth(smoothing=0.6),
	Guide.xlabel("Weather Station Distance (m)"),
	Guide.title("Electricity Degredation with Distance - Residential")
)

# ╔═╡ 28d2660e-258d-4583-8224-c2b4190f4140
draw(PNG(joinpath(output_dir, "learning_results_electricity.png"), 
	20cm, 
	10cm, 
	dpi=600
), p₇)

# ╔═╡ fbe43a2a-f9b0-487e-a579-645c2f40736e
# ∮ = filter(
# 	x -> ∈(x["model"], ["Basic","NOAA","ERA5","EPW","Complete"]),
# 	vₛ,
# );

# ╔═╡ 7b05ec74-6edc-4f44-bf9d-9389d4029494
# p₅ = Gadfly.plot(
# 	∮,
# 	x=:weather_station_distance,
# 	y=:cvrmse,
# 	yintercept=[0, 15],
# 	color=:model,
# 	Coord.cartesian(ymin=-5, ymax=120),
# 	# Scale.y_log,
# 	Guide.ylabel("CVRMSE"),
# 	Guide.xlabel("Distance from Weather Station (m)"),
# 	Guide.title("Model Electricity Predictions"),
# 	Geom.smooth(smoothing=0.5),
# 	Geom.hline(style=[:dash, :dash], color=["pink","red"])
# )

# ╔═╡ c546d3af-8f24-40bd-abfa-e06c708c244e
md"""
#### Want to explore how the predictions may have benefitted
"""

# ╔═╡ 5db964da-1ebc-478a-a63a-cf4f713c7aa8
md"""
For this study, going to explore how SAR and CMIP might be used in collaboration
"""

# ╔═╡ 778dc2a7-4e9b-4f97-b34d-6a7adc38abc2


# ╔═╡ 5953c75a-0e6d-40c4-8e4b-b8de6920acf3


# ╔═╡ c7874fdb-2ba4-4dbf-89b4-0b3af493b256


# ╔═╡ 23525c22-3fc4-4a4e-b2c6-601463a5df31


# ╔═╡ b98baad1-dad8-464c-a18a-1baf01962164
md"""
## Natural Gas
"""

# ╔═╡ 3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
md"""Null Model"""

# ╔═╡ d437be5f-ec26-466b-b099-5e1cc8816cb5
# function gastrain(modelname::String, terms::Vector{String})
# 	tᵥ = DataFrames.select(tᵧ′, terms)
# 	mᵥ = training_pipeline(
# 		DataFrames.select(tᵥ, exclusion_terms),
# 		"naturalgas_mwh",
# 		m
# 	)

# 	# validation
# 	vᵥ = DataFrames.select(vᵧ′, terms)
# 	vᵥ.prediction = validation_pipeline(
# 		DataFrames.select(vᵥ, exclusion_terms),
# 		"naturalgas_mwh",
# 		mᵥ
# 	);
	
# 	vᵥ.recorded = vᵥ.naturalgas_mwh
# 	vᵥ.model = repeat([modelname], nrow(vᵥ))
	
# 	# test
# 	teᵥ = DataFrames.select(teᵧ′, terms)
# 	teᵥ.prediction = validation_pipeline(
# 		DataFrames.select(teᵥ, exclusion_terms),
# 		"naturalgas_mwh",
# 		mᵥ
# 	);
	
# 	teᵥ.recorded = teᵥ.naturalgas_mwh
# 	teᵥ.model = repeat([modelname], nrow(teᵥ))

# 	leftjoin!(
# 		teᵥ,
# 		building_zones,
# 		on="Property Id"
# 	)

# 	return mᵥ, vᵥ, teᵥ
# end

# ╔═╡ dd8e709c-b945-447f-9b3d-8bf9d9e0249c
function gastrain(modelname::String, terms::Vector{String})
pterm = "naturalgas_mwh"
tᵥ = DataFrames.select(
	DataFrames.select(tᵧ′, terms), 
	exclusion_terms
)

X = DataFrames.select(tᵥ, Not(pterm))
y = tᵥ[:,pterm]

mach = machine(m_tree_tuningᵧ, X, y);
fit!(mach, verbosity=0)

# test
teᵣ = DataFrames.select(teᵧ′, terms)
teᵥ = DataFrames.select(
	teᵣ, 
	exclusion_terms
)
Xₑ = DataFrames.select(teᵥ, Not(pterm))
yₑ = teᵥ[:,pterm]

teᵣ.prediction = MLJ.predict(mach, teᵥ)
teᵣ.recorded = yₑ
teᵣ.model = repeat([modelname], nrow(teᵥ))

return (mach, 0, teᵣ)
end

# ╔═╡ 2426e2d6-e364-4e97-bee8-7defb1e88745
md"""
##### Null
"""

# ╔═╡ 9c8f603f-33c6-4988-9efd-83864e871907
termᵧ₀ = unique([naturalgas_terms...])

# ╔═╡ 2d337665-65e8-4f81-83a4-7328fd186352
begin
pterm = "naturalgas_mwh"
tᵥ = DataFrames.select(
	DataFrames.select(tᵧ′, termᵧ₀), 
	exclusion_terms
)

X = DataFrames.select(tᵥ, Not(pterm))
y = tᵥ[:,pterm]
end

# ╔═╡ 2bd22bd5-b6e2-450a-899d-41d647a9b3e3
# begin
# mach = machine(m_tree_tuning, X, y);
# fit!(mach, verbosity=0)
# end

# ╔═╡ ecbb707c-bea0-4b86-a7b8-a4f55027c05d


# ╔═╡ 20a6b910-4f0b-4c78-af6b-d76e55025297
mᵧ₀, vᵧ′₀, teᵧ′₀ = gastrain("Null", termᵧ₀);

# ╔═╡ 8c401b96-6de6-48cb-9a85-ff0620fec402
describe(filter( x -> x.recorded > 0, teᵧ′₀))

# ╔═╡ 991815d7-09da-423c-b14a-a8a3fcf662e4
md"""
##### ERA5
"""

# ╔═╡ bc64b624-d8d2-480a-a698-092aea0a74b2
termᵧ₁ = unique([names(era5)..., naturalgas_terms...])

# ╔═╡ 0e876bab-6648-4a67-b571-dc82a7bdf8f1
md"""##### NOAA"""

# ╔═╡ 3763ad63-003e-495f-aa90-0db525412c62
termᵧ₂ = unique([names(noaa)..., naturalgas_terms...])

# ╔═╡ 061203a0-9ccc-4988-b4f0-11743ecec084
fitted_params(mᵧ₀).best_model

# ╔═╡ a9a9ea9d-5a97-4260-8e02-a27732928e61
mᵧ₂, vᵧ′₂, teᵧ′₂ = gastrain("NOAA", termᵧ₂);

# ╔═╡ b1e7c19f-e2f3-4f17-9933-e5c5e90e386d
fitted_params(mᵧ₂).best_model

# ╔═╡ 8f849bc4-addc-4a0d-9105-48307068febc
Plots.plot(mᵧ₂)

# ╔═╡ 5ed2db5f-c499-4749-87f2-4bb881accf16
begin
noaaᵧₜ = DataFrames.select(teᵧ′₂, ["Property Id","date","prediction","recorded"])
noaaᵧₜ.error = noaaᵧₜ.prediction .- noaaᵧₜ.recorded
select!(noaaᵧₜ, Not([:prediction, :recorded]))
end;

# ╔═╡ c4076bd0-8718-4641-9d09-2c3b71aff1e3


# ╔═╡ 202c9072-0a7b-454d-9112-4ecc0a03c61b
md"""##### MODIS"""

# ╔═╡ 434d7e0d-583d-498b-9b6c-a72fa3775b3c
termᵧ₃ = unique([names(lst)..., naturalgas_terms...])

# ╔═╡ 2857461c-d24e-4b04-91f1-80cd842eeaa4
# mᵧ₃, vᵧ′₃, teᵧ′₃ = gastrain("LST", termᵧ₃);

# ╔═╡ a78a6fd5-4b45-43fc-a733-aef4fd14eb42
md""" ##### EPW"""

# ╔═╡ 1a4471e7-24ad-4652-9f3f-6eef92c781d5
termᵧ₄ = unique([names(epw)..., naturalgas_terms...])

# ╔═╡ 8c074d96-ee63-45fb-bbbf-135c40b66a09
mᵧ₄, vᵧ′₄, teᵧ′₄ = gastrain("EPW", termᵧ₄);

# ╔═╡ 19af17cf-193d-48a7-8db6-e20a3139692f
report(mᵧ₄)

# ╔═╡ 9ef995fb-a591-45df-af67-df63808b119d
fitted_params(mᵧ₄).best_model

# ╔═╡ 399d45a5-a217-413a-8ba8-76b93e246a89
maximum(teᵧ′.date)

# ╔═╡ 5b852b2d-d86e-401b-8251-b21fa080ed1f
begin
epwᵧₜ = DataFrames.select(teᵧ′₄, ["Property Id","date","prediction","recorded"])
epwᵧₜ.error = epwᵧₜ.prediction .- epwᵧₜ.recorded
select!(epwᵧₜ, Not([:prediction, :recorded]))
end

# ╔═╡ e4e7cfed-d150-4615-bb77-a8db4b395011
begin
ŷᵧ = leftjoin(
	noaaᵧₜ,
	epwᵧₜ,
	on=["Property Id","date"],
	makeunique=true
)
ŷᵧ.d = ŷᵧ.error_1 .- ŷᵧ.error
end

# ╔═╡ e0ff3466-1a61-45aa-a5b8-9300e81ccc9a
combine(groupby(ŷᵧ, :date), :d => mean)

# ╔═╡ e1226518-0341-4a7c-bdfc-cc93be354638
# comprehensive_datalist = [
# 	dynam,
# 	noaa,
# 	era5,
# 	epw, 
# 	lst, 
# 	sentinel_1C,
# 	sar,
# 	viirs
# ];

# ╔═╡ 04d720ce-1de5-4c8b-bd2f-0d0a5e8ed271
md"""
##### Dynamic World
"""

# ╔═╡ 6109d56f-ce01-4b10-bb13-1e2eb0ccf990
termᵧ₅ = unique([names(dynam)..., naturalgas_terms...])

# ╔═╡ bc6c39e0-f878-4bea-8453-c845f0d3eba9
mᵧ₅, vᵧ′₅, teᵧ′₅ = gastrain("Dynamic World", termᵧ₅);

# ╔═╡ 8ae63dda-a9d2-47e2-a89d-05fe8c11383b
md"""
##### Sentinel-2
"""

# ╔═╡ 7e8458e4-15be-47ac-8c08-1ff042c5c9b3
termᵧ₆ = unique([names(landsat8)..., naturalgas_terms...])

# ╔═╡ ad7d7a32-861d-48be-af41-833e74023ef6
mᵧ₆, vᵧ′₆, teᵧ′₆ = gastrain("Landsat8", termᵧ₆);

# ╔═╡ edceb483-eb33-4bf9-975d-3dc6f18cffe9
md"""
##### SAR
"""

# ╔═╡ b5178580-013b-4686-8bfc-c1f7395620b2


# ╔═╡ 79811ec5-d713-4dc4-b1a6-b0ea656633fd
termᵧ₇ = unique([names(sar)..., naturalgas_terms...])

# ╔═╡ 7563fe9c-5142-455e-a7ce-559a14b92f28
mᵧ₇, vᵧ′₇, teᵧ′₇ = gastrain("SAR", termᵧ₇);

# ╔═╡ bafb366d-c0fb-428d-8188-7a2c6e100617
md"""
##### VIIRS
"""

# ╔═╡ 222de0a7-0cef-4b01-a683-3bdd8f892f88
termᵧ₈ = unique([names(viirs)..., naturalgas_terms...])

# ╔═╡ eb08a786-9f78-49cc-86b2-418e881a8b2a
mᵧ₈, vᵧ′₈, teᵧ′₈ = gastrain("VIIRS", termᵧ₈);

# ╔═╡ bac2785d-d692-4524-86c5-dc183f07fe86
md"""
##### CMIP
"""

# ╔═╡ c8452580-4680-4ca1-ae58-4c17611e4351
# termᵧ₉ = unique([names(cmip)..., naturalgas_terms...])

# ╔═╡ 73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
# mᵧ₉, vᵧ′₉, teᵧ′₉ = gastrain("CMIP", termᵧ₉);

# ╔═╡ 26086070-08a3-4ce8-b8e6-cd2cfc83e44d
md"""
##### Full Data
"""

# ╔═╡ 1af87fef-8c88-4e34-b1d4-8bccd7881473
mᵧₑ, vᵧ′ₑ, teᵧ′ₑ = gastrain("Full Data", names(tᵧ′));

# ╔═╡ da9a6ba8-083b-40b3-9bd6-089688ff7f73
# begin
# 	# now want to explore how permuted weather data might influence the quality
# 	termᵧₒ = DataFrames.select(tᵧ′₄, naturalgas_terms)
# 	epwᵧₒ = DataFrames.select(tᵧ′₄, Not(naturalgas_terms))
# 	epwᵧₒ′ = epwᵧₒ[shuffle(1:nrow(epwᵧₒ)), :]

# 	tᵧₒ′₄ = hcat(termᵧₒ, epwᵧₒ′)


# 	termᵥᵧₒ = DataFrames.select(vᵧ′₄, naturalgas_terms)
# 	epwᵥᵧₒ = DataFrames.select(vᵧ′₄, Not(naturalgas_terms))
# 	epwᵥᵧₒ′ = epwᵥᵧₒ[shuffle(1:nrow(epwᵥᵧₒ)), :]

# 	vᵧₒ′₄ = hcat(termᵥᵧₒ, epwᵥᵧₒ′)
# end;

# ╔═╡ 433d51c9-9f31-49db-a5a2-e6565888831e


# ╔═╡ c25b404b-0d89-4c4d-bc61-c361fd9d7038
test_termsᵧ = [teᵧ′₀,teᵧ′₂,teᵧ′₄,teᵧ′₅,teᵧ′₆,teᵧ′₇,teᵧ′₈,teᵧ′ₑ];

# ╔═╡ c15b52c1-5d06-4fea-8645-91f92bfa716b
begin
resultsᵧ = vcat([ DataFrames.select(x, common_terms) for x in test_termsᵧ ]...);
resultsᵧ.diff = abs.(resultsᵧ.prediction .- resultsᵧ.recorded);
end;

# ╔═╡ 07c78fb1-bcf4-40fd-9209-d3220d014912
begin
resultsᵧ_combined = combine(groupby(resultsᵧ, ["month","model","zone"]), 
	:diff => mean,
	nrow,
	renamecols=false
)
resultsᵧ_combined.fuel = repeat(["Natural Gas"], nrow(resultsᵧ_combined))
end;

# ╔═╡ 15b3535f-6776-409d-8e6a-457b4c6eba78
resultsᵧ

# ╔═╡ 0c61beac-7fdb-4f05-a04c-f2a042ba34b2
combine(groupby(resultsᵧ, ["Property Id","model"]), :prediction => median)

# ╔═╡ d96c7a96-0875-4ebd-a1ba-e40db8008aed


# ╔═╡ 43a39165-1f9b-49e7-a4bc-c47583c25b10
interest_terms

# ╔═╡ f2067017-32c8-493b-a9fb-3a89f9e549e4
test_termsₑᵧ = vcat(
	[ DataFrames.select(x, prediction_terms, interest_terms, "Property Id", "area") for x in test_termsᵧ]...
);

# ╔═╡ ee0d413c-a107-4bdc-a08a-52cda6d81573
unique(test_termsₑᵧ.model)

# ╔═╡ 77456db6-b0fa-4ba6-9d41-1c393f7ddee1
# Gadfly.plot(
# 	test_termsₑᵧ,
# 	x=:area,
# 	y=test_termsₑᵧ.prediction .- test_termsₑᵧ.recorded,
# 	Geom.point,
# 	Theme(point_size=1pt)
# )

# ╔═╡ 08b888e3-9d37-43a6-9656-4bf6cb62d324
building_distancesᵧ = unique(DataFrames.select(teᵧ, ["Property Id", "weather_station_distance"]));

# ╔═╡ 4c99c655-ae95-4a28-95c8-e7ca38ddf55f
# Gadfly.plot(
# 	stack(test_suiteₜᵧđ, 2:4),
# 	x=:weather_station_distance,
# 	y=:value,
# 	color=:model,
# 	ygroup=:variable,
# 	Geom.subplot_grid(
# 		Geom.line,
# 		free_y_axis=true
# 	),
# )

# ╔═╡ 372df76a-fd5d-4b84-ad83-988af1e78cf7
md"""
##### And this plot gives intuition into the relative benefit of adding each dataset with distance for natural gas prediction
"""

# ╔═╡ 459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
pᵧ₁ = Gadfly.plot(
	test_termsₑᵧ,
	color=:model,
	x=:weather_station_distance,
	y=abs.(test_termsₑᵧ.prediction .- test_termsₑᵧ.recorded),
	Geom.smooth(smoothing=0.9),
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Natural Gas - MAE vs Weather Station Distance"),
	Scale.x_continuous(minvalue=0, maxvalue=2.5e4),
	Scale.y_log,
	Coord.Cartesian(xmin=0,xmax=2.5e4)
)

# ╔═╡ c05dd607-1ce4-4931-9e23-2a9dad7cc039


# ╔═╡ ae386c52-ff6b-4493-96ff-6c14d1c46db8
draw(PNG(joinpath(output_dir, "naturalgas-mae.png"), 14cm, 10cm, dpi=600), pᵧ₁)

# ╔═╡ 94580726-88ed-4a69-a101-9c792e3ed44b


# ╔═╡ a3a49ba6-7272-4739-b11d-f09692afa46f
test_termsₑ.energy = repeat(["Electricity"],nrow(test_termsₑ));

# ╔═╡ cb96e206-8779-4c02-abd2-74948e13f263
test_termsₑᵧ.energy = repeat(["Natural Gas"],nrow(test_termsₑᵧ));

# ╔═╡ daf769b3-bcd3-4476-bdfb-492a90cd1225
distanceresults = vcat(test_termsₑ, DataFrames.select(test_termsₑᵧ, names(test_termsₑ)));

# ╔═╡ 1ddf99ee-0dfa-4dc2-82e4-8e065e78c992
begin
l₁ = Gadfly.layer(train, x=:weather_station_distance, Geom.density)
l₂ = Gadfly.layer(test, x=:weather_station_distance, Geom.density, Theme(default_color="indianred"))
l₃ = Gadfly.plot(l₁, l₂, Scale.x_continuous(minvalue=0, maxvalue=2.5e4),
	Coord.Cartesian(xmin=0,xmax=2.5e4)
)
end

# ╔═╡ 979e8ca8-105c-4f84-a20b-e3bd729495b4
vstack(pᵧ₁, l₃)

# ╔═╡ 467a06fe-2360-442d-b8b6-bbcbb09681f8
distanceresults.distance_bucket = distanceresults.weather_station_distance .- (distanceresults.weather_station_distance .% 2500);

# ╔═╡ 5d7e9b7a-aa1b-4737-94e4-a5df1de8ecc6
begin
# dr_reduced = filter( x -> x.weather_station_distance < 2e4, distanceresults )
dr_reduced = distanceresults
dr_reduced.err = abs.(dr_reduced.prediction .- dr_reduced.recorded)
end;

# ╔═╡ 6cc61a93-1143-42a1-9b8d-9a3f506db3df
typeof(Vector(ColorSchemes.okabe_ito[1:5]))

# ╔═╡ 20cc6945-d767-41e1-8a6b-1d88084a9781
unique(dr_reduced.model)

# ╔═╡ 4acabffb-ed34-4847-9ce2-c30031392ecc
palette

# ╔═╡ a7ef8e5a-ce04-4a3c-8ef8-421990f176a3
dr = Gadfly.plot(
	dr_reduced,
	color=:model,
	x=:weather_station_distance,
	y=:err,
	ygroup=:energy,
	Geom.subplot_grid(
		Geom.smooth(smoothing=0.75),
		Coord.cartesian(ymin=-3.5, ymax=-2.0),
		free_y_axis=true,
		# Guide.xticks(ticks=collect(0:2.5e3:25e3), orientation=:vertical),
	),
	Guide.ylabel("log(MAE) - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Weather Station Distance (m) vs MAE"),
	Scale.y_log,
	Scale.discrete_color_manual(palette(:tableau_superfishel_stone, 7)...),
	Theme(plot_padding=[2mm])
)

# ╔═╡ 50fee719-874e-42f6-8375-b0bcb9689c3c
dr_reduced

# ╔═╡ faecba7d-a52c-4d8f-a4a5-14f3610cb827
edata = filter(
	x -> x.energy == "Natural Gas" && x.zone == "Residential", 
	dr_reduced
);

# ╔═╡ 923da0d6-8333-4335-ad95-1fdc47cde9cb
eplot = Gadfly.plot(
	edata,
	color=:model,
	x=:weather_station_distance,
	y=:err,
	xgroup=:zone,
	Geom.subplot_grid(
		Geom.smooth(smoothing=0.9),
		# Coord.cartesian(ymin=-3.15, ymax=-3.05),
		free_y_axis=true,
		# Guide.xticks(ticks=collect(0:2.5e3:25e3), orientation=:vertical),
	),
	Guide.ylabel("log(MAE) - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Weather Station Distance (m) vs MAE - Commercial Electricity"),
	Scale.y_log,
	Scale.discrete_color_manual(palette(:tab10)...),
	Theme(plot_padding=[2mm])
)

# ╔═╡ 8dec16f2-f672-4a37-b6c6-6d2e7fad9b7c
draw(PNG(joinpath(output_dir, "model-distance-quality-electricity.png"), 
	18cm,
	15cm,
	dpi=600), eplot)

# ╔═╡ a52dae0c-ce46-4a24-b6fc-02fb5226dcda
ℯ^-3.0

# ╔═╡ ac0375e0-131d-49d8-900e-754fa167c13b
clip

# ╔═╡ 44a94b75-ea19-40a8-a863-e362c651c8a2
dr_reduced[dr_reduced.err .> ℯ^-2.5, :]

# ╔═╡ adbe4ee5-62e7-42c6-baab-e42a3c4af4cc
gdata = filter(
	x -> x.energy == "Natural Gas", 
	dr_reduced
);

# ╔═╡ ab0f2a21-2ce6-4fa5-960b-325ef778304f
gplot = Gadfly.plot(
	gdata,
	color=:model,
	x=:weather_station_distance,
	y=:err,
	xgroup=:zone,
	Geom.subplot_grid(
		Geom.smooth(smoothing=0.9),
		# Coord.cartesian(ymin=-3.0, ymax=-2.5),
		free_y_axis=true,
		# Guide.xticks(ticks=collect(0:2.5e3:25e3), orientation=:vertical),
	),
	Guide.ylabel("log(MAE) - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Weather Station Distance (m) vs MAE - Natural Gas"),
	Scale.y_log,
	Scale.discrete_color_manual(palette(:tab10)...),
	Theme(plot_padding=[2mm])
)

# ╔═╡ 81f70fa7-0240-4868-ac9a-73a119b4e831
draw(PNG(joinpath(output_dir, "model-distance-quality-gas.png"), 
	32cm,
	15cm,
	dpi=600), gplot)

# ╔═╡ cf5ef024-edbb-4468-99af-06064358a8ab
begin
disbu = combine(groupby(distanceresults, [:distance_bucket, :model, :energy]), [:prediction, :recorded] .=> mean, :model => first, :energy => first, renamecols=false);
disbu.delta = disbu.prediction .- disbu.recorded
end;

# ╔═╡ d41e7488-5655-487c-ba31-19be428fff2b
disbu_plot = Gadfly.plot(
	disbu,
	x=:distance_bucket,
	y=:model,
	color=abs.(disbu.prediction .- disbu.recorded),
	ygroup=:energy,
	Geom.subplot_grid(
		Geom.rectbin,
		Guide.xticks(
			ticks=collect(0:2500:25000),
			# orientation=:vertical
		),
	),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("abs(MAE) w.r.t. Weather Station Distance"),
	Guide.colorkey(title="abs(mae)"),
	Scale.ContinuousColorScale(
		palette -> get(ColorSchemes.matter, palette),
	),
)

# ╔═╡ b67fab52-d0e0-45b9-8eba-99b4bca34182
draw(PNG(joinpath(output_dir, "model-distance-quality.png"), 
	15cm,
	12cm, 
	dpi=600), disbu_plot)

# ╔═╡ 62cdf00d-86c4-4ad3-bf64-dfbaa4d6c6e6
disbu

# ╔═╡ 67e2b019-47bd-4473-974d-b5fbfcc55f18
disbu_condensed = combine(groupby(disbu, [:distance_bucket, :energy, :model]), [:prediction, :recorded] .=> mean, renamecols=false)

# ╔═╡ 2dc6205a-f370-4727-bc79-5ac6716e2299


# ╔═╡ 61e30268-26f3-4940-8e9d-b9d5c54f4da0
Gadfly.plot(
	filter( x -> x.model ∈ ["EPW","NOAA", "SAR"], disbu),
	x=:distance_bucket,
	y=:delta,
	color=:model,
	xgroup=:model,
	ygroup=:energy,
	Geom.subplot_grid(Geom.point, Geom.hair),
	Theme(boxplot_spacing=2pt, point_size=1.9pt) 
	# Scale.ContinuousColorScale(
	# 	palette -> get(ColorSchemes.tab10, palette),
	# ),
)

# ╔═╡ 6f78dd2a-c225-41c7-b6a9-5f0925f87d14
md"""
## Combined graphs for both results
"""

# ╔═╡ ea5f7773-0697-4ead-adc9-0eabbd0fa05e
seasonal_results = vcat(resultsₑ_combined, resultsᵧ_combined);

# ╔═╡ b2bf0ddc-a327-4b13-9722-130a0ddebffb
resultsₑ_combined

# ╔═╡ 3ec1c94b-ac88-4f01-af9d-abb956dff6f1
nrow(seasonal_results)

# ╔═╡ 3b0a8743-20b6-4533-a598-09a2c94d6528
seasonal_resultsgraph = Gadfly.plot(
	seasonal_results,
	x=:month,
	y=:model,
	color=:diff,
	xgroup=:fuel,
	ygroup=:zone,
	Scale.ContinuousColorScale(
		palette -> get(ColorSchemes.tableau_orange_blue_white, palette),
		minvalue=-0.05,
		maxvalue=0.05
	),
	Geom.subplot_grid(Geom.rectbin),
	Guide.title("Challenges of Predicting Accurately"),
	Guide.colorkey(title="Δ(%)")
)

# ╔═╡ 62b6cf95-266b-46d4-b478-2e11cdf1acb0
baseline_seasonal_results = filter( x -> x.model == "EPW", seasonal_results );

# ╔═╡ ec90e969-6183-4e98-b2ab-3a53eee7b13b
begin
seasonal_results_norm = leftjoin(
	seasonal_results,
	DataFrames.select(
		rename(baseline_seasonal_results, :diff => :baseline),
		:month,
		:zone,
		:fuel,
		:baseline
	),
	on=[:month, :fuel, :zone]
);

seasonal_results_norm.delta = clamp.(100 .* ((seasonal_results_norm.diff .- seasonal_results_norm.baseline) ./ abs.((seasonal_results_norm.baseline))), -100, 0)
end;

# ╔═╡ f54d3095-8e78-4f36-8f81-a022c6c501d6
# filter(
# 	x -> (x.zone == "Manufacturing" && x.fuel == "Natural Gas" && x.model == "Landsat8"),
# 	seasonal_results_norm
# )

# ╔═╡ 93dd3b75-b24a-4a03-853a-cc6d0a5188e4
sort(seasonal_results_norm, :month)

# ╔═╡ 051fc252-5267-4c8b-8ead-75f37a49a440
seaonsal_predictionresults = Gadfly.plot(
	sort(seasonal_results_norm, :month),
	x=:month,
	y=:model,
	color=:delta,
	xgroup=:fuel,
	ygroup=:zone,
	Scale.ContinuousColorScale(
		palette -> get(reverse(ColorSchemes.matter), palette),
		# minvalue=-0.01,
		# maxvalue=0.01
	),
	Geom.subplot_grid(Geom.rectbin),
	Guide.title("Relative Performace of Data Class compared to EPW"),
	Guide.colorkey(title="Δ(%)")
)

# ╔═╡ 148923dc-8802-4af2-9c4b-ec41ad842366
sort(filter(x -> x.zone == "Residential" && x.fuel == "Electricity", seasonal_results_norm), :delta)

# ╔═╡ be498601-d19d-43ab-8bbb-aed717728e63
describe(seasonal_results_norm.nrow)

# ╔═╡ 96dd4c14-6791-4c8c-80c1-8f0301e34b35
sort(filter( 
	x -> x.fuel == "Electricity" && x.zone == "Residential" && x.model == "SAR"
	, seasonal_results_norm ), [:month])

# ╔═╡ 4df27234-b5b1-4607-b955-c2710257ffd6
draw(PNG(joinpath(output_dir, "seasonal-predictionresults.png"), 
	19cm,
	12cm, 
	dpi=600), seaonsal_predictionresults)

# ╔═╡ 0691bffd-6b86-41ce-8044-5474545f9417
draw(
	PNG(
		joinpath(output_dir, "seasonal_results.png"), 
		20cm, 
		20cm,
		dpi=500
	), seasonal_resultsgraph
)

# ╔═╡ 9e2aef0c-0121-4652-ad40-62ad5e8d0a0e
names(test)

# ╔═╡ c5899e25-25b2-43ec-8e9a-3e7e9af8f50d
testbuilding_information = unique(DataFrames.select(test, ["Property Id","zipcode","zone","weather_station_distance"]));

# ╔═╡ 4eb5ef29-57d2-42e8-8579-411964792f72
function dropna(df)
	filter(:cvstd => x -> !any(f -> f(x), (ismissing, isnothing, isnan)), df)
end

# ╔═╡ 08c44240-6481-4af1-8473-b9cf1d91ef3c
errterm = ["cvrmse","nmbe","rmse"]

# ╔═╡ e85de5cc-219b-4e1d-a16d-1e84d657b870
names(teₐ′₀)

# ╔═╡ 85375b75-1bff-4f4a-a093-4bb55e752c0f
teₐ′₀

# ╔═╡ 03ce5f2e-e11d-4a4a-95c4-37b1c931ec18


# ╔═╡ 329d55b8-eb72-4a1e-a4e8-200fee0e0b9d
md"""
##### Now the test set
---
"""

# ╔═╡ b5ce80a3-e177-4c4f-920b-5dee87f2bc3b
# ∮ₜ = filter(
# 	x -> ∈(x["model"], ["Basic","NOAA","ERA5","EPW","Complete"]),
# 	teₛ,
# );

# ╔═╡ c92ab000-dbe9-457d-97f3-88ae31b57a27
# p₆ = Gadfly.plot(
# 	∮ₜ,
# 	x=:weather_station_distance,
# 	y=:cvrmse,
# 	yintercept=[0, 15],
# 	color=:model,
# 	Coord.cartesian(ymin=-5, ymax=150),
# 	# Scale.y_log,
# 	Guide.ylabel("CVRMSE"),
# 	Guide.xlabel("Distance from Weather Station (m)"),
# 	Guide.title("Model Electricity Predictions"),
# 	Geom.smooth(smoothing=0.1),
# 	Geom.hline(style=[:dash, :dash], color=["pink","red"])
# )

# ╔═╡ c089c975-96e1-4281-b5ad-c53e738834a1
# combine(groupby(clean(teₛ), :model), [:cvrmse, :nmbe, :cvstd] .=> mean∘skipmissing, renamecols=false)

# ╔═╡ e4344d50-425b-4bea-b28e-0c3b45debfb1
# teₛ̇ = stack(teₛ, [:cvrmse, :nmbe, :cvstd]);

# ╔═╡ f9ccee0a-9e6b-4070-a15c-ff5d5c324649
# Τ = Gadfly.plot(
# 	teₛ̇,
# 	x=:weather_station_distance,
# 	y=:value,
# 	ygroup=:variable,
# 	color=:model,
# 	# Coord.cartesian(xmin=0, xmax=4500, ymin=40, ymax=120),
# 	Geom.subplot_grid(
# 		Geom.smooth(smoothing=0.5),
# 		# Geom.point,
# 		free_y_axis=true,
# 	),
# 	Guide.ylabel(""),
# 	Guide.xlabel("Distance from Weather Station (m)"),
# 	Guide.title("Metric Errors vs. Weather Station Distance - Electricity")
# )

# ╔═╡ 624ab4a3-5c3b-42f3-be37-89d6382fdfdd
# Gadfly.plot(
# 	filter(
# 		x -> x["model"] ∈ ["NOAA","MODIS","EPW","Basic"] && x["variable"] == "cvrmse",
# 		teₛ̇
# 	),
# 	x=:weather_station_distance,
# 	y=:value,
# 	color=:model,
# 	Geom.beeswarm(padding=0.4mm),
# 	Guide.ylabel("CVRMSE"),
# 	Guide.xlabel("Distance - log(meters)"),
# 	Scale.x_log,
# 	Scale.y_log,
# 	Theme(point_size=2pt)
# )

# ╔═╡ ec97d987-651d-4efa-a36f-e6be9f18e0fd
md"""
##### Now getting into potential climate scenarios
---
first want to run an analysis on a comprehensive dataset to see which variables offer the highest impact
"""

# ╔═╡ b1f8755d-8130-4492-859e-781225f8bb45
md"""
Baseline Existing Scenario
"""

# ╔═╡ 64b19240-6e2a-4bb9-b681-5d4fdcc98935
# begin
# tₚ¹ = DataFrames.select(DataFrames.select(dataₑ¹, names(cmip19)), Not("Property Id"));
# ʒ¹ = combine(groupby(tₚ¹, :date), names(tₚ¹, Union{Real, Missing}) .=> mean, renamecols=false);
# end;

# ╔═╡ fbdc69a6-7227-4df8-a9a9-eca836565a11
# tₚ⁶ = DataFrames.select(DataFrames.select(dataₑ⁶, names(cmip60)), Not("Property Id"));

# ╔═╡ 92bd44dc-c382-4835-b6ed-351756703389
# ʒ⁶ = combine(groupby(tₚ⁶, :date), names(tₚ⁶, Union{Real, Missing}) .=> mean, renamecols=false);

# ╔═╡ 95f04c20-6adb-4122-9d4a-ccda1663fc83
# Gadfly.plot(
# 	Gadfly.layer(
# 		ʒ¹,
# 		x=:date,
# 		y=:tasmin_mean_f₃,
# 		ymin=:tasmin_quartile25_f₃,
# 		ymax=:tasmin_quartile75_f₃,
# 		Geom.point,
# 		Geom.line,
# 		Geom.errorbar
# 	),
# 	Gadfly.layer(
# 		ʒ⁶,
# 		x=:date,
# 		y=:tasmin_mean_f₃,
# 		ymin=:tasmin_quartile25_f₃,
# 		ymax=:tasmin_quartile75_f₃,
# 		Geom.point,
# 		Geom.line,
# 		Geom.errorbar,
# 		Theme(default_color="indianred")
# 	)
# )

# ╔═╡ b8c478a1-c908-423f-bb30-c54d0d5a2f8e
# begin
# t¹ = combine(groupby(dataₑ¹, :month), agg_term => mean => :t₁)
# t² = combine(groupby(dataₑ⁶, :month), agg_term => mean => :t₂)

# tₛ = leftjoin(t¹, t², on=:month, makeunique=true)
# tₛ.diff = 100 .* ( tₛ.t₂ .- tₛ.t₁ ) ./ abs.( tₛ.t₁ )
# end

# ╔═╡ 5591c99f-fccf-42cc-a8ce-db7a4b7ddf08
# Gadfly.plot(
# 	tₛ,
# 	x=:month,
# 	y=:diff,
# 	Geom.point,
# 	Geom.line,
# 	Theme(default_color="indianred")
# )

# ╔═╡ 95ed19ae-63ba-46e4-ade7-56aa84faccda
# m₈ = training_pipeline(
# 	tₐ̇,
# 	comprehensive_datalist,
# 	["Property Id","date"],
# 	"electricity_mwh",
# 	m
# );

# ╔═╡ 393f1361-94bf-4dee-8665-39085f0db729
# feature_importances(m₈)

# ╔═╡ c57efb98-5eca-4139-bcd4-4aec1323694e
# v₈,_ = validation_pipeline(
# 	vₐ̇,
# 	comprehensive_datalist,
# 	["Property Id","date"],
# 	"electricity_mwh",
# 	m₈
# );

# ╔═╡ 55e18a25-e68d-47d1-a8b1-581c8fa8761c
md"""
time to inspect this global model
"""

# ╔═╡ b5fb3739-a983-4801-98db-cd269a7e9e28
# v₈̇ = leftjoin(v₈, validation_metadata, on="Property Id");

# ╔═╡ cf2021eb-f995-4575-9aec-ed8c0fd9c477
# v₈̇[:,"model"] = repeat(["Complete"], nrow(v₈̇));

# ╔═╡ a8344dde-62ea-4697-b06a-d057d8c335a5
# begin
# te₈,teₓ₈ = validation_pipeline(
# 	teₐ,
# 	comprehensive_datalist,
# 	["Property Id","date"],
# 	"electricity_mwh",
# 	m₈
# );
# te₈̇ = leftjoin(te₈, test_metadata, on="Property Id");
# te₈̇[:,"model"] = repeat(["comprehensive"], nrow(te₈̇))
# end;

# ╔═╡ d60cd919-0208-43a2-934a-b880bf95fd69
md"""
Loading data will be accomlished via innerjoins with selected datasets, with learning taking place on top of this
"""

# ╔═╡ dff8b33a-7513-42c5-8d98-4d56445e65d6
function reproject_points!(
	geom_obj::ArchGDAL.IGeometry,
	source::ArchGDAL.ISpatialRef,
	target::ArchGDAL.ISpatialRef)

	ArchGDAL.createcoordtrans(source, target) do transform
		ArchGDAL.transform!(geom_obj, transform)
	end
	return geom_obj
end

# ╔═╡ 1ad32726-a3ae-461b-8e30-ec289c8ff373
function reproject_points!(
	geom_obj::Union{
		Vector{ArchGDAL.IGeometry},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPoint}},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbMultiPolygon}},
		Vector{ArchGDAL.IGeometry{ArchGDAL.wkbPolygon}}
	},
	source::ArchGDAL.ISpatialRef,
	target::ArchGDAL.ISpatialRef)

	ArchGDAL.createcoordtrans(source, target) do transform
		ArchGDAL.transform!.(geom_obj, Ref(transform))
	end
	return geom_obj
end

# ╔═╡ 4881f16e-5692-4e65-9848-38c8da424179
begin
nyc_boundaries_path = joinpath(
	data_path, 
	"council-boundaries.geojson"
)
nyc_boundaries = GeoDataFrames.read(nyc_boundaries_path)
reproject_points!(nyc_boundaries.geometry, source, target)
end

# ╔═╡ dec54a04-6c59-41ae-946e-937092eaaa43
begin
	sample_size = 6000
	building_classes_plot = Plots.plot(
		centroid.(building_metrics.geometry),
		color_palette = distance_colormap,
		markersize=7,
		alpha=0.6,
		markerstrokewidth=0,
		dpi=500,
		size=(700,500),
	)
	Plots.plot!(
		nyc_boundaries.geometry,
		color="transparent"
	)
	# Plots.plot!(
	# 	epw_local.geometry,
	# 	color="white", 
	# 	markersize=8,
	# 	markerstrokewidth=2
	# )
end

# ╔═╡ 812d7cda-ce12-495c-be38-52c8f0f23747
# Plots.savefig(joinpath(output_dir, "sample_location.png"))

# ╔═╡ 01e27ad1-9b45-40b3-8d03-26400cd153f9
# sample_maxepw = filter( row -> row["Property Id"] == sample_id, max_epw );

# ╔═╡ 4eec8fe5-2c77-489d-8d1f-fef0ad388a50
# sample_minepw = filter( row -> row["Property Id"] == sample_id, min_epw );

# ╔═╡ 09a85206-3f09-4ba4-8fe3-85d32c8e8793
# begin
# 	lst_daymeans = sample_training.LST_Day_1km .* 0.02 .- 273.15
# 	lst_nightmeans = sample_training.LST_Night_1km .* 0.02 .- 273.15
# end

# ╔═╡ 6ec97827-015d-4da4-8e58-4564db02fedf
# default_colors = cgrad(:redblue, 5, categorical = true, rev=false);

# ╔═╡ 87b55c18-6842-4321-b2c7-c1abd8fef6fc
# temp_plot = Gadfly.plot(
# 	Gadfly.layer(
# 		x = sample_training.date,
# 		y = lst_daymeans,
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[1],
# 			point_size=2pt
# 		)
# 	),
# 	Gadfly.layer(
# 		x = sample_maxepw.date,
# 		y = sample_maxepw[:,"Drybulb Temperature (°C)"],
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[2],
# 			point_size=2pt
# 		)
# 	),
# 	Gadfly.layer(
# 		x = sample_training.date,
# 		y = sample_training.TMP,
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[3],
# 			point_size=2pt
# 		)
# 	),
# 	Gadfly.layer(
# 		x = sample_training.date,
# 		y = lst_nightmeans,
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[4],
# 			point_size=2pt
# 		)
# 	),
# 	Gadfly.layer(
# 		x = sample_minepw.date,
# 		y = sample_minepw[:,"Drybulb Temperature (°C)"],
# 		Geom.point,
# 		Geom.line,
# 		Theme(
# 			default_color=default_colors[5],
# 			point_size=2pt
# 		)
# 	),
# 	Guide.ylabel("Measured Temperature (°C)"),
# 	Guide.xlabel("Date"),
# 	Guide.xticks(
# 		ticks=DateTime("2018-01-1"):Month(1):DateTime("2021-01-01"),
# 		orientation=:vertical
# 	),
# 	Guide.title("Temperature Measurements - Sample Location"),
# 	Guide.manual_color_key(
# 		"Data Type",
# 		[ 
# 			"MODIS Daytime LST", 
# 			"EPW Drybulb Temp Daily Maximum",
# 			"NOAA Reanalysis",
# 			"MODIS Nighttime LST", 
# 			"EPW Drybulb Temp Daily Minimum"
# 		],
# 		[
# 			default_colors[1], 
# 			default_colors[2], 
# 			default_colors[3], 
# 			default_colors[4], 
# 			default_colors[5]
# 		]
# 	)
# )

# ╔═╡ e81eacd1-22ff-4890-989e-e4ec638f06b5
# begin
# sample_daylst_df = DataFrame(
# 	date=sample_training.date,
# 	lst_temp=lst_daymeans
# )

# lst_epw_tempdiff = DataFrames.select(
# 		innerjoin(
# 		sample_daylst_df,
# 		sample_maxepw,
# 		on="date"
# 		),
# 	["date","lst_temp","Drybulb Temperature (°C)"]
# )
# end;

# ╔═╡ f37dd092-aa81-4669-8925-665415c90aaf
# Gadfly.plot(
# 	Gadfly.layer(
# 		x=lst_epw_tempdiff.date,
# 		y=lst_epw_tempdiff.lst_temp .- lst_epw_tempdiff[:,"Drybulb Temperature (°C)"],
# 		Theme(default_color="black"),
# 		Geom.point,
# 		Geom.line
# 	),
# 	Guide.xlabel("Date"),
# 	Guide.ylabel("Temperature-Δ (°C)"),
# 	Guide.title("Temperature-Δ of Max Readings (MODIS LST - EPW)")
# )

# ╔═╡ 512cad43-28ca-4b07-afa5-20571a31b311
# draw(
# 	PNG(
# 		joinpath(output_dir, "temperature_deviation_example.png"), 
# 		20cm, 
# 		10cm,
# 		dpi=500
# 	), temp_plot
# )

# ╔═╡ 388d9daf-d6e3-4551-84bb-56906f013900
# # looking at general trends
# temperature_trends = combine(
# 	groupby(
# 		epw_training_data,
# 		:date
# 	), :electricity_mwh .=> [mean∘skipmissing,std∘skipmissing] .=> ["mean","std"]
# );

# ╔═╡ 99c5d986-0cdb-4321-82bf-49ced0502430
# begin
# 	ymins = temperature_trends.mean .- (temperature_trends.std)
# 	ymaxs = temperature_trends.mean .+ (temperature_trends.std)

# 	Gadfly.plot(
# 		temperature_trends,
# 		x=:date,
# 		y=:mean,
# 		ymin = ymins,
# 		ymax = ymaxs,
# 		Geom.point,
# 		Geom.errorbar,
# 		Theme(default_color="black")
# 	)
# end

# ╔═╡ b92c6f37-2329-4ce8-bcf9-eb53f09f7266
md"""
#### now exploring a bit about this UHI effect
"""

# ╔═╡ 1c3c1fc5-f1f1-44e7-9700-96f1342b5f9f
# begin
# max_epw_r = combine(
# 	groupby(epw_r, ["Property Id", "date"]), 
# 	names(epw_r, Real) .=> maximum, 
# 	renamecols=false
# );
# max_epw = monthly_average(
# 	max_epw_r,
# 	agg_terms
# );
# min_epw = monthly_average(
# 	combine(groupby(epw_r, ["Property Id", "date"]), names(epw_r, Real) .=> minimum, renamecols=false),
# 	agg_terms
# );
# end;

# ╔═╡ e6cb1988-a03f-4931-84f6-a1bff6da1c4e
# lhi_captured = filter( row -> ~isnan(row.LST_Day_1km), training_data );

# ╔═╡ 030c5f35-1f1d-4bc0-9bd6-c97c193ac835
# lhi_captured.mean_2m_air_temperature;

# ╔═╡ df7e5922-5d1e-4a55-8ae6-9e3e60e0fa9b
# uhi_difference = ( lhi_captured.LST_Day_1km .* 0.02 .- 273.15 ) .- ( lhi_captured.mean_2m_air_temperature .- 273.15 );

# ╔═╡ c2f4541e-d453-4e44-99ed-229a126cda7e
# begin
# 	train_electric = DataFrames.select(train, Not(:naturalgas_mwh))
# 	validate_electric = DataFrames.select(validate, Not(:naturalgas_mwh))
# 	test_electric = DataFrames.select(test, Not(:naturalgas_mwh))

# 	dropmissing!(train_electric, :electricity_mwh)
# 	dropmissing!(validate_electric, :electricity_mwh)
# 	dropmissing!(test_electric, :electricity_mwh)
# end;

# ╔═╡ f825b6a1-0950-461b-80a8-b0fc9924dd53
# train_electric

# ╔═╡ 8806127c-8bbd-4567-b32e-946fb473b4c7
# begin
# 	train_env_electric_aux = DataFrames.select(train_electric, env_dropping_cols)
# 	validate_env_electric_aux = DataFrames.select(validate_electric_env, env_dropping_cols)
# 	test_env_electric_aux = DataFrames.select(test_electric_env, env_dropping_cols)

# 	select!(train_electric_env, Not(env_dropping_cols))
# 	select!(validate_electric_env, Not(env_dropping_cols))

# 	train_xe = DataFrames.select(train_electric_env, Not(:electricity_mwh))
# 	train_ye = train_electric_env.electricity_mwh

# 	validate_xe = DataFrames.select(validate_electric_env, Not(:electricity_mwh))
# 	validate_ye = validate_electric_env.electricity_mwh
# end

# ╔═╡ a8d09777-65ed-4178-b3f2-12a269ccfbbb
# train_env

# ╔═╡ 63031393-40e1-4b16-aaa6-34c7fe9fb35a
# Gadfly.plot(x=validate_env.weather_station_distance, Geom.histogram)

# ╔═╡ ea8e4429-b797-4452-bb7d-73ebbd58af76
md"""
error functions, as defined by [ASHRAE](https://upgreengrade.ir/admin_panel/assets/images/books/ASHRAE%20Guideline%2014-2014.pdf)
"""

# ╔═╡ b3878f52-bbce-433f-a23e-ed5b56d4f5b1
nmbe(ŷ, y, p=1) = 100 * sum(y.-ŷ) / (( length(y)-p ) * mean(y) )

# ╔═╡ 170b929b-ec44-4eb1-bdcf-61a280e54b7d
cvrmse(ŷ, y, p=1) = 100 * (sum((y.-ŷ).^2) / (length(y)-p))^0.5 / mean(y)

# ╔═╡ 732efa0f-8a8a-4c53-a7ad-e90a19c2f637
cvstd(ŷ, y, p=1) = (100 * mean(y)) / ((sum( (y .- mean(y)).^2 )/(length(y)-1))^0.5)

# ╔═╡ c331ec69-2b5c-4ba9-8a7d-7130a27c320f
function test_suite(Ẋ::DataFrame, selectors = ["Property Id", "zone"], report_model = true, region_counts=false)
	# Ẋ = hcat(X, auxiliary)
	if !region_counts
		vᵧ = groupby(Ẋ, selectors)
		results = combine(vᵧ) do vᵢ
		(
		cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
		nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
		cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
		rmse = rmse(vᵢ.prediction, vᵢ.recorded),
		)
		end
	else
		vᵧ = groupby(Ẋ, selectors)
		results = combine(vᵧ) do vᵢ
		(
		cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
		nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
		cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
		rmse = rmse(vᵢ.prediction, vᵢ.recorded),
		count = mean(vᵢ.nrow)
		)
		end	
	end

	if report_model
		results.model = repeat([Ẋ[1,"model"]], nrow(results))
	end
	
	return results
end

# ╔═╡ f770a956-bd1b-41f6-bba4-a228e964965c
describe(test_suite(filter( x -> x.recorded > 0, teₐ′₀), ["Property Id"]))[2:end,:]

# ╔═╡ 92e7c0a0-19df-43e2-90db-4e050ba90c5d
test_suiteₜₑ = vcat([ test_suite(x) for x in test_terms ]...);

# ╔═╡ a0cd5e68-3bd0-4983-809e-ce7acd36a048
modelresultsₑ = combine(
	groupby(
		clean(test_suiteₜₑ), 
		["zone","model"]
	), [:rmse, :cvrmse,:nmbe,:cvstd] .=> mean, 
	renamecols=false
);

# ╔═╡ ecb078da-702d-46f7-846a-4759b3e0f172
baselinesₑ = rename(
	DataFrames.select(filter(x -> x.model == "Null", modelresultsₑ), [:zone,:rmse]),
	:rmse => :baseline
)

# ╔═╡ 02a15a48-70da-467c-84d3-845612c089da
modelresultsₑ

# ╔═╡ 41e55b47-b527-491c-ab94-ecc34a9b64bb
model_baselinesₑ = leftjoin(
	modelresultsₑ,
	baselinesₑ,
	on="zone"
);

# ╔═╡ 342a8606-708b-4b1c-9023-050c81ce345d
model_baselinesₑ.Δ = 100 .* (1 .- model_baselinesₑ.rmse ./ model_baselinesₑ.baseline );

# ╔═╡ 40d7e39c-98c6-4cf8-8461-e54b7a0db696
sort(filter(x -> x.zone == "Residential", model_baselinesₑ), :rmse);

# ╔═╡ 5fb5d0c3-46ab-41dc-be49-64d83cc845f8
model_baselinesₑ.power = repeat(["Electricity"], nrow(model_baselinesₑ));

# ╔═╡ b0035cc3-0806-41ce-8686-faeab40ee41f
Gadfly.plot(
	model_baselinesₑ, 
	x=:zone, 
	y=:model,
	color=:Δ,
	Geom.rectbin
)

# ╔═╡ 86871d26-5612-4166-959f-6fa7d33daaf6
@info "Electricity Aggregate Results" modelresultsₑ

# ╔═╡ cf39d6f0-b592-4a4d-aa17-182f9c794959
describe(test_suite(filter(
	x -> x.recorded > 0.01, teᵧ′₀), ["Property Id"]))[2:end,:]

# ╔═╡ 734ed2d0-4657-4361-9285-97605371af72
test_suiteₜᵧ = vcat([ test_suite(x) for x in test_termsᵧ ]...);

# ╔═╡ d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
modelresultsᵧ = combine(groupby(clean(test_suiteₜᵧ), ["zone","model"]), [:rmse, :cvrmse,:nmbe,:cvstd] .=> mean, renamecols=false);

# ╔═╡ 31a814af-95f1-4d7f-ada0-8bf9df2aee10
baselinesᵧ = rename(
	DataFrames.select(filter(x -> x.model == "EPW", modelresultsᵧ), [:zone,:rmse]),
	:rmse => :baseline
)

# ╔═╡ b9516f6f-bb93-47b3-8669-a4e8adf3db3a
model_baselinesᵧ = leftjoin(
	modelresultsᵧ,
	baselinesᵧ,
	on="zone"
);

# ╔═╡ 9d6a3995-7df0-4552-887e-b306d26a80ad
model_baselinesᵧ.Δ = 100 .* (1 .- (model_baselinesᵧ.rmse ./ model_baselinesᵧ.baseline));

# ╔═╡ 402e380d-a91c-45f7-9fbb-5a2c44e754f7
model_baselinesᵧ.power = repeat(["Natural Gas"], nrow(model_baselinesᵧ));

# ╔═╡ 167d31ad-76f2-432d-8550-ba015cb18e9d
begin
results = vcat(model_baselinesₑ, model_baselinesᵧ);
results.Δ[results.Δ .< 0] .= 0
end;

# ╔═╡ 32bec288-d04b-45ad-8dfc-da4c216ce8e5
combine(groupby(results, :model), :Δ => mean)

# ╔═╡ e38eee70-b9b0-4dca-ae86-697fc5738cfa
model_results = Gadfly.plot(
	results, 
	x=:zone, 
	y=:model,
	color=:Δ,
	ygroup=:power,
	Geom.subplot_grid(Geom.rectbin),
	# Guide.xlabel(""),
	# Guide.ylabel(""),
	Theme(bar_spacing=0pt),
	Scale.ContinuousColorScale(p -> get(ColorSchemes.RdPu, p)),
	Guide.title("Model Comparison to Baseline (EPW )")
)

# ╔═╡ a07ac65e-3e19-411e-9967-af9778d8a45d
draw(PNG(joinpath(output_dir, "model_results.png"), 20cm, 10cm, dpi=600), model_results)

# ╔═╡ 195e6792-8dd7-4122-8036-c2e593768539
DataFrames.select(results, [:zone, :model, :power, :Δ], :)

# ╔═╡ 55f1323b-291e-486d-b3a8-6409db9f7b8e
smooth_seaonsal_predictionresults = Gadfly.plot(
	results,
	x=:power, 
	y=:model,
	color=:Δ,
	ygroup=:zone,
	Geom.subplot_grid(Geom.rectbin),
	Guide.xlabel(""),
	Guide.ylabel(""),
	Theme(bar_spacing=0pt),
	Scale.ContinuousColorScale(p -> get(ColorSchemes.matter, p)),
	Guide.title("Model Comparison to Baseline (Null)")
)

# ╔═╡ d15506e2-5213-48fc-bf6a-fbe6e99af3de
full_seasonal_results = hstack(seaonsal_predictionresults, smooth_seaonsal_predictionresults);

# ╔═╡ f75ae159-d811-47dc-8592-73a8048244fb
draw(
	PNG(
		joinpath(output_dir, "full_seasonal_results.png"), 
		40cm, 
		20cm,
		dpi=500
	), full_seasonal_results
)

# ╔═╡ bf71086c-a8f0-4420-b698-73499fec5257
test_suiteₜᵧđ = leftjoin(
	test_suiteₜᵧ,
	building_distancesᵧ,
	on="Property Id"
);

# ╔═╡ b90dcb6c-70fe-4b7f-b91a-15a8e78d8411
begin
resultsₑₔ = test_suite(resultsₑ, ["Property Id","model"], false, false);
resultsₑₔ[:,"energy"] = repeat(["Electricity"], nrow(resultsₑₔ));
filter!(x -> x.cvrmse ≠ Inf && x.cvstd ≠ Inf, resultsₑₔ)
resultsₑₔ
end

# ╔═╡ e008e66b-40f1-474c-ad3b-f64e542c7499
combine(groupby(resultsₑₔ, ["model","energy"]), [:cvrmse, :nmbe, :cvstd, :rmse] .=>   mean ∘ skipmissing, renamecols=false)

# ╔═╡ 33a3ddb2-898e-4bec-bc45-3cd65e8ddbd0
begin
resultsᵧₔ = test_suite(filter(x -> x.recorded > 0., resultsᵧ), ["Property Id","model"], false, false);
resultsᵧₔ[:,"energy"] = repeat(["Natural Gas"], nrow(resultsᵧₔ))
filter!(x -> x.cvrmse ≠ Inf && x.cvstd ≠ Inf, resultsᵧₔ)
resultsᵧₔ
end;

# ╔═╡ c66a453b-c9b6-427f-ac13-a392626955e1
describe(vcat(resultsₑₔ, resultsᵧₔ))

# ╔═╡ 9cbf5154-b634-44e5-8646-74468f8977f5
begin
resultsₒₒ = combine(
	groupby(vcat(resultsₑₔ, resultsᵧₔ), [:model, :energy]),
	[:cvrmse, :nmbe, :cvstd, :rmse] .=> mean,
	renamecols=false
)
end

# ╔═╡ f4d83c51-650f-4ed3-856b-f39f74fd6e0e
latexify(resultsₒₒ, latex=false, fmt = FancyNumberFormatter(4), env=:table)

# ╔═╡ 3e8fd197-59fd-4547-8437-45456dec8f8d
names(test_suite(teₐ′₀, ["Property Id"]))

# ╔═╡ aba25e76-c18f-44de-b57b-c3aa8f5be42d
function aggregate_councilerr(df)
	test_suite(combine(groupby(df, ["zipcode","date", "model"]),
		[:prediction, :recorded] .=> sum,
		nrow,
		renamecols=false
	), ["zipcode"], true, true)
end

# ╔═╡ d9ac0584-db8f-497e-807f-9b207b0a6151
councilₐ₀ = aggregate_councilerr(teₐ′₀);

# ╔═╡ 5a4110cf-6700-4b82-bdcf-4c724e342d1c
councilₐ₂ = aggregate_councilerr(teₐ′₂);

# ╔═╡ f254e3f8-a5a3-4d83-b2d8-50552bcd2edf
councilₐ₄ = aggregate_councilerr(teₐ′₄);

# ╔═╡ 44e249fc-dea4-464b-b8a8-94ef304f5769
councilₐ₅ = aggregate_councilerr(teₐ′₅);

# ╔═╡ c5903324-c203-43d5-9f0c-b5c7ba03af07
councilₐ₆ = aggregate_councilerr(teₐ′₆);

# ╔═╡ 38d0794b-a374-4a18-94d3-ba0ff3d8bba0
councilₐ₇ = aggregate_councilerr(teₐ′₇);

# ╔═╡ b7dcfb1b-0f90-4fad-9e3d-032944a24eca
begin
council_results = vcat(
	councilₐ₀, 
	councilₐ₂,
	councilₐ₄,
	councilₐ₅,
	councilₐ₆,
	councilₐ₇
);
council_results.energy = repeat(["Electricity"], nrow(council_results))
end;

# ╔═╡ c1287f32-e270-47fc-8b18-38394380e76e
council_results.zipcode

# ╔═╡ 14357463-a585-409b-bbe1-a02bfd086c61
council_results

# ╔═╡ 1732eaee-3259-4982-9dae-69eed1c8b56d
cr₁ = combine(groupby(council_results, [:model]), errterm .=> mean, renamecols=false)

# ╔═╡ a440743b-6595-4bfa-93a7-36784dc965a3
latexify(cr₁, latex=true, fmt = FancyNumberFormatter(4), env=:table)

# ╔═╡ 61186b87-03ed-4cbd-bbfb-cecb98ab9313
Gadfly.plot(
	stack(council_results, [:cvrmse, :nmbe, :cvstd, :rmse]),
	x=:count,
	y=:value,
	# color=:model,
	yintercept=[0],
	ygroup=:variable,
	Geom.subplot_grid(
		Geom.point,
		Geom.smooth(smoothing=0.15),
		# Geom.smooth(smoothing=0.5, color="red"),
		Geom.hline(color="pink"),
		Guide.xrug,
		free_y_axis=true
	),
	# Scale.x_log,
	Theme(
		point_size=0.5pt,
		alphas=[0.5],
		default_color="black", 
		highlight_width = 0pt
	)
)

# ╔═╡ add11e32-0299-4d2b-8d12-6c7436f3b49d
councilᵧ₀ = aggregate_councilerr(teᵧ′₀);

# ╔═╡ cdd6fcc4-0e63-4e9d-ac8b-e8a366f83625
councilᵧ₂ = aggregate_councilerr(teᵧ′₂);

# ╔═╡ e4cfe664-c46b-4626-a6f3-42d87b98e868
councilᵧ₄ = aggregate_councilerr(teᵧ′₄);

# ╔═╡ 82d00718-cc23-42b6-9902-bb0032fde744
councilᵧ₅ = aggregate_councilerr(teᵧ′₅);

# ╔═╡ dbe1f66c-e9ab-4d29-8846-20031c93abf1
councilᵧ₆ = aggregate_councilerr(teᵧ′₆);

# ╔═╡ f2300c70-9eba-4711-a35b-cec5ae8340f5
councilᵧ₇ = aggregate_councilerr(teᵧ′₇);

# ╔═╡ c4716c23-2924-4751-a303-e688b71199ab
begin
council_resultsᵧ = vcat(
	councilᵧ₀, 
	councilᵧ₂,
	councilᵧ₄,
	councilᵧ₅,
	councilᵧ₆,
	councilᵧ₇
);
council_resultsᵧ.energy = repeat(["Natural Gas"], nrow(council_resultsᵧ))
end;

# ╔═╡ 64fde861-20a5-4cdf-ab7e-d1409018ffe2
Gadfly.plot(
	stack(council_resultsᵧ, [:cvrmse, :nmbe, :cvstd, :rmse]),
	x=:count,
	y=:value,
	# color=:model,
	yintercept=[0],
	ygroup=:variable,
	Geom.subplot_grid(Geom.smooth,Geom.hline(color="pink"),free_y_axis=true),
	Theme(default_color="black")
)

# ╔═╡ 1938a66f-9f6e-4b5c-9f67-6ba5c99d8bd1
aggregated_results = stack(vcat(council_results, council_resultsᵧ), [:cvrmse, :nmbe, :cvstd, :rmse]);

# ╔═╡ b02061e3-e42a-4581-b085-96b02fb19b52
filter( x -> x.model == "SAR", aggregated_results)

# ╔═╡ 9d677b71-dcd9-45ba-907c-b3b12d375140
aggregated_results

# ╔═╡ d36a7409-2441-4e04-b704-c389b4e690d0
agg_results = Gadfly.plot(
	filter( x -> x.model ∈ ["EPW"] && x.value < 500 && x.value > -100, aggregated_results),
	x=:count,
	y=:value,
	# color=:model,
	# color=:model,
	yintercept=[0],
	xgroup=:energy,
	ygroup=:variable,
	Guide.title("EPW Model Aggregation Characteristics"),
	Geom.subplot_grid(
		Guide.xticks(ticks=collect(0:5:30)),
		Geom.point,
		Geom.smooth(smoothing=0.3),
		Geom.hline(color="pink"),
		# Guide.xrug,
		free_y_axis=true
	),
	Theme(
		point_size=0.7pt,
		alphas=[0.2],
		highlight_width = 0pt,
		default_color="black"
	)
)

# ╔═╡ 7ac9f47d-c661-46ee-8f03-4029d248550d
draw(
	PNG(
		joinpath(output_dir, "aggregated_benefits_chart.png"), 
		12cm, 
		15cm,
		dpi=500
	), agg_results
)

# ╔═╡ 8ee0146c-e3c3-4a16-b527-ee1e3da663fa
mean(filter( x -> x.model == "EPW", aggregated_results).count)

# ╔═╡ e89c179f-3de0-4613-bbf5-fd936572ce5c
full_aggregation_results = combine(groupby(aggregated_results, [:model, :energy, :variable]), :value => mean, renamecols=false);

# ╔═╡ 0b50394f-5800-449b-89bb-d62db694d174
Gadfly.plot(
	full_aggregation_results,
	y=:model,
	xgroup=:variable,
	x=:energy,
	color=:value,
	Geom.subplot_grid(Geom.rectbin)
)

# ╔═╡ 0915512e-f2fc-432f-9ba7-ba1b3f8cb1ef
begin
	cvagg = filter( x -> x.variable == "cvrmse", full_aggregation_results)
	nmagg = filter( x -> x.variable == "nmbe", full_aggregation_results)
	stagg = filter( x -> x.variable == "cvstd", full_aggregation_results)
	rmagg = filter( x -> x.variable == "rmse", full_aggregation_results)

	ua₁ = Gadfly.plot(
		cvagg,
		y=:energy,
		x=:model,
		color=:value,
		Geom.rectbin,
		Guide.xlabel(""),
		Guide.ylabel(""),
		Guide.title("CVRMSE"),
		Guide.colorkey(title=""),
		Scale.ContinuousColorScale(
			palette -> get(ColorSchemes.Reds, palette),
		),
	)

	ua₂ = Gadfly.plot(
		nmagg,
		y=:energy,
		x=:model,
		color=:value,
		Geom.rectbin,
		Guide.xlabel(""),
		Guide.ylabel(""),
		Guide.title("NMBE"),
		Guide.colorkey(title=""),
		Scale.ContinuousColorScale(
			palette -> get(ColorSchemes.YlOrBr, palette),
		),
	)

	ua₃ = Gadfly.plot(
		stagg,
		y=:energy,
		x=:model,
		color=:value,
		Geom.rectbin,
		Guide.xlabel(""),
		Guide.ylabel(""),
		Guide.title("CVSTD"),
		Guide.colorkey(title=""),
		Scale.ContinuousColorScale(
			palette -> get(ColorSchemes.Blues, palette),
		),
	)

	ua₄ = Gadfly.plot(
		rmagg,
		y=:energy,
		x=:model,
		color=:value,
		Geom.rectbin,
		Guide.xlabel(""),
		Guide.ylabel(""),
		Guide.title("RMSE"),
		Guide.colorkey(title=""),
		Scale.ContinuousColorScale(
			palette -> get(ColorSchemes.Purples, palette),
		),
	)

	uaoo = vstack(ua₁, ua₂, ua₃, ua₄)
end

# ╔═╡ 107b60fa-692b-4c87-944b-ae606a3a353a
draw(
	PNG(
		joinpath(output_dir, "aggregated_benefits_map.png"), 
		15cm, 
		20cm,
		dpi=500
	), uaoo
)

# ╔═╡ f00265ca-fa8a-43f2-bde5-8924934a9406
yuu = combine(groupby(DataFrames.select(vcat(council_results, council_resultsᵧ), Not(:zipcode)), [:model,:energy]), [:cvrmse, :nmbe, :rmse] .=> mean, renamecols=false)

# ╔═╡ 9e6f7e27-a4bb-434b-8daf-7a0d831118ee
latexify(yuu, latex=false, fmt = FancyNumberFormatter(4), env=:table)

# ╔═╡ 3b26f240-30f8-4894-9dab-8ce65a7cac4b
o₀ = test_suite(combine(groupby(teₐ′₀, ["zipcode","date", "model"]),
	[:prediction, :recorded] .=> sum,
	nrow,
	renamecols=false
), ["zipcode"], true, true)

# ╔═╡ 0e4ed949-b051-4ae5-8b7b-b1c8d6d7fc94
function electricaggregation(individual_results)
	# t = innerjoin(
	# 	individual_results,
	# 	testcouncils,
	# 	on="Property Id"
	# )
	ṫ = combine(groupby(individual_results, ["date","zipcode"]), [:prediction, :recorded] .=> sum, :model => first, renamecols=false)
	
	res = combine(groupby(ṫ, "zipcode")) do vᵢ
		(
		cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
		nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
		cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
		rmse = rmse(vᵢ.prediction, vᵢ.recorded),
		model = first(vᵢ.model)
		)
		end
	
	return ṫ, res
end

# ╔═╡ bfd0653f-b0ca-4db2-b095-6d50698fb997
teₐ′₀̇, res₀ = electricaggregation(teₐ′₀);

# ╔═╡ a4f1cfc8-64c3-40ce-b67c-6548f88a44cb
begin
u₀ = combine(groupby(teₐ′₀̇, :date), [:prediction, :recorded] .=> sum, renamecols=false);
u₀.model = repeat(["null"], nrow(u₀))
end;

# ╔═╡ 9309e41a-debd-4e8f-b891-18fa6bf15391
describe(test_suite(teₐ′₀̇, ["zipcode"]))[2:end,:]

# ╔═╡ 7c6d8344-fd9b-4e9c-888e-a57eec39ba04
teₐ′₂̇, res₂ = electricaggregation(teₐ′₂);

# ╔═╡ 32812780-ff19-4650-9d2d-496849a546ae
begin
u₂ = combine(groupby(teₐ′₂̇, :date), [:prediction, :recorded] .=> sum, renamecols=false);
u₂.model = repeat(["noaa"], nrow(u₂))
end;

# ╔═╡ ac06853c-4c63-4395-a30f-3cb06c552d81
electricaggregation(teₐ′₄);

# ╔═╡ 1d081311-5c63-41b5-8db5-2f18fb7ed6b7
teₐ′₅̇, res₅ = electricaggregation(teₐ′₅);

# ╔═╡ f8742eb7-ecc2-4f7d-90bf-acfe6700b80c
begin
u₅ = combine(groupby(teₐ′₅̇, :date), [:prediction, :recorded] .=> sum, renamecols=false);
u₅.model = repeat(["landsat8"], nrow(u₅))
end

# ╔═╡ 798a89f2-dca0-42eb-9091-ec6b8748fc96
teₐ′₇̇, res₇ = electricaggregation(teₐ′₇);

# ╔═╡ 6adf9b7c-5044-48f6-b562-de3544dac55d
begin
u₇ = combine(groupby(teₐ′₇̇, :date), [:prediction, :recorded] .=> sum, renamecols=false);
u₇.model = repeat(["sar"], nrow(u₇))
end;

# ╔═╡ 9ea32e28-94ac-4bfc-902b-c930296f683a
u = vcat(u₀, u₂, u₅, u₇);

# ╔═╡ 29672872-98e2-454f-b5c2-49966f11e15f
Gadfly.plot(
	u,
	x=:date,
	y=abs.(u.prediction .- u.recorded),
	color=:model,
	Geom.point,
	Geom.smooth(smoothing=0.3),
	Guide.ylabel("Aggregated Electricity Error - MWh"),
	Guide.xlabel(""),
	Guide.title("Monthly Prediction with Microclimate Datasets - Electricity")
)

# ╔═╡ 47f4e899-f441-4a68-80b4-2b540f738f3f
teᵧ′₀̇, resᵧ₀ = electricaggregation(teᵧ′₀);

# ╔═╡ 95bfd1a2-3bd6-43f6-b2b1-fb94139212ad
begin
γ₀ = combine(groupby(teᵧ′₀̇, :date), [:prediction, :recorded] .=> sum, renamecols=false);
γ₀.model = repeat(["null"], nrow(γ₀))
end;

# ╔═╡ ba765277-25e9-46a5-ba68-1c33c55c9755
describe(test_suite(teᵧ′₀̇, ["zipcode"]))[2:end,:]

# ╔═╡ a282de0c-aa3b-494a-84a2-9d37e20d1adc
begin
	resultsₑ₁ = groupby(resultsₑ, ["Property Id"])
	resultsₑ₂ = combine(resultsₑ₁) do vᵢ
	(
	cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
	nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
	cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
	rmse = rmse(vᵢ.prediction, vᵢ.recorded),
	)
	end
	# results.model = repeat([Ẋ[1,"model"]], nrow(results))
end;

# ╔═╡ 93c45e9c-0231-49fd-9a94-2ef0b03ecddb
begin
	resultsₑ₁′ = groupby(resultsₑ, ["Property Id"])
	resultsₑ₂′ = combine(resultsₑ₁) do vᵢ
	(
	cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
	nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
	cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
	rmse = rmse(vᵢ.prediction, vᵢ.recorded),
	)
	end
	# results.model = repeat([Ẋ[1,"model"]], nrow(results))
end;

# ╔═╡ Cell order:
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
# ╠═d75e2748-fa12-45f2-a380-e53ba1826893
# ╠═3bf1db0d-fb5e-4212-8dac-5e9c009ad77f
# ╠═aeb3597f-ef6a-4d21-bff5-d2ad359bc1a2
# ╠═9d5f897f-7f25-43ed-a30c-064f21e50174
# ╠═786c2441-7abb-4caa-9f50-c6078fff0f56
# ╠═1c7bfba6-5e1d-457d-bd92-8ba445353e0b
# ╠═5bfda8cf-487b-4930-b212-9cd1622367de
# ╟─9b3790d3-8d5d-403c-8495-45def2c6f8ba
# ╠═bf772ea4-c9ad-4fe7-9436-9799dcb0ad04
# ╠═020b96e3-d218-470d-b4b0-fc9b708ffdf3
# ╠═52ee36d9-eb72-4f2a-a3ed-a48e8e0d7931
# ╠═4881f16e-5692-4e65-9848-38c8da424179
# ╠═6daef876-af37-4284-9489-181d8320b653
# ╠═ec1c88ff-bb90-4de9-bff2-6c7f531b725e
# ╠═2a94df22-2a14-4ba6-821f-a0c125c93e07
# ╠═b8a9e591-1318-4ec4-b608-164c6425187c
# ╠═af422ea0-0f62-4233-a28d-d96a46009a14
# ╠═a5ad08d5-08ee-46d9-b2fd-af8ee904e766
# ╠═0b9c6880-9912-4268-9245-4cc8c7cad66f
# ╠═60943880-2aa3-4d39-adc9-9d10338427e7
# ╠═9aa06073-d43e-4658-adb9-bbc11425978d
# ╠═9deff2bf-61a3-48c8-be41-5c3b501d604f
# ╠═c8a8830a-a861-4b35-ae13-2a5cccecbe50
# ╠═ced7e799-c5e9-490a-943e-533d2d1b4f2a
# ╠═8c97ee63-98a3-4640-8e1c-69fa0cf3810b
# ╠═7b53849c-d51e-48a3-a43c-a20392356400
# ╠═b1526350-b68a-4899-adb6-bf981adf26e0
# ╠═4f1c0eae-e637-40f8-95a9-61088e423725
# ╠═a8a03990-fd30-48b9-9b76-ce33dd90ceb3
# ╠═44f02f33-9044-4355-ba82-b35595a82bdd
# ╠═c2560a7b-ec09-4cdb-bdec-68a65301249f
# ╠═56f43ba6-568b-436d-85a5-a8da5a0a3956
# ╠═8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
# ╠═36699ba6-6472-4cf3-bef6-7c4757fb69b7
# ╠═5c8708ff-bc87-44ed-b39d-8f6a87365a3a
# ╠═79fed5b6-3842-47b7-8918-63f918e070bb
# ╠═cfc6d000-3338-468a-a1f8-e3d0b3c9881d
# ╠═dc5167ab-6c65-4916-be18-c635b04f0c0d
# ╠═e33201c0-678e-4ffc-9310-2420ea65aced
# ╠═4b2ce2f1-fdcd-4d30-80b2-38d909270975
# ╠═5b7ae19f-d341-466a-996d-e709e9006a42
# ╠═f204fb81-916c-4c85-b13d-9ccf4a70fa43
# ╠═a95245b0-afa8-443a-9a16-b01de7294f13
# ╠═179e28b1-4102-4e8e-bd54-12baea4044f7
# ╠═412dd805-7fc4-4c63-9854-0ab98fcb2c8a
# ╠═a0180e4b-4f32-42a7-bd93-9a8aaa0d4979
# ╠═38c7cb84-cb66-4108-8ef3-06767ee15110
# ╠═1a37f7f0-b112-4b0b-b869-713b84f0a1a8
# ╠═a2e624f7-5626-431a-9680-f62ed86b61aa
# ╠═db7c092f-5fa8-4038-b9cf-d40d822a4b9a
# ╠═d3d814ee-ad4f-47bf-966a-08cabc79bf90
# ╠═00acb065-2378-4181-b76a-488071f43a7e
# ╠═fe529a32-ab71-4e5d-a593-45085d69f580
# ╠═a536094d-3894-4ce7-95cd-f38a3666e07e
# ╠═175ec879-2c34-477a-9359-38f8f9992b72
# ╠═217c69fd-380b-4240-8078-68a54e8eafde
# ╠═067fc936-5eac-4082-80f8-c50f194f1721
# ╠═f589009c-fd53-4f5a-a5e3-844442665e8b
# ╠═06672789-8a1f-44f4-b901-903c3abec016
# ╠═5e053e99-2501-4167-8368-da1720c6d0dd
# ╠═b0547ee9-51cc-4427-8269-d69c4b263bbb
# ╠═2c29917b-d1ae-40b5-9eff-14c672bf6a1e
# ╠═7faa0480-bf01-4786-ab6f-4bb772775750
# ╠═4df4b299-b9ee-46a2-9622-37c430e867a1
# ╠═fdd41dc5-1439-458c-ad41-3d10f3a8478f
# ╠═5e26802a-0a84-4dc1-926d-d51ac589dc5e
# ╠═65c72331-c36e-4e15-b530-13069b8cc070
# ╠═cc20d207-016e-408c-baf1-83d68c4c0fde
# ╠═0a22f19d-c662-4071-b4e2-6e8103a0f359
# ╠═a9f2d94d-cbf8-4d47-a4b2-438f451882e5
# ╠═ca12dd08-29af-4ce3-a2cc-d3bf1fa9e3c7
# ╠═348c4307-94dc-4d5f-82b0-77dc535c1650
# ╠═09a4789c-cbe7-496e-98b5-a2c2db3102b6
# ╠═22494217-9254-4374-8a7d-02528bdd0df3
# ╠═d23500b2-53ba-436c-b26d-187f60821a43
# ╠═dcde8c56-7294-47fd-aad1-2204de6c904b
# ╠═e87d641a-9555-4a93-9fe8-f39f8964ce84
# ╠═a6e81108-c335-4e14-8c23-dd33867e45a9
# ╠═e03565fe-83eb-4076-ac21-f425ded9f39b
# ╠═2f1fec21-76a2-4365-b305-0f24505b1ccc
# ╠═86d465e3-7916-479d-a29c-2b93ae54ab6d
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
# ╠═cbd44ab9-6346-46f0-bb22-1c19b471ccf7
# ╠═8191b635-e556-47fe-bb39-93ce96ee82bd
# ╠═3da877c2-159b-4d0d-8b97-34da4dbf2ac3
# ╠═b5b2e5b1-af5e-46fd-9c63-2251b9539f62
# ╠═6f73795c-cb72-4a75-adde-17f7361cc452
# ╠═766cadfe-acac-402d-8c80-d186b6fef2e6
# ╠═e9c65a69-4bf1-48df-8190-f988c7442e38
# ╠═c614fd3b-363f-436e-b6e9-2366d1cf87b9
# ╠═f40de5ec-5260-498d-a102-461e4ba24178
# ╠═0c7528dd-f857-47b8-a1c9-1a7fd5a316a9
# ╠═073ca242-0c60-4f54-a0f2-f5d2bed88421
# ╠═8d0a2791-0f06-48d7-8be9-d7663226f9c2
# ╠═a7d8d342-0829-45fe-ae5e-f4b74cfc29b4
# ╠═0cea9f55-9486-4166-a987-67db0c09da2a
# ╠═3e648045-0915-4eb6-b037-4c09f1f0036e
# ╠═5f739960-1b18-4804-ba3e-986207146849
# ╟─e4cda53e-7fe2-4cdb-8a9c-c9bdf67dd66f
# ╠═3b980465-9b75-404d-a41f-06ad351d12ae
# ╠═a693145c-8552-44ff-8b46-485c8c8fb738
# ╟─b6da63b0-417c-4a84-83f2-7357bf81fd4d
# ╠═17218b4f-64d2-4de3-a5a9-1bbe9194e9d6
# ╠═3d66f852-2a68-4804-bc73-0747b349cf22
# ╠═dcfc75b1-1951-4db1-9d4c-18b46fb21ffd
# ╠═5eddb220-0264-4ce0-856f-ef040c9af06b
# ╟─86bd203f-83ca-42d9-9150-a141ec163e3f
# ╠═5bd94ed4-a413-418b-a67c-21e3c73ea36b
# ╠═31705a91-8f1b-4e1c-a912-d79e5ce349b4
# ╠═a7b58da6-5e51-4c31-8dbc-82b9fe54e609
# ╠═77a3bb94-19f5-417a-ba34-9270a9054b5c
# ╠═a62b17d9-3d2b-4985-b982-78b557a5889c
# ╠═32e235c8-212e-40bb-9428-74ea2a78b900
# ╠═9e3961f4-fa02-4ce6-9655-484fb7f7e6dc
# ╠═055aaebf-84ea-49dc-8f15-85a0e2e48b49
# ╠═36cc54b3-1e48-4f9a-9b34-b3bac0394eb8
# ╠═c331ec69-2b5c-4ba9-8a7d-7130a27c320f
# ╠═b5f8ce76-008a-491d-9537-f1b8f4943880
# ╠═5ddf5811-9d4c-4375-8ae9-268bcb9e7ed5
# ╠═36bf3b43-4cf6-4a48-8233-1f49b9f7fac3
# ╠═10620dd0-2ef4-40d0-97db-3b6dc1411036
# ╠═94d6e626-965d-4101-b31c-08c983678f92
# ╠═6a02a599-d4f7-4505-b4a8-be8253b47374
# ╠═74046c2f-7ad2-41b3-bb9f-2c6e30dc6a5a
# ╠═f73ae203-056b-400b-8457-6245e9283ead
# ╠═f3da6fb8-a85b-4dfc-a307-bc987239abc6
# ╟─54f438b0-893e-42d3-a0f5-2364723be84e
# ╠═c6f4308d-6dce-4a57-b411-6327f4aa87a7
# ╠═c6c038f0-cf8b-4234-a64c-75616fdc07a5
# ╠═1a5ab262-0493-470f-ab58-baa5fa1a69af
# ╠═b5dff4f2-3088-4301-8dfe-96fb8c6999c7
# ╠═df3549d2-4183-4690-9e04-b665a9286792
# ╠═992c77d0-1b6d-4e07-bd3a-f648ef023870
# ╠═70576040-eec4-4b80-8d23-c2da2e65b2d2
# ╠═e6ea0a65-b79f-400f-be61-951b08b5ce88
# ╠═77aad1ea-16cf-4c5f-9a6b-345f7168afb3
# ╠═fa79e50c-5d5e-47a1-b2e7-fe3341ecbf4f
# ╠═8ef3db5b-f7cd-4d09-8817-35dbf96629d8
# ╠═055a08c0-abc9-465b-bb64-24744e87da5d
# ╠═4947232e-568d-4724-8ebd-5f000bdc9483
# ╠═c9e49960-f659-432b-9338-8390e41b447a
# ╠═0429f5c6-d5c7-461b-b42c-04bfd0f205ab
# ╠═07a187c7-bc2e-47ce-ba06-55620c9f516b
# ╠═46df5f17-15f0-4646-b3f7-653122f0d3b7
# ╠═1256b894-473c-42a8-8451-2b9bb0142d23
# ╠═4a5e2b32-fa51-4146-b0bd-793078b30b95
# ╠═413cbb00-bc81-4d17-948c-76707bfb589a
# ╠═4b37d389-bc6d-4d02-83f3-4d5879646c16
# ╠═6c088aab-1d2b-435b-8a13-a82d5879200e
# ╠═40af6472-77ee-419e-b1b1-d77c29f31c46
# ╠═d34c91f0-75e5-4174-96c0-1ecb3e26d335
# ╠═bdcb3e10-f7b5-49a6-b424-0185f6908301
# ╠═3c945664-6536-428b-a7eb-5f85f189c12f
# ╟─3892a1f6-e205-43ef-957e-3f7ba4fa1bd1
# ╟─de82b60b-9335-4acb-bd1f-84c5539ef04c
# ╟─217bfbed-1e13-4147-b4e4-c58eaed29382
# ╟─61238caa-0247-4cf6-8fb7-3b9db84afcee
# ╠═0486a516-1210-4f42-9eac-b78433065365
# ╠═0bc7b14b-ccaf-48f6-90fa-3006737727ed
# ╠═58b4d31f-5340-4cd9-8c9e-6c504479897f
# ╠═850ddd0b-f6ea-4743-99ca-720d9ac538a0
# ╠═90227375-6661-4696-8abd-9562e08040bf
# ╠═aeeab107-8e68-4d0f-a3e3-8b1e0c12e8e6
# ╠═c5e5a5e3-e324-4c8b-8449-caccc87cf8b6
# ╠═ea1a797c-da0a-4133-bde5-366607964754
# ╠═e5f7e860-4802-42a1-822d-51fcc983dfae
# ╟─8edfe547-f860-418f-8c68-e8fe0c16162f
# ╠═ecb078da-702d-46f7-846a-4759b3e0f172
# ╠═02a15a48-70da-467c-84d3-845612c089da
# ╠═6d5c77c8-2596-45eb-8163-c8e14249948d
# ╠═ba8c8058-7efc-48f2-b3c5-38500524856d
# ╠═49f04d2e-f8fc-4d4e-b9a9-bb97bdf559a6
# ╠═01312a8c-8205-4c63-b617-84067b5c0bc2
# ╠═1b63c7e3-3021-469e-b3d8-d898fa3a755b
# ╠═8a462521-c2bc-4b8a-b58f-4fc5743fe09c
# ╠═41e55b47-b527-491c-ab94-ecc34a9b64bb
# ╠═342a8606-708b-4b1c-9023-050c81ce345d
# ╠═40d7e39c-98c6-4cf8-8461-e54b7a0db696
# ╠═5fb5d0c3-46ab-41dc-be49-64d83cc845f8
# ╠═b0035cc3-0806-41ce-8686-faeab40ee41f
# ╠═ef2f7716-f622-4bf2-ada2-14e76900619b
# ╟─6c387352-ae9c-405d-8f16-562b562c4a4b
# ╠═31a814af-95f1-4d7f-ada0-8bf9df2aee10
# ╠═b9516f6f-bb93-47b3-8669-a4e8adf3db3a
# ╠═9d6a3995-7df0-4552-887e-b306d26a80ad
# ╠═402e380d-a91c-45f7-9fbb-5a2c44e754f7
# ╠═167d31ad-76f2-432d-8550-ba015cb18e9d
# ╠═32bec288-d04b-45ad-8dfc-da4c216ce8e5
# ╠═a89ec65a-951c-445c-a0de-8e465dd5c6b7
# ╟─eea75399-d25c-46b6-a471-2cb8676e4db7
# ╠═e38eee70-b9b0-4dca-ae86-697fc5738cfa
# ╠═195e6792-8dd7-4122-8036-c2e593768539
# ╠═a07ac65e-3e19-411e-9967-af9778d8a45d
# ╠═40907e02-e67d-453d-ae28-6db4543db9d8
# ╠═9b0f5fdb-450e-4a51-a1e9-c8c4a884a4e7
# ╠═3d130a66-4d0a-4d02-bec5-ad19a1ebabff
# ╠═0546ace9-f503-4340-9c1d-2ae58460f32f
# ╠═026dd361-efd3-42dc-a6c9-b27efe6dfd47
# ╠═657008c0-997a-4e01-b9a2-f514dc3550b6
# ╠═d1a85a16-dd24-4ce8-9d8f-df575a585dfa
# ╠═1ddb5277-57cb-4fd6-8fe7-632e23d702f1
# ╠═c27b1508-f7f6-47d0-a04e-f816b4b619de
# ╠═9fe1f154-8174-4ae7-82ca-542b123a5201
# ╠═521a5979-e206-44a1-9b4e-e5348e070125
# ╠═a8d738bb-0424-4df0-aa22-2a299fd994b1
# ╟─dc8fb980-6500-4849-b498-b39454dd3ffa
# ╟─bb078875-611c-46f6-8631-1befde358054
# ╟─df753575-b121-4c3b-a456-cfe4e535c2aa
# ╠═a157969b-100a-4794-a78e-2f40439e28d9
# ╠═09b8527a-afb9-4453-80ff-1026fe2f1bfa
# ╠═9239f8dc-fd07-435f-9d88-e24bb1f6faa2
# ╠═d8395037-8305-4852-8d13-d0fff7005a5a
# ╠═f8478f59-7393-472b-b7d0-5cf18a3d4b29
# ╠═ea14aaee-f5ae-4449-b319-7e20416d2b72
# ╠═0c89dbf1-c714-43da-be89-3f82ccc2373a
# ╠═8cbf28f7-a3a9-4fde-9156-935c25b001e7
# ╟─960d4a47-c701-42c5-934e-a80d74b7ddbd
# ╟─6d873e8f-d3ee-4423-86e7-e8d75abaad38
# ╟─cb23bb0f-87ad-4a1b-8491-c6db384824da
# ╟─28ac0a1f-e8cd-4e27-9852-38e712488b81
# ╟─0c9c56b4-4394-4457-b184-23ac8c35d7e6
# ╟─b81a31cc-5024-418c-8b69-61a6011385ff
# ╠═3d513e7f-a1f1-4668-9a0f-87cf7e7a68c6
# ╟─e81ce097-79e0-4e5e-b7f9-3956b14d5db3
# ╟─bfab2fc0-aee6-4e8d-a7e6-dcbac516dedd
# ╠═98da20c8-c530-4ed6-acb2-3c98019a36a9
# ╟─7ff9cc78-44ae-4baa-b3af-d10487c11920
# ╟─5f260a93-7a7e-4c62-8a52-55371a2093c9
# ╟─294cd133-af00-4025-a182-6e2f057107a6
# ╟─7073082d-a2fa-4419-996c-a38cf3ee044c
# ╟─ec365574-6e66-4284-9723-5634ba73d52a
# ╠═b7983d6f-2dab-4279-b7a5-222e40ce968b
# ╠═ce3e2b3b-a6d6-4494-8168-665e878edae3
# ╠═617aa3dd-045f-4e78-bfcf-2d6c45dfe138
# ╠═f2bbbe98-9b3f-46c2-9630-2b3997549742
# ╠═4e0f81be-141f-4882-83f6-a443febc0592
# ╠═4a0f370f-4331-46de-8636-f93dc6172251
# ╠═d37ff7b0-fb28-4f55-b5c1-f4040f1387f6
# ╠═5cbbd305-513c-42e3-b6b4-5d969d62a3f3
# ╟─ea359342-7685-441d-876e-6562f1770504
# ╟─0cc43266-fd12-447c-ab55-4e8c3d359062
# ╟─fd61e191-3b66-4ebc-b5ec-02a128548ffb
# ╟─7ea0c4fd-a2b2-4a4b-975e-8fd010007788
# ╠═97657a35-0407-4acc-b761-e6ebc06a3764
# ╠═82121c42-b4eb-4495-90f9-3eb34f986d15
# ╠═9ea32e28-94ac-4bfc-902b-c930296f683a
# ╠═a084f5b5-05f4-421c-aa22-dc61511c8002
# ╠═764c858a-8810-4498-b0fe-300a3ecf8488
# ╠═38c79d97-e5da-415b-a1f1-45a093aeeb2f
# ╠═29672872-98e2-454f-b5c2-49966f11e15f
# ╠═47827a31-25cd-4cd3-845d-62c7bf4491ba
# ╠═3a2cf01a-ce41-42c6-9cc2-8a6ff0da2016
# ╠═1a524b30-c267-4551-813a-4fa5eda2bee1
# ╟─61705c8f-a3e2-4d4b-bd52-a334a0a9f5bd
# ╠═0e4ed949-b051-4ae5-8b7b-b1c8d6d7fc94
# ╟─3c41548c-da78-49c5-89a3-455df77bf4fa
# ╠═a149faa9-bee4-42a0-93fe-5adff459e0e9
# ╠═748321c3-0956-4439-90d4-e74598d83f20
# ╠═f1306758-b100-4fca-a030-a0eb2efc34ac
# ╠═10abc1fa-c350-4473-9d72-8e9333a0bee4
# ╠═e10d17be-95f8-4ef9-9387-23a46702c27f
# ╠═19c8d652-f42c-4952-ac16-d4752dafdfea
# ╠═7f4f0941-495d-4309-bccf-da82aab641da
# ╠═f99924cb-8619-4cf5-b026-499f3fd7289e
# ╠═6dd0acce-a14c-41d5-b85d-c2945751983f
# ╠═a346a657-5fe7-4f63-a331-c469e51e34c1
# ╠═9ec28f12-d242-48fa-be75-c96af3c9a324
# ╠═88e0db0e-28bb-457a-9a12-3f020074f4a4
# ╠═b22da8d5-9899-4569-bb82-482ee8f72253
# ╠═f2c3d4ac-5843-47ad-b78b-ca08c9c4df2a
# ╠═fc367d8c-0f9b-4d01-a130-be869801412f
# ╠═16792372-04b2-4a63-b869-c6e37521a324
# ╠═f301972b-feff-485a-b385-943fb434f0fd
# ╠═425e4473-5e8b-4264-8c24-6e06ec2a053b
# ╠═a138dfc8-5e9b-4b64-864a-5c292b75186d
# ╠═8653930c-43d6-4062-8be0-764ea5303abe
# ╠═22660260-d2d5-4f3c-bb8f-173f06a4689e
# ╠═7d312518-a642-4e28-942d-e1eb4de69121
# ╠═a43eb841-aeb8-4ee3-b8b2-72a83741b04f
# ╠═0d3141a3-6fd9-4a8a-9531-56d15fae5d37
# ╠═801fdb3b-698e-4555-8686-98822ef0028b
# ╠═e548a3d5-578d-477f-a192-14d8e4c83853
# ╠═0c3fbc2d-f6cf-4bd8-be25-492ae1bffc07
# ╠═0767fb27-a9d2-4382-9c87-cfe230351a90
# ╠═e7e41d5f-8319-433e-8cfa-be502f84bfd5
# ╠═901c6640-feba-4242-aaeb-687fd2a9f371
# ╠═dbae822c-24fa-4767-aaaa-d6bd8ad700ac
# ╠═808adca1-648b-4831-968f-5951eb024477
# ╠═325d32d2-646f-494b-b747-8e0cc703cb3d
# ╠═bfd0653f-b0ca-4db2-b095-6d50698fb997
# ╠═a4f1cfc8-64c3-40ce-b67c-6548f88a44cb
# ╟─d566fcaf-53ed-4a4d-bf71-891a6fdfe311
# ╠═f770a956-bd1b-41f6-bba4-a228e964965c
# ╠═9309e41a-debd-4e8f-b891-18fa6bf15391
# ╟─dee3a9ec-79f5-4917-ad40-2e4ddcdd423d
# ╠═ad174123-117e-4365-9ede-50456d445fce
# ╟─ee57d5d5-545e-4b70-91d9-b82a108f854b
# ╠═ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
# ╠═30e3c82a-fc70-4922-a7f7-cc1bec0e7d1c
# ╠═448c4152-ca34-458e-a35d-3a3a569d96ec
# ╠═af170bc6-cbd9-4f45-aca8-0a1900dd4ccf
# ╠═7c6d8344-fd9b-4e9c-888e-a57eec39ba04
# ╠═32812780-ff19-4650-9d2d-496849a546ae
# ╟─51ecb564-06d5-4767-aa41-3030ca08a6c7
# ╠═2119a637-98d1-4e1f-b25a-27d3dc42e636
# ╠═763edcce-0696-4670-a8cf-4963cfe70975
# ╠═4c5c4ad3-2bbc-4e26-bea9-ff254b737ca8
# ╠═cde0836c-3dbd-43a9-90fd-c30e5985acf7
# ╠═5caecfab-3874-4989-8b3d-c65b53361c62
# ╠═7fdd86d3-1520-4ff4-8d84-87b4585bce65
# ╠═ac06853c-4c63-4395-a30f-3cb06c552d81
# ╠═b9b8a050-1824-414e-928d-b7797760f176
# ╠═e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
# ╠═e783c591-ea50-4508-8d28-524287466621
# ╠═1d081311-5c63-41b5-8db5-2f18fb7ed6b7
# ╠═f8742eb7-ecc2-4f7d-90bf-acfe6700b80c
# ╟─a6cada88-c7c9-495d-8806-2503e674ec39
# ╠═7c84422b-d522-4f11-9465-058f41a4266f
# ╠═e44de134-6bb6-4d26-9657-963cd587e40a
# ╟─03d2381d-e844-4809-b5a9-048c7612b7e2
# ╠═98d04357-be23-4882-b5b5-8a6d924b7876
# ╠═8f3bcc9f-80e4-455a-bea1-c52feea40191
# ╠═5440fd5f-dfc4-40ed-a614-076904f10220
# ╠═798a89f2-dca0-42eb-9091-ec6b8748fc96
# ╠═6adf9b7c-5044-48f6-b562-de3544dac55d
# ╠═447e0a54-a6b4-492b-8db5-aea294c6d45d
# ╠═e84033ac-3b34-4e1f-a72b-9dfd937382c1
# ╠═13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
# ╠═183175cd-b909-4e03-84d0-e7169a822f89
# ╠═0c1da8de-971b-44dd-84b6-0a236fafe027
# ╠═7b95effc-8729-425e-8443-c9ebf7b02b97
# ╠═3962b939-ff46-4362-a30a-dda8cf84133d
# ╟─6c09c553-0c77-410a-aae9-2004c9768d8b
# ╠═4c6dbaa4-0549-4775-b2d8-7c58cdafc24d
# ╠═291d3eb9-df30-4df4-bd9a-dce4603f6fd2
# ╠═7fbfb83f-e9e1-4701-a7ec-e60cb79b806c
# ╠═ff177d30-ecd8-4935-ba10-dc7ea0b79a06
# ╠═82a444a7-6bdb-4c1a-b9dd-aa85e5056faa
# ╟─bb329d66-c51e-423d-b28b-a15cb38d7ed7
# ╠═590138f1-7d6f-4e78-836d-258f7b4f617e
# ╠═1e09be56-ef2a-4ef1-a181-d1f07dcc4ced
# ╠═685169a0-e65c-439c-a5ba-068c83258200
# ╠═d0c234cc-9620-4c9c-bfae-f32b68b9d31f
# ╠═ae507fc7-5a13-42ed-a477-79d0b11c2efb
# ╠═982b25d3-fba6-4dac-9340-d5e87075f95a
# ╠═19ffec1f-43f7-49af-91be-2553d8998951
# ╠═0773b0fe-5a8a-4079-9e14-1ea5520a4bdb
# ╠═fe9f7509-2ba7-45f2-bcf5-84312660d754
# ╠═2be996a7-d960-4939-91c7-9f1a42d35b49
# ╠═a282de0c-aa3b-494a-84a2-9d37e20d1adc
# ╠═93c45e9c-0231-49fd-9a94-2ef0b03ecddb
# ╠═9b3f7260-58f5-45d5-8d53-cb43d739acd9
# ╠═5eebab7e-7cc4-410e-a9ae-a628644edbeb
# ╠═55915fe9-b1f4-4b45-b418-65c7af973f4f
# ╠═a5a9d0e7-2a1f-4b83-a790-6148a2029490
# ╠═d395b477-b536-4603-bc45-b105b3463042
# ╠═dec54a04-6c59-41ae-946e-937092eaaa43
# ╠═92e7c0a0-19df-43e2-90db-4e050ba90c5d
# ╠═ec420749-2c5e-4b70-a476-ff8be50f6fca
# ╠═a0cd5e68-3bd0-4983-809e-ce7acd36a048
# ╠═86871d26-5612-4166-959f-6fa7d33daaf6
# ╠═ed696113-737b-46f5-bfb1-c06d430a83ac
# ╠═8b3175d4-6938-4823-8a91-77cde7c31c2a
# ╠═9d1ea09f-3c42-437f-afbc-21a70296a3ba
# ╠═05a040cf-f6ac-42b2-bcb1-599af5a26038
# ╠═41751922-2646-45fd-b6e0-7ca2109cc642
# ╠═9156e2e0-1a15-4176-a664-5a03b725c24e
# ╠═c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
# ╟─e98994c4-78ab-4d08-a1a0-e8fb3f3e79a7
# ╠═11277ca6-3bbf-403f-8cfc-2aabf94069f7
# ╠═b0f7e467-684b-41dd-b046-ed378dded683
# ╠═38b3911a-8db5-47f4-bd1b-0d77a207188f
# ╠═28d2660e-258d-4583-8224-c2b4190f4140
# ╠═fbe43a2a-f9b0-487e-a579-645c2f40736e
# ╠═7b05ec74-6edc-4f44-bf9d-9389d4029494
# ╟─c546d3af-8f24-40bd-abfa-e06c708c244e
# ╟─5db964da-1ebc-478a-a63a-cf4f713c7aa8
# ╠═778dc2a7-4e9b-4f97-b34d-6a7adc38abc2
# ╠═5953c75a-0e6d-40c4-8e4b-b8de6920acf3
# ╠═c7874fdb-2ba4-4dbf-89b4-0b3af493b256
# ╠═23525c22-3fc4-4a4e-b2c6-601463a5df31
# ╟─b98baad1-dad8-464c-a18a-1baf01962164
# ╟─3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
# ╠═d437be5f-ec26-466b-b099-5e1cc8816cb5
# ╠═dd8e709c-b945-447f-9b3d-8bf9d9e0249c
# ╟─2426e2d6-e364-4e97-bee8-7defb1e88745
# ╠═9c8f603f-33c6-4988-9efd-83864e871907
# ╠═2d337665-65e8-4f81-83a4-7328fd186352
# ╠═2bd22bd5-b6e2-450a-899d-41d647a9b3e3
# ╠═ecbb707c-bea0-4b86-a7b8-a4f55027c05d
# ╠═20a6b910-4f0b-4c78-af6b-d76e55025297
# ╠═47f4e899-f441-4a68-80b4-2b540f738f3f
# ╠═95bfd1a2-3bd6-43f6-b2b1-fb94139212ad
# ╠═8c401b96-6de6-48cb-9a85-ff0620fec402
# ╠═cf39d6f0-b592-4a4d-aa17-182f9c794959
# ╠═ba765277-25e9-46a5-ba68-1c33c55c9755
# ╟─991815d7-09da-423c-b14a-a8a3fcf662e4
# ╠═bc64b624-d8d2-480a-a698-092aea0a74b2
# ╟─0e876bab-6648-4a67-b571-dc82a7bdf8f1
# ╠═3763ad63-003e-495f-aa90-0db525412c62
# ╠═061203a0-9ccc-4988-b4f0-11743ecec084
# ╠═a9a9ea9d-5a97-4260-8e02-a27732928e61
# ╠═b1e7c19f-e2f3-4f17-9933-e5c5e90e386d
# ╠═19af17cf-193d-48a7-8db6-e20a3139692f
# ╠═8f849bc4-addc-4a0d-9105-48307068febc
# ╠═5ed2db5f-c499-4749-87f2-4bb881accf16
# ╠═c4076bd0-8718-4641-9d09-2c3b71aff1e3
# ╟─202c9072-0a7b-454d-9112-4ecc0a03c61b
# ╠═434d7e0d-583d-498b-9b6c-a72fa3775b3c
# ╠═2857461c-d24e-4b04-91f1-80cd842eeaa4
# ╟─a78a6fd5-4b45-43fc-a733-aef4fd14eb42
# ╠═1a4471e7-24ad-4652-9f3f-6eef92c781d5
# ╠═8c074d96-ee63-45fb-bbbf-135c40b66a09
# ╠═9ef995fb-a591-45df-af67-df63808b119d
# ╠═399d45a5-a217-413a-8ba8-76b93e246a89
# ╠═5b852b2d-d86e-401b-8251-b21fa080ed1f
# ╠═e4e7cfed-d150-4615-bb77-a8db4b395011
# ╠═e0ff3466-1a61-45aa-a5b8-9300e81ccc9a
# ╠═e1226518-0341-4a7c-bdfc-cc93be354638
# ╟─04d720ce-1de5-4c8b-bd2f-0d0a5e8ed271
# ╠═6109d56f-ce01-4b10-bb13-1e2eb0ccf990
# ╠═bc6c39e0-f878-4bea-8453-c845f0d3eba9
# ╟─8ae63dda-a9d2-47e2-a89d-05fe8c11383b
# ╠═7e8458e4-15be-47ac-8c08-1ff042c5c9b3
# ╠═ad7d7a32-861d-48be-af41-833e74023ef6
# ╟─edceb483-eb33-4bf9-975d-3dc6f18cffe9
# ╠═b5178580-013b-4686-8bfc-c1f7395620b2
# ╠═79811ec5-d713-4dc4-b1a6-b0ea656633fd
# ╠═7563fe9c-5142-455e-a7ce-559a14b92f28
# ╠═bafb366d-c0fb-428d-8188-7a2c6e100617
# ╠═222de0a7-0cef-4b01-a683-3bdd8f892f88
# ╠═eb08a786-9f78-49cc-86b2-418e881a8b2a
# ╟─bac2785d-d692-4524-86c5-dc183f07fe86
# ╠═c8452580-4680-4ca1-ae58-4c17611e4351
# ╠═73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
# ╟─26086070-08a3-4ce8-b8e6-cd2cfc83e44d
# ╠═1af87fef-8c88-4e34-b1d4-8bccd7881473
# ╠═da9a6ba8-083b-40b3-9bd6-089688ff7f73
# ╠═433d51c9-9f31-49db-a5a2-e6565888831e
# ╠═c25b404b-0d89-4c4d-bc61-c361fd9d7038
# ╠═c15b52c1-5d06-4fea-8645-91f92bfa716b
# ╠═07c78fb1-bcf4-40fd-9209-d3220d014912
# ╠═15b3535f-6776-409d-8e6a-457b4c6eba78
# ╠═0c61beac-7fdb-4f05-a04c-f2a042ba34b2
# ╠═d96c7a96-0875-4ebd-a1ba-e40db8008aed
# ╠═734ed2d0-4657-4361-9285-97605371af72
# ╠═d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
# ╠═43a39165-1f9b-49e7-a4bc-c47583c25b10
# ╠═f2067017-32c8-493b-a9fb-3a89f9e549e4
# ╠═ee0d413c-a107-4bdc-a08a-52cda6d81573
# ╠═77456db6-b0fa-4ba6-9d41-1c393f7ddee1
# ╠═08b888e3-9d37-43a6-9656-4bf6cb62d324
# ╠═bf71086c-a8f0-4420-b698-73499fec5257
# ╠═4c99c655-ae95-4a28-95c8-e7ca38ddf55f
# ╟─372df76a-fd5d-4b84-ad83-988af1e78cf7
# ╠═459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
# ╠═c05dd607-1ce4-4931-9e23-2a9dad7cc039
# ╠═ae386c52-ff6b-4493-96ff-6c14d1c46db8
# ╠═94580726-88ed-4a69-a101-9c792e3ed44b
# ╠═a3a49ba6-7272-4739-b11d-f09692afa46f
# ╠═cb96e206-8779-4c02-abd2-74948e13f263
# ╠═daf769b3-bcd3-4476-bdfb-492a90cd1225
# ╠═1ddf99ee-0dfa-4dc2-82e4-8e065e78c992
# ╠═979e8ca8-105c-4f84-a20b-e3bd729495b4
# ╠═467a06fe-2360-442d-b8b6-bbcbb09681f8
# ╠═5d7e9b7a-aa1b-4737-94e4-a5df1de8ecc6
# ╠═6cc61a93-1143-42a1-9b8d-9a3f506db3df
# ╠═20cc6945-d767-41e1-8a6b-1d88084a9781
# ╠═4acabffb-ed34-4847-9ce2-c30031392ecc
# ╠═a7ef8e5a-ce04-4a3c-8ef8-421990f176a3
# ╠═50fee719-874e-42f6-8375-b0bcb9689c3c
# ╠═faecba7d-a52c-4d8f-a4a5-14f3610cb827
# ╠═923da0d6-8333-4335-ad95-1fdc47cde9cb
# ╠═8dec16f2-f672-4a37-b6c6-6d2e7fad9b7c
# ╠═a52dae0c-ce46-4a24-b6fc-02fb5226dcda
# ╠═ac0375e0-131d-49d8-900e-754fa167c13b
# ╠═44a94b75-ea19-40a8-a863-e362c651c8a2
# ╠═adbe4ee5-62e7-42c6-baab-e42a3c4af4cc
# ╠═ab0f2a21-2ce6-4fa5-960b-325ef778304f
# ╠═81f70fa7-0240-4868-ac9a-73a119b4e831
# ╠═cf5ef024-edbb-4468-99af-06064358a8ab
# ╠═d41e7488-5655-487c-ba31-19be428fff2b
# ╠═b67fab52-d0e0-45b9-8eba-99b4bca34182
# ╠═62cdf00d-86c4-4ad3-bf64-dfbaa4d6c6e6
# ╠═67e2b019-47bd-4473-974d-b5fbfcc55f18
# ╠═2dc6205a-f370-4727-bc79-5ac6716e2299
# ╠═61e30268-26f3-4940-8e9d-b9d5c54f4da0
# ╠═b90dcb6c-70fe-4b7f-b91a-15a8e78d8411
# ╠═e008e66b-40f1-474c-ad3b-f64e542c7499
# ╠═33a3ddb2-898e-4bec-bc45-3cd65e8ddbd0
# ╠═c66a453b-c9b6-427f-ac13-a392626955e1
# ╠═9cbf5154-b634-44e5-8646-74468f8977f5
# ╠═f4d83c51-650f-4ed3-856b-f39f74fd6e0e
# ╠═6f78dd2a-c225-41c7-b6a9-5f0925f87d14
# ╠═ea5f7773-0697-4ead-adc9-0eabbd0fa05e
# ╠═b2bf0ddc-a327-4b13-9722-130a0ddebffb
# ╠═3ec1c94b-ac88-4f01-af9d-abb956dff6f1
# ╠═3b0a8743-20b6-4533-a598-09a2c94d6528
# ╠═62b6cf95-266b-46d4-b478-2e11cdf1acb0
# ╠═ec90e969-6183-4e98-b2ab-3a53eee7b13b
# ╟─f54d3095-8e78-4f36-8f81-a022c6c501d6
# ╠═93dd3b75-b24a-4a03-853a-cc6d0a5188e4
# ╠═051fc252-5267-4c8b-8ead-75f37a49a440
# ╠═148923dc-8802-4af2-9c4b-ec41ad842366
# ╠═be498601-d19d-43ab-8bbb-aed717728e63
# ╟─96dd4c14-6791-4c8c-80c1-8f0301e34b35
# ╠═4df27234-b5b1-4607-b955-c2710257ffd6
# ╠═0691bffd-6b86-41ce-8044-5474545f9417
# ╠═55f1323b-291e-486d-b3a8-6409db9f7b8e
# ╠═d15506e2-5213-48fc-bf6a-fbe6e99af3de
# ╠═f75ae159-d811-47dc-8592-73a8048244fb
# ╠═9e2aef0c-0121-4652-ad40-62ad5e8d0a0e
# ╠═c5899e25-25b2-43ec-8e9a-3e7e9af8f50d
# ╠═3e8fd197-59fd-4547-8437-45456dec8f8d
# ╠═4eb5ef29-57d2-42e8-8579-411964792f72
# ╠═08c44240-6481-4af1-8473-b9cf1d91ef3c
# ╠═aba25e76-c18f-44de-b57b-c3aa8f5be42d
# ╠═e85de5cc-219b-4e1d-a16d-1e84d657b870
# ╠═3b26f240-30f8-4894-9dab-8ce65a7cac4b
# ╠═85375b75-1bff-4f4a-a093-4bb55e752c0f
# ╠═d9ac0584-db8f-497e-807f-9b207b0a6151
# ╠═5a4110cf-6700-4b82-bdcf-4c724e342d1c
# ╠═f254e3f8-a5a3-4d83-b2d8-50552bcd2edf
# ╠═44e249fc-dea4-464b-b8a8-94ef304f5769
# ╠═c5903324-c203-43d5-9f0c-b5c7ba03af07
# ╠═38d0794b-a374-4a18-94d3-ba0ff3d8bba0
# ╠═b7dcfb1b-0f90-4fad-9e3d-032944a24eca
# ╠═c1287f32-e270-47fc-8b18-38394380e76e
# ╠═14357463-a585-409b-bbe1-a02bfd086c61
# ╠═1732eaee-3259-4982-9dae-69eed1c8b56d
# ╠═a440743b-6595-4bfa-93a7-36784dc965a3
# ╠═61186b87-03ed-4cbd-bbfb-cecb98ab9313
# ╠═add11e32-0299-4d2b-8d12-6c7436f3b49d
# ╠═cdd6fcc4-0e63-4e9d-ac8b-e8a366f83625
# ╠═e4cfe664-c46b-4626-a6f3-42d87b98e868
# ╠═82d00718-cc23-42b6-9902-bb0032fde744
# ╠═dbe1f66c-e9ab-4d29-8846-20031c93abf1
# ╠═f2300c70-9eba-4711-a35b-cec5ae8340f5
# ╠═c4716c23-2924-4751-a303-e688b71199ab
# ╠═64fde861-20a5-4cdf-ab7e-d1409018ffe2
# ╠═1938a66f-9f6e-4b5c-9f67-6ba5c99d8bd1
# ╠═b02061e3-e42a-4581-b085-96b02fb19b52
# ╠═f00265ca-fa8a-43f2-bde5-8924934a9406
# ╠═9e6f7e27-a4bb-434b-8daf-7a0d831118ee
# ╠═9d677b71-dcd9-45ba-907c-b3b12d375140
# ╠═d36a7409-2441-4e04-b704-c389b4e690d0
# ╠═8ee0146c-e3c3-4a16-b527-ee1e3da663fa
# ╠═7ac9f47d-c661-46ee-8f03-4029d248550d
# ╠═e89c179f-3de0-4613-bbf5-fd936572ce5c
# ╠═0b50394f-5800-449b-89bb-d62db694d174
# ╠═0915512e-f2fc-432f-9ba7-ba1b3f8cb1ef
# ╠═107b60fa-692b-4c87-944b-ae606a3a353a
# ╠═03ce5f2e-e11d-4a4a-95c4-37b1c931ec18
# ╟─329d55b8-eb72-4a1e-a4e8-200fee0e0b9d
# ╠═b5ce80a3-e177-4c4f-920b-5dee87f2bc3b
# ╠═c92ab000-dbe9-457d-97f3-88ae31b57a27
# ╠═c089c975-96e1-4281-b5ad-c53e738834a1
# ╠═e4344d50-425b-4bea-b28e-0c3b45debfb1
# ╠═f9ccee0a-9e6b-4070-a15c-ff5d5c324649
# ╠═624ab4a3-5c3b-42f3-be37-89d6382fdfdd
# ╟─ec97d987-651d-4efa-a36f-e6be9f18e0fd
# ╟─b1f8755d-8130-4492-859e-781225f8bb45
# ╠═64b19240-6e2a-4bb9-b681-5d4fdcc98935
# ╠═fbdc69a6-7227-4df8-a9a9-eca836565a11
# ╠═92bd44dc-c382-4835-b6ed-351756703389
# ╠═95f04c20-6adb-4122-9d4a-ccda1663fc83
# ╠═b8c478a1-c908-423f-bb30-c54d0d5a2f8e
# ╠═5591c99f-fccf-42cc-a8ce-db7a4b7ddf08
# ╠═95ed19ae-63ba-46e4-ade7-56aa84faccda
# ╠═393f1361-94bf-4dee-8665-39085f0db729
# ╠═c57efb98-5eca-4139-bcd4-4aec1323694e
# ╟─55e18a25-e68d-47d1-a8b1-581c8fa8761c
# ╠═b5fb3739-a983-4801-98db-cd269a7e9e28
# ╠═cf2021eb-f995-4575-9aec-ed8c0fd9c477
# ╠═a8344dde-62ea-4697-b06a-d057d8c335a5
# ╟─d60cd919-0208-43a2-934a-b880bf95fd69
# ╟─dff8b33a-7513-42c5-8d98-4d56445e65d6
# ╟─1ad32726-a3ae-461b-8e30-ec289c8ff373
# ╠═812d7cda-ce12-495c-be38-52c8f0f23747
# ╠═01e27ad1-9b45-40b3-8d03-26400cd153f9
# ╠═4eec8fe5-2c77-489d-8d1f-fef0ad388a50
# ╠═09a85206-3f09-4ba4-8fe3-85d32c8e8793
# ╠═6ec97827-015d-4da4-8e58-4564db02fedf
# ╟─87b55c18-6842-4321-b2c7-c1abd8fef6fc
# ╟─e81eacd1-22ff-4890-989e-e4ec638f06b5
# ╟─f37dd092-aa81-4669-8925-665415c90aaf
# ╠═512cad43-28ca-4b07-afa5-20571a31b311
# ╠═388d9daf-d6e3-4551-84bb-56906f013900
# ╟─99c5d986-0cdb-4321-82bf-49ced0502430
# ╟─b92c6f37-2329-4ce8-bcf9-eb53f09f7266
# ╠═1c3c1fc5-f1f1-44e7-9700-96f1342b5f9f
# ╠═e6cb1988-a03f-4931-84f6-a1bff6da1c4e
# ╠═030c5f35-1f1d-4bc0-9bd6-c97c193ac835
# ╠═df7e5922-5d1e-4a55-8ae6-9e3e60e0fa9b
# ╠═c2f4541e-d453-4e44-99ed-229a126cda7e
# ╠═f825b6a1-0950-461b-80a8-b0fc9924dd53
# ╠═8806127c-8bbd-4567-b32e-946fb473b4c7
# ╠═a8d09777-65ed-4178-b3f2-12a269ccfbbb
# ╠═63031393-40e1-4b16-aaa6-34c7fe9fb35a
# ╟─ea8e4429-b797-4452-bb7d-73ebbd58af76
# ╠═b3878f52-bbce-433f-a23e-ed5b56d4f5b1
# ╠═170b929b-ec44-4eb1-bdcf-61a280e54b7d
# ╠═732efa0f-8a8a-4c53-a7ad-e90a19c2f637
