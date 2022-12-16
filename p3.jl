### A Pluto.jl notebook ###
# v0.19.16

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

# ╔═╡ aeb3597f-ef6a-4d21-bff5-d2ad359bc1a2
using Chain

# ╔═╡ 9d5f897f-7f25-43ed-a30c-064f21e50174
using Pipe: @pipe

# ╔═╡ 786c2441-7abb-4caa-9f50-c6078fff0f56
using ArchGDAL

# ╔═╡ 1c7bfba6-5e1d-457d-bd92-8ba445353e0b
using MLJ

# ╔═╡ a95245b0-afa8-443a-9a16-b01de7294f13
using LinearAlgebra

# ╔═╡ 0486a516-1210-4f42-9eac-b78433065365
using Random

# ╔═╡ 97657a35-0407-4acc-b761-e6ebc06a3764
using StatsBase

# ╔═╡ 02e93f83-be74-4614-a0ff-2b1044198975
using ColorSchemes

# ╔═╡ 89b9cce5-b297-48c9-a4b6-3a7b43952294
using Plots

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

# ╔═╡ 9aa06073-d43e-4658-adb9-bbc11425978d
begin
	t = CSV.read(joinpath(input_dir, "train.csv"), DataFrame)
	v = CSV.read(joinpath(input_dir, "validate.csv"), DataFrame)

	train = vcat(t, v)
	validate = v
	test = CSV.read(joinpath(input_dir, "test.csv"), DataFrame)
	@info unique(sort(filter(x -> x.date >= DateTime(2020), test).date))

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

# ╔═╡ 9deff2bf-61a3-48c8-be41-5c3b501d604f


# ╔═╡ c8a8830a-a861-4b35-ae13-2a5cccecbe50
building_zones = select(unique(vcat(train,test), Symbol("Property Id")), [Symbol("Property Id"),:zone]);

# ╔═╡ ced7e799-c5e9-490a-943e-533d2d1b4f2a
testcouncils = select(test, ["Property Id","council_region"]);

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
validation_metadata = select(unique(validate, "Property Id"), [
	"Property Id",
	"heightroof",
	"cnstrct_yr",
	"groundelev",
	"area",
	"council_region",
	"weather_station_distance"
]);

# ╔═╡ 44f02f33-9044-4355-ba82-b35595a82bdd
test_metadata = select(unique(test, "Property Id"), [
	"Property Id",
	"heightroof",
	"cnstrct_yr",
	"groundelev",
	"area",
	"council_region",
	"weather_station_distance"
]);

# ╔═╡ c2560a7b-ec09-4cdb-bdec-68a65301249f
cmip_p = joinpath(input_dir_environmental, "cmip.csv");

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

# ╔═╡ 04349795-8a8d-40b7-a515-cb4806a5776f
cmip_r = CSV.read(cmip_p, DataFrame; dateformat=date_f);

# ╔═╡ bf38b421-99ea-48fa-a548-49c9ffd52758


# ╔═╡ 8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
epw_r = CSV.read(epw_p, DataFrame);

# ╔═╡ 79fed5b6-3842-47b7-8918-63f918e070bb
begin
	dynam_p = joinpath(input_dir_environmental, "dynamicworld.csv")
	dynam_r = CSV.read(dynam_p, DataFrame; dateformat=date_f)
end

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
noaa_r = select(CSV.read(noaa_p, DataFrame; dateformat=date_f), Not(:ACPC01));

# ╔═╡ 0a22f19d-c662-4071-b4e2-6e8103a0f359
sentinel_1C_r = CSV.read(sentinel_1C_p, DataFrame; dateformat=date_f);

# ╔═╡ a9f2d94d-cbf8-4d47-a4b2-438f451882e5
sentinel_2A_r = CSV.read(sentinel_2A_p, DataFrame; dateformat=date_f);

# ╔═╡ ca12dd08-29af-4ce3-a2cc-d3bf1fa9e3c7
viirs_r = CSV.read(viirs_p, DataFrame; dateformat=date_f)

# ╔═╡ d4680498-1966-48f7-8a56-296578559d53
maximum(viirs_r.date)

# ╔═╡ 348c4307-94dc-4d5f-82b0-77dc535c1650
function strip_month!(data::DataFrame)
	data[!,"date"] = Date.(Dates.Year.(data.date), Dates.Month.(data.date))
end

# ╔═╡ 09a4789c-cbe7-496e-98b5-a2c2db3102b6
begin
	# also want to get the building data in a uniform format for matching
	strip_month!(train)
	strip_month!(validate)
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

# ╔═╡ f8783987-88a7-47a5-8f7e-fea72f321e60
strip_month!(cmip_r);

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

# ╔═╡ ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
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

# ╔═╡ 2f1fec21-76a2-4365-b305-0f24505b1ccc
describe(sar_r, :nmissing)

# ╔═╡ 86d465e3-7916-479d-a29c-2b93ae54ab6d
agg_terms = 	["Property Id","date"]

# ╔═╡ 637220ba-c76a-4210-8c08-fde56b86366a
functional_terms = [f₃ f₅]

# ╔═╡ c0d3f3b7-60b1-4b73-86cd-2b10b30e3f57
begin
	# CMIP Suite
	cmip30_p = joinpath(input_dir_environmental, "cmip_30.csv")
	cmip40_p = joinpath(input_dir_environmental, "cmip_40.csv")
	cmip50_p = joinpath(input_dir_environmental, "cmip_50.csv")
	cmip60_p = joinpath(input_dir_environmental, "cmip_60.csv")
	
	cmip30_r = CSV.read(cmip30_p, DataFrame; dateformat=date_f);
	cmip40_r = CSV.read(cmip40_p, DataFrame; dateformat=date_f);
	cmip50_r = CSV.read(cmip50_p, DataFrame; dateformat=date_f);
	cmip60_r = CSV.read(cmip60_p, DataFrame; dateformat=date_f);

	strip_month!(cmip30_r)
	strip_month!(cmip40_r)
	strip_month!(cmip50_r)
	strip_month!(cmip60_r)

	cmip30 = monthly_aggregation(cmip30_r, agg_terms, functional_terms);
	cmip40 = monthly_aggregation(cmip40_r, agg_terms, functional_terms);
	cmip50 = monthly_aggregation(cmip50_r, agg_terms, functional_terms);
	cmip60 = monthly_aggregation(cmip60_r, agg_terms, functional_terms);

	# we want to match dates here but keep the data unique - matching on 2019
	cmip30.date = cmip30.date .- Dates.Year(11)
	cmip40.date = cmip40.date .- Dates.Year(21)
	cmip50.date = cmip50.date .- Dates.Year(31)
	cmip60.date = cmip60.date .- Dates.Year(41)
end;

# ╔═╡ cbd44ab9-6346-46f0-bb22-1c19b471ccf7
cmip = 			monthly_aggregation(cmip_r, agg_terms, functional_terms);

# ╔═╡ 3da877c2-159b-4d0d-8b97-34da4dbf2ac3
epw = 			monthly_aggregation(epw_r, agg_terms, functional_terms);

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
					select(sar_r, Not([:HH,:HV])), 
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
	"weather_station_distance"
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

X = select(data_clean, Not(join_on))
aux = select(data_clean, join_on)

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

X = select(data_augmented, Not(join_on))
aux = select(data_augmented, join_on)

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
# 	training_prepared = select(training_data, Not(join_on))
# end

# println(nrow(training_prepared))
	
X = select(training_data, Not(prediction_term))
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
# 	validating_prepared = select(validation_data, Not(join_on))
# 	auxiliary = select(validation_data, join_on)
# end

X = select(validation_data, Not(prediction_term))
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

# ╔═╡ b5f8ce76-008a-491d-9537-f1b8f4943880
begin
tₐ = clean(dropmissing(select(train, electricity_terms)))
tₐ.month = coerce(tₐ.month, OrderedFactor)

tᵧ = clean(dropmissing(select(train, naturalgas_terms)))
tᵧ.month = coerce(tᵧ.month, OrderedFactor)
end;

# ╔═╡ 77a3bb94-19f5-417a-ba34-9270a9054b5c
minimum(tₐ.date)

# ╔═╡ a62b17d9-3d2b-4985-b982-78b557a5889c
maximum(tₐ.date)

# ╔═╡ 5ddf5811-9d4c-4375-8ae9-268bcb9e7ed5
naturalgas_terms

# ╔═╡ 36bf3b43-4cf6-4a48-8233-1f49b9f7fac3
begin
vₐ = clean(dropmissing(select(validate, electricity_terms)))
vₐ.month = coerce(vₐ.month, OrderedFactor)

vᵧ = clean(dropmissing(select(validate, naturalgas_terms)))
vᵧ.month = coerce(vᵧ.month, OrderedFactor)
end;

# ╔═╡ 94d6e626-965d-4101-b31c-08c983678f92
begin
teₐ = clean(dropmissing(select(test, electricity_terms)))
teₐ.month = coerce(teₐ.month, OrderedFactor)

teᵧ = clean(dropmissing(select(test, naturalgas_terms)))
teᵧ.month = coerce(teᵧ.month, OrderedFactor)
end;

# ╔═╡ 6a02a599-d4f7-4505-b4a8-be8253b47374
dataₑ = vcat(tₐ, teₐ);

# ╔═╡ 74046c2f-7ad2-41b3-bb9f-2c6e30dc6a5a
dataᵧ = vcat(tᵧ, teᵧ);

# ╔═╡ f73ae203-056b-400b-8457-6245e9283ead
schema(tₐ)

# ╔═╡ f3da6fb8-a85b-4dfc-a307-bc987239abc6
schema(vₐ)

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

# ╔═╡ 76e4aac5-be47-49e3-8359-3d2c9f24500b
Η₁

# ╔═╡ 992c77d0-1b6d-4e07-bd3a-f648ef023870
begin
tₐ₂ = clean(dropmissing(select(train, electricity_terms, "zone")))
tₐ₂.month = coerce(tₐ₂.month, OrderedFactor)

tᵧ₂ = clean(dropmissing(select(train, naturalgas_terms, "zone")))
tᵧ₂.month = coerce(tᵧ₂.month, OrderedFactor)

ᵞ₂ = combine(groupby(tᵧ₂, [:zone,:month]), :naturalgas_mwh => mean => "Natural Gas");
p₂ = combine(groupby(tₐ₂, [:zone,:month]), :electricity_mwh => mean => "Electricity");

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
		Guide.yticks(ticks=0.0:0.05:0.35),
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

# ╔═╡ 12f035a7-73a9-4d3b-9004-f223600d5059
names(sar′, Union{Real, Missing})

# ╔═╡ 07fac431-8103-451f-b4f2-9e000bdbc418
sample_sars = unique(sar′[:,"Property Id"])[1:2000]

# ╔═╡ 59ee8ba9-97a1-4a3b-99a4-cddc0a667a3e
sample_filtered = filter( x-> x["Property Id"] in sample_sars, sar′ );

# ╔═╡ 932e6c04-d0e5-4517-b80a-2e468def88d6
t₁ = Gadfly.plot(
	sample_filtered,
	Gadfly.layer(
		x=:LST_Night_1km_f₃,
		y=sample_filtered.LST_Day_1km_f₃,
		intercept=[mean(sample_filtered.LST_Day_1km_f₃ .- sample_filtered.LST_Night_1km_f₃) - 5], 
		slope=[1],
		Geom.abline(color="pink", style=:solid),
	),
	Gadfly.layer(
		x=:LST_Night_1km_f₃,
		y=:LST_Day_1km_f₃,
		Geom.smooth,
		intercept=[0], 
		slope=[1],
		Geom.abline(color="red", style=:dot),
	),
	Gadfly.layer(
		x=:LST_Night_1km_f₃,
		y=:LST_Day_1km_f₃,
		color="Property Id",
		# Geom.line,
		# Geom.point,
		# Geom.line,
		Geom.smooth(smoothing=0.7),
		Theme(line_width=0.08pt, alphas=[0.5])
	),
	Scale.discrete_color_hue,
	Guide.ylabel("Daytime Temperature - °C"),
	Guide.xlabel("Nighttime Temperature - °C"),
	Guide.title("Diurnal Temperature Trends - Landsat8"),
	Theme(default_color="black", key_position = :none),
)

# ╔═╡ 4b37d389-bc6d-4d02-83f3-4d5879646c16
draw(
	PNG(
		joinpath(output_dir, "temperature_nonlinearity.png"), 
		13cm, 
		9cm,
		dpi=500
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
Tree

# ╔═╡ ea1a797c-da0a-4133-bde5-366607964754
# m_tree = EvoTree(
# 	rng=rng, 
# 	max_depth=3,
# 	lambda=0.1,
# 	gamma=0.1,
# 	nrounds=20, 
# 	rowsample=0.9, 
# 	colsample=0.9,
# 	device="gpu"
# )

# ╔═╡ e5f7e860-4802-42a1-822d-51fcc983dfae
m_tree = Tree(
	rng=rng,
	max_depth=3,
	n_trees=20,
	sampling_fraction=0.85
)

# ╔═╡ 019f1c79-440c-42aa-9483-d64389876336
m = EnsembleModel(model=m_tree, n=5)

# ╔═╡ 8edfe547-f860-418f-8c68-e8fe0c16162f
md"""
###### Electric Results
"""

# ╔═╡ 6d5c77c8-2596-45eb-8163-c8e14249948d
# baselineₑ = filter(x -> x.model == "Null", modelresultsₑ).rmse[1]

# ╔═╡ ef2f7716-f622-4bf2-ada2-14e76900619b
# feature_importances(mₑ)[1:10]

# ╔═╡ 6c387352-ae9c-405d-8f16-562b562c4a4b
md"""
###### Gas Results
"""

# ╔═╡ eea75399-d25c-46b6-a471-2cb8676e4db7
md"""
##### 1. Initial Results - Comparison to Baseline
"""

# ╔═╡ efbe759f-ec6c-46a3-b4aa-8fa051f31151
# Gadfly.plot(
# 	modelresultsᵧ,
# 	x=:model,
# 	y=:rmse,
# 	color=:zone,
# 	Geom.point,
# 	Geom.line
# 	# Geom.subplot_grid(Geom.point)
# )

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

# ╔═╡ 9239f8dc-fd07-435f-9d88-e24bb1f6faa2
for term in comprehensive_datalist
	@info nrow(term)
end

# ╔═╡ ea14aaee-f5ae-4449-b319-7e20416d2b72
tₐ

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
# here is where we want to inject all of the data and drop missing terms
vₐ′ = clean(dropmissing(innerjoin(
	vₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

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
extra_omission_features = ["weather_station_distance"]

# ╔═╡ 617aa3dd-045f-4e78-bfcf-2d6c45dfe138
tᵧ′ = dropmissing(clean(innerjoin(
	tᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

# ╔═╡ f2bbbe98-9b3f-46c2-9630-2b3997549742
vᵧ′ = clean(dropmissing(innerjoin(
	vᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

# ╔═╡ 4e0f81be-141f-4882-83f6-a443febc0592
maximum(teᵧ.date)

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

# ╔═╡ 61705c8f-a3e2-4d4b-bd52-a334a0a9f5bd
function electrictrain(modelname::String, terms::Vector{String})
	tᵥ = select(tₐ′, terms)
	mᵥ = training_pipeline(
		select(tᵥ, exclusion_terms),
		"electricity_mwh",
		m
	)

	# validation
	vᵥ = select(vₐ′, terms)
	vᵥ.prediction = validation_pipeline(
		select(vᵥ, exclusion_terms),
		"electricity_mwh",
		mᵥ
	);
	
	vᵥ.recorded = vᵥ.electricity_mwh
	vᵥ.model = repeat([modelname], nrow(vᵥ))
	
	# test
	teᵥ = select(teₐ′, terms)
	teᵥ.prediction = validation_pipeline(
		select(teᵥ, exclusion_terms),
		"electricity_mwh",
		mᵥ
	);
	
	teᵥ.recorded = teᵥ.electricity_mwh
	teᵥ.model = repeat([modelname], nrow(teᵥ))

	leftjoin!(
		teᵥ,
		building_zones,
		on="Property Id"
	)

	return mᵥ, vᵥ, teᵥ
end

# ╔═╡ 3c41548c-da78-49c5-89a3-455df77bf4fa
md"""
Null model
"""

# ╔═╡ a149faa9-bee4-42a0-93fe-5adff459e0e9
## as a preliminary introduction - thinking about the overall dataset
@info "Number of data points" nrow(tₐ′)

# ╔═╡ 748321c3-0956-4439-90d4-e74598d83f20
term₀ = unique([electricity_terms...])

# ╔═╡ dbae822c-24fa-4767-aaaa-d6bd8ad700ac
m₀, vₐ′₀, teₐ′₀ = electrictrain("Null", term₀);

# ╔═╡ 808adca1-648b-4831-968f-5951eb024477
t₁₁ = combine(groupby(teₐ′₀, :date), [:prediction, :recorded] .=> sum, renamecols=false);

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
m₂, vₐ′₂, teₐ′₂ = electrictrain("NOAA", term₂);

# ╔═╡ 448c4152-ca34-458e-a35d-3a3a569d96ec
t₁₂ = combine(groupby(teₐ′₂, :date), [:prediction, :recorded] .=> sum, renamecols=false);

# ╔═╡ af170bc6-cbd9-4f45-aca8-0a1900dd4ccf
p₁₂ = Gadfly.plot(
	t₁₂,
	Gadfly.layer(
		x=:date,
		y=:prediction,
		Theme(default_color="indianred"),
		Geom.point,
		Geom.line
	),
	Gadfly.layer(
		x=:date,
		y=:recorded,
		Geom.point,
		Geom.line
	),
	Guide.title("Prediction vs Recorded"),
	Guide.ylabel("Electricity MWh")
)

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
m₅, vₐ′₅, teₐ′₅ = electrictrain("Landsat8", term₅);

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

# ╔═╡ 8f3bcc9f-80e4-455a-bea1-c52feea40191
m₇, vₐ′₇, teₐ′₇ = electrictrain("SAR", term₇);

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
term₉ = unique([names(cmip)..., electricity_terms...])

# ╔═╡ 3962b939-ff46-4362-a30a-dda8cf84133d
# m₉, vₐ′₉, teₐ′₉ = electrictrain("CMIP", term₉);

# ╔═╡ 6c09c553-0c77-410a-aae9-2004c9768d8b
md"""
###### CMIP + SAR
"""

# ╔═╡ 4c6dbaa4-0549-4775-b2d8-7c58cdafc24d
cmip_α = rename(
	select(cmip, ["Property Id","date","tasmin_median_f₃","tasmax_median_f₃"]),
	:tasmin_median_f₃ => :tmin,
	:tasmax_median_f₃ => :tmax
)

# ╔═╡ 291d3eb9-df30-4df4-bd9a-dce4603f6fd2
termᵪₛ = unique([names(cmip_α)...,names(sar)...,electricity_terms...])

# ╔═╡ 7fbfb83f-e9e1-4701-a7ec-e60cb79b806c
tₔ = clean(dropmissing(innerjoin(
	tₐ,
	[cmip_α, sar]...,
	on=["Property Id", "date"]
)));

# ╔═╡ ff177d30-ecd8-4935-ba10-dc7ea0b79a06
teₔ = clean(dropmissing(innerjoin(
	teₐ,
	[cmip_α, sar]...,
	on=["Property Id", "date"]
)));

# ╔═╡ 82a444a7-6bdb-4c1a-b9dd-aa85e5056faa
exclusion_terms

# ╔═╡ 8cb63cae-a783-4a86-8a51-ae3064d02a32
begin
function cmip_train()
	mᵥ = training_pipeline(
		select(select(tₔ, termᵪₛ), exclusion_terms),
		"electricity_mwh",
		m
	)
	
	return mᵥ
end
end

# ╔═╡ 2ca59d38-d1ee-4fbd-a0a0-88de89d645b3
mᵥ = cmip_train()

# ╔═╡ 7eca8b57-5d0d-4f12-a883-8115e6a745ce
begin
function cmip_evaluate(testdata, model, modelname)
	teᵥ = select(testdata, termᵪₛ)
	teᵥ.prediction = validation_pipeline(
		select(teᵥ, exclusion_terms),
		"electricity_mwh",
		model
	);
	
	teᵥ.recorded = teᵥ.electricity_mwh
	t = combine(groupby(teᵥ,["date"]),
		[:prediction, :recorded] .=> mean,
		renamecols=false
	)
	t.model = repeat([modelname], nrow(t))
	return select(t, :prediction, :recorded, :)
end
end

# ╔═╡ 793692f3-75f6-4d87-9223-497da0d88d28
cmip_evaluate(teₔ, mᵥ, "testdata")

# ╔═╡ 48e1a2ac-ceb3-4229-b56b-c1e0c2c10075
# mᵪₛ, vₐ′ᵪₛ, teₐ′ᵪₛ = electrictrain("CMIP-SAR", termᵪₛ);

# ╔═╡ bb329d66-c51e-423d-b28b-a15cb38d7ed7
md"""
###### Full Dataset
"""

# ╔═╡ 590138f1-7d6f-4e78-836d-258f7b4f617e
termₑ = names(tₐ′)

# ╔═╡ 1e09be56-ef2a-4ef1-a181-d1f07dcc4ced
mₑ, vₐ′ₑ, teₐ′ₑ = electrictrain("Full Data", termₑ);

# ╔═╡ 7dcd284f-3ed4-47bd-aabf-a91e7f939910
mₑ

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
resultsₑ = vcat([ select(x, common_terms) for x in test_terms ]...);
resultsₑ.diff = resultsₑ.prediction .- resultsₑ.recorded;
end;

# ╔═╡ 19ffec1f-43f7-49af-91be-2553d8998951
begin
resultsₑ_combined = combine(groupby(resultsₑ, ["month","model","zone"]), 
	:diff => mean,
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
	Geom.subplot_grid(Geom.smooth),
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

# ╔═╡ 5b136482-1013-4c51-8e52-cdbf0ab96735


# ╔═╡ f5b41025-e072-4d42-b7aa-0982ddf01982
rmse

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
test_termsₑ = vcat([ select(x, prediction_terms, interest_terms) for x in test_terms ]...);

# ╔═╡ 41751922-2646-45fd-b6e0-7ca2109cc642
function mse(x)
	(sum(x.^2) / length(x))^0.5
end

# ╔═╡ c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
# begin
# # Null
# Ξ₀ = hcat(select(teₐ′₀, prediction_terms), select(teₐ′, interest_terms))

# # ERA5
# Ξ₅ = hcat(select(teₐ′₅, prediction_terms), select(teₐ′, interest_terms))

# #MODIS
# Ξ₃ = hcat(select(teₐ′₃, prediction_terms), select(teₐ′, interest_terms))
	
# #EPW
# Ξ₄ = hcat(select(teₐ′₄, prediction_terms), select(teₐ′, interest_terms))
	
# Ξ = vcat(Ξ₅, Ξ₃, Ξ₄)

# Ξ.error = Ξ.prediction .- Ξ.recorded
# Ξ′ = combine(groupby(Ξ, [:month, :model]), :error => mean, renamecols=false)

# p₄ = Gadfly.plot(
# 	Ξ′,
# 	x=:month,
# 	y=:error,
# 	color=map(x-> seasons[x], Ξ′.month),
# 	xgroup=:model,
# 	yintercept=[0],
# 	Guide.ylabel("Δ Prediction - Recorded"),
# 	Guide.xlabel("Month"),
# 	Guide.title("Mean Electricity Error by Season"),
# 	Guide.colorkey(title="Season"),
# 	Geom.subplot_grid(
# 		# Guide.yticks(ticks=0.0:0.001:0.01),
# 		Guide.xticks(ticks=1:2:12),
# 		Geom.point,
# 		Geom.line,
# 		Geom.hline(color=["pink"], style=:dash),
# 		# Geom.point,
# 		# free_y_axis=true,
# 	),	# Coord.cartesian(ymin=-0.1, ymax=0.1, aspect_ratio=1.5),
# 	# Theme(default_color="black")
# )
# end

# ╔═╡ 11277ca6-3bbf-403f-8cfc-2aabf94069f7
p₇ = Gadfly.plot(
	test_termsₑ,
	color=:model,
	x=:weather_station_distance,
	y=test_termsₑ.prediction .- test_termsₑ.recorded,
	xgroup=:zone,
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Electricity - MAE vs Weather Station Distance"),
	# Scale.y_log,
	Geom.subplot_grid(
		Geom.smooth(smoothing=0.7),
		# Guide.yticks(ticks=-0.15:0.1:0.15),
		# Coord.cartesian(ymin=-0.2, ymax=0.2),
	),
)

# ╔═╡ b0f7e467-684b-41dd-b046-ed378dded683
resₑ = filter( x -> x.zone == "Residential", test_termsₑ );

# ╔═╡ 38b3911a-8db5-47f4-bd1b-0d77a207188f
Gadfly.plot(
	resₑ,
	x=:weather_station_distance,
	y=resₑ.prediction .- resₑ.recorded,
	ygroup=:zone,
	color=:model,
	Geom.subplot_grid(Geom.smooth(smoothing=0.6)),
	Guide.xlabel("Weather Station Distance (m)"),
	Guide.title("Electricity Degredation with Distance")
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

# ╔═╡ fa72b6f2-de9b-430c-9ab6-f799157f1570
validate

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
function gastrain(modelname::String, terms::Vector{String})
	tᵥ = select(tᵧ′, terms)
	mᵥ = training_pipeline(
		select(tᵥ, exclusion_terms),
		"naturalgas_mwh",
		m
	)

	# validation
	vᵥ = select(vᵧ′, terms)
	vᵥ.prediction = validation_pipeline(
		select(vᵥ, exclusion_terms),
		"naturalgas_mwh",
		mᵥ
	);
	
	vᵥ.recorded = vᵥ.naturalgas_mwh
	vᵥ.model = repeat([modelname], nrow(vᵥ))
	
	# test
	teᵥ = select(teᵧ′, terms)
	teᵥ.prediction = validation_pipeline(
		select(teᵥ, exclusion_terms),
		"naturalgas_mwh",
		mᵥ
	);
	
	teᵥ.recorded = teᵥ.naturalgas_mwh
	teᵥ.model = repeat([modelname], nrow(teᵥ))

	leftjoin!(
		teᵥ,
		building_zones,
		on="Property Id"
	)

	return mᵥ, vᵥ, teᵥ
end

# ╔═╡ 2426e2d6-e364-4e97-bee8-7defb1e88745
md"""
##### Null
"""

# ╔═╡ 9c8f603f-33c6-4988-9efd-83864e871907
termᵧ₀ = unique([naturalgas_terms...])

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

# ╔═╡ a9a9ea9d-5a97-4260-8e02-a27732928e61
mᵧ₂, vᵧ′₂, teᵧ′₂ = gastrain("NOAA", termᵧ₂);

# ╔═╡ 5ed2db5f-c499-4749-87f2-4bb881accf16
begin
noaaᵧₜ = select(teᵧ′₂, ["Property Id","date","prediction","recorded"])
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

# ╔═╡ 399d45a5-a217-413a-8ba8-76b93e246a89
maximum(teᵧ′.date)

# ╔═╡ 5b852b2d-d86e-401b-8251-b21fa080ed1f
begin
epwᵧₜ = select(teᵧ′₄, ["Property Id","date","prediction","recorded"])
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
termᵧ₉ = unique([names(cmip)..., naturalgas_terms...])

# ╔═╡ 73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
# mᵧ₉, vᵧ′₉, teᵧ′₉ = gastrain("CMIP", termᵧ₉);

# ╔═╡ 26086070-08a3-4ce8-b8e6-cd2cfc83e44d
md"""
##### Full Data
"""

# ╔═╡ 1af87fef-8c88-4e34-b1d4-8bccd7881473
# mᵧₑ, vᵧ′ₑ, teᵧ′ₑ = gastrain("Full Data", names(tᵧ′));

# ╔═╡ da9a6ba8-083b-40b3-9bd6-089688ff7f73
# begin
# 	# now want to explore how permuted weather data might influence the quality
# 	termᵧₒ = select(tᵧ′₄, naturalgas_terms)
# 	epwᵧₒ = select(tᵧ′₄, Not(naturalgas_terms))
# 	epwᵧₒ′ = epwᵧₒ[shuffle(1:nrow(epwᵧₒ)), :]

# 	tᵧₒ′₄ = hcat(termᵧₒ, epwᵧₒ′)


# 	termᵥᵧₒ = select(vᵧ′₄, naturalgas_terms)
# 	epwᵥᵧₒ = select(vᵧ′₄, Not(naturalgas_terms))
# 	epwᵥᵧₒ′ = epwᵥᵧₒ[shuffle(1:nrow(epwᵥᵧₒ)), :]

# 	vᵧₒ′₄ = hcat(termᵥᵧₒ, epwᵥᵧₒ′)
# end;

# ╔═╡ 433d51c9-9f31-49db-a5a2-e6565888831e


# ╔═╡ c25b404b-0d89-4c4d-bc61-c361fd9d7038
test_termsᵧ = [teᵧ′₀,teᵧ′₂,teᵧ′₄,teᵧ′₅,teᵧ′₆,teᵧ′₇,teᵧ′₈];

# ╔═╡ c15b52c1-5d06-4fea-8645-91f92bfa716b
begin
resultsᵧ = vcat([ select(x, common_terms) for x in test_termsᵧ ]...);
resultsᵧ.diff = resultsᵧ.prediction .- resultsᵧ.recorded;
end;

# ╔═╡ 07c78fb1-bcf4-40fd-9209-d3220d014912
begin
resultsᵧ_combined = combine(groupby(resultsᵧ, ["month","model","zone"]), 
	:diff => mean,
	renamecols=false
)
resultsᵧ_combined.fuel = repeat(["Natural Gas"], nrow(resultsᵧ_combined))
end;

# ╔═╡ 15b3535f-6776-409d-8e6a-457b4c6eba78


# ╔═╡ d96c7a96-0875-4ebd-a1ba-e40db8008aed


# ╔═╡ 43a39165-1f9b-49e7-a4bc-c47583c25b10
interest_terms

# ╔═╡ f2067017-32c8-493b-a9fb-3a89f9e549e4
test_termsₑᵧ = vcat(
	[ select(x, prediction_terms, interest_terms, "Property Id", "area") for x in test_termsᵧ]...
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
building_distancesᵧ = unique(select(teᵧ, ["Property Id", "weather_station_distance"]));

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

# ╔═╡ 459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
pᵧ₁ = Gadfly.plot(
	test_termsₑᵧ,
	Gadfly.layer(
		color=:model,
		x=:weather_station_distance,
		y=abs.(test_termsₑᵧ.prediction .- test_termsₑᵧ.recorded),
		Geom.smooth(smoothing=0.9),
	),
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Natural Gas - MAE vs Weather Station Distance"),
	Scale.y_log
)

# ╔═╡ ae386c52-ff6b-4493-96ff-6c14d1c46db8
draw(PNG(joinpath(output_dir, "naturalgas-mae.png"), 14cm, 10cm, dpi=600), pᵧ₁)

# ╔═╡ 94580726-88ed-4a69-a101-9c792e3ed44b


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
	select(
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

# ╔═╡ 051fc252-5267-4c8b-8ead-75f37a49a440
seaonsal_predictionresults = Gadfly.plot(
	seasonal_results_norm,
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

# ╔═╡ 96dd4c14-6791-4c8c-80c1-8f0301e34b35
sort(filter( 
	x -> x.fuel == "Electricity" && x.zone == "Residential" && x.model == "SAR"
	, seasonal_results_norm ), [:month])

# ╔═╡ 4df27234-b5b1-4607-b955-c2710257ffd6
draw(PNG(joinpath(output_dir, "seasonal-predictionresults.png"), 
	20cm,
	15cm, 
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

# ╔═╡ f32949e9-d644-4cbc-aed6-74a5e0825664
cmip19 = filter( row -> Dates.year(row.date) == 2020, cmip );

# ╔═╡ b8764c1b-4fc7-4efc-b706-3daad124dfde
cmip19

# ╔═╡ b2d422e0-d2b4-4b60-8e79-a9b4754f7e27
function build_cmip(cmipdata, minterm, maxterm)
	c = rename(
		select(
			cmipdata, 
			[
				Symbol("Property Id"),
				Symbol("date"),
				minterm,
				maxterm
			]),
		minterm => :tmin,
		maxterm => :tmax
	)

	t = clean(dropmissing(innerjoin(
		vcat(tₐ, teₐ),
		[c, sar]...,
		on=["Property Id", "date"]
	)))
	return c, t
end

# ╔═╡ 77d37af6-212e-4d8b-9e29-c4873d10ca78
crows = ["prediction","recorded","date","model"]

# ╔═╡ 08844605-1dd5-42f6-ac05-bcb57a1ff985
cmip_19, t19 = build_cmip(
	filter( row -> Dates.year(row.date) == 2019, cmip ),
	:tasmin_mean_f₃,
	:tasmax_mean_f₃
);

# ╔═╡ 38672535-b7ca-46e7-afc0-4d8aa2200c95
c19 = select(cmip_evaluate(t19, mᵥ, "cmip19"), crows);

# ╔═╡ 68d4b11c-2c00-4de1-9be8-099128ee0629
function deviation(old_cmip, new_cmip)
	new_cmip.deviation = 100 .* (new_cmip.prediction .- old_cmip.prediction) ./ old_cmip.prediction
	return new_cmip
end

# ╔═╡ e8fbf241-8eda-4723-a6b3-f76a9fadc487
cmip_30, t30 = build_cmip(
	cmip30,
	:tasmin_mean_f₃,
	:tasmax_mean_f₃
);

# ╔═╡ fcc6904a-af26-44eb-9e03-2d6ba83bb111
names(cmip30)

# ╔═╡ 9a279e3b-a26d-4b16-bcee-4a712f81bdd6
cmip_30ₗ, t30ₗ = build_cmip(
	cmip30,
	:tasmin_quartile25_f₃,
	:tasmax_quartile25_f₃
);

# ╔═╡ ce8ce8f4-7e5e-4a07-b434-74933a48d112
cmip_30ᵤ, t30ᵤ = build_cmip(
	cmip30,
	:tasmin_quartile75_f₃,
	:tasmax_quartile75_f₃
);

# ╔═╡ c3825758-c3d3-4d2e-ac20-37acd7d3ff16
c30 = deviation(c19, select(cmip_evaluate(t30, mᵥ, "2030"), crows));

# ╔═╡ e2bcd8f3-75fc-4e02-9d4a-eba3a9d01b30
c30ₗ = rename(
	deviation(c19, select(cmip_evaluate(t30ₗ, mᵥ, "2030"), crows)),
	"deviation" => "lower"
);

# ╔═╡ 06903f32-5069-42df-8ca3-4cac0011bd64
c30ᵤ = rename(
	deviation(c19, select(cmip_evaluate(t30ᵤ, mᵥ, "2030"), crows)),
	"deviation" => "upper"
);

# ╔═╡ 35eb8005-c75b-462a-aa7b-b1322ca55664
c30ₐ = innerjoin(
	select(c30, ["date","model","deviation"]),
	select(c30ₗ, ["date","model","lower"]),
	select(c30ᵤ, ["date","model","upper"]),
	on=["model","date"]
)

# ╔═╡ 99acd1ef-17ef-47ca-84c3-0c7956e1c27d
cmip_40, t40 = build_cmip(
	cmip40,
	:tasmin_mean_f₃,
	:tasmax_mean_f₃
);

# ╔═╡ 5617fff4-d3a6-4852-832e-fad36c8c1ec1
c40 = deviation(c19, select(cmip_evaluate(t40, mᵥ, "2040"), crows));

# ╔═╡ 9da17695-6f7a-4a37-90ef-34d6adedbd8c
cmip_40ₗ, t40ₗ = build_cmip(
	cmip40,
	:tasmin_quartile25_f₃,
	:tasmax_quartile25_f₃
);

# ╔═╡ 8baea97a-f9a3-43a3-be07-6f0ffda6e60a
cmip_40ᵤ, t40ᵤ = build_cmip(
	cmip40,
	:tasmin_quartile75_f₃,
	:tasmax_quartile75_f₃
);

# ╔═╡ 3b00cc15-14fd-471f-b48c-8db92496684e
c40ₗ = rename(
	deviation(c19, select(cmip_evaluate(t40ₗ, mᵥ, "2040"), crows)),
	"deviation" => "lower"
);

# ╔═╡ 6c4d5d37-4cc8-4bc4-9842-d18c0ce7ec4f
c40ᵤ = rename(
	deviation(c19, select(cmip_evaluate(t40ᵤ, mᵥ, "2040"), crows)),
	"deviation" => "upper"
);

# ╔═╡ 3612d185-8a7c-409f-952b-b17532a484dd
c40ₐ = innerjoin(
	select(c40, ["date","model","deviation"]),
	select(c40ₗ, ["date","model","lower"]),
	select(c40ᵤ, ["date","model","upper"]),
	on=["model","date"]
);

# ╔═╡ 479e8b29-ab02-4075-9711-599ded8807b3


# ╔═╡ bfaf76d3-562f-498b-a031-fa07f459c4b4
cmip_50, t50 = build_cmip(
	cmip50,
	:tasmin_mean_f₃,
	:tasmax_mean_f₃
);

# ╔═╡ cb292cfb-32c8-441f-ab7e-5cc908234d17
c50 = deviation(c19, select(cmip_evaluate(t50, mᵥ, "2050"), crows));

# ╔═╡ b2c52c2e-9972-4a52-9345-b4c768b76494
cmip_50ₗ, t50ₗ = build_cmip(
	cmip50,
	:tasmin_quartile25_f₃,
	:tasmax_quartile25_f₃
);

# ╔═╡ c1fbea75-d24f-4049-b9be-93d1d4f391c9
cmip_50ᵤ, t50ᵤ = build_cmip(
	cmip50,
	:tasmin_quartile75_f₃,
	:tasmax_quartile75_f₃
);

# ╔═╡ 4130dfc2-56bc-41ef-83e1-ad9a0f2ff1fe
c50ₗ = rename(
	deviation(c19, select(cmip_evaluate(t50ₗ, mᵥ, "2050"), crows)),
	"deviation" => "lower"
);

# ╔═╡ 648a99e0-6ee6-4489-a5bb-be1fa853fb23
c50ᵤ = rename(
	deviation(c19, select(cmip_evaluate(t50ᵤ, mᵥ, "2050"), crows)),
	"deviation" => "upper"
);

# ╔═╡ 300b043c-e73a-41ae-b041-707e5365f597
c50ₐ = innerjoin(
	select(c50, ["date","model","deviation"]),
	select(c50ₗ, ["date","model","lower"]),
	select(c50ᵤ, ["date","model","upper"]),
	on=["model","date"]
);

# ╔═╡ 1ddfcf7e-2958-4f5c-b43e-6f8393b5bdbd


# ╔═╡ b2cc7582-8878-4ac1-a12b-79117c0dad93
cmip_60, t60 = build_cmip(
	cmip60,
	:tasmin_mean_f₃,
	:tasmax_mean_f₃
);

# ╔═╡ 604d254b-3d86-48bb-a22f-5d56771afdb4
c60 = deviation(c19, select(cmip_evaluate(t60, mᵥ, "2060"), crows));

# ╔═╡ ad41ec1e-c21c-46aa-8da1-e5bbf6fedcb5
cmip_60ₗ, t60ₗ = build_cmip(
	cmip60,
	:tasmin_quartile25_f₃,
	:tasmax_quartile25_f₃
);

# ╔═╡ cde6098d-9ec2-47e1-ad7f-02b910caa196
cmip_60ᵤ, t60ᵤ = build_cmip(
	cmip60,
	:tasmin_quartile75_f₃,
	:tasmax_quartile75_f₃
);

# ╔═╡ c01746c5-391a-497c-b9d3-10ddd2b1f7f5
c60ₗ = rename(
	deviation(c19, select(cmip_evaluate(t60ₗ, mᵥ, "2060"), crows)),
	"deviation" => "lower"
);

# ╔═╡ 261a1d4e-adbf-4fdc-b3cc-b7e3ab8a1974
c60ᵤ = rename(
	deviation(c19, select(cmip_evaluate(t60ᵤ, mᵥ, "2060"), crows)),
	"deviation" => "upper"
);

# ╔═╡ 53571901-8805-4c9c-b222-274c6a4cfa4f
c60ₐ = innerjoin(
	select(c60, ["date","model","deviation"]),
	select(c60ₗ, ["date","model","lower"]),
	select(c60ᵤ, ["date","model","upper"]),
	on=["model","date"]
);

# ╔═╡ 053a7bc6-bf65-48be-bca5-f0ff615abc40


# ╔═╡ a5841fe0-3f2d-4c86-976c-69eff2fc04b5


# ╔═╡ 0f0a2e2e-599d-45cd-b997-0fd7f77f68e1
begin
cmip_forecast = vcat(c30ₐ,c40ₐ,c50ₐ,c60ₐ)
end

# ╔═╡ e71a83ea-2035-488d-8639-9837e79d1c30
begin
cmip_forecast_β = combine(
	groupby(cmip_forecast, ["date","model"]),
	[:deviation,:lower,:upper] .=> median,
	renamecols=false
)
cmip_forecast_β.month = Dates.month.(cmip_forecast_β.date)
end;

# ╔═╡ 861d9d1d-e542-4de2-a4df-8ce3771dad64
Gadfly.Scale.DiscreteColorScale

# ╔═╡ 32715449-730c-4201-9c61-96b68ba6635f
typeof(palette(:matter, 5)[1])

# ╔═╡ 5b29638f-47fd-4b3a-bc51-cd35bcecf88e
theme_colors = palette(:viridis, 5)[2:5];

# ╔═╡ c5575618-9027-484f-85c0-aa239f239dbe
electricity_forecast = Gadfly.plot(
	cmip_forecast_β,
	x=:month,
	y=:deviation,
	ymin=:lower,
	ymax=:upper,
	color=:model,
	Geom.point,
	Geom.ribbon,
	Guide.xticks(ticks=1:12),
	# Guide.yticks(ticks=-4:2:10),
	Guide.title("NYC Building Electricity Predictions under RCP60"),
	Guide.xlabel("Month"),
	Guide.ylabel("Mean Electricity Change (%) - MWh"),
	Scale.color_discrete_manual(theme_colors...),
	Theme(alphas=[0.5])
)

# ╔═╡ 95d01f67-5b46-46c9-9403-53070829571e
draw(
	PNG(
		joinpath(output_dir, "electricity_forecast.png"), 
		13cm, 
		9cm,
		dpi=500
	), electricity_forecast
)

# ╔═╡ cb6a77d5-cfdf-4a04-bafc-c5051568ff1d


# ╔═╡ 05346d5e-081e-4139-97cb-99283ba1342b
c60

# ╔═╡ 0409ff93-8ece-4cef-a44d-1c36ea4a9c5f
t19

# ╔═╡ ddb913c3-e62c-48f8-8b65-4b281bdd33ab
begin
dataₑ¹ = dropmissing(clean(innerjoin(
	dataₑ,
	[cmip19, sar]...,
	on=joining_terms
)));

dataₑ¹.prediction = validation_pipeline(
	select(dataₑ¹, exclusion_terms),
	"electricity_mwh",
	mᵪₛ
)
end

# ╔═╡ f07cf5b3-acc0-40ef-b997-76e4d5313aec
## now want to make custom dates for the SAR data for matching

# ╔═╡ cc58368b-bbcc-4849-ad3f-bec12ad3e16f
md"""
Climate scenario 60
"""

# ╔═╡ f4a21906-1857-418c-ba74-1d7d4f11509b
begin
dataₑ⁶ = dropmissing(clean(innerjoin(
	dataₑ,
	[cmip60, sar]...,
	on=joining_terms
)));
dataₑ⁶.prediction = validation_pipeline(
	select(dataₑ⁶, exclusion_terms),
	"electricity_mwh",
	mᵪₛ
)
end

# ╔═╡ bd742e20-2fe6-40f6-9d0e-9922a907e7d3
agg_func = sum

# ╔═╡ 38c6250b-1c74-4c6a-be38-7612bd2cd06b
d¹ = combine(groupby(dataₑ¹, :month), :prediction => agg_func => :p₁);

# ╔═╡ ed2dccce-905e-4503-8c70-5cecd9d80734
d² = combine(groupby(dataₑ⁶, :month), :prediction => agg_func => :p₂);

# ╔═╡ 83ed5fe5-5f0e-4b0e-9ac4-f82e06940726
begin
pₛ = leftjoin(d¹, d², on=:month, makeunique=true)
pₛ.diff = 100 .* ( pₛ.p₂ .- pₛ.p₁ ) ./ abs.( pₛ.p₁ )
end

# ╔═╡ 7b7cf78f-efb0-44b7-ae43-b413b79eb9d2
Gadfly.plot(
	pₛ,
	x=:month,
	y=:diff,
	# Guide.yticks(ticks=0.0:0.3:2.5),
	Geom.point,
	Geom.line
)

# ╔═╡ b810a51c-7372-40df-a613-4112b53e547b
agg_term = :tasmax_mean_f₃

# ╔═╡ 64b19240-6e2a-4bb9-b681-5d4fdcc98935
begin
tₚ¹ = select(select(dataₑ¹, names(cmip19)), Not("Property Id"));
ʒ¹ = combine(groupby(tₚ¹, :date), names(tₚ¹, Union{Real, Missing}) .=> mean, renamecols=false);
end;

# ╔═╡ fbdc69a6-7227-4df8-a9a9-eca836565a11
tₚ⁶ = select(select(dataₑ⁶, names(cmip60)), Not("Property Id"));

# ╔═╡ 92bd44dc-c382-4835-b6ed-351756703389
ʒ⁶ = combine(groupby(tₚ⁶, :date), names(tₚ⁶, Union{Real, Missing}) .=> mean, renamecols=false);

# ╔═╡ 95f04c20-6adb-4122-9d4a-ccda1663fc83
Gadfly.plot(
	Gadfly.layer(
		ʒ¹,
		x=:date,
		y=:tasmin_mean_f₃,
		ymin=:tasmin_quartile25_f₃,
		ymax=:tasmin_quartile75_f₃,
		Geom.point,
		Geom.line,
		Geom.errorbar
	),
	Gadfly.layer(
		ʒ⁶,
		x=:date,
		y=:tasmin_mean_f₃,
		ymin=:tasmin_quartile25_f₃,
		ymax=:tasmin_quartile75_f₃,
		Geom.point,
		Geom.line,
		Geom.errorbar,
		Theme(default_color="indianred")
	)
)

# ╔═╡ b8c478a1-c908-423f-bb30-c54d0d5a2f8e
begin
t¹ = combine(groupby(dataₑ¹, :month), agg_term => mean => :t₁)
t² = combine(groupby(dataₑ⁶, :month), agg_term => mean => :t₂)

tₛ = leftjoin(t¹, t², on=:month, makeunique=true)
tₛ.diff = 100 .* ( tₛ.t₂ .- tₛ.t₁ ) ./ abs.( tₛ.t₁ )
end

# ╔═╡ 5591c99f-fccf-42cc-a8ce-db7a4b7ddf08
Gadfly.plot(
	tₛ,
	x=:month,
	y=:diff,
	Geom.point,
	Geom.line,
	Theme(default_color="indianred")
)

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

# lst_epw_tempdiff = select(
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
# 	train_electric = select(train, Not(:naturalgas_mwh))
# 	validate_electric = select(validate, Not(:naturalgas_mwh))
# 	test_electric = select(test, Not(:naturalgas_mwh))

# 	dropmissing!(train_electric, :electricity_mwh)
# 	dropmissing!(validate_electric, :electricity_mwh)
# 	dropmissing!(test_electric, :electricity_mwh)
# end;

# ╔═╡ f825b6a1-0950-461b-80a8-b0fc9924dd53
# train_electric

# ╔═╡ 8806127c-8bbd-4567-b32e-946fb473b4c7
# begin
# 	train_env_electric_aux = select(train_electric, env_dropping_cols)
# 	validate_env_electric_aux = select(validate_electric_env, env_dropping_cols)
# 	test_env_electric_aux = select(test_electric_env, env_dropping_cols)

# 	select!(train_electric_env, Not(env_dropping_cols))
# 	select!(validate_electric_env, Not(env_dropping_cols))

# 	train_xe = select(train_electric_env, Not(:electricity_mwh))
# 	train_ye = train_electric_env.electricity_mwh

# 	validate_xe = select(validate_electric_env, Not(:electricity_mwh))
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
function test_suite(Ẋ::DataFrame, selectors = ["Property Id", "zone"], report_model = true)
	# Ẋ = hcat(X, auxiliary)	
	vᵧ = groupby(Ẋ, selectors)
	results = combine(vᵧ) do vᵢ
	(
	cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
	nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
	cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
	rmse = rmse(vᵢ.prediction, vᵢ.recorded),
	)
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
	select(filter(x -> x.model == "EPW", modelresultsₑ), [:zone,:rmse]),
	:rmse => :baseline
)

# ╔═╡ 41e55b47-b527-491c-ab94-ecc34a9b64bb
model_baselinesₑ = leftjoin(
	modelresultsₑ,
	baselinesₑ,
	on="zone"
);

# ╔═╡ 342a8606-708b-4b1c-9023-050c81ce345d
model_baselinesₑ.percent_improvement = 100 .* (1 .- model_baselinesₑ.rmse ./ model_baselinesₑ.baseline );

# ╔═╡ 40d7e39c-98c6-4cf8-8461-e54b7a0db696
sort(filter(x -> x.zone == "Residential", model_baselinesₑ), :rmse);

# ╔═╡ 5fb5d0c3-46ab-41dc-be49-64d83cc845f8
model_baselinesₑ.power = repeat(["Electricity"], nrow(model_baselinesₑ));

# ╔═╡ b0035cc3-0806-41ce-8686-faeab40ee41f
Gadfly.plot(
	model_baselinesₑ, 
	x=:zone, 
	y=:model,
	color=:percent_improvement,
	Geom.rectbin
);

# ╔═╡ cf39d6f0-b592-4a4d-aa17-182f9c794959
describe(test_suite(filter(
	x -> x.recorded > 0.01, teᵧ′₀), ["Property Id"]))[2:end,:]

# ╔═╡ 734ed2d0-4657-4361-9285-97605371af72
test_suiteₜᵧ = vcat([ test_suite(x) for x in test_termsᵧ ]...);

# ╔═╡ 42360e3a-afae-4fd0-a919-d9f1668865ce
test_suiteₜᵧ

# ╔═╡ d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
modelresultsᵧ = combine(groupby(clean(test_suiteₜᵧ), ["zone","model"]), [:rmse, :cvrmse,:nmbe,:cvstd] .=> mean, renamecols=false)

# ╔═╡ 31a814af-95f1-4d7f-ada0-8bf9df2aee10
baselinesᵧ = rename(
	select(filter(x -> x.model == "EPW", modelresultsᵧ), [:zone,:rmse]),
	:rmse => :baseline
)

# ╔═╡ b9516f6f-bb93-47b3-8669-a4e8adf3db3a
model_baselinesᵧ = leftjoin(
	modelresultsᵧ,
	baselinesᵧ,
	on="zone"
);

# ╔═╡ 9d6a3995-7df0-4552-887e-b306d26a80ad
model_baselinesᵧ.percent_improvement = 100 .* (1 .- (model_baselinesᵧ.rmse ./ model_baselinesᵧ.baseline));

# ╔═╡ 402e380d-a91c-45f7-9fbb-5a2c44e754f7
model_baselinesᵧ.power = repeat(["Natural Gas"], nrow(model_baselinesᵧ));

# ╔═╡ 167d31ad-76f2-432d-8550-ba015cb18e9d
begin
results = vcat(model_baselinesₑ, model_baselinesᵧ);
results.percent_improvement[results.percent_improvement .< 0] .= 0
end;

# ╔═╡ e38eee70-b9b0-4dca-ae86-697fc5738cfa
model_results = Gadfly.plot(
	results, 
	x=:zone, 
	y=:model,
	color=:percent_improvement,
	xgroup=:power,
	Geom.subplot_grid(Geom.rectbin),
	Guide.xlabel(""),
	Guide.ylabel(""),
	Theme(bar_spacing=0pt),
	Scale.ContinuousColorScale(p -> get(ColorSchemes.Reds, p)),
	Guide.title("Model Comparison to Baseline (Null)")
)

# ╔═╡ a07ac65e-3e19-411e-9967-af9778d8a45d
draw(PNG(joinpath(output_dir, "model_results.png"), 20cm, 10cm, dpi=600), model_results)

# ╔═╡ 55f1323b-291e-486d-b3a8-6409db9f7b8e
smooth_seaonsal_predictionresults = Gadfly.plot(
	results,
	x=:power, 
	y=:model,
	color=:percent_improvement,
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

# ╔═╡ 8df28d8b-443e-48a3-89e4-a5824d3d66c8
stack(test_suiteₜᵧđ, 2:4)

# ╔═╡ 0e4ed949-b051-4ae5-8b7b-b1c8d6d7fc94
function electricaggregation(individual_results)
	t = innerjoin(
		individual_results,
		testcouncils,
		on="Property Id"
	)
	ṫ = combine(groupby(t, ["date","council_region"]), [:prediction, :recorded] .=> sum, :model => first, renamecols=false)
	
	res = combine(groupby(ṫ, "council_region")) do vᵢ
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
describe(test_suite(teₐ′₀̇, ["council_region"]))[2:end,:]

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
	y=u.prediction .- u.recorded,
	color=:model,
	Geom.point,
	Geom.line,
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
describe(test_suite(teᵧ′₀̇, ["council_region"]))[2:end,:]

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
end

# ╔═╡ Cell order:
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
# ╠═aeb3597f-ef6a-4d21-bff5-d2ad359bc1a2
# ╠═9d5f897f-7f25-43ed-a30c-064f21e50174
# ╠═786c2441-7abb-4caa-9f50-c6078fff0f56
# ╠═1c7bfba6-5e1d-457d-bd92-8ba445353e0b
# ╟─9b3790d3-8d5d-403c-8495-45def2c6f8ba
# ╠═bf772ea4-c9ad-4fe7-9436-9799dcb0ad04
# ╠═020b96e3-d218-470d-b4b0-fc9b708ffdf3
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
# ╠═d4680498-1966-48f7-8a56-296578559d53
# ╠═04349795-8a8d-40b7-a515-cb4806a5776f
# ╠═bf38b421-99ea-48fa-a548-49c9ffd52758
# ╠═c0d3f3b7-60b1-4b73-86cd-2b10b30e3f57
# ╠═8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
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
# ╠═f8783987-88a7-47a5-8f7e-fea72f321e60
# ╠═22494217-9254-4374-8a7d-02528bdd0df3
# ╠═d23500b2-53ba-436c-b26d-187f60821a43
# ╠═dcde8c56-7294-47fd-aad1-2204de6c904b
# ╠═e87d641a-9555-4a93-9fe8-f39f8964ce84
# ╠═ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
# ╠═2f1fec21-76a2-4365-b305-0f24505b1ccc
# ╠═86d465e3-7916-479d-a29c-2b93ae54ab6d
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
# ╠═cbd44ab9-6346-46f0-bb22-1c19b471ccf7
# ╠═3da877c2-159b-4d0d-8b97-34da4dbf2ac3
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
# ╠═c331ec69-2b5c-4ba9-8a7d-7130a27c320f
# ╠═b5f8ce76-008a-491d-9537-f1b8f4943880
# ╠═5ddf5811-9d4c-4375-8ae9-268bcb9e7ed5
# ╠═36bf3b43-4cf6-4a48-8233-1f49b9f7fac3
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
# ╠═76e4aac5-be47-49e3-8359-3d2c9f24500b
# ╠═992c77d0-1b6d-4e07-bd3a-f648ef023870
# ╠═70576040-eec4-4b80-8d23-c2da2e65b2d2
# ╠═e6ea0a65-b79f-400f-be61-951b08b5ce88
# ╠═77aad1ea-16cf-4c5f-9a6b-345f7168afb3
# ╠═fa79e50c-5d5e-47a1-b2e7-fe3341ecbf4f
# ╠═8ef3db5b-f7cd-4d09-8817-35dbf96629d8
# ╠═055a08c0-abc9-465b-bb64-24744e87da5d
# ╠═12f035a7-73a9-4d3b-9004-f223600d5059
# ╠═07fac431-8103-451f-b4f2-9e000bdbc418
# ╠═59ee8ba9-97a1-4a3b-99a4-cddc0a667a3e
# ╟─932e6c04-d0e5-4517-b80a-2e468def88d6
# ╟─4b37d389-bc6d-4d02-83f3-4d5879646c16
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
# ╠═ea1a797c-da0a-4133-bde5-366607964754
# ╠═e5f7e860-4802-42a1-822d-51fcc983dfae
# ╠═019f1c79-440c-42aa-9483-d64389876336
# ╟─8edfe547-f860-418f-8c68-e8fe0c16162f
# ╠═ecb078da-702d-46f7-846a-4759b3e0f172
# ╠═6d5c77c8-2596-45eb-8163-c8e14249948d
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
# ╟─eea75399-d25c-46b6-a471-2cb8676e4db7
# ╟─e38eee70-b9b0-4dca-ae86-697fc5738cfa
# ╟─a07ac65e-3e19-411e-9967-af9778d8a45d
# ╟─efbe759f-ec6c-46a3-b4aa-8fa051f31151
# ╠═7dcd284f-3ed4-47bd-aabf-a91e7f939910
# ╠═a8d738bb-0424-4df0-aa22-2a299fd994b1
# ╟─dc8fb980-6500-4849-b498-b39454dd3ffa
# ╟─bb078875-611c-46f6-8631-1befde358054
# ╟─df753575-b121-4c3b-a456-cfe4e535c2aa
# ╠═a157969b-100a-4794-a78e-2f40439e28d9
# ╠═9239f8dc-fd07-435f-9d88-e24bb1f6faa2
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
# ╟─29672872-98e2-454f-b5c2-49966f11e15f
# ╟─61705c8f-a3e2-4d4b-bd52-a334a0a9f5bd
# ╟─0e4ed949-b051-4ae5-8b7b-b1c8d6d7fc94
# ╟─3c41548c-da78-49c5-89a3-455df77bf4fa
# ╠═a149faa9-bee4-42a0-93fe-5adff459e0e9
# ╠═748321c3-0956-4439-90d4-e74598d83f20
# ╠═dbae822c-24fa-4767-aaaa-d6bd8ad700ac
# ╠═808adca1-648b-4831-968f-5951eb024477
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
# ╟─af170bc6-cbd9-4f45-aca8-0a1900dd4ccf
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
# ╠═2ca59d38-d1ee-4fbd-a0a0-88de89d645b3
# ╠═8cb63cae-a783-4a86-8a51-ae3064d02a32
# ╠═7eca8b57-5d0d-4f12-a883-8115e6a745ce
# ╠═793692f3-75f6-4d87-9223-497da0d88d28
# ╠═48e1a2ac-ceb3-4229-b56b-c1e0c2c10075
# ╠═bb329d66-c51e-423d-b28b-a15cb38d7ed7
# ╠═590138f1-7d6f-4e78-836d-258f7b4f617e
# ╠═1e09be56-ef2a-4ef1-a181-d1f07dcc4ced
# ╠═685169a0-e65c-439c-a5ba-068c83258200
# ╠═d0c234cc-9620-4c9c-bfae-f32b68b9d31f
# ╠═ae507fc7-5a13-42ed-a477-79d0b11c2efb
# ╠═19ffec1f-43f7-49af-91be-2553d8998951
# ╟─0773b0fe-5a8a-4079-9e14-1ea5520a4bdb
# ╠═fe9f7509-2ba7-45f2-bcf5-84312660d754
# ╠═2be996a7-d960-4939-91c7-9f1a42d35b49
# ╠═a282de0c-aa3b-494a-84a2-9d37e20d1adc
# ╠═5b136482-1013-4c51-8e52-cdbf0ab96735
# ╠═92e7c0a0-19df-43e2-90db-4e050ba90c5d
# ╠═f5b41025-e072-4d42-b7aa-0982ddf01982
# ╠═a0cd5e68-3bd0-4983-809e-ce7acd36a048
# ╠═ed696113-737b-46f5-bfb1-c06d430a83ac
# ╠═8b3175d4-6938-4823-8a91-77cde7c31c2a
# ╠═9d1ea09f-3c42-437f-afbc-21a70296a3ba
# ╠═05a040cf-f6ac-42b2-bcb1-599af5a26038
# ╠═41751922-2646-45fd-b6e0-7ca2109cc642
# ╠═c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
# ╠═11277ca6-3bbf-403f-8cfc-2aabf94069f7
# ╠═b0f7e467-684b-41dd-b046-ed378dded683
# ╠═38b3911a-8db5-47f4-bd1b-0d77a207188f
# ╠═28d2660e-258d-4583-8224-c2b4190f4140
# ╠═fbe43a2a-f9b0-487e-a579-645c2f40736e
# ╠═7b05ec74-6edc-4f44-bf9d-9389d4029494
# ╟─c546d3af-8f24-40bd-abfa-e06c708c244e
# ╟─5db964da-1ebc-478a-a63a-cf4f713c7aa8
# ╠═fa72b6f2-de9b-430c-9ab6-f799157f1570
# ╠═778dc2a7-4e9b-4f97-b34d-6a7adc38abc2
# ╠═5953c75a-0e6d-40c4-8e4b-b8de6920acf3
# ╠═c7874fdb-2ba4-4dbf-89b4-0b3af493b256
# ╠═23525c22-3fc4-4a4e-b2c6-601463a5df31
# ╟─b98baad1-dad8-464c-a18a-1baf01962164
# ╟─3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
# ╠═d437be5f-ec26-466b-b099-5e1cc8816cb5
# ╟─2426e2d6-e364-4e97-bee8-7defb1e88745
# ╠═9c8f603f-33c6-4988-9efd-83864e871907
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
# ╠═a9a9ea9d-5a97-4260-8e02-a27732928e61
# ╠═5ed2db5f-c499-4749-87f2-4bb881accf16
# ╠═c4076bd0-8718-4641-9d09-2c3b71aff1e3
# ╟─202c9072-0a7b-454d-9112-4ecc0a03c61b
# ╠═434d7e0d-583d-498b-9b6c-a72fa3775b3c
# ╠═2857461c-d24e-4b04-91f1-80cd842eeaa4
# ╟─a78a6fd5-4b45-43fc-a733-aef4fd14eb42
# ╠═1a4471e7-24ad-4652-9f3f-6eef92c781d5
# ╠═8c074d96-ee63-45fb-bbbf-135c40b66a09
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
# ╠═d96c7a96-0875-4ebd-a1ba-e40db8008aed
# ╠═734ed2d0-4657-4361-9285-97605371af72
# ╠═42360e3a-afae-4fd0-a919-d9f1668865ce
# ╠═d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
# ╠═43a39165-1f9b-49e7-a4bc-c47583c25b10
# ╠═f2067017-32c8-493b-a9fb-3a89f9e549e4
# ╠═ee0d413c-a107-4bdc-a08a-52cda6d81573
# ╠═77456db6-b0fa-4ba6-9d41-1c393f7ddee1
# ╠═08b888e3-9d37-43a6-9656-4bf6cb62d324
# ╠═bf71086c-a8f0-4420-b698-73499fec5257
# ╟─8df28d8b-443e-48a3-89e4-a5824d3d66c8
# ╠═4c99c655-ae95-4a28-95c8-e7ca38ddf55f
# ╠═459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
# ╠═ae386c52-ff6b-4493-96ff-6c14d1c46db8
# ╠═94580726-88ed-4a69-a101-9c792e3ed44b
# ╠═6f78dd2a-c225-41c7-b6a9-5f0925f87d14
# ╠═ea5f7773-0697-4ead-adc9-0eabbd0fa05e
# ╠═b2bf0ddc-a327-4b13-9722-130a0ddebffb
# ╠═3ec1c94b-ac88-4f01-af9d-abb956dff6f1
# ╟─3b0a8743-20b6-4533-a598-09a2c94d6528
# ╠═62b6cf95-266b-46d4-b478-2e11cdf1acb0
# ╠═ec90e969-6183-4e98-b2ab-3a53eee7b13b
# ╟─f54d3095-8e78-4f36-8f81-a022c6c501d6
# ╠═051fc252-5267-4c8b-8ead-75f37a49a440
# ╟─96dd4c14-6791-4c8c-80c1-8f0301e34b35
# ╠═4df27234-b5b1-4607-b955-c2710257ffd6
# ╠═0691bffd-6b86-41ce-8044-5474545f9417
# ╠═55f1323b-291e-486d-b3a8-6409db9f7b8e
# ╠═d15506e2-5213-48fc-bf6a-fbe6e99af3de
# ╠═f75ae159-d811-47dc-8592-73a8048244fb
# ╠═9e2aef0c-0121-4652-ad40-62ad5e8d0a0e
# ╟─329d55b8-eb72-4a1e-a4e8-200fee0e0b9d
# ╠═b5ce80a3-e177-4c4f-920b-5dee87f2bc3b
# ╠═c92ab000-dbe9-457d-97f3-88ae31b57a27
# ╠═c089c975-96e1-4281-b5ad-c53e738834a1
# ╠═e4344d50-425b-4bea-b28e-0c3b45debfb1
# ╠═f9ccee0a-9e6b-4070-a15c-ff5d5c324649
# ╠═624ab4a3-5c3b-42f3-be37-89d6382fdfdd
# ╟─ec97d987-651d-4efa-a36f-e6be9f18e0fd
# ╟─b1f8755d-8130-4492-859e-781225f8bb45
# ╠═f32949e9-d644-4cbc-aed6-74a5e0825664
# ╠═b8764c1b-4fc7-4efc-b706-3daad124dfde
# ╠═b2d422e0-d2b4-4b60-8e79-a9b4754f7e27
# ╠═77d37af6-212e-4d8b-9e29-c4873d10ca78
# ╠═08844605-1dd5-42f6-ac05-bcb57a1ff985
# ╠═38672535-b7ca-46e7-afc0-4d8aa2200c95
# ╠═68d4b11c-2c00-4de1-9be8-099128ee0629
# ╠═e8fbf241-8eda-4723-a6b3-f76a9fadc487
# ╠═fcc6904a-af26-44eb-9e03-2d6ba83bb111
# ╠═9a279e3b-a26d-4b16-bcee-4a712f81bdd6
# ╠═ce8ce8f4-7e5e-4a07-b434-74933a48d112
# ╠═c3825758-c3d3-4d2e-ac20-37acd7d3ff16
# ╠═e2bcd8f3-75fc-4e02-9d4a-eba3a9d01b30
# ╠═06903f32-5069-42df-8ca3-4cac0011bd64
# ╠═35eb8005-c75b-462a-aa7b-b1322ca55664
# ╠═99acd1ef-17ef-47ca-84c3-0c7956e1c27d
# ╠═5617fff4-d3a6-4852-832e-fad36c8c1ec1
# ╠═9da17695-6f7a-4a37-90ef-34d6adedbd8c
# ╠═8baea97a-f9a3-43a3-be07-6f0ffda6e60a
# ╠═3b00cc15-14fd-471f-b48c-8db92496684e
# ╠═6c4d5d37-4cc8-4bc4-9842-d18c0ce7ec4f
# ╠═3612d185-8a7c-409f-952b-b17532a484dd
# ╠═479e8b29-ab02-4075-9711-599ded8807b3
# ╠═bfaf76d3-562f-498b-a031-fa07f459c4b4
# ╠═cb292cfb-32c8-441f-ab7e-5cc908234d17
# ╠═b2c52c2e-9972-4a52-9345-b4c768b76494
# ╠═c1fbea75-d24f-4049-b9be-93d1d4f391c9
# ╠═4130dfc2-56bc-41ef-83e1-ad9a0f2ff1fe
# ╠═648a99e0-6ee6-4489-a5bb-be1fa853fb23
# ╠═300b043c-e73a-41ae-b041-707e5365f597
# ╠═1ddfcf7e-2958-4f5c-b43e-6f8393b5bdbd
# ╠═b2cc7582-8878-4ac1-a12b-79117c0dad93
# ╠═604d254b-3d86-48bb-a22f-5d56771afdb4
# ╠═ad41ec1e-c21c-46aa-8da1-e5bbf6fedcb5
# ╠═cde6098d-9ec2-47e1-ad7f-02b910caa196
# ╠═c01746c5-391a-497c-b9d3-10ddd2b1f7f5
# ╠═261a1d4e-adbf-4fdc-b3cc-b7e3ab8a1974
# ╠═53571901-8805-4c9c-b222-274c6a4cfa4f
# ╠═053a7bc6-bf65-48be-bca5-f0ff615abc40
# ╠═a5841fe0-3f2d-4c86-976c-69eff2fc04b5
# ╠═0f0a2e2e-599d-45cd-b997-0fd7f77f68e1
# ╠═e71a83ea-2035-488d-8639-9837e79d1c30
# ╠═861d9d1d-e542-4de2-a4df-8ce3771dad64
# ╠═02e93f83-be74-4614-a0ff-2b1044198975
# ╠═89b9cce5-b297-48c9-a4b6-3a7b43952294
# ╠═32715449-730c-4201-9c61-96b68ba6635f
# ╠═5b29638f-47fd-4b3a-bc51-cd35bcecf88e
# ╠═c5575618-9027-484f-85c0-aa239f239dbe
# ╠═95d01f67-5b46-46c9-9403-53070829571e
# ╠═cb6a77d5-cfdf-4a04-bafc-c5051568ff1d
# ╠═05346d5e-081e-4139-97cb-99283ba1342b
# ╠═0409ff93-8ece-4cef-a44d-1c36ea4a9c5f
# ╠═ddb913c3-e62c-48f8-8b65-4b281bdd33ab
# ╠═f07cf5b3-acc0-40ef-b997-76e4d5313aec
# ╟─cc58368b-bbcc-4849-ad3f-bec12ad3e16f
# ╠═f4a21906-1857-418c-ba74-1d7d4f11509b
# ╠═bd742e20-2fe6-40f6-9d0e-9922a907e7d3
# ╠═38c6250b-1c74-4c6a-be38-7612bd2cd06b
# ╠═ed2dccce-905e-4503-8c70-5cecd9d80734
# ╠═83ed5fe5-5f0e-4b0e-9ac4-f82e06940726
# ╠═7b7cf78f-efb0-44b7-ae43-b413b79eb9d2
# ╠═b810a51c-7372-40df-a613-4112b53e547b
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
