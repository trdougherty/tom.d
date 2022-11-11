### A Pluto.jl notebook ###
# v0.19.14

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

# ╔═╡ b6d7cd90-0d59-4194-91c8-6d8f40a4a9c3
using StatsBase

# ╔═╡ 982250e8-58ad-483d-87b5-f6aff464bd10
begin
end

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
	train = CSV.read(joinpath(input_dir, "train.csv"), DataFrame)
	validate = CSV.read(joinpath(input_dir, "validate.csv"), DataFrame)
	test = CSV.read(joinpath(input_dir, "test.csv"), DataFrame)
end;

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

# ╔═╡ 44e4ebf2-f3b8-4be4-a6b9-06822230d947


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
Gadfly.plot(
	sar_colorsample,
	x=:date,
	y=:VV,
	color=:colorgroups,
	# Geom.beeswarm(padding=1pt),
	Geom.smooth(smoothing=0.2),
	Guide.title("VV Polarization Over Time"),
	Guide.Theme(default_color="black", point_size=0.5pt, line_width=0.5pt),
	# Coord.cartesian(ymin=-10, ymax=10),
)

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
landsat8_r = CSV.read(landsat8_p, DataFrame; dateformat=date_f);

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
noaa_r = CSV.read(noaa_p, DataFrame; dateformat=date_f);

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
	combine(
		groupby(
			data, 
			agg_terms
			), 
		numericterms .=> functional_terms,
		renamecols=true
	)
end

# ╔═╡ 2f1fec21-76a2-4365-b305-0f24505b1ccc
describe(sar_r, :nmissing)

# ╔═╡ 86d465e3-7916-479d-a29c-2b93ae54ab6d
agg_terms = 	["Property Id","date"]

# ╔═╡ 637220ba-c76a-4210-8c08-fde56b86366a
functional_terms = [f₁ f₂ f₃ f₄ f₅]

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
p₂ = Gadfly.plot(
	train,
	x=:council_region,
	y=:weather_station_id,
	Guide.xlabel("Council Region"),
	Guide.ylabel("Weather Station ID"),
	Guide.title("Data Points Histogram"),
	Guide.yticks(ticks=725020:4:725060),
	Geom.histogram2d
)

# ╔═╡ c6c038f0-cf8b-4234-a64c-75616fdc07a5
draw(
	PNG(
		joinpath(output_dir, "datapoint_histogram.png"), 
		20cm, 
		10cm,
		dpi=500
	), p₂
)

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

# ╔═╡ e6ea0a65-b79f-400f-be61-951b08b5ce88
p₃ = Gadfly.plot(
	Η₁,
	x=:month,
	y=:value,
	color=:variable,
	Geom.point,
	Geom.line,
	Scale.color_discrete(),
	Guide.xlabel("Month"),
	Guide.ylabel("Avg. Energy - MWh"),
	Guide.xticks(ticks=1:12),
	Guide.yticks(ticks=0.0:0.02:0.15),
	Guide.colorkey(title="Term"),
	Guide.title("New York Energy Consumption Habits"),
	Theme(point_size=3.5pt),
	Scale.color_discrete_manual("indianred","lightblue")
)

# ╔═╡ 77aad1ea-16cf-4c5f-9a6b-345f7168afb3
draw(
	PNG(
		joinpath(output_dir, "energy_trends.png"), 
		13cm, 
		9cm,
		dpi=500
	), p₃
)

# ╔═╡ fa79e50c-5d5e-47a1-b2e7-fe3341ecbf4f
## still playing around a little bit with the visualizations

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
		x=:LST_Day_1km_f₁,
		y=:LST_Night_1km_f₁,
		Geom.smooth,
		intercept=[0], 
		slope=[1],
		Geom.abline(color="pink", style=:dash),
	),
	Gadfly.layer(
		x=:LST_Day_1km_f₁,
		y=:LST_Night_1km_f₁,
		color="Property Id",
		# Geom.line,
		# Geom.point,
		# Geom.line,
		Geom.smooth(smoothing=0.7),
		Theme(line_width=0.08pt, alphas=[0.5])
	),
	Scale.discrete_color_hue,
	Guide.ylabel("Nightime Temperature - °C"),
	Guide.xlabel("Daytime Temperature - °C"),
	Guide.title("Diurnal Temperature Trends"),
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
begin
bincount = 100
Gadfly.plot(
	x=sample_filtered.VH_f₁,
	y=sample_filtered.LST_Day_1km_f₃,
	Geom.histogram2d(xbincount=bincount, ybincount=bincount),
	Guide.xlabel("Polarization"),
	Guide.ylabel("Daytime Temperature - °C"),
	Guide.title("Polarization vs. Daytime Temps"),
	Guide.Theme(panel_line_width=0pt),
	# Coord.cartesian(xmin=-18, xmax=-13),
)
end

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
Gadfly.plot(
	filter( x -> x["Property Id"] in lst_names, lst′ ),
	x=:date,
	y=:trees_f₁,
	color="Property Id",
	# Geom.point,
	Geom.smooth(smoothing=0.2),
	Scale.discrete_color,
	Theme(key_position = :none)
)

# ╔═╡ 217bfbed-1e13-4147-b4e4-c58eaed29382
md"""
#### Now jumping into the training process
"""

# ╔═╡ 61238caa-0247-4cf6-8fb7-3b9db84afcee
md"""
Base model - this m₁ term will be used for almost all of the anlaysis
"""

# ╔═╡ 0bc7b14b-ccaf-48f6-90fa-3006737727ed
rng = MersenneTwister(100);

# ╔═╡ 58b4d31f-5340-4cd9-8c9e-6c504479897f
EvoTree = @load EvoTreeRegressor pkg=EvoTrees verbosity=0

# ╔═╡ 850ddd0b-f6ea-4743-99ca-720d9ac538a0
Tree = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# ╔═╡ 90227375-6661-4696-8abd-9562e08040bf
"Invalid loss: EvoTrees.L1(). Only [`:linear`, `:logistic`, `:L1`, `:quantile`] are supported at the moment by EvoTreeRegressor.";

# ╔═╡ ea1a797c-da0a-4133-bde5-366607964754
m = EvoTree(
	rng=rng, 
	max_depth=6,
	lambda=0.0,
	gamma=0.0,
	nrounds=300, 
	rowsample=0.9, 
	colsample=0.9,
	device="gpu"
)

# ╔═╡ 8edfe547-f860-418f-8c68-e8fe0c16162f
md"""
###### Electric Results
"""

# ╔═╡ 6c387352-ae9c-405d-8f16-562b562c4a4b
md"""
###### Gas Results
"""

# ╔═╡ ad63a1dc-1091-4c0c-b8a5-18a2ffe19a31
# ╠═╡ disabled = true
#=╠═╡
# m = Tree(rng=rng, max_depth=6, n_trees=50, device="gpu")
  ╠═╡ =#

# ╔═╡ 59d97a5f-bcaf-4cf2-874c-405ad8cdadab
n_imp = 8

# ╔═╡ d8531d5a-e843-45e5-a498-0891d057d393
# m = EvoTree(loss=:L1)

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
	dynam,
	noaa,
	era5,
	epw, 
	lst, 
	sentinel_1C,
	sar,
	viirs
];

# ╔═╡ 9239f8dc-fd07-435f-9d88-e24bb1f6faa2
for term in comprehensive_datalist
	@info nrow(term)
end

# ╔═╡ 0c89dbf1-c714-43da-be89-3f82ccc2373a
# here is where we want to inject all of the data and drop missing terms
tₐ′ = clean(dropmissing(innerjoin(
	tₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

# ╔═╡ 8cbf28f7-a3a9-4fde-9156-935c25b001e7
nrow(tₐ′)

# ╔═╡ e16ea364-ee01-46d2-ab1b-9da4dd057616
describe(tₐ′, :nmissing)

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

# ╔═╡ 07fe88a1-bc61-42e1-b951-e902864efc4e


# ╔═╡ 3d513e7f-a1f1-4668-9a0f-87cf7e7a68c6
# here is where we want to inject all of the data and drop missing terms
vₐ′ = clean(dropmissing(innerjoin(
	vₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

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

# ╔═╡ d37ff7b0-fb28-4f55-b5c1-f4040f1387f6
teᵧ′ = clean(dropmissing(innerjoin(
	teᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

# ╔═╡ 5cbbd305-513c-42e3-b6b4-5d969d62a3f3
exclusion_terms = Not([joining_terms..., extra_omission_features...])

# ╔═╡ 73ec227e-2a7f-4f32-98b6-9000c5cdea4c


# ╔═╡ fd61e191-3b66-4ebc-b5ec-02a128548ffb
md"""
#### Electricity
"""

# ╔═╡ 3c41548c-da78-49c5-89a3-455df77bf4fa
md"""
Null model
"""

# ╔═╡ a149faa9-bee4-42a0-93fe-5adff459e0e9
## as a preliminary introduction - thinking about the overall dataset
@info "Number of data points" nrow(tₐ′)

# ╔═╡ 748321c3-0956-4439-90d4-e74598d83f20
term₀ = unique([electricity_terms...])

# ╔═╡ 05084413-9620-40ca-9f9f-98efea6f65a0
describe(tₐ′, :nmissing)

# ╔═╡ 844b35d3-6ff2-42bd-9fbe-d20cfd9d856f
begin
tₐ′₀ = select(tₐ′, term₀)
m₀ = training_pipeline(
	select(tₐ′₀, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ 43f847df-5247-48b5-8310-986dc7ccb60d
begin
vₐ′₀ = select(vₐ′, term₀)

vₐ′₀.prediction = validation_pipeline(
	select(vₐ′₀, exclusion_terms),
	"electricity_mwh",
	m₀
);

vₐ′₀.recorded = vₐ′₀.electricity_mwh
vₐ′₀.model = repeat(["Null"], nrow(vₐ′₀))
end;

# ╔═╡ f76ee669-8dd1-42ae-aa30-3cb0d1541aa3
begin
teₐ′₀ = select(teₐ′, term₀)

teₐ′₀.prediction = validation_pipeline(
	select(teₐ′₀, exclusion_terms),
	"electricity_mwh",
	m₀
);

teₐ′₀.recorded = teₐ′₀.electricity_mwh
teₐ′₀.model = repeat(["Null"], nrow(teₐ′₀))
end;

# ╔═╡ dee3a9ec-79f5-4917-ad40-2e4ddcdd423d
md"""
ERA5
"""

# ╔═╡ ad174123-117e-4365-9ede-50456d445fce
term₁ = unique([names(era5)..., electricity_terms...])

# ╔═╡ 3f37f92d-119b-4020-b4a3-c9e64e00c264
tₐ′₁ = select(tₐ′, term₁);

# ╔═╡ e4ccf0ac-0b82-4462-bbb3-9fc1ae09dc2b
begin
m₁ = training_pipeline(
	select(tₐ′₁, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ 99280172-b38b-4598-99a8-e091a72ae52c
begin
vₐ′₁ = select(vₐ′, term₁)

vₐ′₁.prediction = validation_pipeline(
	select(vₐ′₁, exclusion_terms),
	"electricity_mwh",
	m₁
);

vₐ′₁.recorded = vₐ′₁.electricity_mwh
vₐ′₁.model = repeat(["ERA5"], nrow(vₐ′₁))
end;

# ╔═╡ a57efaa9-89c5-46a0-b310-892890068561
begin
teₐ′₁ = select(teₐ′, term₁)

teₐ′₁.prediction = validation_pipeline(
	select(teₐ′₁, exclusion_terms),
	"electricity_mwh",
	m₁
);

teₐ′₁.recorded = teₐ′₁.electricity_mwh
teₐ′₁.model = repeat(["ERA5"], nrow(teₐ′₁))
end;

# ╔═╡ ee57d5d5-545e-4b70-91d9-b82a108f854b
md"""
###### NOAA
"""

# ╔═╡ ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
term₂ = unique([names(noaa)..., electricity_terms...])

# ╔═╡ 30e3c82a-fc70-4922-a7f7-cc1bec0e7d1c
tₐ′₂ = select(tₐ′, term₂);

# ╔═╡ 704d3436-b0f0-4bf4-8f5c-2f57786f248b
# describe(tₐ′₂, :nmissing)

# ╔═╡ 87a6d9a5-7957-44c4-8592-3eef5b945782
begin
m₂ = training_pipeline(
	select(tₐ′₂, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ eb2c58ed-6388-47a5-a6a4-c516acc9a4bd
begin
vₐ′₂ = select(vₐ′, term₂)

vₐ′₂.prediction = validation_pipeline(
	select(vₐ′₂, exclusion_terms),
	"electricity_mwh",
	m₂
);

vₐ′₂.recorded = vₐ′₂.electricity_mwh
vₐ′₂.model = repeat(["NOAA"], nrow(vₐ′₂))
end;

# ╔═╡ c0e28ffb-b49f-4b2d-919e-f63076cd8485
begin
teₐ′₂ = select(teₐ′, term₂)

teₐ′₂.prediction = validation_pipeline(
	select(teₐ′₂, exclusion_terms),
	"electricity_mwh",
	m₂
);

teₐ′₂.recorded = teₐ′₂.electricity_mwh
teₐ′₂.model = repeat(["NOAA"], nrow(teₐ′₂))
end;

# ╔═╡ 6958cff8-3c8d-4a77-bbff-f8cb17afd632


# ╔═╡ 51ecb564-06d5-4767-aa41-3030ca08a6c7
md"""
###### MODIS
"""

# ╔═╡ 2119a637-98d1-4e1f-b25a-27d3dc42e636
nrow(tₐ′)

# ╔═╡ 763edcce-0696-4670-a8cf-4963cfe70975
term₃ = unique([names(lst)..., electricity_terms...])

# ╔═╡ 8a337442-2f6b-4381-902b-93c52f6d8981
begin
tₐ′₃ = dropmissing(select(tₐ′, term₃))
m₃ = training_pipeline(
	select(tₐ′₃, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ d8e9d7bc-2b19-46d0-b918-554ecc003924
begin
vₐ′₃ = select(vₐ′, term₃)

vₐ′₃.prediction = validation_pipeline(
	select(vₐ′₃, exclusion_terms),
	"electricity_mwh",
	m₃
);

vₐ′₃.recorded = vₐ′₃.electricity_mwh
vₐ′₃.model = repeat(["MODIS"], nrow(vₐ′₃))
end;

# ╔═╡ 8b7c57c8-a525-432f-b51a-9d14196f30be
begin
teₐ′₃ = select(teₐ′, term₃)

teₐ′₃.prediction = validation_pipeline(
	select(teₐ′₃, exclusion_terms),
	"electricity_mwh",
	m₃
);

teₐ′₃.recorded = teₐ′₃.electricity_mwh
teₐ′₃.model = repeat(["MODIS"], nrow(teₐ′₃))
end;

# ╔═╡ cde0836c-3dbd-43a9-90fd-c30e5985acf7
md"""
###### EPW
"""

# ╔═╡ 5caecfab-3874-4989-8b3d-c65b53361c62
term₄ = unique([names(epw)..., electricity_terms...])

# ╔═╡ 79d05a1e-add2-42b4-81bb-680d98a1747d
begin
tₐ′₄ = dropmissing(select(tₐ′, term₄))
m₄ = training_pipeline(
	select(tₐ′₄, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ e876b1e6-1c80-4bfd-9397-f0611e47ac81
begin
vₐ′₄ = select(vₐ′, term₄)

vₐ′₄.prediction = validation_pipeline(
	select(vₐ′₄, exclusion_terms),
	"electricity_mwh",
	m₄
);

vₐ′₄.recorded = vₐ′₄.electricity_mwh
vₐ′₄.model = repeat(["EPW"], nrow(vₐ′₄))
end;

# ╔═╡ acdb3544-c64f-405a-afd7-fa0c40d27844
begin
teₐ′₄ = select(teₐ′, term₄)

teₐ′₄.prediction = validation_pipeline(
	select(teₐ′₄, exclusion_terms),
	"electricity_mwh",
	m₄
);

teₐ′₄.recorded = teₐ′₄.electricity_mwh
teₐ′₄.model = repeat(["EPW"], nrow(teₐ′₄))
end;

# ╔═╡ b9b8a050-1824-414e-928d-b7797760f176
md"""
###### Sentinel-2
"""



# ╔═╡ e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
term₅ = unique([names(sentinel_1C)..., electricity_terms...])

# ╔═╡ 358c345c-108e-4bdf-8a79-9c704b529ce3
begin
tₐ′₅ = dropmissing(select(tₐ′, term₅))
m₅ = training_pipeline(
	select(tₐ′₅, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ ad27236b-945e-4ef9-a056-968f5bb9fa93
begin
vₐ′₅ = select(vₐ′, term₅)

vₐ′₅.prediction = validation_pipeline(
	select(vₐ′₅, exclusion_terms),
	"electricity_mwh",
	m₅
);

vₐ′₅.recorded = vₐ′₅.electricity_mwh
vₐ′₅.model = repeat(["Sentinel-2"], nrow(vₐ′₅))
end;

# ╔═╡ 156ddaf3-c417-49f9-ab9b-382ca47031de
begin
teₐ′₅ = select(teₐ′, term₅)

teₐ′₅.prediction = validation_pipeline(
	select(teₐ′₅, exclusion_terms),
	"electricity_mwh",
	m₅
);

teₐ′₅.recorded = teₐ′₅.electricity_mwh
teₐ′₅.model = repeat(["Sentinel-2"], nrow(teₐ′₅))
end;

# ╔═╡ a6cada88-c7c9-495d-8806-2503e674ec39
md"""
###### VIIRS
"""



# ╔═╡ 7c84422b-d522-4f11-9465-058f41a4266f
term₆ = unique([names(viirs)..., electricity_terms...])

# ╔═╡ 9666648a-42e2-4237-a4a4-71f0d5abf46c
begin
tₐ′₆ = select(tₐ′, term₆)
m₆ = training_pipeline(
	select(tₐ′₆, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ c5b5b147-22ec-432b-ab20-111f6a759101
begin
vₐ′₆ = select(vₐ′, term₆)

vₐ′₆.prediction = validation_pipeline(
	select(vₐ′₆, exclusion_terms),
	"electricity_mwh",
	m₆
);

vₐ′₆.recorded = vₐ′₆.electricity_mwh
vₐ′₆.model = repeat(["VIIRS"], nrow(vₐ′₆))
end;

# ╔═╡ 6066527e-c75a-4305-8a34-0cc98c4b3a91
begin
teₐ′₆ = select(teₐ′, term₆)

teₐ′₆.prediction = validation_pipeline(
	select(teₐ′₆, exclusion_terms),
	"electricity_mwh",
	m₆
);

teₐ′₆.recorded = teₐ′₆.electricity_mwh
teₐ′₆.model = repeat(["VIIRS"], nrow(teₐ′₆))
end;

# ╔═╡ 03d2381d-e844-4809-b5a9-048c7612b7e2
md"""
###### SAR
"""

# ╔═╡ 98d04357-be23-4882-b5b5-8a6d924b7876
term₇ = unique([names(sar)..., electricity_terms...])

# ╔═╡ 51a6b9f1-d87e-4279-9374-18f51c08c6c4
begin
tₐ′₇ = select(tₐ′, term₇)
m₇ = training_pipeline(
	select(tₐ′₇, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ 0c897845-76fe-4b3b-b3ec-f844bffb9829
begin
vₐ′₇ = select(vₐ′, term₇)

vₐ′₇.prediction = validation_pipeline(
	select(vₐ′₇, exclusion_terms),
	"electricity_mwh",
	m₇
);

vₐ′₇.recorded = vₐ′₇.electricity_mwh
vₐ′₇.model = repeat(["SAR"], nrow(vₐ′₇))
end;

# ╔═╡ e684132c-130b-4ed5-b04c-a855f05e0e41
begin
teₐ′₇ = select(teₐ′, term₇)

teₐ′₇.prediction = validation_pipeline(
	select(teₐ′₇, exclusion_terms),
	"electricity_mwh",
	m₇
);

teₐ′₇.recorded = teₐ′₇.electricity_mwh
teₐ′₇.model = repeat(["SAR"], nrow(teₐ′₇))
end;

# ╔═╡ e84033ac-3b34-4e1f-a72b-9dfd937382c1
md"""
###### Dynamic World
"""

# ╔═╡ 13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
term₈ = unique([names(dynam)..., electricity_terms...])

# ╔═╡ 80d6f1f6-d4b2-460f-8f49-763ea1e32983
names(tₐ′)

# ╔═╡ cec1ec7b-b585-4509-85b1-f3a32d435341
begin
tₐ′₈ = select(tₐ′, term₈)
m₈ = training_pipeline(
	select(tₐ′₈, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ 197656d7-63de-4e07-83b7-d62e89e86374
begin
vₐ′₈ = select(vₐ′, term₈)

vₐ′₈.prediction = validation_pipeline(
	select(vₐ′₈, exclusion_terms),
	"electricity_mwh",
	m₈
);

vₐ′₈.recorded = vₐ′₈.electricity_mwh
vₐ′₈.model = repeat(["Dynamic World"], nrow(vₐ′₈))
end;

# ╔═╡ d69927ce-4a03-4a05-8d48-ed77b5800c78
begin
teₐ′₈ = select(teₐ′, term₈)

teₐ′₈.prediction = validation_pipeline(
	select(teₐ′₈, exclusion_terms),
	"electricity_mwh",
	m₈
);

teₐ′₈.recorded = teₐ′₈.electricity_mwh
teₐ′₈.model = repeat(["Dynamic World"], nrow(teₐ′₈))
end;

# ╔═╡ bb329d66-c51e-423d-b28b-a15cb38d7ed7
md"""
###### Full Dataset
"""

# ╔═╡ 10ef1866-2a8b-4bc6-8fdd-41f5b3335e18
begin
mₑ = training_pipeline(
	select(tₐ′, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ ef2f7716-f622-4bf2-ada2-14e76900619b
feature_importances(mₑ)[1:10]

# ╔═╡ 43e57bc2-e04e-4fad-a922-56c0756641e6
feature_importances(mₑ)[1:10]

# ╔═╡ a8d738bb-0424-4df0-aa22-2a299fd994b1
imp = feature_importances(mₑ);

# ╔═╡ d15f99c3-b97d-4b6f-9e2b-bc1d4dbae892
imp_vars = map(x -> x[1], imp)

# ╔═╡ 91576be3-5a84-459a-bf66-94016676170d
imp_values = map(x ->x[2], imp)

# ╔═╡ dc8fb980-6500-4849-b498-b39454dd3ffa
Gadfly.plot(
	x=imp_vars[1:n_imp],
	y=imp_values[1:n_imp],
	Geom.bar
)

# ╔═╡ 6c154508-b660-4070-b4b0-e152b486c64f
feature_importances(mₑ)

# ╔═╡ e9bf3f64-816b-47ca-b79a-e65ddb4f0a80
begin
vₐ′ₑ = deepcopy(vₐ′)

vₐ′ₑ.prediction = validation_pipeline(
	select(vₐ′ₑ, exclusion_terms),
	"electricity_mwh",
	mₑ
);

vₐ′ₑ.recorded = vₐ′ₑ.electricity_mwh
vₐ′ₑ.model = repeat(["Full Data"], nrow(vₐ′ₑ))
end;

# ╔═╡ 8627fcb3-2f48-4c7c-a07a-f670e836a605
begin
teₐ′ₑ = deepcopy(teₐ′)

teₐ′ₑ.prediction = validation_pipeline(
	select(teₐ′ₑ, exclusion_terms),
	"electricity_mwh",
	mₑ
);

teₐ′ₑ.recorded = teₐ′ₑ.electricity_mwh
teₐ′ₑ.model = repeat(["Full Data"], nrow(teₐ′ₑ))
end;

# ╔═╡ 3baf6654-4517-4874-9007-9a1033a6f753


# ╔═╡ 9a439ab8-87bc-41c8-94fd-17da1f329a78
# ╠═╡ disabled = true
#=╠═╡
test_terms = [vₐ′₀,vₐ′₁,vₐ′₂,vₐ′₃,vₐ′₄,vₐ′₅,vₐ′₆,vₐ′₇,vₐ′₈,vₐ′ₑ];
  ╠═╡ =#

# ╔═╡ 685169a0-e65c-439c-a5ba-068c83258200
test_terms = [teₐ′₀,teₐ′₁,teₐ′₂,teₐ′₃,teₐ′₄,teₐ′₅,teₐ′₆,teₐ′₇,teₐ′₈, teₐ′ₑ];

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
prediction_terms = [:prediction, :recorded, :month, :model]

# ╔═╡ 9d1ea09f-3c42-437f-afbc-21a70296a3ba
interest_terms = [:weather_station_distance]

# ╔═╡ 05a040cf-f6ac-42b2-bcb1-599af5a26038
test_termsₑ = vcat([ select(x, prediction_terms, interest_terms) for x in test_terms ]...);

# ╔═╡ 41751922-2646-45fd-b6e0-7ca2109cc642
function mse(x)
	(sum(x.^2) / length(x))^0.5
end

# ╔═╡ 4ae273f6-80e8-4858-97b4-3c803c727a08
# ╠═╡ disabled = true
#=╠═╡
a = collect(4:0.3:8)
  ╠═╡ =#

# ╔═╡ c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
begin
# Null
Ξ₀ = hcat(select(teₐ′₀, prediction_terms), select(teₐ′, interest_terms))

# ERA5
Ξ₅ = hcat(select(teₐ′₅, prediction_terms), select(teₐ′, interest_terms))

#MODIS
Ξ₃ = hcat(select(teₐ′₃, prediction_terms), select(teₐ′, interest_terms))
	
#EPW
Ξ₄ = hcat(select(teₐ′₄, prediction_terms), select(teₐ′, interest_terms))
	
Ξ = vcat(Ξ₅, Ξ₃, Ξ₄)

Ξ.error = Ξ.prediction .- Ξ.recorded
Ξ′ = combine(groupby(Ξ, [:month, :model]), :error => mean, renamecols=false)

p₄ = Gadfly.plot(
	Ξ′,
	x=:month,
	y=:error,
	color=map(x-> seasons[x], Ξ′.month),
	xgroup=:model,
	yintercept=[0],
	Guide.ylabel("Δ Prediction - Recorded"),
	Guide.xlabel("Month"),
	Guide.title("Mean Electricity Error by Season"),
	Guide.colorkey(title="Season"),
	Geom.subplot_grid(
		# Guide.yticks(ticks=0.0:0.001:0.01),
		Guide.xticks(ticks=1:2:12),
		Geom.point,
		Geom.line,
		Geom.hline(color=["pink"], style=:dash),
		# Geom.point,
		# free_y_axis=true,
	),	# Coord.cartesian(ymin=-0.1, ymax=0.1, aspect_ratio=1.5),
	# Theme(default_color="black")
)
end

# ╔═╡ 187a03ba-c9a4-47cf-be7e-ae024d1fff72
draw(PNG(joinpath(output_dir, "seasonal_accuracy.png"), 14cm, 10cm, dpi=600), p₄)

# ╔═╡ 11277ca6-3bbf-403f-8cfc-2aabf94069f7
p₇ = Gadfly.plot(
	test_termsₑ,
	color=:model,
	x=:weather_station_distance,
	y=abs.(test_termsₑ.prediction .- test_termsₑ.recorded),
	Geom.smooth(smoothing=0.8),
	Guide.yticks(ticks=0.015:0.01:0.035),
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Electricity - MAE vs Weather Station Distance"),
)

# ╔═╡ 28d2660e-258d-4583-8224-c2b4190f4140
draw(PNG(joinpath(output_dir, "learning_results_electricity.png"), 20cm, 10cm, dpi=600), p₇)

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


# ╔═╡ fa72b6f2-de9b-430c-9ab6-f799157f1570


# ╔═╡ 778dc2a7-4e9b-4f97-b34d-6a7adc38abc2


# ╔═╡ b98baad1-dad8-464c-a18a-1baf01962164
md"""
#### Natural Gas
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

# ╔═╡ 991815d7-09da-423c-b14a-a8a3fcf662e4
md"""
##### ERA5
"""

# ╔═╡ bc64b624-d8d2-480a-a698-092aea0a74b2
termᵧ₁ = unique([names(era5)..., naturalgas_terms...])

# ╔═╡ 287c4356-e85d-4457-88f7-cc9814c39c30
mᵧ₁, vᵧ′₁, teᵧ′₁ = gastrain("ERA5", termᵧ₁);

# ╔═╡ 0e876bab-6648-4a67-b571-dc82a7bdf8f1
md"""##### NOAA"""

# ╔═╡ 3763ad63-003e-495f-aa90-0db525412c62
termᵧ₂ = unique([names(noaa)..., naturalgas_terms...])

# ╔═╡ a9a9ea9d-5a97-4260-8e02-a27732928e61
mᵧ₂, vᵧ′₂, teᵧ′₂ = gastrain("NOAA", termᵧ₂);

# ╔═╡ 202c9072-0a7b-454d-9112-4ecc0a03c61b
md"""##### MODIS"""

# ╔═╡ 434d7e0d-583d-498b-9b6c-a72fa3775b3c
termᵧ₃ = unique([names(lst)..., naturalgas_terms...])

# ╔═╡ 2857461c-d24e-4b04-91f1-80cd842eeaa4
mᵧ₃, vᵧ′₃, teᵧ′₃ = gastrain("LST", termᵧ₃);

# ╔═╡ a78a6fd5-4b45-43fc-a733-aef4fd14eb42
md""" ##### EPW"""

# ╔═╡ 1a4471e7-24ad-4652-9f3f-6eef92c781d5
termᵧ₄ = unique([names(epw)..., naturalgas_terms...])

# ╔═╡ 8c074d96-ee63-45fb-bbbf-135c40b66a09
mᵧ₄, vᵧ′₄, teᵧ′₄ = gastrain("EPW", termᵧ₄);

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
termᵧ₆ = unique([names(sentinel_1C)..., naturalgas_terms...])

# ╔═╡ ad7d7a32-861d-48be-af41-833e74023ef6
mᵧ₆, vᵧ′₆, teᵧ′₆ = gastrain("Sentinel-2", termᵧ₆);

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
##### Full Data
"""

# ╔═╡ 73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
mᵧ₉, vᵧ′₉, teᵧ′₉ = gastrain("Full Data", names(tᵧ′));

# ╔═╡ 1af87fef-8c88-4e34-b1d4-8bccd7881473


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


# ╔═╡ 59144efe-6073-4d1b-b976-b25d6bdd15e0
# epwᵧₒ[shuffle(1:nrow(epwᵧₒ)), :]

# ╔═╡ 4d12e794-2054-46c4-9fcf-73ba21b79794
# ╠═╡ disabled = true
#=╠═╡
test_termsᵧ = [vᵧ′₀,vᵧ′₁,vᵧ′₂,vᵧ′₃,vᵧ′₄,vᵧ′₅,vᵧ′₆,vᵧ′₇,vᵧ′₈,vᵧ′₉];
  ╠═╡ =#

# ╔═╡ c25b404b-0d89-4c4d-bc61-c361fd9d7038
test_termsᵧ = [teᵧ′₀,teᵧ′₁,teᵧ′₂,teᵧ′₃,teᵧ′₄,teᵧ′₅,teᵧ′₆,teᵧ′₇,teᵧ′₈,teᵧ′₉];

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

# ╔═╡ 21e706e9-3ce3-4c9b-b279-7abc7b9f8c94
begin
# Null
Γ₀ = hcat(select(teᵧ′₀, prediction_terms), select(teᵧ′, interest_terms))

#MODIS
Γ₁ = hcat(select(teᵧ′₁, prediction_terms), select(teᵧ′, interest_terms))
	
#EPW
Γ₂ = hcat(select(teᵧ′₂, prediction_terms), select(teᵧ′, interest_terms))

#MODIS
Γ₃ = hcat(select(teᵧ′₃, prediction_terms), select(teᵧ′, interest_terms))

# EPW
Γ₄ = hcat(select(teᵧ′₄, prediction_terms), select(teᵧ′, interest_terms))

Γ = vcat(Γ₁, Γ₂, Γ₃, Γ₄)

Γ.error = Γ.prediction .- Γ.recorded
Γ′ = combine(groupby(Γ, [:month, :model]), :error => mse, renamecols=false)

p₈ = Gadfly.plot(
	Γ′,
	x=:month,
	y=:error,
	color=map(x-> seasons[x], Γ′.month),
	xgroup=:model,
	yintercept=[0],
	Guide.ylabel("Δ Prediction - Recorded"),
	Guide.xlabel("Month"),
	Guide.title("Mean Natural Gas Prediction Error by Season"),
	Guide.colorkey(title="Season"),
	Geom.subplot_grid(
		Guide.yticks(ticks=0.0:0.025:0.15),
		Guide.xticks(ticks=1:2:12),
		Geom.point,
		Geom.line,
		Geom.hline(color=["pink"], style=:dash),
		# Geom.point,
		# free_y_axis=true,
	),	# Coord.cartesian(ymin=-0.1, ymax=0.1, aspect_ratio=1.5),
	# Theme(default_color="black")
)
end

# ╔═╡ a1e9f1c1-6764-4d03-a14f-a362c0fba808
draw(PNG(joinpath(output_dir, "naturalgas_seasonal.png"), 14cm, 10cm, dpi=600), p₈)

# ╔═╡ 08b888e3-9d37-43a6-9656-4bf6cb62d324
building_distancesᵧ = unique(select(teᵧ, ["Property Id", "weather_station_distance"]));

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
##### Now getting into combination data sets
---
first want to run an analysis on a comprehensive dataset to see which variables offer the highest impact
"""

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
function test_suite(Ẋ::DataFrame)
	# Ẋ = hcat(X, auxiliary)	
	vᵧ = groupby(Ẋ, "Property Id")
	results = combine(vᵧ) do vᵢ
	(
	cvrmse = cvrmse(vᵢ.prediction, vᵢ.recorded),
	nmbe = nmbe(vᵢ.prediction, vᵢ.recorded),
	cvstd = cvstd(vᵢ.prediction, vᵢ.recorded),
	rmse = rmse(vᵢ.prediction, vᵢ.recorded)
	)
	end

	results.model = repeat([Ẋ[1,"model"]], nrow(results))
	return results
end

# ╔═╡ 92e7c0a0-19df-43e2-90db-4e050ba90c5d
test_suiteₜₑ = vcat([ test_suite(x) for x in test_terms ]...);

# ╔═╡ a0cd5e68-3bd0-4983-809e-ce7acd36a048
modelresultsₑ = combine(
	groupby(
		clean(test_suiteₜₑ), 
		["model"]
	), [:rmse, :cvrmse,:nmbe,:cvstd] .=> mean, 
	renamecols=false
);

# ╔═╡ 6d5c77c8-2596-45eb-8163-c8e14249948d
baselineₑ = filter(x -> x.model == "Null", modelresultsₑ).rmse[1]

# ╔═╡ 342a8606-708b-4b1c-9023-050c81ce345d
modelresultsₑ.percent_improvement = 100 .* (1 .- modelresultsₑ.rmse ./ baselineₑ);

# ╔═╡ 40d7e39c-98c6-4cf8-8461-e54b7a0db696
sort(modelresultsₑ, :rmse)

# ╔═╡ 734ed2d0-4657-4361-9285-97605371af72
test_suiteₜᵧ = vcat([ test_suite(x) for x in test_termsᵧ ]...);

# ╔═╡ d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
modelresultsᵧ = combine(groupby(clean(test_suiteₜᵧ), ["model"]), [:rmse, :cvrmse,:nmbe,:cvstd] .=> mean, renamecols=false);

# ╔═╡ 31a814af-95f1-4d7f-ada0-8bf9df2aee10
baselineᵧ = filter(x -> x.model == "Null", modelresultsᵧ).rmse[1]

# ╔═╡ 9d6a3995-7df0-4552-887e-b306d26a80ad
modelresultsᵧ.percent_improvement = 100 .* (1 .- modelresultsᵧ.rmse ./ baselineᵧ);

# ╔═╡ 2984d75a-5264-4663-a08b-3c50e01d1670
sort(modelresultsᵧ, :rmse)

# ╔═╡ bf71086c-a8f0-4420-b698-73499fec5257
test_suiteₜᵧđ = leftjoin(
	test_suiteₜᵧ,
	building_distancesᵧ,
	on="Property Id"
);

# ╔═╡ 8df28d8b-443e-48a3-89e4-a5824d3d66c8
stack(test_suiteₜᵧđ, 2:4)

# ╔═╡ 4c99c655-ae95-4a28-95c8-e7ca38ddf55f
Gadfly.plot(
	stack(test_suiteₜᵧđ, 2:4),
	x=:weather_station_distance,
	y=:value,
	color=:model,
	ygroup=:variable,
	Geom.subplot_grid(
		Geom.line,
		free_y_axis=true
	),
)

# ╔═╡ Cell order:
# ╠═982250e8-58ad-483d-87b5-f6aff464bd10
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
# ╠═aeb3597f-ef6a-4d21-bff5-d2ad359bc1a2
# ╠═9d5f897f-7f25-43ed-a30c-064f21e50174
# ╠═786c2441-7abb-4caa-9f50-c6078fff0f56
# ╠═1c7bfba6-5e1d-457d-bd92-8ba445353e0b
# ╟─9b3790d3-8d5d-403c-8495-45def2c6f8ba
# ╠═bf772ea4-c9ad-4fe7-9436-9799dcb0ad04
# ╠═020b96e3-d218-470d-b4b0-fc9b708ffdf3
# ╠═9aa06073-d43e-4658-adb9-bbc11425978d
# ╠═4f1c0eae-e637-40f8-95a9-61088e423725
# ╠═a8a03990-fd30-48b9-9b76-ce33dd90ceb3
# ╠═44f02f33-9044-4355-ba82-b35595a82bdd
# ╠═56f43ba6-568b-436d-85a5-a8da5a0a3956
# ╠═8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
# ╠═44e4ebf2-f3b8-4be4-a6b9-06822230d947
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
# ╠═22494217-9254-4374-8a7d-02528bdd0df3
# ╠═d23500b2-53ba-436c-b26d-187f60821a43
# ╠═dcde8c56-7294-47fd-aad1-2204de6c904b
# ╠═e87d641a-9555-4a93-9fe8-f39f8964ce84
# ╠═ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
# ╠═2f1fec21-76a2-4365-b305-0f24505b1ccc
# ╠═86d465e3-7916-479d-a29c-2b93ae54ab6d
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
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
# ╠═36bf3b43-4cf6-4a48-8233-1f49b9f7fac3
# ╠═94d6e626-965d-4101-b31c-08c983678f92
# ╠═f73ae203-056b-400b-8457-6245e9283ead
# ╠═f3da6fb8-a85b-4dfc-a307-bc987239abc6
# ╟─54f438b0-893e-42d3-a0f5-2364723be84e
# ╠═c6f4308d-6dce-4a57-b411-6327f4aa87a7
# ╠═c6c038f0-cf8b-4234-a64c-75616fdc07a5
# ╠═1a5ab262-0493-470f-ab58-baa5fa1a69af
# ╠═b5dff4f2-3088-4301-8dfe-96fb8c6999c7
# ╠═df3549d2-4183-4690-9e04-b665a9286792
# ╠═e6ea0a65-b79f-400f-be61-951b08b5ce88
# ╠═77aad1ea-16cf-4c5f-9a6b-345f7168afb3
# ╠═fa79e50c-5d5e-47a1-b2e7-fe3341ecbf4f
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
# ╟─217bfbed-1e13-4147-b4e4-c58eaed29382
# ╟─61238caa-0247-4cf6-8fb7-3b9db84afcee
# ╠═0486a516-1210-4f42-9eac-b78433065365
# ╠═0bc7b14b-ccaf-48f6-90fa-3006737727ed
# ╠═58b4d31f-5340-4cd9-8c9e-6c504479897f
# ╠═850ddd0b-f6ea-4743-99ca-720d9ac538a0
# ╠═90227375-6661-4696-8abd-9562e08040bf
# ╠═ea1a797c-da0a-4133-bde5-366607964754
# ╟─8edfe547-f860-418f-8c68-e8fe0c16162f
# ╠═6d5c77c8-2596-45eb-8163-c8e14249948d
# ╠═342a8606-708b-4b1c-9023-050c81ce345d
# ╠═40d7e39c-98c6-4cf8-8461-e54b7a0db696
# ╠═ef2f7716-f622-4bf2-ada2-14e76900619b
# ╟─6c387352-ae9c-405d-8f16-562b562c4a4b
# ╠═31a814af-95f1-4d7f-ada0-8bf9df2aee10
# ╠═9d6a3995-7df0-4552-887e-b306d26a80ad
# ╠═2984d75a-5264-4663-a08b-3c50e01d1670
# ╠═43e57bc2-e04e-4fad-a922-56c0756641e6
# ╠═ad63a1dc-1091-4c0c-b8a5-18a2ffe19a31
# ╠═a8d738bb-0424-4df0-aa22-2a299fd994b1
# ╠═d15f99c3-b97d-4b6f-9e2b-bc1d4dbae892
# ╠═91576be3-5a84-459a-bf66-94016676170d
# ╠═59d97a5f-bcaf-4cf2-874c-405ad8cdadab
# ╠═dc8fb980-6500-4849-b498-b39454dd3ffa
# ╠═d8531d5a-e843-45e5-a498-0891d057d393
# ╟─bb078875-611c-46f6-8631-1befde358054
# ╠═b6d7cd90-0d59-4194-91c8-6d8f40a4a9c3
# ╟─df753575-b121-4c3b-a456-cfe4e535c2aa
# ╠═a157969b-100a-4794-a78e-2f40439e28d9
# ╠═9239f8dc-fd07-435f-9d88-e24bb1f6faa2
# ╠═0c89dbf1-c714-43da-be89-3f82ccc2373a
# ╠═8cbf28f7-a3a9-4fde-9156-935c25b001e7
# ╠═e16ea364-ee01-46d2-ab1b-9da4dd057616
# ╠═960d4a47-c701-42c5-934e-a80d74b7ddbd
# ╠═6d873e8f-d3ee-4423-86e7-e8d75abaad38
# ╠═cb23bb0f-87ad-4a1b-8491-c6db384824da
# ╠═28ac0a1f-e8cd-4e27-9852-38e712488b81
# ╠═0c9c56b4-4394-4457-b184-23ac8c35d7e6
# ╠═b81a31cc-5024-418c-8b69-61a6011385ff
# ╟─07fe88a1-bc61-42e1-b951-e902864efc4e
# ╠═3d513e7f-a1f1-4668-9a0f-87cf7e7a68c6
# ╠═98da20c8-c530-4ed6-acb2-3c98019a36a9
# ╠═7ff9cc78-44ae-4baa-b3af-d10487c11920
# ╠═5f260a93-7a7e-4c62-8a52-55371a2093c9
# ╠═294cd133-af00-4025-a182-6e2f057107a6
# ╠═7073082d-a2fa-4419-996c-a38cf3ee044c
# ╟─ec365574-6e66-4284-9723-5634ba73d52a
# ╠═b7983d6f-2dab-4279-b7a5-222e40ce968b
# ╠═ce3e2b3b-a6d6-4494-8168-665e878edae3
# ╠═617aa3dd-045f-4e78-bfcf-2d6c45dfe138
# ╠═f2bbbe98-9b3f-46c2-9630-2b3997549742
# ╠═d37ff7b0-fb28-4f55-b5c1-f4040f1387f6
# ╠═5cbbd305-513c-42e3-b6b4-5d969d62a3f3
# ╠═73ec227e-2a7f-4f32-98b6-9000c5cdea4c
# ╟─fd61e191-3b66-4ebc-b5ec-02a128548ffb
# ╟─3c41548c-da78-49c5-89a3-455df77bf4fa
# ╠═a149faa9-bee4-42a0-93fe-5adff459e0e9
# ╠═748321c3-0956-4439-90d4-e74598d83f20
# ╠═05084413-9620-40ca-9f9f-98efea6f65a0
# ╠═844b35d3-6ff2-42bd-9fbe-d20cfd9d856f
# ╠═43f847df-5247-48b5-8310-986dc7ccb60d
# ╠═f76ee669-8dd1-42ae-aa30-3cb0d1541aa3
# ╟─dee3a9ec-79f5-4917-ad40-2e4ddcdd423d
# ╠═ad174123-117e-4365-9ede-50456d445fce
# ╠═3f37f92d-119b-4020-b4a3-c9e64e00c264
# ╠═e4ccf0ac-0b82-4462-bbb3-9fc1ae09dc2b
# ╠═99280172-b38b-4598-99a8-e091a72ae52c
# ╠═a57efaa9-89c5-46a0-b310-892890068561
# ╟─ee57d5d5-545e-4b70-91d9-b82a108f854b
# ╠═ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
# ╠═30e3c82a-fc70-4922-a7f7-cc1bec0e7d1c
# ╠═704d3436-b0f0-4bf4-8f5c-2f57786f248b
# ╠═87a6d9a5-7957-44c4-8592-3eef5b945782
# ╠═eb2c58ed-6388-47a5-a6a4-c516acc9a4bd
# ╠═c0e28ffb-b49f-4b2d-919e-f63076cd8485
# ╠═6958cff8-3c8d-4a77-bbff-f8cb17afd632
# ╟─51ecb564-06d5-4767-aa41-3030ca08a6c7
# ╠═2119a637-98d1-4e1f-b25a-27d3dc42e636
# ╠═763edcce-0696-4670-a8cf-4963cfe70975
# ╠═8a337442-2f6b-4381-902b-93c52f6d8981
# ╠═d8e9d7bc-2b19-46d0-b918-554ecc003924
# ╠═8b7c57c8-a525-432f-b51a-9d14196f30be
# ╠═cde0836c-3dbd-43a9-90fd-c30e5985acf7
# ╠═5caecfab-3874-4989-8b3d-c65b53361c62
# ╠═79d05a1e-add2-42b4-81bb-680d98a1747d
# ╠═e876b1e6-1c80-4bfd-9397-f0611e47ac81
# ╠═acdb3544-c64f-405a-afd7-fa0c40d27844
# ╟─b9b8a050-1824-414e-928d-b7797760f176
# ╠═e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
# ╠═358c345c-108e-4bdf-8a79-9c704b529ce3
# ╠═ad27236b-945e-4ef9-a056-968f5bb9fa93
# ╠═156ddaf3-c417-49f9-ab9b-382ca47031de
# ╟─a6cada88-c7c9-495d-8806-2503e674ec39
# ╠═7c84422b-d522-4f11-9465-058f41a4266f
# ╠═9666648a-42e2-4237-a4a4-71f0d5abf46c
# ╠═c5b5b147-22ec-432b-ab20-111f6a759101
# ╠═6066527e-c75a-4305-8a34-0cc98c4b3a91
# ╟─03d2381d-e844-4809-b5a9-048c7612b7e2
# ╠═98d04357-be23-4882-b5b5-8a6d924b7876
# ╠═51a6b9f1-d87e-4279-9374-18f51c08c6c4
# ╠═0c897845-76fe-4b3b-b3ec-f844bffb9829
# ╠═e684132c-130b-4ed5-b04c-a855f05e0e41
# ╠═e84033ac-3b34-4e1f-a72b-9dfd937382c1
# ╠═13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
# ╠═80d6f1f6-d4b2-460f-8f49-763ea1e32983
# ╠═cec1ec7b-b585-4509-85b1-f3a32d435341
# ╠═197656d7-63de-4e07-83b7-d62e89e86374
# ╠═d69927ce-4a03-4a05-8d48-ed77b5800c78
# ╠═bb329d66-c51e-423d-b28b-a15cb38d7ed7
# ╠═10ef1866-2a8b-4bc6-8fdd-41f5b3335e18
# ╠═6c154508-b660-4070-b4b0-e152b486c64f
# ╠═e9bf3f64-816b-47ca-b79a-e65ddb4f0a80
# ╠═8627fcb3-2f48-4c7c-a07a-f670e836a605
# ╠═3baf6654-4517-4874-9007-9a1033a6f753
# ╠═9a439ab8-87bc-41c8-94fd-17da1f329a78
# ╠═685169a0-e65c-439c-a5ba-068c83258200
# ╠═92e7c0a0-19df-43e2-90db-4e050ba90c5d
# ╠═f5b41025-e072-4d42-b7aa-0982ddf01982
# ╠═a0cd5e68-3bd0-4983-809e-ce7acd36a048
# ╠═ed696113-737b-46f5-bfb1-c06d430a83ac
# ╠═8b3175d4-6938-4823-8a91-77cde7c31c2a
# ╠═9d1ea09f-3c42-437f-afbc-21a70296a3ba
# ╠═05a040cf-f6ac-42b2-bcb1-599af5a26038
# ╠═41751922-2646-45fd-b6e0-7ca2109cc642
# ╠═4ae273f6-80e8-4858-97b4-3c803c727a08
# ╠═c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
# ╠═187a03ba-c9a4-47cf-be7e-ae024d1fff72
# ╠═11277ca6-3bbf-403f-8cfc-2aabf94069f7
# ╠═28d2660e-258d-4583-8224-c2b4190f4140
# ╠═fbe43a2a-f9b0-487e-a579-645c2f40736e
# ╠═7b05ec74-6edc-4f44-bf9d-9389d4029494
# ╠═c546d3af-8f24-40bd-abfa-e06c708c244e
# ╠═fa72b6f2-de9b-430c-9ab6-f799157f1570
# ╠═778dc2a7-4e9b-4f97-b34d-6a7adc38abc2
# ╟─b98baad1-dad8-464c-a18a-1baf01962164
# ╟─3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
# ╟─d437be5f-ec26-466b-b099-5e1cc8816cb5
# ╟─2426e2d6-e364-4e97-bee8-7defb1e88745
# ╠═9c8f603f-33c6-4988-9efd-83864e871907
# ╠═20a6b910-4f0b-4c78-af6b-d76e55025297
# ╟─991815d7-09da-423c-b14a-a8a3fcf662e4
# ╠═bc64b624-d8d2-480a-a698-092aea0a74b2
# ╠═287c4356-e85d-4457-88f7-cc9814c39c30
# ╟─0e876bab-6648-4a67-b571-dc82a7bdf8f1
# ╠═3763ad63-003e-495f-aa90-0db525412c62
# ╠═a9a9ea9d-5a97-4260-8e02-a27732928e61
# ╟─202c9072-0a7b-454d-9112-4ecc0a03c61b
# ╠═434d7e0d-583d-498b-9b6c-a72fa3775b3c
# ╠═2857461c-d24e-4b04-91f1-80cd842eeaa4
# ╟─a78a6fd5-4b45-43fc-a733-aef4fd14eb42
# ╠═1a4471e7-24ad-4652-9f3f-6eef92c781d5
# ╠═8c074d96-ee63-45fb-bbbf-135c40b66a09
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
# ╠═73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
# ╠═1af87fef-8c88-4e34-b1d4-8bccd7881473
# ╠═da9a6ba8-083b-40b3-9bd6-089688ff7f73
# ╠═433d51c9-9f31-49db-a5a2-e6565888831e
# ╠═59144efe-6073-4d1b-b976-b25d6bdd15e0
# ╠═4d12e794-2054-46c4-9fcf-73ba21b79794
# ╠═c25b404b-0d89-4c4d-bc61-c361fd9d7038
# ╠═734ed2d0-4657-4361-9285-97605371af72
# ╠═d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
# ╠═f2067017-32c8-493b-a9fb-3a89f9e549e4
# ╠═ee0d413c-a107-4bdc-a08a-52cda6d81573
# ╠═77456db6-b0fa-4ba6-9d41-1c393f7ddee1
# ╟─21e706e9-3ce3-4c9b-b279-7abc7b9f8c94
# ╠═a1e9f1c1-6764-4d03-a14f-a362c0fba808
# ╠═08b888e3-9d37-43a6-9656-4bf6cb62d324
# ╠═bf71086c-a8f0-4420-b698-73499fec5257
# ╟─8df28d8b-443e-48a3-89e4-a5824d3d66c8
# ╠═4c99c655-ae95-4a28-95c8-e7ca38ddf55f
# ╠═459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
# ╠═ae386c52-ff6b-4493-96ff-6c14d1c46db8
# ╟─329d55b8-eb72-4a1e-a4e8-200fee0e0b9d
# ╠═b5ce80a3-e177-4c4f-920b-5dee87f2bc3b
# ╠═c92ab000-dbe9-457d-97f3-88ae31b57a27
# ╠═c089c975-96e1-4281-b5ad-c53e738834a1
# ╠═e4344d50-425b-4bea-b28e-0c3b45debfb1
# ╠═f9ccee0a-9e6b-4070-a15c-ff5d5c324649
# ╠═624ab4a3-5c3b-42f3-be37-89d6382fdfdd
# ╟─ec97d987-651d-4efa-a36f-e6be9f18e0fd
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
