### A Pluto.jl notebook ###
# v0.19.9

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

# ╔═╡ 786c2441-7abb-4caa-9f50-c6078fff0f56
using ArchGDAL

# ╔═╡ 1c7bfba6-5e1d-457d-bd92-8ba445353e0b
using MLJ

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
sar_r = CSV.read(sar_p, DataFrame; dateformat=date_f)

# ╔═╡ a2e624f7-5626-431a-9680-f62ed86b61aa
combine(groupby(sar_r, :date), first)

# ╔═╡ db7c092f-5fa8-4038-b9cf-d40d822a4b9a
Set(values(countmap(dropmissing(sar_r, [:HH] )[:,"Property Id"])))

# ╔═╡ d3d814ee-ad4f-47bf-966a-08cabc79bf90
Gadfly.plot(
	filter( x -> x.date == DateTime("2019-04-20T22:51:04"), sar_r ),
	x=:HH,
	Geom.histogram
)

# ╔═╡ 00acb065-2378-4181-b76a-488071f43a7e


# ╔═╡ 7d13349f-c3a4-40e7-b7d3-be2e0e951dd5
filter( x -> x["Property Id"] == 6683659, sar_r )

# ╔═╡ 44d399f7-268b-4e66-8ba9-fe2e8e17a19d
Gadfly.plot(
	filter( x -> x["Property Id"] == 6683659, dropmissing(sar_r, :VV)),
	x=:date,
	y=:VV,
	Geom.point,
	Geom.line,
	# Coord.cartesian(ymin=37.57, ymax=37.6),
	Theme(default_color="black", point_size=2pt)
)

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
VV_councilmedian = combine(groupby(innerjoin(dropmissing(sar_r, :VH), train[:,["Property Id","council_region"]], on="Property Id"), ["date","council_region"]), :VH => mean, renamecols=false)

# ╔═╡ 175ec879-2c34-477a-9359-38f8f9992b72
Gadfly.plot(
	filter(x -> 
		x.date > DateTime("2020-09-15T22:58:27") &&
		x.council_region > 45, VV_councilmedian),
	x=:date,
	y=:council_region,
	color=:VH,
	# Geom.point,
	Geom.rectbin
)

# ╔═╡ 217c69fd-380b-4240-8078-68a54e8eafde
describe(sar_r, :nmissing)

# ╔═╡ c65e56e9-0bde-4278-819c-f3148d71668b
begin
	# the _r prefix is meant to denote that these are closer to "raw" data
	era5_r = CSV.read(era5_p, DataFrame; dateformat=date_f)
	# landsat8_r = CSV.read(landsat8_p, DataFrame; dateformat=date_f)
	lst_aqua_r = CSV.read(lst_aqua_p, DataFrame; dateformat=date_f)
	lst_terra_r = CSV.read(lst_terra_p, DataFrame; dateformat=date_f)
	lst_r = vcat(lst_aqua_r, lst_terra_r)
	
	noaa_r = CSV.read(noaa_p, DataFrame; dateformat=date_f)
	# sentinel_1C_r = CSV.read(sentinel_1C_p, DataFrame; dateformat=date_f)
	# sentinel_2A_r = CSV.read(sentinel_2A_p, DataFrame; dateformat=date_f)
	# viirs_r = CSV.read(viirs_p, DataFrame; dateformat=date_f)
end;

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
	# strip_month!(landsat8_r)
	strip_month!(lst_r)
	strip_month!(noaa_r)
	# strip_month!(sentinel_1C_r)
	# strip_month!(sentinel_2A_r)
	# strip_month!(viirs_r)
	strip_month!(sar_r)
end;

# ╔═╡ 22494217-9254-4374-8a7d-02528bdd0df3
strip_month!(dynam_r)

# ╔═╡ e87d641a-9555-4a93-9fe8-f39f8964ce84


# ╔═╡ ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
function monthly_average(data::DataFrame, agg_terms::Vector{String})
	meanterms = filter(term -> term ∉ agg_terms, names(data))
	combine(groupby(data, agg_terms), meanterms .=> mean ∘ skipmissing, renamecols=false)
end

# ╔═╡ 637220ba-c76a-4210-8c08-fde56b86366a
begin
	agg_terms = 	["Property Id","date"]

	epw = 			monthly_average(epw_r, agg_terms)

	era5 = 			monthly_average(era5_r, agg_terms)	
	# landsat8 = 		monthly_average(landsat8_r, agg_terms)
	lst = 			monthly_average(lst_r, agg_terms)
	noaa = 			monthly_average(noaa_r, agg_terms)
	# sentinel_1C = 	monthly_average(sentinel_1C_r, agg_terms)
	# sentinel_2A = 	monthly_average(sentinel_2A_r, agg_terms)
	# viirs = 		monthly_average(viirs_r, agg_terms)
	sar = 			monthly_average(sar_r, agg_terms)
	dynam = 		monthly_average(dynam_r, agg_terms)
end;

# ╔═╡ ee7cd99c-88a3-43d7-8fe6-02285598bd1e
names(dynam)

# ╔═╡ 09a2d2d3-a468-4760-83de-8c423d4e962b
Gadfly.plot(
	filter( x -> x["Property Id"] == 7365, dynam),
	x=:date,
	y=:trees,
	Geom.point,
	Geom.line,
	Theme(default_color="forestgreen")
)

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
end

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

# ╔═╡ 217bfbed-1e13-4147-b4e4-c58eaed29382
md"""
#### Now jumping into the training process
"""

# ╔═╡ 61238caa-0247-4cf6-8fb7-3b9db84afcee
md"""
Base model - this m₁ term will be used for almost all of the anlaysis
"""

# ╔═╡ 0bc7b14b-ccaf-48f6-90fa-3006737727ed
# rng = MersenneTwister(100);

# ╔═╡ 58b4d31f-5340-4cd9-8c9e-6c504479897f
EvoTree = @load EvoTreeRegressor pkg=EvoTrees verbosity=0

# ╔═╡ 850ddd0b-f6ea-4743-99ca-720d9ac538a0
Tree = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# ╔═╡ 90227375-6661-4696-8abd-9562e08040bf
"Invalid loss: EvoTrees.L1(). Only [`:linear`, `:logistic`, `:L1`, `:quantile`] are supported at the moment by EvoTreeRegressor.";

# ╔═╡ ea1a797c-da0a-4133-bde5-366607964754
m = Tree(rng=100, max_depth=20, n_trees=30)

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
	noaa,
	era5,
	epw, 
	lst, 
	# sentinel_1C, 
	# viirs
];

# ╔═╡ 0c89dbf1-c714-43da-be89-3f82ccc2373a
# here is where we want to inject all of the data and drop missing terms
tₐ′ = dropmissing(clean(innerjoin(
	tₐ,
	comprehensive_datalist...,
	on=["Property Id", "date"]
)));

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

# ╔═╡ 748321c3-0956-4439-90d4-e74598d83f20
term₀ = unique([electricity_terms...])

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

# ╔═╡ e4ccf0ac-0b82-4462-bbb3-9fc1ae09dc2b
begin
tₐ′₁ = select(tₐ′, term₁)
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

# ╔═╡ 87a6d9a5-7957-44c4-8592-3eef5b945782
begin
tₐ′₂ = select(tₐ′, term₂)
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

# ╔═╡ 763edcce-0696-4670-a8cf-4963cfe70975
term₃ = unique([names(lst)..., electricity_terms...])

# ╔═╡ 8a337442-2f6b-4381-902b-93c52f6d8981
begin
tₐ′₃ = select(tₐ′, term₃)
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
tₐ′₄ = select(tₐ′, term₄)
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
# md"""
# ###### Sentinel-2
# """

# # ╔═╡ e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
# term₅ = unique([names(sentinel_1C)..., electricity_terms...])

# # ╔═╡ 358c345c-108e-4bdf-8a79-9c704b529ce3
# begin
# tₐ′₅ = select(tₐ′, term₅)
# m₅ = training_pipeline(
# 	select(tₐ′₅, exclusion_terms),
# 	"electricity_mwh",
# 	m
# );
# end

# # ╔═╡ ad27236b-945e-4ef9-a056-968f5bb9fa93
# begin
# vₐ′₅ = select(vₐ′, term₅)

# vₐ′₅.prediction = validation_pipeline(
# 	select(vₐ′₅, exclusion_terms),
# 	"electricity_mwh",
# 	m₅
# );

# vₐ′₅.recorded = vₐ′₅.electricity_mwh
# vₐ′₅.model = repeat(["Sentinel-2"], nrow(vₐ′₅))
# end;

# # ╔═╡ 156ddaf3-c417-49f9-ab9b-382ca47031de
# begin
# teₐ′₅ = select(teₐ′, term₅)

# teₐ′₅.prediction = validation_pipeline(
# 	select(teₐ′₅, exclusion_terms),
# 	"electricity_mwh",
# 	m₅
# );

# teₐ′₅.recorded = teₐ′₅.electricity_mwh
# teₐ′₅.model = repeat(["Sentinel-2"], nrow(teₐ′₅))
# end;

# # ╔═╡ a6cada88-c7c9-495d-8806-2503e674ec39
# md"""
# ###### VIIRS
# """

# # ╔═╡ 7c84422b-d522-4f11-9465-058f41a4266f
# term₆ = unique([names(viirs)..., electricity_terms...])

# # ╔═╡ 9666648a-42e2-4237-a4a4-71f0d5abf46c
# begin
# tₐ′₆ = select(tₐ′, term₆)
# m₆ = training_pipeline(
# 	select(tₐ′₆, exclusion_terms),
# 	"electricity_mwh",
# 	m
# );
# end

# # ╔═╡ c5b5b147-22ec-432b-ab20-111f6a759101
# begin
# vₐ′₆ = select(vₐ′, term₆)

# vₐ′₆.prediction = validation_pipeline(
# 	select(vₐ′₆, exclusion_terms),
# 	"electricity_mwh",
# 	m₆
# );

# vₐ′₆.recorded = vₐ′₆.electricity_mwh
# vₐ′₆.model = repeat(["VIIRS"], nrow(vₐ′₆))
# end;

# # ╔═╡ 6066527e-c75a-4305-8a34-0cc98c4b3a91
# begin
# teₐ′₆ = select(teₐ′, term₆)

# teₐ′₆.prediction = validation_pipeline(
# 	select(teₐ′₆, exclusion_terms),
# 	"electricity_mwh",
# 	m₆
# );

# teₐ′₆.recorded = teₐ′₆.electricity_mwh
# teₐ′₆.model = repeat(["VIIRS"], nrow(teₐ′₆))
# end;

# ╔═╡ 24f60a4e-7c69-4ff9-b284-eb9418b7a496
test_terms = [teₐ′₀,teₐ′₁,teₐ′₂,teₐ′₃,teₐ′₄] #,teₐ′₅];

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

# ╔═╡ c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
begin
# Null
Ξ₀ = hcat(select(teₐ′₀, prediction_terms), select(teₐ′, interest_terms))

#MODIS
Ξ₃ = hcat(select(teₐ′₃, prediction_terms), select(teₐ′, interest_terms))
	
#EPW
Ξ₄ = hcat(select(teₐ′₄, prediction_terms), select(teₐ′, interest_terms))
	
Ξ = vcat(Ξ₀, Ξ₃, Ξ₄)

Ξ.error = Ξ.prediction .- Ξ.recorded
Ξ′ = combine(groupby(Ξ, [:month, :model]), :error => mean, renamecols=false)

p₄ = Gadfly.plot(
	Ξ′,
	x=:month,
	y=:error,
	color=map(x-> seasons[x], Ξ′.month),
	ygroup=:model,
	yintercept=[0],
	Guide.ylabel("Δ Prediction - Recorded"),
	Guide.xlabel("Month"),
	Guide.title("Mean Electricity Error by Season"),
	Guide.colorkey(title="Season"),
	Geom.subplot_grid(
		Guide.yticks(ticks=-0.005:0.005:0.005),
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

# ╔═╡ 187a03ba-c9a4-47cf-be7e-ae024d1fff72
draw(PNG(joinpath(output_dir, "seasonal_accuracy.png"), 14cm, 10cm, dpi=600), p₄)

# ╔═╡ 11277ca6-3bbf-403f-8cfc-2aabf94069f7
p₇ = Gadfly.plot(
	test_termsₑ,
	color=:model,
	x=:weather_station_distance,
	y=abs.(test_termsₑ.prediction .- test_termsₑ.recorded),
	Geom.smooth(smoothing=0.6),
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Electricity - MAE vs Weather Station Distance")
)

# ╔═╡ 28d2660e-258d-4583-8224-c2b4190f4140
draw(PNG(joinpath(output_dir, "learning_results_electricity.png"), 20cm, 10cm, dpi=600), p₇)

# ╔═╡ 211723e6-5ce8-4ff1-a2f6-4751df955989
test_termsₑ

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


# ╔═╡ b98baad1-dad8-464c-a18a-1baf01962164
md"""
#### Natural Gas
"""

# ╔═╡ 3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
md"""Null Model"""

# ╔═╡ 9c8f603f-33c6-4988-9efd-83864e871907
termᵧ₀ = unique([naturalgas_terms...])

# ╔═╡ fa1e71e4-4a53-436a-9678-8c058f0bf474
begin
tᵧ′₀ = select(tᵧ′, termᵧ₀)
mᵧ₀ = training_pipeline(
	select(tₐ′₀, exclusion_terms),
	"electricity_mwh",
	m
);
end

# ╔═╡ a70b0eb5-e246-4db1-865f-3952319287f9
begin
vᵧ′₀ = select(vᵧ′, termᵧ₀)

vᵧ′₀.prediction = validation_pipeline(
	select(vᵧ′₀, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₀
);

vᵧ′₀.recorded = vᵧ′₀.naturalgas_mwh
vᵧ′₀.model = repeat(["Null"], nrow(vᵧ′₀))
end;

# ╔═╡ 607d1798-14c6-46d3-a405-5e6c8816fdd4
begin
teᵧ′₀ = select(teᵧ′, termᵧ₀)

teᵧ′₀.prediction = validation_pipeline(
	select(teᵧ′₀, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₀
);

teᵧ′₀.recorded = teᵧ′₀.naturalgas_mwh
teᵧ′₀.model = repeat(["Null"], nrow(teᵧ′₀))
end;

# ╔═╡ be35f0e7-e40f-4485-bd71-ec589be309ac
md"""ERA5"""

# ╔═╡ 7a78fad4-b5cc-4964-b2e5-1338be6b0c9d
termᵧ₁ = unique([names(era5)..., naturalgas_terms...])

# ╔═╡ 2e4893c8-f86e-41ab-99ef-76adeaeb3040
begin
tᵧ′₁ = select(tᵧ′, termᵧ₁)
mᵧ₁ = training_pipeline(
	select(tᵧ′₁ , exclusion_terms),
	"naturalgas_mwh",
	m
);

vᵧ′₁ = select(vᵧ′, termᵧ₁)
vᵧ′₁.prediction = validation_pipeline(
	select(vᵧ′₁, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₁
);

vᵧ′₁.recorded = vᵧ′₁.naturalgas_mwh
vᵧ′₁.model = repeat(["ERA5"], nrow(vᵧ′₁))

teᵧ′₁ = select(teᵧ′, termᵧ₁)
teᵧ′₁.prediction = validation_pipeline(
	select(teᵧ′₁, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₁
);

teᵧ′₁.recorded = teᵧ′₁.naturalgas_mwh
teᵧ′₁.model = repeat(["ERA5"], nrow(teᵧ′₁))
end;

# ╔═╡ 0e876bab-6648-4a67-b571-dc82a7bdf8f1
md"""##### NOAA"""

# ╔═╡ 3763ad63-003e-495f-aa90-0db525412c62
termᵧ₂ = unique([names(noaa)..., naturalgas_terms...])

# ╔═╡ f2e8dbb5-75f1-4e6c-b11a-5622992de4e7
begin
tᵧ′₂ = select(tᵧ′, termᵧ₂)
mᵧ₂ = training_pipeline(
	select(tᵧ′₂, exclusion_terms),
	"naturalgas_mwh",
	m
);

vᵧ′₂ = select(vᵧ′, termᵧ₂)
vᵧ′₂.prediction = validation_pipeline(
	select(vᵧ′₂, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₂
);

vᵧ′₂.recorded = vᵧ′₂.naturalgas_mwh
vᵧ′₂.model = repeat(["NOAA"], nrow(vᵧ′₂))

teᵧ′₂ = select(teᵧ′, termᵧ₂)
teᵧ′₂.prediction = validation_pipeline(
	select(teᵧ′₂, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₂
);

teᵧ′₂.recorded = teᵧ′₂.naturalgas_mwh
teᵧ′₂.model = repeat(["NOAA"], nrow(teᵧ′₂))
end;

# ╔═╡ 202c9072-0a7b-454d-9112-4ecc0a03c61b
md"""###### MODIS"""

# ╔═╡ 434d7e0d-583d-498b-9b6c-a72fa3775b3c
termᵧ₃ = unique([names(lst)..., naturalgas_terms...])

# ╔═╡ d7291354-9040-4039-acb2-eef807ec404e
begin
tᵧ′₃ = select(tᵧ′, termᵧ₃)
mᵧ₃ = training_pipeline(
	select(tᵧ′₃, exclusion_terms),
	"naturalgas_mwh",
	m
);

vᵧ′₃ = select(vᵧ′, termᵧ₃)
vᵧ′₃.prediction = validation_pipeline(
	select(vᵧ′₃, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₃
);

vᵧ′₃.recorded = vᵧ′₃.naturalgas_mwh
vᵧ′₃.model = repeat(["MODIS"], nrow(vᵧ′₃))

teᵧ′₃ = select(teᵧ′, termᵧ₃)
teᵧ′₃.prediction = validation_pipeline(
	select(teᵧ′₃, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₃
);

teᵧ′₃.recorded = teᵧ′₃.naturalgas_mwh
teᵧ′₃.model = repeat(["MODIS"], nrow(teᵧ′₃))
end;

# ╔═╡ a78a6fd5-4b45-43fc-a733-aef4fd14eb42
md""" ##### EPW"""

# ╔═╡ 1a4471e7-24ad-4652-9f3f-6eef92c781d5
termᵧ₄ = unique([names(epw)..., naturalgas_terms...])

# ╔═╡ 3126516f-51d9-49c0-aac6-baf14703f0bb
begin
tᵧ′₄ = select(tᵧ′, termᵧ₄)
mᵧ₄ = training_pipeline(
	select(tᵧ′₄, exclusion_terms),
	"naturalgas_mwh",
	m
);

vᵧ′₄ = select(vᵧ′, termᵧ₄)
vᵧ′₄.prediction = validation_pipeline(
	select(vᵧ′₄, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₄
);

vᵧ′₄.recorded = vᵧ′₄.naturalgas_mwh
vᵧ′₄.model = repeat(["EPW"], nrow(vᵧ′₄))

teᵧ′₄ = select(teᵧ′, termᵧ₄)
teᵧ′₄.prediction = validation_pipeline(
	select(teᵧ′₄, exclusion_terms),
	"naturalgas_mwh",
	mᵧ₄
);

teᵧ′₄.recorded = teᵧ′₄.naturalgas_mwh
teᵧ′₄.model = repeat(["EPW"], nrow(teᵧ′₄))
end;

# ╔═╡ da9a6ba8-083b-40b3-9bd6-089688ff7f73
begin
	# now want to explore how permuted weather data might influence the quality
	termᵧₒ = select(tᵧ′₄, naturalgas_terms)
	epwᵧₒ = select(tᵧ′₄, Not(naturalgas_terms))
	epwᵧₒ′ = epwᵧₒ[shuffle(1:nrow(epwᵧₒ)), :]

	tᵧₒ′₄ = hcat(termᵧₒ, epwᵧₒ′)


	termᵥᵧₒ = select(vᵧ′₄, naturalgas_terms)
	epwᵥᵧₒ = select(vᵧ′₄, Not(naturalgas_terms))
	epwᵥᵧₒ′ = epwᵥᵧₒ[shuffle(1:nrow(epwᵥᵧₒ)), :]

	vᵧₒ′₄ = hcat(termᵥᵧₒ, epwᵥᵧₒ′)
end;

# ╔═╡ 433d51c9-9f31-49db-a5a2-e6565888831e


# ╔═╡ 59144efe-6073-4d1b-b976-b25d6bdd15e0
epwᵧₒ[shuffle(1:nrow(epwᵧₒ)), :]

# ╔═╡ c25b404b-0d89-4c4d-bc61-c361fd9d7038
test_termsᵧ = [teᵧ′₀,teᵧ′₁,teᵧ′₃,teᵧ′₄];

# ╔═╡ f2067017-32c8-493b-a9fb-3a89f9e549e4
test_termsₑᵧ = vcat(
	[ select(x, prediction_terms, interest_terms, "Property Id", "area") for x in test_termsᵧ]...
);

# ╔═╡ ee0d413c-a107-4bdc-a08a-52cda6d81573
test_termsₑᵧ

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

Γ = vcat(Γ₁, Γ₃, Γ₄)

Γ.error = Γ.prediction .- Γ.recorded
Γ′ = combine(groupby(Γ, [:month, :model]), :error => mean, renamecols=false)

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
		Guide.yticks(ticks=-0.0025:0.0025:0.015),
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

# ╔═╡ a1e9f1c1-6764-4d03-a14f-a362c0fba808
draw(PNG(joinpath(output_dir, "naturalgas_seasonal.png"), 14cm, 10cm, dpi=600), p₈)

# ╔═╡ 08b888e3-9d37-43a6-9656-4bf6cb62d324
building_distancesᵧ = unique(select(teᵧ, ["Property Id", "weather_station_distance"]));

# ╔═╡ 9da9c2e2-1932-4fb4-9541-9c022b2680b2
test[:,"Property Id"]

# ╔═╡ 66bd7173-15a4-436a-ad63-fe94de054863
# Gadfly.plot(
# 	leftjoin(stack(test_suiteₜᵧđ, 2:4), select(test, ["area","Property Id"] ), on="Property Id"),
# 	x=:area,
# 	y=:value,
# 	color=:model,
# 	ygroup=:variable,
# 	Geom.subplot_grid(
# 		Geom.smooth(smoothing=0.08),
# 		free_y_axis=true
# 	),
# )

# ╔═╡ f3d171bd-1d0f-4657-b243-b499092eaaf3


# ╔═╡ 459dbe8b-08f7-4bde-ba6d-ee4d05d1836f
pᵧ₁ = Gadfly.plot(
	test_termsₑᵧ,
	Gadfly.layer(
		color=:model,
		x=:weather_station_distance,
		y=abs.(test_termsₑᵧ.prediction .- test_termsₑᵧ.recorded),
		Geom.smooth(smoothing=0.75),
	),
	Guide.ylabel("Mean Absolute Error - Smoothed"),
	Guide.xlabel("Distance from Weather Station (m)"),
	Guide.title("Natural Gas - MAE vs Weather Station Distance")
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
default_colors = cgrad(:redblue, 5, categorical = true, rev=false);

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
	)
	end

	results.model = repeat([Ẋ[1,"model"]], nrow(results))
	return results
end

# ╔═╡ be217ff7-856e-4919-bcd8-5d2dff20944a
test_suiteₜₑ = vcat([ test_suite(x) for x in test_terms ]...);

# ╔═╡ 734ed2d0-4657-4361-9285-97605371af72
test_suiteₜᵧ = vcat([ test_suite(x) for x in test_termsᵧ ]...);

# ╔═╡ d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
combine(groupby(clean(test_suiteₜᵧ), ["model"]), [:cvrmse,:nmbe,:cvstd] .=> mean, renamecols=false)

# ╔═╡ bf71086c-a8f0-4420-b698-73499fec5257
test_suiteₜᵧđ = leftjoin(
	test_suiteₜᵧ,
	building_distancesᵧ,
	on="Property Id"
);

# ╔═╡ 669a8a80-9bff-432d-b227-cd81ea90cb01
stack(test_suiteₜᵧđ, 2:4)

# ╔═╡ 4c99c655-ae95-4a28-95c8-e7ca38ddf55f
Gadfly.plot(
	stack(test_suiteₜᵧđ, 2:4),
	x=:weather_station_distance,
	y=:value,
	color=:model,
	ygroup=:variable,
	Geom.subplot_grid(
		Geom.smooth(smoothing=0.08),
		free_y_axis=true
	),
)

# ╔═╡ Cell order:
# ╠═982250e8-58ad-483d-87b5-f6aff464bd10
# ╠═ac97e0d6-2cfa-11ed-05b5-13b524a094e3
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
# ╠═a2e624f7-5626-431a-9680-f62ed86b61aa
# ╠═db7c092f-5fa8-4038-b9cf-d40d822a4b9a
# ╠═d3d814ee-ad4f-47bf-966a-08cabc79bf90
# ╠═00acb065-2378-4181-b76a-488071f43a7e
# ╠═7d13349f-c3a4-40e7-b7d3-be2e0e951dd5
# ╠═44d399f7-268b-4e66-8ba9-fe2e8e17a19d
# ╠═fe529a32-ab71-4e5d-a593-45085d69f580
# ╠═a536094d-3894-4ce7-95cd-f38a3666e07e
# ╠═175ec879-2c34-477a-9359-38f8f9992b72
# ╠═217c69fd-380b-4240-8078-68a54e8eafde
# ╠═c65e56e9-0bde-4278-819c-f3148d71668b
# ╠═348c4307-94dc-4d5f-82b0-77dc535c1650
# ╠═09a4789c-cbe7-496e-98b5-a2c2db3102b6
# ╠═22494217-9254-4374-8a7d-02528bdd0df3
# ╠═e87d641a-9555-4a93-9fe8-f39f8964ce84
# ╠═ac31e0ac-b35b-494f-814c-3f9eaf26e8b1
# ╠═637220ba-c76a-4210-8c08-fde56b86366a
# ╠═ee7cd99c-88a3-43d7-8fe6-02285598bd1e
# ╠═09a2d2d3-a468-4760-83de-8c423d4e962b
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
# ╟─c6f4308d-6dce-4a57-b411-6327f4aa87a7
# ╠═c6c038f0-cf8b-4234-a64c-75616fdc07a5
# ╠═1a5ab262-0493-470f-ab58-baa5fa1a69af
# ╠═b5dff4f2-3088-4301-8dfe-96fb8c6999c7
# ╠═df3549d2-4183-4690-9e04-b665a9286792
# ╠═e6ea0a65-b79f-400f-be61-951b08b5ce88
# ╠═77aad1ea-16cf-4c5f-9a6b-345f7168afb3
# ╟─217bfbed-1e13-4147-b4e4-c58eaed29382
# ╟─61238caa-0247-4cf6-8fb7-3b9db84afcee
# ╠═0bc7b14b-ccaf-48f6-90fa-3006737727ed
# ╠═58b4d31f-5340-4cd9-8c9e-6c504479897f
# ╠═850ddd0b-f6ea-4743-99ca-720d9ac538a0
# ╠═90227375-6661-4696-8abd-9562e08040bf
# ╠═ea1a797c-da0a-4133-bde5-366607964754
# ╠═d8531d5a-e843-45e5-a498-0891d057d393
# ╟─bb078875-611c-46f6-8631-1befde358054
# ╠═b6d7cd90-0d59-4194-91c8-6d8f40a4a9c3
# ╟─df753575-b121-4c3b-a456-cfe4e535c2aa
# ╠═a157969b-100a-4794-a78e-2f40439e28d9
# ╠═0c89dbf1-c714-43da-be89-3f82ccc2373a
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
# ╠═748321c3-0956-4439-90d4-e74598d83f20
# ╠═844b35d3-6ff2-42bd-9fbe-d20cfd9d856f
# ╠═43f847df-5247-48b5-8310-986dc7ccb60d
# ╠═f76ee669-8dd1-42ae-aa30-3cb0d1541aa3
# ╟─dee3a9ec-79f5-4917-ad40-2e4ddcdd423d
# ╠═ad174123-117e-4365-9ede-50456d445fce
# ╠═e4ccf0ac-0b82-4462-bbb3-9fc1ae09dc2b
# ╠═99280172-b38b-4598-99a8-e091a72ae52c
# ╠═a57efaa9-89c5-46a0-b310-892890068561
# ╟─ee57d5d5-545e-4b70-91d9-b82a108f854b
# ╠═ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
# ╠═87a6d9a5-7957-44c4-8592-3eef5b945782
# ╠═eb2c58ed-6388-47a5-a6a4-c516acc9a4bd
# ╠═c0e28ffb-b49f-4b2d-919e-f63076cd8485
# ╠═6958cff8-3c8d-4a77-bbff-f8cb17afd632
# ╟─51ecb564-06d5-4767-aa41-3030ca08a6c7
# ╠═763edcce-0696-4670-a8cf-4963cfe70975
# ╠═8a337442-2f6b-4381-902b-93c52f6d8981
# ╠═d8e9d7bc-2b19-46d0-b918-554ecc003924
# ╠═8b7c57c8-a525-432f-b51a-9d14196f30be
# ╟─cde0836c-3dbd-43a9-90fd-c30e5985acf7
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
# ╠═24f60a4e-7c69-4ff9-b284-eb9418b7a496
# ╠═be217ff7-856e-4919-bcd8-5d2dff20944a
# ╠═ed696113-737b-46f5-bfb1-c06d430a83ac
# ╠═8b3175d4-6938-4823-8a91-77cde7c31c2a
# ╠═9d1ea09f-3c42-437f-afbc-21a70296a3ba
# ╠═05a040cf-f6ac-42b2-bcb1-599af5a26038
# ╠═c1e6a0a2-5252-413a-adf9-fa316d0a6b0a
# ╠═187a03ba-c9a4-47cf-be7e-ae024d1fff72
# ╠═11277ca6-3bbf-403f-8cfc-2aabf94069f7
# ╠═28d2660e-258d-4583-8224-c2b4190f4140
# ╠═211723e6-5ce8-4ff1-a2f6-4751df955989
# ╠═fbe43a2a-f9b0-487e-a579-645c2f40736e
# ╠═7b05ec74-6edc-4f44-bf9d-9389d4029494
# ╠═c546d3af-8f24-40bd-abfa-e06c708c244e
# ╟─b98baad1-dad8-464c-a18a-1baf01962164
# ╟─3a8209d5-30fb-4a6e-8e0f-b46ebc9f8611
# ╠═9c8f603f-33c6-4988-9efd-83864e871907
# ╠═fa1e71e4-4a53-436a-9678-8c058f0bf474
# ╠═a70b0eb5-e246-4db1-865f-3952319287f9
# ╠═607d1798-14c6-46d3-a405-5e6c8816fdd4
# ╠═be35f0e7-e40f-4485-bd71-ec589be309ac
# ╠═7a78fad4-b5cc-4964-b2e5-1338be6b0c9d
# ╠═2e4893c8-f86e-41ab-99ef-76adeaeb3040
# ╟─0e876bab-6648-4a67-b571-dc82a7bdf8f1
# ╠═3763ad63-003e-495f-aa90-0db525412c62
# ╠═f2e8dbb5-75f1-4e6c-b11a-5622992de4e7
# ╟─202c9072-0a7b-454d-9112-4ecc0a03c61b
# ╠═434d7e0d-583d-498b-9b6c-a72fa3775b3c
# ╠═d7291354-9040-4039-acb2-eef807ec404e
# ╟─a78a6fd5-4b45-43fc-a733-aef4fd14eb42
# ╠═1a4471e7-24ad-4652-9f3f-6eef92c781d5
# ╠═3126516f-51d9-49c0-aac6-baf14703f0bb
# ╠═da9a6ba8-083b-40b3-9bd6-089688ff7f73
# ╠═433d51c9-9f31-49db-a5a2-e6565888831e
# ╠═59144efe-6073-4d1b-b976-b25d6bdd15e0
# ╠═c25b404b-0d89-4c4d-bc61-c361fd9d7038
# ╠═734ed2d0-4657-4361-9285-97605371af72
# ╠═d1eaa0ee-8c7d-4b1b-a3c6-030fffd320c4
# ╠═f2067017-32c8-493b-a9fb-3a89f9e549e4
# ╠═ee0d413c-a107-4bdc-a08a-52cda6d81573
# ╠═77456db6-b0fa-4ba6-9d41-1c393f7ddee1
# ╠═21e706e9-3ce3-4c9b-b279-7abc7b9f8c94
# ╠═a1e9f1c1-6764-4d03-a14f-a362c0fba808
# ╠═08b888e3-9d37-43a6-9656-4bf6cb62d324
# ╠═bf71086c-a8f0-4420-b698-73499fec5257
# ╠═669a8a80-9bff-432d-b227-cd81ea90cb01
# ╠═4c99c655-ae95-4a28-95c8-e7ca38ddf55f
# ╠═9da9c2e2-1932-4fb4-9541-9c022b2680b2
# ╠═66bd7173-15a4-436a-ad63-fe94de054863
# ╠═f3d171bd-1d0f-4657-b243-b499092eaaf3
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
