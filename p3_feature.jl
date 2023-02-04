using Markdown
using InteractiveUtils

# ╔═╡ ac97e0d6-2cfa-11ed-05b5-13b524a094e3
import Pkg
Pkg.activate(Base.current_project())

using Cairo
using Fontconfig

using CSV
using DataFrames
using DataStructures
using Dates
using Formatting
using Gadfly
using Statistics
using YAML

using ColorSchemes
using MLJ
using MLJParticleSwarmOptimization

using Latexify
using LinearAlgebra
using Random
using EvoTrees
using StatsBase
using JSON
using ComputationalResources

Random.seed!(1)
rng = MersenneTwister(1);

## utility function
function commas(num::Integer)
    str = string(num)
    return replace(str, r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")
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
sources_file = joinpath(pwd(), "sources.yml")
sources = YAML.load_file(sources_file)
data_destination = sources["output-destination"]
data_path = joinpath(data_destination, "data", "nyc")

input_dir = joinpath(data_path, "p1_o")
input_dir_environmental = joinpath(data_path, "p2_o")

energydata = CSV.read(joinpath(input_dir, "alldata.csv"), DataFrame)
energydata.date = Date.(energydata.date)

select!(energydata, Not([
	:council_region,
	:electricity_kbtu,
	:naturalgas_kbtu
]))

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
	rng,
	unique_buildings, 
	trunc(Int, length(unique_buildings) * 0.20)
);

# ╔═╡ a5ad08d5-08ee-46d9-b2fd-af8ee904e766
@info "Number of Test Buildings:" length(test_buildings)

# ╔═╡ 0b9c6880-9912-4268-9245-4cc8c7cad66f
train_buildings = setdiff(unique_buildings, test_buildings);

# ╔═╡ af422ea0-0f62-4233-a28d-d96a46009a14
@info "Number of Train Buildings:" length(train_buildings)

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

building_zones = DataFrames.select(unique(vcat(train,test), Symbol("Property Id")), [Symbol("Property Id"),:zone]);
testcouncils = DataFrames.select(test, ["Property Id","zipcode"]);

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

epw_p = joinpath(input_dir, "epw.csv")
# building the paths for all of the environmental features

cmip_p = joinpath(input_dir_environmental, "cmip.csv")
cmip_r = CSV.read(cmip_p, DataFrame; dateformat=date_f);

era5_p = joinpath(input_dir_environmental, "era5.csv")
landsat8_p = joinpath(input_dir_environmental, "landsat8.csv")
# lst_aqua_p = joinpath(input_dir_environmental, "lst_aqua.csv")
# lst_terra_p = joinpath(input_dir_environmental, "lst_terra.csv")
noaa_p = joinpath(input_dir_environmental, "noaa.csv")
sentinel_1C_p = joinpath(input_dir_environmental, "sentinel_1C.csv")
# sentinel_2A_p = joinpath(input_dir_environmental, "sentinel_2A.csv")
viirs_p = joinpath(input_dir_environmental, "viirs.csv")

# ╔═╡ 8883d4ac-9ec4-40b5-a885-e1f3c5cbd4b9
epw_r = CSV.read(epw_p, DataFrame);
epw_sample = CSV.read(joinpath(input_dir, "epw_sample.csv"), DataFrame);
property_map = CSV.read(joinpath(input_dir, "property_mapping.csv"), DataFrame);
dynam_p = joinpath(input_dir_environmental, "dynamicworld.csv")
dynam_r = CSV.read(dynam_p, DataFrame; dateformat=date_f)

sar_p = joinpath(input_dir_environmental, "sar.csv")
sar_r = CSV.read(sar_p, DataFrame; dateformat=date_f);
sar_ids = unique(sar_r[:,"Property Id"])[400:800]
sar_sample = dropmissing(filter( x -> x["Property Id"] ∈ sar_ids, sar_r ), :VV);
sar_sample.month = Dates.month.(sar_sample.date);
group_sar = combine(groupby(sar_sample, "Property Id"), [:VV,:VH] .=> mean, renamecols=false)

# ╔═╡ 067fc936-5eac-4082-80f8-c50f194f1721
era5_r = CSV.read(era5_p, DataFrame; dateformat=date_f);
landsat8_r = CSV.read(landsat8_p, DataFrame; dateformat=date_f);
landsat8_r.ST_B10 = landsat8_r.ST_B10 .* 0.00341802 .+ 149.0 .- 273.15


# # ╔═╡ 4df4b299-b9ee-46a2-9622-37c430e867a1
# lst_aqua_r = CSV.read(lst_aqua_p, DataFrame; dateformat=date_f);

# # ╔═╡ fdd41dc5-1439-458c-ad41-3d10f3a8478f
# lst_terra_r = CSV.read(lst_terra_p, DataFrame; dateformat=date_f);

# # ╔═╡ 5e26802a-0a84-4dc1-926d-d51ac589dc5e
# begin
# lst_r = vcat(lst_aqua_r, lst_terra_r);
# lst_r[!,"LST_Day_1km"] = lst_r[:,"LST_Day_1km"] .* 0.02 .-273.15
# lst_r[!,"LST_Night_1km"] = lst_r[:,"LST_Night_1km"] .* 0.02 .-273.15
# end;

# # ╔═╡ 65c72331-c36e-4e15-b530-13069b8cc070
# lst_r

# ╔═╡ cc20d207-016e-408c-baf1-83d68c4c0fde
noaa_r = DataFrames.select(CSV.read(noaa_p, DataFrame; dateformat=date_f), Not(:ACPC01));

# renaming this because landsat also has an ndvi metric
sentinel_1C_r = rename!(CSV.read(sentinel_1C_p, DataFrame; dateformat=date_f), :NDVI => :NDVI_S);

# # ╔═╡ a9f2d94d-cbf8-4d47-a4b2-438f451882e5
# sentinel_2A_r = CSV.read(sentinel_2A_p, DataFrame; dateformat=date_f);

# ╔═╡ ca12dd08-29af-4ce3-a2cc-d3bf1fa9e3c7
viirs_r = CSV.read(viirs_p, DataFrame; dateformat=date_f)

# ╔═╡ 348c4307-94dc-4d5f-82b0-77dc535c1650
function strip_month!(data::DataFrame)
	data[!,"date"] = Date.(Dates.Year.(data.date), Dates.Month.(data.date))
end

# also want to get the building data in a uniform format for matching
strip_month!(train)
# strip_month!(validate)
strip_month!(test)

strip_month!(epw_r)

# strip_month!(era5_r)
strip_month!(landsat8_r)
strip_month!(cmip_r)
strip_month!(noaa_r)
strip_month!(sentinel_1C_r)
# strip_month!(sentinel_2A_r)
strip_month!(viirs_r)
strip_month!(sar_r)
strip_month!(dynam_r)

# ╔═╡ dcde8c56-7294-47fd-aad1-2204de6c904b
function skip_function(values)
	valuearr::Vector{Any} = (collect∘skipmissing)(values)
	return valuearr
end

# ╔═╡ e87d641a-9555-4a93-9fe8-f39f8964ce84
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

# for the comprehensive case
# functional_terms = [f₁ f₃ f₅]

# for the explainable case
functional_terms = f₃

# ╔═╡ 3da877c2-159b-4d0d-8b97-34da4dbf2ac3
cmip = 			monthly_aggregation(cmip_r, agg_terms, functional_terms);
epw = 			monthly_aggregation(epw_r, agg_terms, functional_terms);
era5 = 			monthly_aggregation(era5_r, agg_terms, functional_terms);	
landsat8 = 		monthly_aggregation(landsat8_r, agg_terms, functional_terms);
# lst = 			monthly_aggregation(lst_r, agg_terms, functional_terms);
noaa = 			monthly_aggregation(noaa_r, agg_terms, functional_terms);
sentinel_1C = 	monthly_aggregation(sentinel_1C_r, agg_terms, functional_terms);
# sentinel_2A = 	monthly_aggregation(sentinel_2A_r, agg_terms, functional_terms);
viirs = 		monthly_aggregation(viirs_r, agg_terms, functional_terms);
sar = 			monthly_aggregation(
					DataFrames.select(sar_r, Not([:HH,:HV])), 
					agg_terms, 
					functional_terms
);
dynam = 		monthly_aggregation(dynam_r, agg_terms, functional_terms);
sample_sar = innerjoin(train, sar, on=["Property Id", "date"])

# ╔═╡ 3b980465-9b75-404d-a41f-06ad351d12ae
@info "Training data points prior to merge" nrow(train)

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
data_clean = clean(dropmissing(data_augmented))

X = DataFrames.select(data_augmented, Not(join_on))
aux = DataFrames.select(data_augmented, join_on)

return X, aux
end

begin
tₐ = clean(dropmissing(DataFrames.select(train, electricity_terms)))
tₐ.month = coerce(tₐ.month, OrderedFactor)
tₐ.electricity_mwh = tₐ.electricity_mwh ./ Dates.daysinmonth.(tₐ.date)
@info "tₐ dates" length(unique(tₐ.date))

tᵧ = clean(dropmissing(DataFrames.select(train, naturalgas_terms)))
tᵧ.month = coerce(tᵧ.month, OrderedFactor)
tᵧ.naturalgas_mwh = tᵧ.naturalgas_mwh ./ Dates.daysinmonth.(tᵧ.date)
@info "tᵧ dates" length(unique(tᵧ.date))

end;

begin
teₐ = clean(dropmissing(DataFrames.select(test, electricity_terms)))
teₐ.month = coerce(teₐ.month, OrderedFactor)
teₐ.electricity_mwh = teₐ.electricity_mwh ./ Dates.daysinmonth.(teₐ.date)
@info "teₐ dates" length(unique(teₐ.date))

teᵧ = clean(dropmissing(DataFrames.select(test, naturalgas_terms)))
teᵧ.month = coerce(teᵧ.month, OrderedFactor)
teᵧ.naturalgas_mwh = teᵧ.naturalgas_mwh ./ Dates.daysinmonth.(teᵧ.date)
@info "teᵧ dates" length(unique(teᵧ.date))

end;

dataₑ = vcat(tₐ, teₐ);
dataᵧ = vcat(tᵧ, teᵧ);

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
	Guide.ylabel("Avg. Daily Energy - MWh"),
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


# ╔═╡ 055a08c0-abc9-465b-bb64-24744e87da5d
# sar′ = dropmissing(clean(innerjoin(
# 	tₐ,
# 	sar, 
# 	lst,
# 	on=["Property Id", "date"]
# )));

# ╔═╡ 4947232e-568d-4724-8ebd-5f000bdc9483
# names(sar′, Union{Real, Missing})

# ╔═╡ c9e49960-f659-432b-9338-8390e41b447a
selected_weather_station = 725033
weatherstation_buildings = filter( x -> x.weather_station_id == selected_weather_station, property_map)[:,"Property Id"]
sample_sars = sample(rng, filter( x -> x ∈ weatherstation_buildings, unique(landsat8[:,"Property Id"])), 100)
sample_filtered = filter( x-> x["Property Id"] in sample_sars, landsat8_r );
dropmissing!(sample_filtered, :ST_B10)

single_building = filter( x-> x["Property Id"] == 6405415, landsat8_r );
dropmissing!(single_building, :ST_B10)

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

# ╔═╡ 413cbb00-bc81-4d17-948c-76707bfb589a
t₁ = Gadfly.plot(
	Gadfly.layer(
		single_building,
		x=:date,
		y=:ST_B10,
		Geom.point,
		Geom.line,
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
		Geom.smooth(smoothing=0.15),
		Theme(line_width=0.08pt, alphas=[0.2])
	),
	Scale.discrete_color_hue,
	Guide.ylabel("Temperature °C"),
	Guide.xlabel("Date"),
	Guide.yticks(ticks=-10:5:50),
	Guide.xticks(ticks=DateTime("2018-01-1"):Month(1):DateTime("2020-12-31"), orientation=:vertical),
	Guide.title("Temperature Discrepancies near Central Park - Landsat8 vs. EPW"),
	Theme(default_color="black", key_position = :none),
	Guide.manual_color_key(
		"Legend", ["Landsat8", "EPW"], ["lightcoral", "gray"];
	)
)

# ╔═╡ 217bfbed-1e13-4147-b4e4-c58eaed29382
md"""
# Training Pipeline
"""

# ╔═╡ 61238caa-0247-4cf6-8fb7-3b9db84afcee
md"""
Base model - this m₁ term will be used for almost all of the anlaysis
"""

# ╔═╡ 0bc7b14b-ccaf-48f6-90fa-3006737727ed
rng = MersenneTwister(1);

# ╔═╡ 58b4d31f-5340-4cd9-8c9e-6c504479897f
EvoTree = @load EvoTreeRegressor pkg=EvoTrees verbosity=0

begin
loss_function = :tweedie
m_tree = EvoTree(
	loss=loss_function,
	max_depth=2,
	lambda=0.2,
	gamma=0.05,
	eta=0.175,
	nrounds=2,
	rowsample=0.95, 
	colsample=0.95,
	device="gpu"
)
end

# ╔═╡ a157969b-100a-4794-a78e-2f40439e28d9
comprehensive_datalist = [
	cmip,
	dynam,
	noaa,
	# era5,
	epw, 
	# lst,
	landsat8,
	sentinel_1C,
	sar,
	viirs
];

# ╔═╡ 9239f8dc-fd07-435f-9d88-e24bb1f6faa2
for term in comprehensive_datalist
	@info "Maintained Features in Dataset: " nrow(term)
end

joining_terms = ["Property Id", "date"]

tₐ′ = clean(dropmissing(innerjoin(
	tₐ,
	comprehensive_datalist...,
	on=joining_terms
)));

electric_train_buildings = unique(tₐ′[:,"Property Id"])
@info "Number of training buildings - electricity: " length(electric_train_buildings)

teₐ′ = clean(dropmissing(innerjoin(
	teₐ,
	comprehensive_datalist...,
	on=joining_terms
)));

electric_test_buildings = unique(teₐ′[:,"Property Id"])
@info "Number of testing buildings - electricity: " length(electric_test_buildings)


### info board
# @info "dynam dates" unique(dynam.date)
# @info "noaa dates" unique(noaa.date)
# @info "epw dates" unique(epw.date)
# @info "landsat8 dates" unique(landsat8.date)
# @info "sentinel dates" unique(sentinel_1C.date)
# @info "sar dates" unique(sar.date)
# @info "viirs dates" unique(viirs.date)

# @info "comprehensive dates " unique(innerjoin(
# 	teₐ,
# 	comprehensive_datalist...,
# 	on=["Property Id", "date"]
# ).date)

# @info "comprehensive dates - dropmissing " unique(dropmissing(innerjoin(
# 	teₐ,
# 	comprehensive_datalist...,
# 	on=["Property Id", "date"]
# )).date)

# @info "comprehensive dates - dropmissing + clean " unique(teₐ′.date)

# ╔═╡ ec365574-6e66-4284-9723-5634ba73d52a
md"""
###### Natural Gas Data Cleaning and Prep
"""

# ╔═╡ b7983d6f-2dab-4279-b7a5-222e40ce968b

# ╔═╡ ce3e2b3b-a6d6-4494-8168-665e878edae3
extra_omission_features = [
	"weather_station_distance",
	"zone",
	"zipcode",
	"month"
]

tᵧ′ = clean(dropmissing(innerjoin(
	tᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

gas_train_buildings = unique(tᵧ′[:,"Property Id"])
n_gas_training_buildings = length(gas_train_buildings)
@info "Number of training buildings - gas: " n_gas_training_buildings

teᵧ′ = clean(dropmissing(innerjoin(
	teᵧ,
	comprehensive_datalist...,
	on=joining_terms
)));

gas_test_buildings = unique(teᵧ′[:,"Property Id"])
n_gas_test_buildings = length(gas_test_buildings)
@info "Number of testing buildings - gas: " n_gas_test_buildings


# exit(86)

exclusion_terms = Not([joining_terms..., extra_omission_features...])

# ╔═╡ 3c41548c-da78-49c5-89a3-455df77bf4fa
md"""
Null model
"""

# ╔═╡ a149faa9-bee4-42a0-93fe-5adff459e0e9
## as a preliminary introduction - thinking about the overall dataset
@info "Number of data points in comprehensive set: " nrow(tₐ′)

# ╔═╡ 748321c3-0956-4439-90d4-e74598d83f20

# ╔═╡ 7f4f0941-495d-4309-bccf-da82aab641da
n_buildings = trunc(Int, n_gas_test_buildings * 0.3)
@info "Number of buildings in each CV set: " n_buildings

# ╔═╡ f99924cb-8619-4cf5-b026-499f3fd7289e


# ╔═╡ 6dd0acce-a14c-41d5-b85d-c2945751983f
begin
sampling_strategy::Vector{Tuple{Vector{Int64}, Vector{Int64}}} = []
sampling_strategyᵧ::Vector{Tuple{Vector{Int64}, Vector{Int64}}} = []

training_indexlist = collect(1:nrow(tₐ′))
training_indexlistᵧ = collect(1:nrow(tᵧ′))

for i in 1:15
sample_buildings = sample(
	MersenneTwister(i+50),
	electric_train_buildings, 
	n_buildings
)
sample_buildingsᵧ = sample(
	MersenneTwister(i+50),
	gas_train_buildings, 
	n_buildings
)

# @info "Sample of building ids in CV fold: " sample_buildings[1:3]
	
building_boolmap = map( x -> x ∈ sample_buildings, tₐ′[:,"Property Id"] );
building_boolmapᵧ = map( x -> x ∈ sample_buildingsᵧ, tᵧ′[:,"Property Id"] );

building_cv = training_indexlist[building_boolmap]
building_cvᵧ = training_indexlistᵧ[building_boolmapᵧ]

building_train = training_indexlist[.!building_boolmap]
building_trainᵧ = training_indexlistᵧ[.!building_boolmapᵧ]

push!(sampling_strategy, (building_train, building_cv))
push!(sampling_strategyᵧ, (building_trainᵧ, building_cvᵧ))
end
end

# ╔═╡ c3255235-d457-49c5-b992-471a159fb34e
# length(sampling_strategy)

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
# m_tree

# ╔═╡ 88e0db0e-28bb-457a-9a12-3f020074f4a4
begin
# training_ground
n = 75

roundsᵣ = range(m_tree, :nrounds, lower=50, upper=150, scale=:linear);
max_depthᵣ = range(m_tree, :max_depth, lower=4, upper=10, scale=:linear);
ηᵣ = range(m_tree, :eta, lower=0.07, upper=0.4, scale=:linear);
γᵣ = range(m_tree, :gamma, lower=0, upper=50.0, scale=:linear);
αᵣ = range(m_tree, :alpha, lower=0, upper=30, scale=:linear);
λᵣ = range(m_tree, :lambda, lower=0.0, upper=1.0, scale=:linear);
rowsampleᵣ = range(m_tree, :rowsample, lower=0.3, upper=1.0, scale=:linear);
colsampleᵣ = range(m_tree, :colsample, lower=0.3, upper=1.0, scale=:linear);
nbinsᵣ = range(m_tree, :nbins, lower=16, upper=64, scale=:linear);
measure_terms = [rms,mae];

fitting_terms = [
	roundsᵣ,
	max_depthᵣ,
	ηᵣ,
	γᵣ,
	αᵣ,
	λᵣ,
	rowsampleᵣ,
	colsampleᵣ
]

# space = Dict(
#     :nrounds => HP.QuantUniform(:num_round, 50., 100., 10.),
#     :eta => HP.LogUniform(:eta, -3., 0.),
#     :gamma => HP.LogUniform(:gamma, -3., 1.),
#     :max_depth => HP.QuantUniform(:max_depth, 4.0, 10.0, 1.0),
#     :rowsample => HP.QuantUniform(:rowsample, 0.35, 1.0, 0.1),
# 	:colsample => HP.QuantUniform(:colsample, 0.35, 1.0, 0.1),
#     :lambda => HP.LogUniform(:lambda, -5., 3.),
#     :alpha => HP.LogUniform(:alpha, -5., 3.),
# )

end;

begin
modelname = string(hash((loss_function, fitting_terms, n, sampling_strategy, sampling_strategyᵧ, functional_terms)))[end-5:end]
output_dir = joinpath(joinpath(data_path, "p3_o"), modelname)
smlink_path = joinpath(joinpath(data_path, "p3_o"), "recent")
mkpath(output_dir)

# for convenience, we can link the most recent file for easy indexing
if isdir(smlink_path)
	rm(smlink_path)
end
symlink(output_dir, smlink_path; dir_target = true)

machines_dir = joinpath(output_dir, "machines")
dataout_dir = joinpath(output_dir, "data_out")

mkpath(machines_dir)
mkpath(dataout_dir)
end;

draw(
	PNG(
		joinpath(output_dir, "energy_trends.png"), 
		20cm, 
		10cm,
		dpi=500
	), p₃
)

draw(
	PNG(
		joinpath(output_dir, "temperature_nonlinearity.png"), 
		20cm, 
		12cm,
		dpi=600
	), t₁
)

## save the training data files
@info "Writing electric training to file..."
CSV.write(joinpath(output_dir, "training_electric.csv"), tₐ′);

@info "Writing gas training to file..."
CSV.write(joinpath(output_dir, "training_gas.csv"), tᵧ′);

begin
notes = split(string(fitting_terms), "NumericRange")[3:end]
open(joinpath(output_dir, "config.txt"), "w") do f
	println(f, "Model:\n", string(m_tree))
	println(f, "")
	println(f, "Loss function: \t\t\t\t", string(loss_function))
	println(f, "Functions used:\t\t\t\t", string(functional_terms))
	println(f, "")
	println(f, "Electric Training - #Buildings\t\t", Formatting.format(length(electric_train_buildings), commas=true))
	println(f, "Electric Training Data - #Samples:\t", Formatting.format(nrow(tₐ′), commas=true))
	println(f, "Electric Test - #Buildings\t\t\t", Formatting.format(length(electric_test_buildings), commas=true))
	println(f, "Electric Test Data - #Samples:\t\t", Formatting.format(nrow(teₐ′), commas=true))

	println(f, "Gas Training - #Buildings\t\t\t", Formatting.format(n_gas_training_buildings, commas=true))
	println(f, "Gas Training Data - #Samples:\t\t", Formatting.format(nrow(tᵧ′), commas=true))
	println(f, "Gas Test - #Buildings\t\t\t\t", Formatting.format(n_gas_test_buildings, commas=true))
	println(f, "Gas Test Data - #Samples:\t\t\t", Formatting.format(nrow(teᵧ′), commas=true))

	println(f, "Cross-validation n-buildings:\t\t", Formatting.format(n_buildings, commas=true))
	println(f, "Cross-validation folds:\t\t\t\t", Formatting.format(length(sampling_strategy), commas=true))
	println(f, "Hypercube Samples:\t\t\t\t\t", Formatting.format(n, commas=true))
	println(f, "")
	for i in eachindex(notes)
		@info notes[i]
		println(f, notes[i])
	end
end
end

@info "Output Directory:" modelname

m_tree_tuning = TunedModel(
	model=m_tree,
	tuning=AdaptiveParticleSwarm(rng=rng),
	n=n,
	resampling=sampling_strategy,
	range=fitting_terms,
	measure=measure_terms,
    # acceleration=CPUThreads(),
	train_best=true
);

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

daymonths = Dates.daysinmonth.(teₐ′.date)
teᵣ.prediction = MLJ.predict(mach, teᵥ) .* daymonths
teᵣ.recorded = yₑ .* daymonths
teᵣ.model = repeat([modelname], nrow(teᵥ))

return (mach, 0, teᵣ)
end

# ╔═╡ f301972b-feff-485a-b385-943fb434f0fd
m_tree_tuningᵧ = TunedModel(
	model=m_tree,
	tuning=AdaptiveParticleSwarm(rng=rng),
	n=n,
	resampling=sampling_strategyᵧ,
	range=fitting_terms,
	measure=measure_terms,
    # acceleration=CPUThreads(),
	train_best=true
);

# temportary stop before training

# ╔═╡ 2426e2d6-e364-4e97-bee8-7defb1e88745
md"""
##### CMIP
"""

# ╔═╡ 9c8f603f-33c6-4988-9efd-83864e871907
term₃ = unique([names(cmip)..., electricity_terms...])

if !isfile(joinpath(dataout_dir, "tea3.csv"))
m₃, v′₃, teₐ′₃ = electrictrain("CMIP", term₃);
MLJ.save(joinpath(machines_dir, "m3.jlso"), m₃);
CSV.write(joinpath(dataout_dir, "tea3.csv"), teₐ′₃);
@info "CMIP Model:" fitted_params(m₃).best_model
@info "CMIP Quality:" report(m₃).best_history_entry.measurement

m₃ = nothing
teₐ′₃ = nothing
else
	@info "Found CMIP Model."
end

# ╔═╡ dbae822c-24fa-4767-aaaa-d6bd8ad700ac
if !isfile(joinpath(dataout_dir, "tea0.csv"))
term₀ = unique([electricity_terms...])
m₀, vₐ′₀, teₐ′₀ = electrictrain("Null", term₀);

@info "Null Model:" fitted_params(m₀).best_model
@info "Null Quality:" report(m₀).best_history_entry.measurement

MLJ.save(joinpath(machines_dir, "m0.jlso"), m₀);
CSV.write(joinpath(dataout_dir, "tea0.csv"), teₐ′₀);

m₀ = nothing
teₐ′₀ = nothing
else
@info "Found Null Model."
end


# ╔═╡ ee57d5d5-545e-4b70-91d9-b82a108f854b
md"""
###### NOAA
"""

# ╔═╡ ab681a4a-d9d7-4751-bae3-2cfc5d7e997d
if !isfile(joinpath(dataout_dir, "tea2.csv"))
term₂ = unique([names(noaa)..., electricity_terms...])

m₂, vₐ′₂, teₐ′₂ = electrictrain("NOAA", term₂);
@info "NOAA Model:" fitted_params(m₂).best_model
@info "NOAA Quality:" report(m₂).best_history_entry.measurement

# ╔═╡ c0a54c77-d44d-40d0-9ecf-fd90609cae07
MLJ.save(joinpath(machines_dir, "m2.jlso"), m₂);
CSV.write(joinpath(dataout_dir, "tea2.csv"), teₐ′₂);

m₂ = nothing
teₐ′₂ = nothing
else
@info "Found NOAA Model."
end

# ╔═╡ cde0836c-3dbd-43a9-90fd-c30e5985acf7
md"""
###### EPW
"""

# ╔═╡ 5caecfab-3874-4989-8b3d-c65b53361c62
if !isfile(joinpath(dataout_dir, "tea4.csv"))
term₄ = unique([names(epw)..., electricity_terms...])

m₄, vₐ′₄, teₐ′₄ = electrictrain("EPW", term₄);
@info "EPW Model:" fitted_params(m₄).best_model
@info "EPW Quality:" report(m₄).best_history_entry.measurement

# ╔═╡ a56adf6a-bc86-46ad-a392-b4526250b9eb
MLJ.save(joinpath(machines_dir, "m4.jlso"), m₄);
CSV.write(joinpath(dataout_dir, "tea4.csv"), teₐ′₄);

m₄ = nothing
teₐ′₄ = nothing
else
	@info "Found EPW Model."
end

# ╔═╡ b9b8a050-1824-414e-928d-b7797760f176
md"""
###### Landsat8
"""

# ╔═╡ e97fd9dc-edb5-41e4-bbaf-dfbb14e7d461
if !isfile(joinpath(dataout_dir, "tea5.csv"))
term₅ = unique([names(landsat8)..., electricity_terms...])

m₅, vₐ′₅, teₐ′₅ = electrictrain("Landsat8", term₅);
@info "Landsat8 Model:" fitted_params(m₅).best_model
@info "Landsat8 Quality:" report(m₅).best_history_entry.measurement

# ╔═╡ 0ade6f11-7b91-43df-999b-4f9a20e18fef
MLJ.save(joinpath(machines_dir, "m5.jlso"), m₅);
CSV.write(joinpath(dataout_dir, "tea5.csv"), teₐ′₅);

m₅ = nothing
teₐ′₅ = nothing
else
	@info "Found Landsat8 Model."
end

# ╔═╡ a6cada88-c7c9-495d-8806-2503e674ec39
md"""
###### VIIRS
"""

if !isfile(joinpath(dataout_dir, "tea6.csv"))
term₆ = unique([names(viirs)..., electricity_terms...])

m₆, vₐ′₆, teₐ′₆ = electrictrain("VIIRS", term₆);
@info "VIIRS Model:" fitted_params(m₆).best_model
@info "VIIRS Quality:" report(m₆).best_history_entry.measurement

# ╔═╡ 1bbfb1b8-889f-46a9-bac1-4250efffefe5
MLJ.save(joinpath(machines_dir, "m6.jlso"), m₆);
CSV.write(joinpath(dataout_dir, "tea6.csv"), teₐ′₆);

m₆ = nothing
teₐ′₆ = nothing
else
	@info "Found VIIRS Model."
end

# ╔═╡ 03d2381d-e844-4809-b5a9-048c7612b7e2
md"""
###### SAR
"""

# ╔═╡ 98d04357-be23-4882-b5b5-8a6d924b7876
if !isfile(joinpath(dataout_dir, "tea7.csv"))
term₇ = unique([names(sar)..., electricity_terms...])

m₇, vₐ′₇, teₐ′₇ = electrictrain("SAR", term₇);
@info "SAR Model:" fitted_params(m₇).best_model
@info "SAR Quality:" report(m₇).best_history_entry.measurement

# ╔═╡ 91fb0341-e5f4-4736-9b1e-2e833aef72e6
MLJ.save(joinpath(machines_dir, "m7.jlso"), m₇);
CSV.write(joinpath(dataout_dir, "tea7.csv"), teₐ′₇);

m₇ = nothing
teₐ′₇ = nothing
else
	@info "Found SAR Model."
end

# ╔═╡ e84033ac-3b34-4e1f-a72b-9dfd937382c1
md"""
###### Dynamic World
"""

# ╔═╡ 13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
if !isfile(joinpath(dataout_dir, "tea8.csv"))
term₈ = unique([names(dynam)..., electricity_terms...])

m₈, vₐ′₈, teₐ′₈ = electrictrain("Dynamic World", term₈);
@info "Dynamic World Model:" fitted_params(m₈).best_model
@info "Dynamic World Quality:" report(m₈).best_history_entry.measurement

MLJ.save(joinpath(machines_dir, "m8.jlso"), m₈);
CSV.write(joinpath(dataout_dir, "tea8.csv"), teₐ′₈);

m₈ = nothing
teₐ′₈ = nothing
else
	@info "Found Dynamic World Model."
end

# ╔═╡ e84033ac-3b34-4e1f-a72b-9dfd937382c1
md"""
###### Sentinel 2
"""

# ╔═╡ 13f286d1-7e0f-4496-b54a-c6ee74c0cdb5
if !isfile(joinpath(dataout_dir, "tea9.csv"))
term₉ = unique([names(sentinel_1C)..., electricity_terms...])

m₉, vₐ′₉, teₐ′₉ = electrictrain("Sentinel-2", term₉);
@info "Sentinel-2 Model:" fitted_params(m₉).best_model
@info "Sentinel-2 Quality:" report(m₉).best_history_entry.measurement

MLJ.save(joinpath(machines_dir, "m9.jlso"), m₉);
CSV.write(joinpath(dataout_dir, "tea9.csv"), teₐ′₉);

m₉ = nothing
teₐ′₉ = nothing
else
	@info "Found Sentinel-2 Model."
end


md"""
###### Full Dataset
"""

# ╔═╡ 590138f1-7d6f-4e78-836d-258f7b4f617e
if !isfile(joinpath(dataout_dir, "teae.csv"))
termₑ = names(tₐ′)

mₑ, vₐ′ₑ, teₐ′ₑ = electrictrain("Full Data", termₑ);
@info "Full Data Model:" fitted_params(mₑ).best_model
@info "Full Data Quality:" report(mₑ).best_history_entry.measurement

# ╔═╡ 1e920d69-3a6b-4dda-8b20-82dc56ab30e7
MLJ.save(joinpath(machines_dir, "me.jlso"), mₑ);
CSV.write(joinpath(dataout_dir, "teae.csv"), teₐ′ₑ);

mₑ = nothing
teₐ′ₑ = nothing
else
	@info "Found Full Model."
end

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

daymonths = Dates.daysinmonth.(teᵧ′.date)
teᵣ.prediction = MLJ.predict(mach, teᵥ) .* daymonths
teᵣ.recorded = yₑ .* daymonths
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
# pterm = "naturalgas_mwh"
# tᵥ = DataFrames.select(
# 	DataFrames.select(tᵧ′, termᵧ₀), 
# 	exclusion_terms
# )

# X = DataFrames.select(tᵥ, Not(pterm))
# y = tᵥ[:,pterm]
end


if !isfile(joinpath(dataout_dir, "teg0.csv"))
mᵧ₀, vᵧ′₀, teᵧ′₀ = gastrain("Null", termᵧ₀);
MLJ.save(joinpath(machines_dir, "mg0.jlso"), mᵧ₀);
CSV.write(joinpath(dataout_dir, "teg0.csv"), teᵧ′₀);
@info "Null Model - gas:" fitted_params(mᵧ₀).best_model
@info "Null Quality - gas:" report(mᵧ₀).best_history_entry.measurement

mᵧ₀ = nothing
teᵧ′₀ = nothing
else
	@info "Found Null Gas Model."
end

# ╔═╡ 2426e2d6-e364-4e97-bee8-7defb1e88745
md"""
##### CMIP
"""

# ╔═╡ 9c8f603f-33c6-4988-9efd-83864e871907
termᵧ₃ = unique([names(cmip)..., naturalgas_terms...])

if !isfile(joinpath(dataout_dir, "teg3.csv"))
mᵧ₃, vᵧ′₃, teᵧ′₃ = gastrain("CMIP", termᵧ₃);
MLJ.save(joinpath(machines_dir, "mg3.jlso"), mᵧ₃);
CSV.write(joinpath(dataout_dir, "teg3.csv"), teᵧ′₃);
@info "CMIP Model - gas:" fitted_params(mᵧ₃).best_model
@info "CMIP Quality - gas:" report(mᵧ₃).best_history_entry.measurement

mᵧ₃ = nothing
teᵧ′₃ = nothing
else
	@info "Found CMIP Gas Model."
end

md"""##### NOAA"""

if !isfile(joinpath(dataout_dir, "teg2.csv"))
termᵧ₂ = unique([names(noaa)..., naturalgas_terms...])
mᵧ₂, vᵧ′₂, teᵧ′₂ = gastrain("NOAA", termᵧ₂);
MLJ.save(joinpath(machines_dir, "mg2.jlso"), mᵧ₂);
CSV.write(joinpath(dataout_dir, "teg2.csv"), teᵧ′₂);
@info "NOAA Model - gas:" fitted_params(mᵧ₂).best_model
@info "NOAA Quality - gas:" report(mᵧ₂).best_history_entry.measurement

mᵧ₂ = nothing
teᵧ′₂ = nothing
else
	@info "Found NOAA Gas Model."
end


# ╔═╡ a78a6fd5-4b45-43fc-a733-aef4fd14eb42
md""" ##### EPW"""

# ╔═╡ 1a4471e7-24ad-4652-9f3f-6eef92c781d5
if !isfile(joinpath(dataout_dir, "teg4.csv"))
termᵧ₄ = unique([names(epw)..., naturalgas_terms...])
mᵧ₄, vᵧ′₄, teᵧ′₄ = gastrain("EPW", termᵧ₄);
MLJ.save(joinpath(machines_dir, "mg4.jlso"), mᵧ₄);
CSV.write(joinpath(dataout_dir, "teg4.csv"), teᵧ′₄);
@info "EPW Model - gas:" fitted_params(mᵧ₄).best_model
@info "EPW Quality - gas:" report(mᵧ₄).best_history_entry.measurement

mᵧ₄ = nothing
teᵧ′₄ = nothing
else
	@info "Found EPW Gas Model."
end

# ╔═╡ 04d720ce-1de5-4c8b-bd2f-0d0a5e8ed271
md"""
##### Dynamic World
"""

if !isfile(joinpath(dataout_dir, "teg5.csv"))
termᵧ₅ = unique([names(dynam)..., naturalgas_terms...])
mᵧ₅, vᵧ′₅, teᵧ′₅ = gastrain("Dynamic World", termᵧ₅);
MLJ.save(joinpath(machines_dir, "mg5.jlso"), mᵧ₅);
CSV.write(joinpath(dataout_dir, "teg5.csv"), teᵧ′₅);
@info "Dynamic World Model - gas:" fitted_params(mᵧ₅).best_model
@info "Dynamic WOrld Quality - gas:" report(mᵧ₅).best_history_entry.measurement

mᵧ₅ = nothing
teᵧ′₅ = nothing
else
	@info "Found Dynamic World Gas Model."
end

# ╔═╡ 8ae63dda-a9d2-47e2-a89d-05fe8c11383b
md"""
##### Landsat8
"""

if !isfile(joinpath(dataout_dir, "teg6.csv"))
termᵧ₆ = unique([names(landsat8)..., naturalgas_terms...])
mᵧ₆, vᵧ′₆, teᵧ′₆ = gastrain("Landsat8", termᵧ₆);
@info "Landsat8 Model - gas:" fitted_params(mᵧ₆).best_model
@info "Landsat8 Quality - gas:" report(mᵧ₆).best_history_entry.measurement

# ╔═╡ e4fcbcc2-c0d2-4c65-b49d-ab4ef2988dc4
MLJ.save(joinpath(machines_dir, "mg6.jlso"), mᵧ₆);
CSV.write(joinpath(dataout_dir, "teg6.csv"), teᵧ′₆);

mᵧ₆ = nothing
teᵧ′₆ = nothing
else
	@info "Found Landsat8 Gas Model."
end
# ╔═╡ edceb483-eb33-4bf9-975d-3dc6f18cffe9
md"""
##### SAR
"""

# ╔═╡ b5178580-013b-4686-8bfc-c1f7395620b2


# ╔═╡ 79811ec5-d713-4dc4-b1a6-b0ea656633fd
if !isfile(joinpath(dataout_dir, "teg7.csv"))
termᵧ₇ = unique([names(sar)..., naturalgas_terms...])
mᵧ₇, vᵧ′₇, teᵧ′₇ = gastrain("SAR", termᵧ₇);
@info "SAR Model - gas:" fitted_params(mᵧ₇).best_model
@info "SAR Quality - gas:" report(mᵧ₇).best_history_entry.measurement
MLJ.save(joinpath(machines_dir, "mg7.jlso"), mᵧ₇);
CSV.write(joinpath(dataout_dir, "teg7.csv"), teᵧ′₇);

mᵧ₇ = nothing
teᵧ′₇ = nothing
else
	@info "Found SAR Gas Model."
end
# ╔═╡ 50bf1066-95aa-4385-aca7-510d2b2648d2
# @info "SAR Gas Model:" fitted_params(mᵧ₇).best_model

# ╔═╡ bafb366d-c0fb-428d-8188-7a2c6e100617
md"""
##### VIIRS
"""

if !isfile(joinpath(dataout_dir, "teg8.csv"))
termᵧ₈ = unique([names(viirs)..., naturalgas_terms...])
mᵧ₈, vᵧ′₈, teᵧ′₈ = gastrain("VIIRS", termᵧ₈);
MLJ.save(joinpath(machines_dir, "mg8.jlso"), mᵧ₈);
CSV.write(joinpath(dataout_dir, "teg8.csv"), teᵧ′₈);

@info "VIIRS Model - gas:" fitted_params(mᵧ₈).best_model
@info "VIIRS Quality - gas:" report(mᵧ₈).best_history_entry.measurement

mᵧ₈ = nothing
teᵧ′₈ = nothing
else
	@info "Found VIIRS Gas Model."
end

# ╔═╡ bac2785d-d692-4524-86c5-dc183f07fe86
md"""
##### Sentinel
"""

if !isfile(joinpath(dataout_dir, "teg9.csv"))
termᵧ₉ = unique([names(sentinel_1C))..., naturalgas_terms...])

# ╔═╡ 73a62e14-1535-4a5d-b4e7-f20a7a7ff7f7
mᵧ₉, vᵧ′₉, teᵧ′₉ = gastrain("Sentinel-2", termᵧ₉);

MLJ.save(joinpath(machines_dir, "mg9.jlso"), mᵧ₉);
CSV.write(joinpath(dataout_dir, "teg9.csv"), teᵧ′₉);

@info "Sentinel-2 Model - gas:" fitted_params(mᵧ₉).best_model
@info "Sentinel-2 Quality - gas:" report(mᵧ₉).best_history_entry.measurement

mᵧ₉ = nothing
teᵧ′₉ = nothing
else
	@info "Found Sentinel-2 Gas Model."
end

# ╔═╡ 26086070-08a3-4ce8-b8e6-cd2cfc83e44d
md"""
##### Full Data
"""

# ╔═╡ 1af87fef-8c88-4e34-b1d4-8bccd7881473
if !isfile(joinpath(dataout_dir, "tege.csv"))
mᵧₑ, vᵧ′ₑ, teᵧ′ₑ = gastrain("Full Data", names(tᵧ′));
MLJ.save(joinpath(machines_dir, "mge.jlso"), mᵧₑ);
CSV.write(joinpath(dataout_dir, "tege.csv"), teᵧ′ₑ);
@info "Full Data Model - gas:" fitted_params(mᵧₑ).best_model
@info "Full Data Quality - gas:" report(mᵧₑ).best_history_entry.measurement

mᵧₑ = nothing
teᵧ′ₑ = nothing
else
	@info "Found Full Gas Model."
end
