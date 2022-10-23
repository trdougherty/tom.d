### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 1ea68209-380d-4b2e-9239-bedf850b8243
begin
import Pkg
Pkg.activate(Base.current_project())

using ArchGDAL
using CSV
using ColorSchemes
using DataFrames
using Dates
using GeoDataFrames
using GeoFormatTypes
using JSON
using Logging
using Plots
using YAML
using StatsBase
end;

# ╔═╡ 600556af-f7b6-4dfe-a10a-ed80ee98ef25
using Gadfly

# ╔═╡ f8e0a746-8b03-4f23-820b-f8cb66bc00bb
using EnergyPlusWeather

# ╔═╡ a5017265-097a-4be3-926b-54d0044a6b37
using Random

# ╔═╡ 01df78ab-aafd-462c-b5a1-1561fb890acb
using Shapefile

# ╔═╡ f758a6da-9531-4d37-b22b-6902d2466b9b
using Distances

# ╔═╡ 7488a1a4-2b76-48f7-80bb-3b441631068a
import Cairo, Fontconfig

# ╔═╡ 70bc6e0e-fecb-4506-a810-2777d67f73a9
Random.seed!(42)

# ╔═╡ 2de15f2f-cfc8-4416-9cde-a5e63a3b7d95
begin
function reproject_points!(
	geom_obj::ArchGDAL.IGeometry,
	source::ArchGDAL.ISpatialRef,
	target::ArchGDAL.ISpatialRef)

	ArchGDAL.createcoordtrans(source, target) do transform
		ArchGDAL.transform!(geom_obj, transform)
	end
	return geom_obj
end

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
end

# ╔═╡ 7f162842-c24a-4fdf-866d-eb4df50e8d23
begin
	# this pass going to try and keep everything in one coordinate system
	source_num = 4326
	target_num = 32618
	pm_num = 3857
	source = ArchGDAL.importPROJ4("+proj=longlat +datum=WGS84 +no_defs +type=crs")
	target = ArchGDAL.importPROJ4("+proj=utm +zone=18 +datum=WGS84 +units=m +no_defs +type=crs")
	pm = ArchGDAL.importEPSG(pm_num)

	@info "Source EPSG: " source
	@info "Target EPSG: " target
end;

# ╔═╡ 4ce9f094-273d-4497-9255-202726b00c11
md"""
## Chapter 1: New York City
"""

# ╔═╡ c704089f-5a31-403f-89a1-352e90991d72


# ╔═╡ f6ba219d-9c7c-4947-a3fd-5a32e26e89a8
begin
sources_file = joinpath(pwd(), "sources.yml")
sources = YAML.load_file(sources_file)
data_destination = sources["output-destination"]
data_path = joinpath(data_destination, "data", "nyc")
output_dir = joinpath(data_path, "p1_o")
mkpath(output_dir)

nyc_boundaries_path = joinpath(
	data_path, 
	"council-boundaries.geojson"
)
nyc_boundaries = GeoDataFrames.read(nyc_boundaries_path)
reproject_points!(nyc_boundaries.geometry, source, target)

nyc_boundaries[!,"coun_dist"] = parse.(Int64, nyc_boundaries[:,"coun_dist"])
Plots.plot(
	nyc_boundaries.geometry,
	color=:transparent,
	dpi=400
)
end

# ╔═╡ d1ea0a90-7830-4d3f-8fb1-18cf88f3293c
nyc_boundaries

# ╔═╡ f65a1dea-039e-4705-ae78-10a385f85b6b
md"""
The data we have for new york city:
"""

# ╔═╡ 8db8b2ca-ee25-44a8-91d2-26b7c7fb8c0d
md"""
#### Stage 1
1. Load the data used to link the energy data with building footprints
2. Load energy data, remove duplicated terms
3. Load footprint centroids data
4. Check which footprint centroids can be matched to the energy data, remove the ones which aren't
"""

# ╔═╡ f2013cd8-5ce1-47f0-8700-a0550612f943
md"""(1) *linking data*"""

# ╔═╡ e0eae347-e275-4737-b429-4dd2b04cb27c
begin
	annual_data = CSV.read(
		joinpath(
			data_path,
			"annual-energy.csv"
		),
		missingstring=["Not Available","Insufficient access"],
		DataFrame
	)

	linking_data = deepcopy(annual_data)
	rename!(linking_data, "NYC Borough, Block and Lot (BBL)" => "bbl")
	select!(linking_data, ["Property Id", "bbl"])
end

# ╔═╡ 6f6159a2-7910-4483-807b-aeed118f4242
annual_energy = annual_data[:, "Site Energy Use (kBtu)"] .+ annual_data[:, "Source Energy Use (kBtu)"]

# ╔═╡ 0fb0caa1-d988-4920-9f89-8c9431a22371
names(annual_data)

# ╔═╡ dc2196ab-a277-4083-a7b6-4bbe90df3c53
energy_variables = names(annual_data)[Base.contains.(names(annual_data), "Use (kBtu)")]

# ╔═╡ 27bb969d-c79f-4d8d-b780-e672b51138a8
annual_energyterms = select(annual_data, energy_variables, "Electricity Use - Grid Purchase (kBtu)");

# ╔═╡ 8ad4c349-d342-45de-9d44-dc2a40f73baf
names(annual_energyterms)

# ╔═╡ 059b124f-9a9d-4424-a92b-24853aa481af
term_energy = map(eachcol(annual_energyterms)) do col
	sum(skipmissing(col))
end;

# ╔═╡ 01178082-f94d-4cce-a491-f71a5e46355d
total_energyconsumption = DataFrame(
	"term" => names(annual_energyterms),
	"consumption" => term_energy
)

# ╔═╡ a12127c9-3a2d-46db-b01d-b1887390d353
annualₜ = total_energyconsumption[5:end,:];

# ╔═╡ 8a36a580-9d4f-4b41-a28d-d02ca7fd34d5
annualₚ = annualₜ.consumption ./ sum(annualₜ.consumption);

# ╔═╡ b16c61cd-3ad5-4806-8381-f8e43fcb447d
α₁ = Gadfly.plot(
	y=annualₜ.consumption,
	# x=annualₜ.term,
	Geom.bar
)

# ╔═╡ 3164491c-df20-4384-8045-e270d82317d7
draw(PNG(joinpath(output_dir, "energy_usetypes_nyc.png"), 14cm, 7cm, dpi=600), α₁)

# ╔═╡ 6ceeb173-c9b4-4daa-be6b-0097c926885c
# ℓ = Gadfly.plot(
# 	total_energyconsumption[5:end,:],
# 	x=:term,
# 	y= 100 * total_energyconsumption[5:end,:].consumption ./ sum(total_energyconsumption[5:end,:].consumption),
# 	Geom.bar,
# 	Guide.title("Energy Consumption by Type"),
# 	Guide.xlabel(""),
# 	Guide.ylabel("%")
# )

# ╔═╡ fa6b3c0c-8c48-4b99-85b9-e49b9f7993db
# draw(PNG(joinpath(output_dir, "energy_usetypes_nyc.png"), 14cm, 7cm, dpi=600), ℓ);

# ╔═╡ 4197d920-ef37-4284-970a-7fd4331ad9b7
md"""(2) *Full monthly energy data between 2018 and 2020*"""

# ╔═╡ 6fc43b1e-bdd6-44dd-9940-9d96e25a285d
begin
	nyc_monthly_2020 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2020.csv"
		),
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly_2019 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2019.csv"
		),
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly_2018 = CSV.read(
		joinpath(
			data_path,
			"monthly-energy-2018.csv"
		),
		missingstring="Not Available",
		DataFrame
	)
	nyc_monthly = vcat(
		nyc_monthly_2020,
		nyc_monthly_2019,
		nyc_monthly_2018
	)
	nyc_monthly[!,"date"] = DateTime.(nyc_monthly.Month, "u-y") .+ Dates.Year(2000)
	select!(nyc_monthly, Not("Month"))

	select!(
		nyc_monthly, 
		Not(
			["Parent Property Id", "Property Name", "Parent Property Name"]
		)
	)
	rename!(
		nyc_monthly, 
		[
			"Electricity Use  (kBtu)" => "electricity_kbtu",
			"Natural Gas Use  (kBtu)" => "naturalgas_kbtu"
		]
	)
	nyc_monthly[:,"electricity_mwh"] = nyc_monthly.electricity_kbtu ./ 3.412e+6
	nyc_monthly[:,"naturalgas_mwh"] = nyc_monthly.naturalgas_kbtu ./ 3.412e+6
end

# ╔═╡ 66cd75cc-161c-4e85-8d40-ab01547443e1
begin
custom_quantiles = (0.05, 0.95)
	
electric_min, electric_max = quantile(dropmissing(nyc_monthly, :electricity_mwh).electricity_mwh, custom_quantiles)
gas_min, gas_max = quantile(dropmissing(nyc_monthly, :naturalgas_mwh).naturalgas_mwh, custom_quantiles)
end

# ╔═╡ bea53833-7f47-4b4b-afc2-a0bcc9159926
begin
filter!( 
	x -> ismissing(x.electricity_mwh) 
	|| electric_max > x.electricity_mwh > electric_min, 
	nyc_monthly
)
filter!( 
	x -> ismissing(x.naturalgas_mwh) || gas_max > x.naturalgas_mwh > gas_min, nyc_monthly
)
end;

# ╔═╡ ccf8cb9f-88a4-4f41-a635-d02dc479e50d
nyc_idcount = countmap(nyc_monthly[:, "Property Id"])

# ╔═╡ 0f3a58c2-ae4b-4c89-9791-803cb89ab51e
# want to snip any of the weird data which might have snuk in
sneaky_ids = collect(keys(filter(t-> t.second <= 36, nyc_idcount)))

# ╔═╡ 78ace92e-81de-47e7-a19b-41a0c5f35bb4
filtering_idx = in.(nyc_monthly[:, "Property Id"], (sneaky_ids,))

# ╔═╡ 64f587cd-aac9-47c9-a0ff-762b502f8adb
length(unique(nyc_monthly[:, "Property Id"]))

# ╔═╡ bbc1d2c3-393d-4fb9-a039-d54ae414cf16
nyc_monthly_clean = nyc_monthly[filtering_idx, :];

# ╔═╡ 8569cd2f-8b58-4797-ad35-42fbb831a15c
nyc_monthly_clean[!,"month"] = Dates.month.(nyc_monthly_clean.date);

# ╔═╡ 89bb5aae-2139-4a4a-8e11-b5cca12cb4b5
begin
nyc_keys = keys(countmap(Date.(nyc_monthly_clean.date)))
nyc_values = values(countmap(Date.(nyc_monthly_clean.date)))
end;

# ╔═╡ 40b8821e-0c9a-463a-af9a-bd4cfd797011
begin
date_counts = sort(DataFrame("Date" => collect(nyc_keys), "Count" => collect(nyc_values)), :Date)

datacounts_plot = Plots.plot(
	date_counts.Date,
	date_counts.Count,
	ylims = (0,30000),
	linestyle = :dash,
	# color="black",
	thickness_scaling=1.2,
	linewidth = 2.5,
	legend=:bottomright,
	ylabel="N Points - Monthly",
	xlabel="Date",
	title="Temporal Quality of Data"
)
end

# ╔═╡ 211c6ff2-9308-494a-9897-d969123fe873
savefig(datacounts_plot, joinpath(output_dir, "temporal_quality.png"))

# ╔═╡ 69c8b562-1ed2-4c6e-9b60-44a71ac62ed0
CSV.write(
	joinpath(output_dir, "monthly_energy_nyc.csv"),
	nyc_monthly_clean
)

# ╔═╡ 5c1383bb-50ad-43d9-ac99-2e2b14d41f33
md"""(2) *unique monthly property ids - essentially only use the unique property ids seen*"""

# ╔═╡ 9a579e9d-8de1-4b82-9508-a378f08e2955
md"""(3) *NYC building centroids*"""

# ╔═╡ 34db351a-d5cd-4069-add3-9aa3b4162585
begin
	footprints = GeoDataFrames.read(joinpath(data_path, "building-footprints.shp"))

	rename!(
		footprints,
		:mpluto_bbl => :bbl
	)
	select!(
		footprints,
		[
			"geometry",
			"bbl",
			"heightroof",
			"cnstrct_yr",
			"groundelev"
		]
	)
	dropmissing!(footprints)
	
	# footprints[!,"heightroof"] = parse.(Float64, footprints[!,"heightroof"])
	# footprints[!,"cnstrct_yr"] = parse.(Int64, footprints[!,"cnstrct_yr"])
	# footprints[!,"groundelev"] = parse.(Int64, footprints[!,"groundelev"]);

	# want to reproject the coordinates into something which preserves meters
	reproject_points!(footprints.geometry, source, target)
	footprints[!,"area"] = ArchGDAL.geomarea.(footprints.geometry)
	# reproject_points!(footprints.geometry, target, source)

	# finally drop all the buildings without a geometry provided
	filter!(row -> row.area > 0, footprints)
end

# ╔═╡ 6d6a3a4e-ac6f-41e5-8e8c-78f45eb4f618
property_ids_data = select(unique(nyc_monthly, "Property Id"), "Property Id");

# ╔═╡ e54c18df-46fe-4425-bdfc-ca8c529d436a
md"""(4) *Cleaned nyc building centroids*"""

# ╔═╡ 009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
begin
	nyc_building_points_ids = leftjoin(
		footprints,
		linking_data,
		on=:bbl
	)
	
	dropmissing!(
		nyc_building_points_ids,
		"Property Id"
	)
	select!(
		nyc_building_points_ids,
		Not(:bbl)
	)
	
	@info "Buildings with valid linking term:" nrow(nyc_building_points_ids)

	nyc_building_points_data = leftjoin(
		nyc_building_points_ids,
		property_ids_data,
		on="Property Id"
	)

	@info "Unfiltered Buildings with valid property id:" nrow(nyc_building_points_data)

	dropmissing!(nyc_building_points_data)

	@info "Cleaned Buildings with valid property id:" nrow(nyc_building_points_data)

	unique!(nyc_building_points_data, ["Property Id"])
	select(nyc_building_points_data, "Property Id", :)

	# @info "building data and locations" select(nyc_building_points_data, Not(:geometry))
end;

# ╔═╡ a7e02d02-158f-44d5-8d2e-28d5aba1cca5


# ╔═╡ 61c4ad16-99ef-4d21-a4dd-cae897e18659
md"""
##### Cleaned list of buildings in each of the council regions
"""

# ╔═╡ 68c9d819-d948-4bcf-a625-64f2bdc26b62
begin
	let 
		regionlist::Vector{Union{Int64, Missing}} = fill(missing, length(nyc_building_points_data.geometry))
		for (building_index, building_point) in enumerate(nyc_building_points_data.geometry)
			for (boundary_index, boundary_geom) in enumerate(nyc_boundaries.geometry)
				if GeoDataFrames.contains(boundary_geom, building_point)
					regionlist[building_index] = nyc_boundaries[boundary_index, "coun_dist"]
					break
				end
			end
		end
		nyc_building_points_data[!, "council_region"] = regionlist;
		dropmissing!(nyc_building_points_data);
	end
end

# ╔═╡ 2dea8532-505a-4b0e-8b98-966a47808804
begin
	council_countmap = countmap(dropmissing(nyc_building_points_data).council_region)
	council_counts = collect(values(council_countmap))

	@info "Council Building Counts: " council_countmap
end;

# ╔═╡ 07aab43a-697f-4d6a-a8c4-4415116368a7
# want to find some councils for validation which roughly have this many buildings
begin
	council_std = std(council_counts)
	council_median = median(council_counts)
end

# ╔═╡ 878e4be3-1a1a-4f73-8823-fb69fd756437
begin
	median_council_plot = Plots.histogram(
		council_counts, 
		bins=20, 
		color="transparent",
		label="Building Count - Council",
		dpi=400,
		title="Building count per Council - Histogram",
		ylabel="Count of Councils",
		xlabel="Count of Buildings"
	)
	Plots.vline!(
		median(council_counts, dims=1),
		line=(4, :dash, 1.0, :red),
		label="Median Line"
	)
end

# ╔═╡ 27c79591-5b49-4d41-97a5-2eb8b2780e1d
savefig(median_council_plot, joinpath(output_dir, "council_median.png"))

# ╔═╡ d9e0a62c-0528-41e8-9c6d-aa9629ca79c3
nyc_building_points_data

# ╔═╡ 5478f86e-e8a1-4290-8255-3b18a8e36ff5
# station_distances = combine(
# 	groupby(nyc_building_points_data, :council_region),
# 	:weather_station_distance => mean
# );

# ╔═╡ 95dcee91-24ac-4495-a4e1-355590382c37
# station_distances

# ╔═╡ daf1e9a4-6fc0-4b27-b468-44c66ed40940
# begin
# 	median_council_distance = median(
# 		station_distances.weather_station_distance_mean, dims=1)
# 	std_council_distance = std(station_distances.weather_station_distance_mean)
	
# 	Plots.histogram(
# 		station_distances.weather_station_distance_mean, 
# 		bins=10, 
# 		color="transparent",
# 		label="Building Count - Council",
# 		dpi=400,
# 		title="Building count per Council - Histogram",
# 		ylabel="Mean Distance from Weather Station",
# 		xlabel="Count of Councils"
# 	)
# 	Plots.vline!(
# 		median_council_distance,
# 		line=(4, :dash, 1.0, :red),
# 		label="Median Line"
# 	)
# end

# ╔═╡ f378506c-50fb-430d-a26f-2ee99c77dd8d
# std_distance = 6

# ╔═╡ 140fea64-b020-461c-88e1-d46cfa78571d
# candidate_councils = filter( 
# 	row -> 
# 	 median_council_distance[1] + (std_council_distance / std_distance) > 
# 	 row.weather_station_distance_mean > 
# 	 median_council_distance[1] - (std_council_distance / std_distance),
# 	station_distances
# ).council_region

# ╔═╡ 941b6f4a-1359-4084-859a-07b3b3fcc7a2
# begin
# 	candidate_councils = collect(keys(filter(
# 		t -> council_median + (council_std/std_distance) > t.second > council_median - (council_std/std_distance),
# 		council_countmap
# 	)))
# end

# ╔═╡ 45334b27-e117-4e76-bf15-44271775743c
# Plots.histogram(nyc_council_building_density, bins=20)

# ╔═╡ fa11a171-904f-424b-bb42-25f664fe13e6
rng = MersenneTwister(100)

# ╔═╡ eb4e2971-8fa4-4d6a-a175-6554c3122a41
# candidate_councils = rand(1:nrow(nyc_council_counts),6)

# ╔═╡ 35d7d6d9-34bb-430a-9c74-001b446c9d09
council_count_df = DataFrame(
	coun_dist=collect(keys(council_countmap)), 
	building_count=collect(values(council_countmap))
);

# ╔═╡ 691a9409-ab03-4843-9743-5685c563af81
nyc_council_counts = dropmissing(leftjoin(nyc_boundaries, council_count_df, on="coun_dist"));

# ╔═╡ 98e5f297-585b-4d8c-96e3-59b4b7c52d1e
nyc_council_building_density = nyc_council_counts.building_count ./ parse.(Float64, nyc_council_counts.shape_area)

# ╔═╡ 01a6eaca-1294-4487-a224-6c59c1de7899
candidate_councils = sample(
	rng,
	nyc_council_counts.coun_dist,
	6,
	replace=false
)

# ╔═╡ e7f3b286-fe7e-48ce-a01a-25e5e3743267
color_translation = nyc_council_counts.building_count / maximum(nyc_council_counts.building_count);

# ╔═╡ 5f548a26-a95e-4bc7-9dad-7bdefa41611c
council_colors = cgrad(:matter)[color_translation];

# ╔═╡ 9c877f38-ecf8-4acd-9849-200fb01bce00
begin
	council_building_distribution = Plots.plot(
		nyc_council_counts.geometry,
		color_palette=council_colors,
		title="Distribution of Building Counts",
		dpi=500
	)
end

# ╔═╡ 353f0519-69a8-484d-829d-368140ccb545
savefig(
	council_building_distribution, 
	joinpath(output_dir, "council_building_distribution.png")
)

# ╔═╡ 2da9205e-4325-4bf0-943e-10f8d4dd2b86
nyc_council_counts

# ╔═╡ 734c7a3b-10d5-4627-ae4d-1f787509e838
begin
	candidate_regions = filter(row -> row.coun_dist ∈ candidate_councils, nyc_council_counts)
	noncandidate_regions = filter(row -> row.coun_dist ∉ candidate_councils, nyc_council_counts)
end;

# ╔═╡ 70a44334-ae15-4225-9f2e-668f6ee2b965
md"""
#### Candidate Regions
"""

# ╔═╡ 7e0ecfd3-cdea-4470-a799-acbebc67e7dd
# this will be the validation and test set
candidate_councils

# ╔═╡ f66ee36c-b0c5-4536-ad9d-214c13b3a984
midpoint = length(candidate_councils) ÷ 2

# ╔═╡ a465b08a-26f6-4e7f-a189-df7d76395f80
validation_districts  = []

# ╔═╡ bab66f4b-1d1e-46ad-ae79-1c4f5fae6273
test_districts = candidate_councils

# ╔═╡ b66daacc-515e-4137-b1ed-36186dc3fda4
function cascade_union(geometry_list)
	geom = geometry_list[1]
	for geometry in geometry_list[2:end]
		geom = ArchGDAL.union(geom, geometry)
	end
	return geom
end

# ╔═╡ 0bfc6dec-97dd-4a7c-8af5-382068cddbc7
begin
	train_regions = filter(row -> row.coun_dist ∉ candidate_councils, nyc_council_counts)
	validate_regions = filter(row -> row.coun_dist ∈ validation_districts, nyc_council_counts)
	test_regions = filter(row -> row.coun_dist ∈ test_districts, nyc_council_counts)

	datasplit_map = Plots.plot(
		cascade_union(train_regions.geometry),
		color=:transparent,
		dpi=600,
		label="Training / Validating",
		legend=:topleft
	)
	Plots.plot!(
		cascade_union(test_regions.geometry),
		color=:indianred,
		label="Testing"
	)
	# Plots.plot!(
	# 	cascade_union(test_regions.geometry),
	# 	color=:indianred,
	# 	dpi=400,
	# 	label="Testing"
	# )
end

# ╔═╡ 25b4bc79-2d09-4e44-a75e-144c972db3ba
savefig(datasplit_map, joinpath(output_dir, "datapartition_map.png"))

# ╔═╡ f69cb982-573f-43c4-8fb0-a1aab6028021
nyc_building_points_data

# ╔═╡ c97e405d-8601-4634-9a7c-654598dd1200
md"""
#### Stage 2
1. Get the footprints for each of the energy models.
2. Check to see if any of the points from the cleaned building centroid data are found in the boundaries of the energy model footprint. 
3. If it matches multiple footprints, throw away the datapoint."""

# ╔═╡ ab355e93-a064-40e2-a67a-e75cb54efb33
md"""(1) *Testing with just a sample of building footprints for now, here is what they might look like*"""

# ╔═╡ 8e462588-c0c1-4e96-a1fe-f2df50a8ec37
# begin
# 	# want this file to change to the building shapefile before this is actually run
# 	energy_model_footprints = GeoDataFrames.read(
# 		joinpath(
# 			data_path,
# 			"largest_buildings.geojson"
# 		)
# 	)
# 	energy_model_footprints[!,"id"] = collect(1:nrow(energy_model_footprints))
# 	sample_building = energy_model_footprints[5,"geometry"]

# 	select!(energy_model_footprints, ["id","geometry"])
# end

# ╔═╡ 14f9ac4d-e705-41fd-8561-4d1421e41126
# Plots.plot(sample_building, color=:transparent)

# ╔═╡ 2d956f5a-7f40-4a7d-b491-f3de3340cb66
md"""
(2) Need to see if any of the energy points can be found in the buiding shapefile. First going to take the bounding box for all buidings to narrow down the scope of potential points which might be contained, prior to checking if it's actually inside the vectorized box.
"""

# ╔═╡ 630cbcdf-31c6-4a96-9c36-8fea5d64e2b9
md"""
- *gray:* candidate box
- *white:* true geometry
"""

# ╔═╡ 4e00b820-9f99-4134-afd7-2710424fe28e
# begin
# 	Plots.plot(
# 		ArchGDAL.boundingbox(sample_building),
# 		opacity=0.15,
# 		color=:gray
# 	)
# 	Plots.plot!(
# 		sample_building,
# 		color=:white
# 	)
# end

# ╔═╡ c4b107f9-421b-4c5e-b876-10b9efb4d4c6
# begin
# 	points_range = length(nyc_building_points_data.geometry)
# 	buildings_range = length(energy_model_footprints.geometry)
	
# 	building_mapping = Array{Union{Missing, Int}}(missing, buildings_range)
# 	for i = 1:buildings_range
# 		# candidate geom - just a box
# 		candidate_geom = ArchGDAL.boundingbox(energy_model_footprints.geometry[i])
		
# 		for j = 1:points_range
# 			# candidate process to find something in the ballpark
# 			if ArchGDAL.contains(
# 				candidate_geom,
# 				nyc_building_points_data.geometry[j]
# 			)
# 				# if we pass the candidacy process
# 				if ArchGDAL.contains(
# 					energy_model_footprints.geometry[i], 
# 					nyc_building_points_data.geometry[j]
# 				)
# 					# if we already have a mapped value, it means multiple points
# 					# map to the same building. We need to toss it
# 					if ~ismissing(building_mapping[i])
# 						building_mapping[i] = missing
# 						break
# 					end
# 					building_mapping[i] = j
# 				end
# 			end
# 		end
# 	end
# 	building_mapping
# end;

# ╔═╡ 11fd74bd-4ebc-4537-bbd9-958761ac7b01
md"""
Estimated time without candidate process (h):
"""

# ╔═╡ 7df29d94-63b7-46dd-8872-a8dbe0e81e38
md"""
Estimated time with candidate process:
"""

# ╔═╡ 5a35d191-76e7-447e-841d-881b47ab8d6c
md"""
After mapping to the data points we have for the buildings
	"""

# ╔═╡ 0c000a60-6fe3-464a-a8b6-906e1e94f02d
# begin
# 	energy_model_footprints[!,"data_mapping"] = building_mapping
# 	energy_model_footprints
# end

# ╔═╡ 329d43de-17b6-4d11-a0c8-4cd0a6980932
# begin
# 	matched_buildings = dropmissing(energy_model_footprints, :data_mapping)
# 	matched_buildings
# end

# ╔═╡ 9cfc27da-38b5-40cd-b41f-a1d2687011a8
# matched_points = nyc_building_points_data[matched_buildings.data_mapping, :]

# ╔═╡ 69cd9bb6-01d7-447d-afbc-b349d01d9fbf
# begin
# 	matched_points[:, "shapefile_id"] = matched_buildings.id
# 	id_mapping = select(matched_points, ["Property Id","shapefile_id"])
# 	@info "Matched building shapefiles with building IDs:" id_mapping
# end

# ╔═╡ a85a3658-e1b5-4c57-a275-26a69dce572a
# begin
# 	poi = 4
# 	Plots.plot(matched_buildings.geometry[poi], color=:transparent)
# 	Plots.plot!(matched_points.geometry[poi])
# end

# ╔═╡ 2520a4ec-cebd-4f27-8e21-5a5f9b55c8a9
md"""
At this point we should just be able to extract a list of building ids for which we want to run simulations, and we can use the property id to collect the energy data
"""

# ╔═╡ f6e0da2e-1394-4945-aee7-67de4ade486c
# matched_buildings.id

# ╔═╡ 39fc72af-2262-4597-9c44-c9ccd56b6ec9
# matched_points[:, "Property Id"]

# ╔═╡ 3bd5037f-87cf-4988-888f-d62c10de3195
# begin
# 	# now going to output a list of all the idf files we want to run
# 	@info "Creating queue of simulation files" id_mapping.shapefile_id
	
# 	output_dir = joinpath(data_path, "p1_o")
# 	mkpath(output_dir)

# 	simulation_output = joinpath(output_dir, "simulation_queue.txt")
# 	open(simulation_output,"w") do io
# 		for id in id_mapping.shapefile_id
# 	   		println(io,string(id)*".idf")
# 		end
# 	end
# end

# ╔═╡ b4d4e254-5198-4b0f-b9e8-054fe97f7f3a
md"""
#### Stage 3
Extracting what we know about the epw weather files
"""

# ╔═╡ a88489a2-2da2-492e-94d6-7af90feaf92f
typeof(output_dir)

# ╔═╡ 6cabf58a-7082-4a18-9dd3-b9334f2ccd9f
epw_dir = joinpath(data_path, "tmy-files")

# ╔═╡ edd15ef6-80ca-496d-9073-49a430b9d2ed
# first, we need a way of extracting the geographic information from the weather stations
function epw_location(epw_dir::String)
	readdir(epw_dir)
end

# ╔═╡ 621d236a-ce0a-4bf5-9486-d481527d3261
epw_files = filter( x -> endswith(x, ".epw"), epw_location(epw_dir) )

# ╔═╡ abec0385-4725-4a51-8ef4-bda7cfaede64
epw_full_files = joinpath.(epw_dir, epw_files)

# ╔═╡ ca10ae5a-7a94-42eb-97e0-770e7f781a4c
epw_datafile_path = joinpath(output_dir, "epw-geodf.csv")

# ╔═╡ 28a46b6f-ae61-4639-8448-699451ea228c
begin
	# want to make sure we wipe the file before we start
	close(open(epw_datafile_path, "w"))
	
	epw_io = open(epw_datafile_path, "w")
	for epw_file in epw_full_files
		println(epw_io, readline(epw_file))
	end
	close(epw_io)
end

# ╔═╡ c92d5491-a044-4943-835c-cc2ecc418208
epw_geo_headers = [
	"Source",
	"City",
	"State",
	"Country",
	"Type",
	"Location ID",
	"Longitude",
	"Latitude",
	"Time Zone",
	"Elevation"
];

# ╔═╡ 052750a1-4941-47b8-9b54-518a4bc6cb67
begin
	epw_geo = CSV.read(epw_datafile_path, header=epw_geo_headers, DataFrame)
	epw_geo[!,"filename"] = epw_files
	epw_geo[!,"geometry"] = createpoint.(zip(epw_geo.Latitude, epw_geo.Longitude))
	filter!( row -> row.Type == "TMY3", epw_geo )
	reproject_points!(epw_geo.geometry, source, target)
end;

# ╔═╡ 92eab0d2-f601-4f9c-b433-e2285be87ed2
epw_geo

# ╔═╡ 14b73aa8-cdc6-40c9-ac38-39039f19dae7
md"""
Here are the weather stations, projected into the New York UTM zone (which dones't make sense for most of the US...)
"""

# ╔═╡ a1cd7408-c92b-432c-8830-a7d709d7c298
Plots.plot(
	epw_geo.geometry,
	color="white",
	markersize=1.5
)

# ╔═╡ 3e5adcf1-0815-41b6-bd06-113cbd35072c
md"""
##### now extracting a region around the city. 
The city is 783.8km², so let's get all weather stations within 25km² to be safe. Going to find the mean point of the buildings and run a circle of 25km radius around
"""

# ╔═╡ 7e1470f0-d52d-48cb-a62a-07acb02e18a5
nyc_building_centroids = ArchGDAL.centroid.(nyc_building_points_data.geometry);

# ╔═╡ c6e1f5ac-55a0-4175-b05b-236a2eb68561
# want this in KM
projection_dist = 45e3

# ╔═╡ 5c352c0c-0775-446d-b785-e7d9d2677c41
# like to keep these together as reproject changes the original dataframe
begin
	nyc_building_lons = ArchGDAL.getx.(nyc_building_centroids, 0)
	nyc_building_lats = ArchGDAL.gety.(nyc_building_centroids, 0)
	building_locations = hcat(
		nyc_building_lons,
		nyc_building_lats
	)

	mean_lat = mean(nyc_building_lats)
	mean_lon = mean(nyc_building_lons)
	
	weather_region = ArchGDAL.buffer(
		ArchGDAL.createpoint(mean_lon, mean_lat), 
		projection_dist
	)
end;

# ╔═╡ c7fcad00-fa50-4a3f-ab13-8ac392b7abe7
md"""
Finally we can filter for the stations which provide local weather data
"""

# ╔═╡ 643e0bbb-89e5-45a5-bcca-991cfc66e85d
epw_local = filter( row -> ArchGDAL.contains(weather_region, row.geometry), epw_geo );

# ╔═╡ 1b7432eb-bc8d-4bd1-a19c-06ca71c29786
begin
	weather_stations_map = Plots.plot(
		weather_region, 
		opacity=0.15, 
		color="gray",
		size=(650,550),
		dpi=600
	)
	Plots.plot!(
		nyc_building_centroids[1:1000],
		markersize=0.5,
		color="white"
	)
	Plots.plot!(
		epw_local.geometry,
		markerstrokewidth=0,
		color="indianred"
	)
end

# ╔═╡ e0abb89d-4e46-4771-8353-92f8e59dfbbc
savefig(weather_stations_map, joinpath(output_dir, "weather_stations_map.png"))

# ╔═╡ dc40e244-bdda-482c-8a57-09c24d5b9358
md"""
mapping dictionary between epw files and location ids
"""

# ╔═╡ cd5ec182-b189-43ff-81b7-04a92f525678
begin
epw_mapping_dict = select(epw_geo, ["Location ID", "filename"])
epw_mapping = Dict(zip(epw_mapping_dict[:, "filename"], epw_mapping_dict[:, "Location ID"]))
end

# ╔═╡ 51381ab9-d91d-4142-ab6c-43140375fd03
md"""
now want to make a large dataframe with all of the local weather files found
"""

# ╔═╡ 0af09aca-8ea7-46c2-a262-795639c295fb
epw_localfilenames = unique(epw_local.filename)

# ╔═╡ 16275ca0-17ab-42e7-b242-a6a804cd8e4c
begin
epw_dataframelist = []
	
for epw_file in epw_localfilenames
	full_epwfilename = joinpath(epw_dir, epw_file)
	epw_dataframe = EnergyPlusWeather.read(full_epwfilename)
	epw_dataframe[!,"weather_station_id"] = repeat([epw_mapping[epw_file]], nrow(epw_dataframe))
	
	push!(epw_dataframelist, epw_dataframe)
end
# epw_dataframelist[end]
epw_dataframe = vcat(epw_dataframelist...)
epw_dataframe.month = month.(epw_dataframe.Date)
epw_dataframe.day = day.(epw_dataframe.Date)
select!(epw_dataframe, Not(:Date))
end

# ╔═╡ e6c80d02-1841-4598-bb38-bdb7a06e11ab
md"""
### Distances
at this point, now want to find the pairwise distances between each building and the weather stations. Then we want to choose the smallest distance and grab the index of the weather station
"""

# ╔═╡ a68eb98c-3991-452a-b4b8-9564a4890e6a
begin
	local_lon = ArchGDAL.getx.(epw_local.geometry, 0)
	local_lat = ArchGDAL.gety.(epw_local.geometry, 0)
	epw_points = hcat(local_lon, local_lat)
end;

# ╔═╡ e00c4191-6ab4-4144-a877-42834d4bb199
md"""
so we have a number of weather locations
"""

# ╔═╡ b68f0739-3b62-4586-b50f-4654deafce26
size(epw_points)

# ╔═╡ 9d7307e5-af20-44fd-bf34-793594f65bd6
md"""
and we have a big list of buildings
"""

# ╔═╡ 56d1d77d-d1ca-448a-b5b3-0ff83a1c1141
size(building_locations)

# ╔═╡ cabaa3df-3eeb-4982-bd4e-d5b26278302a
building_weather_distances = pairwise(
	Euclidean(),
	epw_points, 
	building_locations,
	dims=1
);

# ╔═╡ d0678256-9f5d-4c1f-bcb3-33f50e9b9f7c
building_weather_distances

# ╔═╡ 95105b4c-64a5-481e-893b-e915bd46fc1f
min_dist, min_index = findmin(building_weather_distances, dims=1)

# ╔═╡ e0dda8c5-fffe-4469-9a60-9a5596e21e3a
begin
	Plots.plot(
		nyc_council_counts.geometry,
		color="transparent",
		title="Weather Stations",
		dpi=500
	)
	Plots.plot!(
		epw_local[unique(map( x -> x[1], min_index )), :].geometry,
		color="red"
	)
end

# ╔═╡ a50cd1a7-4c17-4b71-91e8-82c3b8acac2f
unique(map( x -> x[1], min_index ))

# ╔═╡ 135cd14a-1e9f-4837-a159-a569ac676b0b
min_distances = min_dist[1,:]

# ╔═╡ c705fcf9-369c-44da-852b-4ec457f0d5b4
maximum(min_distances)

# ╔═╡ b993865b-cc70-46a0-a87c-8ec56fd58e2a
minimum(min_distances)

# ╔═╡ de137895-c4f4-4233-a4f2-935be5177d85
building_weatherstation_histogram = Plots.histogram(
	min_distances, 
	color="transparent", 
	linewidth=1.2,
	dpi=600,
	bins=20, 
	title="Distance to Nearest Weather Station",
	xlabel="Distance - meters"
)

# ╔═╡ 92fc5411-985c-4fc0-90b5-8ecf0ab29ebd
savefig(building_weatherstation_histogram, joinpath(output_dir, "building_weatherstation_histogram.png"))

# ╔═╡ 8bcfc861-0479-4763-a455-f09f4df09186
min_station_idx = map( x -> x[1], min_index );

# ╔═╡ 45abdf93-87e4-455d-8500-55adb51c9c0e
min_station_ids = Matrix(select(epw_local, "Location ID"))[min_station_idx][1,:];

# ╔═╡ f5db610a-7704-459b-9fb0-ec22c7dbdaee
begin
	nyc_building_points_data[!, "weather_station_id"] = min_station_ids;
	nyc_building_points_data[!, "weather_station_distance"] = min_distances
end;

# ╔═╡ 30b035ae-4e82-4607-afc3-445a7bf96d97
nyc_building_points_data

# ╔═╡ 9145880b-2b40-49d2-815c-bd8cf7e090c2
#countmap(nyc_building_points_data.weather_station_id)

# ╔═╡ 9f1b44e9-5106-4747-bdbe-84dcda3d9717
#log.(nyc_building_points_data.weather_station_distance)

# ╔═╡ 6085ae4a-6874-4b38-91b3-5b4b60f3806f
distance_colors = cgrad(:matter)

# ╔═╡ 64fefeee-7980-41c2-8a2f-2e539da260a4
distance_colormap = distance_colors[standardize(
	UnitRangeTransform, 
	nyc_building_points_data.weather_station_distance
)]

# ╔═╡ 2626b03d-e2e8-405e-84a4-ccb5d8432e5c
begin
	unique_weatherstations = unique(nyc_building_points_data.weather_station_id)
	unique_weathermap = Dict(zip(
		unique_weatherstations,
		collect(1:length(unique_weatherstations))
	))
end

# ╔═╡ 2f0caff4-5cf0-4c15-a143-9e8ef52a63f0
weather_colorscheme = cgrad(:matter, length(unique_weatherstations), categorical=true)

# ╔═╡ c29ca84d-fa67-48c1-89c3-347b4ed2de24
begin
	weather_colors = weather_colorscheme[map( x -> unique_weathermap[x], nyc_building_points_data.weather_station_id )]
end

# ╔═╡ 0cb33287-09da-47ff-9839-ed7d6b4da7e7
begin
	random_buildings_idx = sample(1:length(nyc_building_centroids), 500)
	random_buildings = nyc_building_centroids[random_buildings_idx]
	random_colors = distance_colormap[random_buildings_idx]
	random_weather_colors = weather_colors[random_buildings_idx]
end;

# ╔═╡ 805be959-76eb-4d96-aed4-b9344af198b0
md"""
##### This function is meant to show how the buildings are getting mapped to the closest weather station
"""

# ╔═╡ efbb87f4-9ecb-411e-a380-61897a89f178
begin
	building_membership_plot = Plots.plot(
		random_buildings,
		color_palette = random_weather_colors,
		markersize=3,
		markerstrokewidth=0,
		dpi=500,
		size=(700,500)
	)
	Plots.plot!(
		nyc_council_counts.geometry,
		color="transparent"
	)
	Plots.plot!(
		epw_local.geometry,
		color="white", 
		markersize=8,
		markerstrokewidth=2
	)
end

# ╔═╡ 3c4e9f90-ebbb-40f9-a30a-c9d900af6bb7
savefig(building_membership_plot, joinpath(output_dir, "building_station_membership.png"))

# ╔═╡ bcf9c256-c6e7-45a2-b76d-12866bbb038b
md"""
##### Or we could instead visualize the distances between the buildings and their closest weather station
"""

# ╔═╡ 9f5a9366-f35a-4639-a141-4341f7ac41f9
begin
	building_distance_plot = Plots.plot(
		random_buildings,
		color_palette = random_colors,
		markersize=3,
		markerstrokewidth=0,
		dpi=500,
		size=(700,500)
	)
	Plots.plot!(
		nyc_council_counts.geometry,
		color="transparent"
	)
	Plots.plot!(
		epw_local.geometry,
		color="white", 
		markersize=8,
		markerstrokewidth=2
	)
end

# ╔═╡ aabde700-9b51-4933-96a2-5a6d37ccddab
savefig(building_distance_plot, joinpath(output_dir, "building_distances.png")) # save the fig referenced by plot_ref as filename_string (such as "output.png")

# ╔═╡ 8928439e-6d4b-4c65-8edb-0e329f86d4ec
# now going to try and map the weather station indicator
@info "Weather Station Closest Counts: " countmap(nyc_building_points_data.weather_station_id)

# ╔═╡ ff729c37-8b94-4f55-bacb-d2d5d017a8b8
id_stationmap = Dict(zip(nyc_building_points_data[:,"Property Id"], nyc_building_points_data[:,"weather_station_id"]))

# ╔═╡ d5319700-fc6b-42e9-bfcf-166e6d162da8
nyc_property_weathermap = select(nyc_building_points_data, ["Property Id", "weather_station_id"])

# ╔═╡ 2252a358-a59b-41e6-b1ab-37054987585c
begin
nyc_monthly_weathermap = leftjoin(
	nyc_monthly_clean,
	nyc_property_weathermap,
	on="Property Id"
)
# nyc_monthly_weathermap.month = month.(nyc_monthly_weathermap.date)
# nyc_monthly_weathermap.day = day.(nyc_monthly_weathermap.date)
# select!(nyc_monthly_weathermap, ["Property Id", "weather_station_id", "month", "day"])
end

# ╔═╡ d993a9a6-107c-40f4-a9f7-2672df4d4e2e
epw_average_dataframe = combine(groupby(epw_dataframe, [:month, :day, :weather_station_id]), names(epw_dataframe, Real) .=> mean, renamecols=false)

# ╔═╡ deac9195-7ffa-4fad-975b-89a05c3997b4
md"""
one final complication - we want to duplicate the epw files for each year of the data we have
"""

# ╔═╡ 653f9d45-6845-4251-bdc9-27ca09ba73e5
min_date = minimum(nyc_monthly_weathermap.date)

# ╔═╡ 5400d8d1-a8d9-47a1-b580-1b3eb2f3af48
max_date = maximum(nyc_monthly_weathermap.date)

# ╔═╡ 8b1a9a22-1258-42da-a3ba-73def9c7d74f
desired_daterange = collect(min_date:Dates.Day(1):max_date)

# ╔═╡ cd34c405-7f39-45ca-87af-12c90e611ff0
custom_datedf = DataFrame(
	date = desired_daterange,
	month = Dates.month.(desired_daterange),
	day = Dates.day.(desired_daterange)
)

# ╔═╡ f4a4c3a9-8ce3-4b08-ab16-484ca96c5037
epw_dataframe_expanded = select(leftjoin(
	epw_average_dataframe,
	custom_datedf,
	on=["month", "day"]
), Not(["month","day"]))

# ╔═╡ 76992d99-5910-4a63-be30-0f7005e74f67
# so this is the goal of what I want to get
# epw_dataframe_expanded

# ╔═╡ ca9f2fa1-8611-4a08-809c-2247bb7cdd29
unique_propertymap = dropmissing(unique(select(nyc_monthly_weathermap, ["Property Id", "weather_station_id"])))

# ╔═╡ 3f64f1cf-243c-4b56-9a5c-ccb741ee967b
nyc_epw_data = select(dropmissing(leftjoin(
	epw_dataframe_expanded,
	unique_propertymap,
	on="weather_station_id"
), ["Property Id"]), ["Property Id","date"], Not("weather_station_id"))

# ╔═╡ d733f31c-b6c6-4a6d-8b80-c0ea2150b8b3
# nyc_epw_data = select(leftjoin(
# 	dropmissing(nyc_monthly_weathermap, ["weather_station_id"]),
# 	epw_dataframe_expanded,
# 	on=["weather_station_id", "date"]
# ), Not([:weather_station_id, :electricity_kbtu, :naturalgas_kbtu, :electricity_mwh, :naturalgas_mwh]))

# ╔═╡ 9a7b7072-0b92-455d-978c-85bd6f8e052f
# filter( x -> x["Property Id"] == 1416310, nyc_epw_data )

# ╔═╡ 2386de40-02d1-4244-a2f5-9a290ff81858
CSV.write(
	joinpath(output_dir, "epw.csv"),
	nyc_epw_data
)

# ╔═╡ 5379b543-b900-4ffb-a7b9-24107d2c74ef
# saving the local epw dataframe to build a data map later
CSV.write(
	joinpath(output_dir, "epw_local.csv"),
	select(epw_local, Not([:geometry, :Source]))
)

# ╔═╡ 9eaefae4-4b09-4738-9e6e-cccd9eefc512
begin
	# this cell now transports the epws for use in the next stage
	local_epws = joinpath(output_dir, "local-epws")
	mkpath(local_epws)
	
	for epw in epw_local.filename
		existing_epw = joinpath(epw_dir, epw)
		cp(existing_epw, joinpath(local_epws, epw), force=true)
	end
end

# ╔═╡ ee1aaec5-0a92-4d34-a853-39346ae107a5


# ╔═╡ 656232a8-af8d-4086-b934-606005fd1359
begin
GeoDataFrames.write(
	joinpath(output_dir, "buildings.geojson"),
	nyc_building_points_data
)
end;

# ╔═╡ 6bb085a2-ab02-4103-9a64-1ada10c8726b
begin
	nyc_buildings_simple = deepcopy(nyc_building_points_data)
	nyc_buildings_simple[!,"geometry"] = ArchGDAL.boundingbox.(nyc_building_points_data.geometry)
	
	GeoDataFrames.write(
		joinpath(output_dir, "buildings_bbox.geojson"), 
		nyc_buildings_simple
	)
end

# ╔═╡ 2abcf02e-1939-43a2-b9db-d6b7aed79bcb
md"""
Here is where the buildings get segmented away
"""

# ╔═╡ c451dc76-cc1f-4ed6-992f-1dda55ab05c5
nyc_data_stripped = select(nyc_building_points_data, Not(:geometry));

# ╔═╡ 09c9e54a-4045-4748-8e5f-4a8d96818fcf
nyc_train_validate_buildings = filter(
	row -> row.council_region ∉ candidate_councils, 
	nyc_data_stripped
);

# ╔═╡ ae3313aa-0708-4bd5-84c9-d1f46af60688
validation_buildings_idx = sample(
	rng,
	1:nrow(nyc_train_validate_buildings), 
	Int(floor(nrow(nyc_train_validate_buildings) / 10)),
	replace=false
)

# ╔═╡ 6355aa82-05a5-4444-9b28-eea13ea76802
training_buildings_idx = filter( 
	x -> x ∉ validation_buildings_idx, 
	1:nrow(nyc_train_validate_buildings)
)

# ╔═╡ 9cfc624e-2277-42f5-88e2-144ae23e178c
length(validation_buildings_idx)

# ╔═╡ db856bad-d666-4972-ac42-af0a5be857ff
length(unique(validation_buildings_idx))

# ╔═╡ 5445f923-6b7b-4ec2-9d6d-bff64a2d980e
begin
training_buildings = nyc_train_validate_buildings[training_buildings_idx, :]
validation_buildings = nyc_train_validate_buildings[validation_buildings_idx, :]
test_buildings = filter(
	row -> row.council_region ∈ test_districts, 
	nyc_data_stripped
)

first(validation_buildings, 3)
end

# ╔═╡ d740b9b2-c261-4816-bfb1-1eb6c8b9a845
first(nyc_building_points_data, 3)

# ╔═╡ 0b1b09b5-d55c-48c1-a876-0783766476cf
begin
	CSV.write(
		joinpath(output_dir, "train_buildings.csv"),
		training_buildings
	);
	CSV.write(
		joinpath(output_dir, "validate_buildings.csv"), 
		validation_buildings
	);
	CSV.write(
		joinpath(output_dir, "test_buildings.csv"), 
		test_buildings
	);
end;

# ╔═╡ a9a4a2a5-2495-4a1e-a551-fd573b0d557b
energy_terms = select(
	nyc_monthly_clean,
	["electricity_mwh","naturalgas_mwh"]
);

# ╔═╡ 99947700-33b5-46fa-8d6a-b9689c6c7ee1
for col in names(energy_terms)
   energy_terms[!,col] = Missings.coalesce.(energy_terms[:,col], 0)
end

# ╔═╡ 21eeb33a-5892-46b7-b1cb-9c19b5d58856
# nyc_monthly_clean.energy_mwh = energy_terms.electricity_mwh .+ energy_terms.naturalgas_mwh;

# ╔═╡ 34c7f174-eb11-4cae-8741-879efe9f36c0
nyc_monthly_clean

# ╔═╡ 55921f40-4103-4606-98cf-287ca8fb9a19
begin
	nyc_train = select(
		leftjoin(training_buildings, nyc_monthly_clean, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_train, :date)
	
	nyc_validate = select(
		leftjoin(validation_buildings, nyc_monthly_clean, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_validate, :date)
	
	nyc_test = select(
		leftjoin(test_buildings, nyc_monthly_clean, on="Property Id"),
		"Property Id", "date", :)
	dropmissing!(nyc_test, :date)
end;

# ╔═╡ b520c694-b9f3-4fa5-9ce8-3f0c04e221e4
nyc_monthly_clean

# ╔═╡ 022a9d72-68fb-4764-adbb-9cbe1a6ab44b
# training percentage
@info "Training Data Size: " nrow(nyc_train)

# ╔═╡ d60c9e77-2a3e-4975-a38c-2102af2b8706
# validation percentage
@info "Validation Data Size: " nrow(nyc_validate)

# ╔═╡ af43f162-0f5c-4c0e-ab8d-c6c2032af565
# validation percentage
@info "Test Data Size: " nrow(nyc_test)

# ╔═╡ 371ee554-2386-45b7-a09b-09b8772cb377
md"""
##### Want to strip the geometries from the objects here and just store it as a csv to improve operation with other data types
"""

# ╔═╡ 4c013c95-cf5f-4d59-b268-6a1aac7c42aa
nyc_train

# ╔═╡ ed6ed540-885a-41e1-9348-27826b74d194
begin
	CSV.write(joinpath(output_dir, "train.csv"), nyc_train);
	CSV.write(joinpath(output_dir, "validate.csv"), nyc_validate);
	CSV.write(joinpath(output_dir, "test.csv"), nyc_test);
end;

# ╔═╡ Cell order:
# ╠═1ea68209-380d-4b2e-9239-bedf850b8243
# ╠═7488a1a4-2b76-48f7-80bb-3b441631068a
# ╠═600556af-f7b6-4dfe-a10a-ed80ee98ef25
# ╠═f8e0a746-8b03-4f23-820b-f8cb66bc00bb
# ╠═a5017265-097a-4be3-926b-54d0044a6b37
# ╠═70bc6e0e-fecb-4506-a810-2777d67f73a9
# ╟─2de15f2f-cfc8-4416-9cde-a5e63a3b7d95
# ╠═7f162842-c24a-4fdf-866d-eb4df50e8d23
# ╟─4ce9f094-273d-4497-9255-202726b00c11
# ╠═c704089f-5a31-403f-89a1-352e90991d72
# ╠═f6ba219d-9c7c-4947-a3fd-5a32e26e89a8
# ╠═d1ea0a90-7830-4d3f-8fb1-18cf88f3293c
# ╟─f65a1dea-039e-4705-ae78-10a385f85b6b
# ╟─8db8b2ca-ee25-44a8-91d2-26b7c7fb8c0d
# ╟─f2013cd8-5ce1-47f0-8700-a0550612f943
# ╠═e0eae347-e275-4737-b429-4dd2b04cb27c
# ╠═6f6159a2-7910-4483-807b-aeed118f4242
# ╠═0fb0caa1-d988-4920-9f89-8c9431a22371
# ╠═dc2196ab-a277-4083-a7b6-4bbe90df3c53
# ╠═27bb969d-c79f-4d8d-b780-e672b51138a8
# ╠═8ad4c349-d342-45de-9d44-dc2a40f73baf
# ╠═059b124f-9a9d-4424-a92b-24853aa481af
# ╠═01178082-f94d-4cce-a491-f71a5e46355d
# ╠═a12127c9-3a2d-46db-b01d-b1887390d353
# ╠═8a36a580-9d4f-4b41-a28d-d02ca7fd34d5
# ╠═b16c61cd-3ad5-4806-8381-f8e43fcb447d
# ╠═3164491c-df20-4384-8045-e270d82317d7
# ╠═6ceeb173-c9b4-4daa-be6b-0097c926885c
# ╠═fa6b3c0c-8c48-4b99-85b9-e49b9f7993db
# ╟─4197d920-ef37-4284-970a-7fd4331ad9b7
# ╠═6fc43b1e-bdd6-44dd-9940-9d96e25a285d
# ╠═66cd75cc-161c-4e85-8d40-ab01547443e1
# ╠═bea53833-7f47-4b4b-afc2-a0bcc9159926
# ╠═ccf8cb9f-88a4-4f41-a635-d02dc479e50d
# ╠═0f3a58c2-ae4b-4c89-9791-803cb89ab51e
# ╠═78ace92e-81de-47e7-a19b-41a0c5f35bb4
# ╠═64f587cd-aac9-47c9-a0ff-762b502f8adb
# ╠═bbc1d2c3-393d-4fb9-a039-d54ae414cf16
# ╠═8569cd2f-8b58-4797-ad35-42fbb831a15c
# ╠═89bb5aae-2139-4a4a-8e11-b5cca12cb4b5
# ╠═40b8821e-0c9a-463a-af9a-bd4cfd797011
# ╠═211c6ff2-9308-494a-9897-d969123fe873
# ╠═69c8b562-1ed2-4c6e-9b60-44a71ac62ed0
# ╟─5c1383bb-50ad-43d9-ac99-2e2b14d41f33
# ╟─9a579e9d-8de1-4b82-9508-a378f08e2955
# ╠═01df78ab-aafd-462c-b5a1-1561fb890acb
# ╟─34db351a-d5cd-4069-add3-9aa3b4162585
# ╠═6d6a3a4e-ac6f-41e5-8e8c-78f45eb4f618
# ╟─e54c18df-46fe-4425-bdfc-ca8c529d436a
# ╠═009bc1d3-6909-4edf-a615-2b1c7bdb1ce8
# ╟─a7e02d02-158f-44d5-8d2e-28d5aba1cca5
# ╟─61c4ad16-99ef-4d21-a4dd-cae897e18659
# ╟─68c9d819-d948-4bcf-a625-64f2bdc26b62
# ╠═2dea8532-505a-4b0e-8b98-966a47808804
# ╠═07aab43a-697f-4d6a-a8c4-4415116368a7
# ╠═878e4be3-1a1a-4f73-8823-fb69fd756437
# ╠═27c79591-5b49-4d41-97a5-2eb8b2780e1d
# ╠═d9e0a62c-0528-41e8-9c6d-aa9629ca79c3
# ╠═5478f86e-e8a1-4290-8255-3b18a8e36ff5
# ╠═95dcee91-24ac-4495-a4e1-355590382c37
# ╠═daf1e9a4-6fc0-4b27-b468-44c66ed40940
# ╠═f378506c-50fb-430d-a26f-2ee99c77dd8d
# ╠═140fea64-b020-461c-88e1-d46cfa78571d
# ╠═941b6f4a-1359-4084-859a-07b3b3fcc7a2
# ╠═98e5f297-585b-4d8c-96e3-59b4b7c52d1e
# ╠═45334b27-e117-4e76-bf15-44271775743c
# ╠═fa11a171-904f-424b-bb42-25f664fe13e6
# ╠═01a6eaca-1294-4487-a224-6c59c1de7899
# ╠═eb4e2971-8fa4-4d6a-a175-6554c3122a41
# ╠═35d7d6d9-34bb-430a-9c74-001b446c9d09
# ╠═691a9409-ab03-4843-9743-5685c563af81
# ╠═e7f3b286-fe7e-48ce-a01a-25e5e3743267
# ╠═5f548a26-a95e-4bc7-9dad-7bdefa41611c
# ╠═9c877f38-ecf8-4acd-9849-200fb01bce00
# ╠═353f0519-69a8-484d-829d-368140ccb545
# ╠═2da9205e-4325-4bf0-943e-10f8d4dd2b86
# ╠═734c7a3b-10d5-4627-ae4d-1f787509e838
# ╟─70a44334-ae15-4225-9f2e-668f6ee2b965
# ╠═7e0ecfd3-cdea-4470-a799-acbebc67e7dd
# ╠═f66ee36c-b0c5-4536-ad9d-214c13b3a984
# ╠═a465b08a-26f6-4e7f-a189-df7d76395f80
# ╠═bab66f4b-1d1e-46ad-ae79-1c4f5fae6273
# ╠═b66daacc-515e-4137-b1ed-36186dc3fda4
# ╠═0bfc6dec-97dd-4a7c-8af5-382068cddbc7
# ╠═25b4bc79-2d09-4e44-a75e-144c972db3ba
# ╠═f69cb982-573f-43c4-8fb0-a1aab6028021
# ╟─c97e405d-8601-4634-9a7c-654598dd1200
# ╟─ab355e93-a064-40e2-a67a-e75cb54efb33
# ╠═8e462588-c0c1-4e96-a1fe-f2df50a8ec37
# ╟─14f9ac4d-e705-41fd-8561-4d1421e41126
# ╟─2d956f5a-7f40-4a7d-b491-f3de3340cb66
# ╟─630cbcdf-31c6-4a96-9c36-8fea5d64e2b9
# ╟─4e00b820-9f99-4134-afd7-2710424fe28e
# ╠═c4b107f9-421b-4c5e-b876-10b9efb4d4c6
# ╟─11fd74bd-4ebc-4537-bbd9-958761ac7b01
# ╟─7df29d94-63b7-46dd-8872-a8dbe0e81e38
# ╟─5a35d191-76e7-447e-841d-881b47ab8d6c
# ╠═0c000a60-6fe3-464a-a8b6-906e1e94f02d
# ╠═329d43de-17b6-4d11-a0c8-4cd0a6980932
# ╠═9cfc27da-38b5-40cd-b41f-a1d2687011a8
# ╠═69cd9bb6-01d7-447d-afbc-b349d01d9fbf
# ╟─a85a3658-e1b5-4c57-a275-26a69dce572a
# ╟─2520a4ec-cebd-4f27-8e21-5a5f9b55c8a9
# ╠═f6e0da2e-1394-4945-aee7-67de4ade486c
# ╠═39fc72af-2262-4597-9c44-c9ccd56b6ec9
# ╠═3bd5037f-87cf-4988-888f-d62c10de3195
# ╟─b4d4e254-5198-4b0f-b9e8-054fe97f7f3a
# ╠═a88489a2-2da2-492e-94d6-7af90feaf92f
# ╠═6cabf58a-7082-4a18-9dd3-b9334f2ccd9f
# ╠═edd15ef6-80ca-496d-9073-49a430b9d2ed
# ╠═621d236a-ce0a-4bf5-9486-d481527d3261
# ╠═abec0385-4725-4a51-8ef4-bda7cfaede64
# ╠═ca10ae5a-7a94-42eb-97e0-770e7f781a4c
# ╠═28a46b6f-ae61-4639-8448-699451ea228c
# ╠═c92d5491-a044-4943-835c-cc2ecc418208
# ╠═052750a1-4941-47b8-9b54-518a4bc6cb67
# ╠═92eab0d2-f601-4f9c-b433-e2285be87ed2
# ╟─14b73aa8-cdc6-40c9-ac38-39039f19dae7
# ╠═a1cd7408-c92b-432c-8830-a7d709d7c298
# ╟─3e5adcf1-0815-41b6-bd06-113cbd35072c
# ╠═7e1470f0-d52d-48cb-a62a-07acb02e18a5
# ╠═5c352c0c-0775-446d-b785-e7d9d2677c41
# ╠═c6e1f5ac-55a0-4175-b05b-236a2eb68561
# ╟─c7fcad00-fa50-4a3f-ab13-8ac392b7abe7
# ╠═643e0bbb-89e5-45a5-bcca-991cfc66e85d
# ╠═1b7432eb-bc8d-4bd1-a19c-06ca71c29786
# ╠═e0abb89d-4e46-4771-8353-92f8e59dfbbc
# ╟─dc40e244-bdda-482c-8a57-09c24d5b9358
# ╟─cd5ec182-b189-43ff-81b7-04a92f525678
# ╟─51381ab9-d91d-4142-ab6c-43140375fd03
# ╠═0af09aca-8ea7-46c2-a262-795639c295fb
# ╠═16275ca0-17ab-42e7-b242-a6a804cd8e4c
# ╟─e6c80d02-1841-4598-bb38-bdb7a06e11ab
# ╠═f758a6da-9531-4d37-b22b-6902d2466b9b
# ╠═a68eb98c-3991-452a-b4b8-9564a4890e6a
# ╟─e00c4191-6ab4-4144-a877-42834d4bb199
# ╠═b68f0739-3b62-4586-b50f-4654deafce26
# ╟─9d7307e5-af20-44fd-bf34-793594f65bd6
# ╠═56d1d77d-d1ca-448a-b5b3-0ff83a1c1141
# ╠═cabaa3df-3eeb-4982-bd4e-d5b26278302a
# ╠═d0678256-9f5d-4c1f-bcb3-33f50e9b9f7c
# ╠═95105b4c-64a5-481e-893b-e915bd46fc1f
# ╟─e0dda8c5-fffe-4469-9a60-9a5596e21e3a
# ╠═a50cd1a7-4c17-4b71-91e8-82c3b8acac2f
# ╠═135cd14a-1e9f-4837-a159-a569ac676b0b
# ╠═c705fcf9-369c-44da-852b-4ec457f0d5b4
# ╠═b993865b-cc70-46a0-a87c-8ec56fd58e2a
# ╠═de137895-c4f4-4233-a4f2-935be5177d85
# ╠═92fc5411-985c-4fc0-90b5-8ecf0ab29ebd
# ╠═8bcfc861-0479-4763-a455-f09f4df09186
# ╠═45abdf93-87e4-455d-8500-55adb51c9c0e
# ╠═f5db610a-7704-459b-9fb0-ec22c7dbdaee
# ╠═30b035ae-4e82-4607-afc3-445a7bf96d97
# ╠═9145880b-2b40-49d2-815c-bd8cf7e090c2
# ╠═9f1b44e9-5106-4747-bdbe-84dcda3d9717
# ╠═6085ae4a-6874-4b38-91b3-5b4b60f3806f
# ╠═64fefeee-7980-41c2-8a2f-2e539da260a4
# ╠═2626b03d-e2e8-405e-84a4-ccb5d8432e5c
# ╠═2f0caff4-5cf0-4c15-a143-9e8ef52a63f0
# ╠═c29ca84d-fa67-48c1-89c3-347b4ed2de24
# ╠═0cb33287-09da-47ff-9839-ed7d6b4da7e7
# ╟─805be959-76eb-4d96-aed4-b9344af198b0
# ╠═efbb87f4-9ecb-411e-a380-61897a89f178
# ╠═3c4e9f90-ebbb-40f9-a30a-c9d900af6bb7
# ╟─bcf9c256-c6e7-45a2-b76d-12866bbb038b
# ╠═9f5a9366-f35a-4639-a141-4341f7ac41f9
# ╠═aabde700-9b51-4933-96a2-5a6d37ccddab
# ╠═8928439e-6d4b-4c65-8edb-0e329f86d4ec
# ╠═ff729c37-8b94-4f55-bacb-d2d5d017a8b8
# ╠═d5319700-fc6b-42e9-bfcf-166e6d162da8
# ╠═2252a358-a59b-41e6-b1ab-37054987585c
# ╠═d993a9a6-107c-40f4-a9f7-2672df4d4e2e
# ╟─deac9195-7ffa-4fad-975b-89a05c3997b4
# ╠═653f9d45-6845-4251-bdc9-27ca09ba73e5
# ╠═5400d8d1-a8d9-47a1-b580-1b3eb2f3af48
# ╠═8b1a9a22-1258-42da-a3ba-73def9c7d74f
# ╠═cd34c405-7f39-45ca-87af-12c90e611ff0
# ╠═f4a4c3a9-8ce3-4b08-ab16-484ca96c5037
# ╠═76992d99-5910-4a63-be30-0f7005e74f67
# ╠═ca9f2fa1-8611-4a08-809c-2247bb7cdd29
# ╠═3f64f1cf-243c-4b56-9a5c-ccb741ee967b
# ╠═d733f31c-b6c6-4a6d-8b80-c0ea2150b8b3
# ╠═9a7b7072-0b92-455d-978c-85bd6f8e052f
# ╠═2386de40-02d1-4244-a2f5-9a290ff81858
# ╠═5379b543-b900-4ffb-a7b9-24107d2c74ef
# ╠═9eaefae4-4b09-4738-9e6e-cccd9eefc512
# ╠═ee1aaec5-0a92-4d34-a853-39346ae107a5
# ╠═656232a8-af8d-4086-b934-606005fd1359
# ╠═6bb085a2-ab02-4103-9a64-1ada10c8726b
# ╠═2abcf02e-1939-43a2-b9db-d6b7aed79bcb
# ╠═c451dc76-cc1f-4ed6-992f-1dda55ab05c5
# ╠═09c9e54a-4045-4748-8e5f-4a8d96818fcf
# ╠═ae3313aa-0708-4bd5-84c9-d1f46af60688
# ╠═6355aa82-05a5-4444-9b28-eea13ea76802
# ╠═9cfc624e-2277-42f5-88e2-144ae23e178c
# ╠═db856bad-d666-4972-ac42-af0a5be857ff
# ╠═5445f923-6b7b-4ec2-9d6d-bff64a2d980e
# ╠═d740b9b2-c261-4816-bfb1-1eb6c8b9a845
# ╠═0b1b09b5-d55c-48c1-a876-0783766476cf
# ╠═a9a4a2a5-2495-4a1e-a551-fd573b0d557b
# ╠═99947700-33b5-46fa-8d6a-b9689c6c7ee1
# ╠═21eeb33a-5892-46b7-b1cb-9c19b5d58856
# ╠═34c7f174-eb11-4cae-8741-879efe9f36c0
# ╠═55921f40-4103-4606-98cf-287ca8fb9a19
# ╠═b520c694-b9f3-4fa5-9ce8-3f0c04e221e4
# ╠═022a9d72-68fb-4764-adbb-9cbe1a6ab44b
# ╠═d60c9e77-2a3e-4975-a38c-2102af2b8706
# ╠═af43f162-0f5c-4c0e-ab8d-c6c2032af565
# ╠═371ee554-2386-45b7-a09b-09b8772cb377
# ╠═4c013c95-cf5f-4d59-b268-6a1aac7c42aa
# ╠═ed6ed540-885a-41e1-9348-27826b74d194
